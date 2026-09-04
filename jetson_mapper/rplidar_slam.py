#!/usr/bin/env python3
"""2D SLAM from a single 360-degree RPLiDAR. Pure Python -- no ROS required.

Runs on the Jetson. Reads raw RPLiDAR bytes (serial, or a TCP stream from an
existing forwarder), builds an occupancy grid of the room, and serves the map
plus the live pose to the laptop viewer over TCP.

There is no IMU and no wheel odometry. The pose comes entirely from
scan-to-map matching: a coarse-to-fine correlative search over x/y/theta
against a blurred likelihood field, then a Gauss-Newton polish on the
bilinearly-interpolated grid (the Hector SLAM objective). Hector was designed
for exactly this case -- a hand-carried LiDAR with no odometry -- so a person
walking the sensor around a room is the intended usage, not a workaround.

    # On the Jetson, straight off the serial port:
    python3 rplidar_slam.py --source serial:/dev/ttyUSB0 --baud 460800

    # Or on the laptop, against the existing rplidar_server.py forwarder:
    python3 rplidar_slam.py --source tcp://192.168.55.1:9000

    # Record raw bytes now, tune the SLAM offline later (no hardware needed):
    python3 rplidar_slam.py --source serial:/dev/ttyUSB0 --record run1.bin
    python3 rplidar_slam.py --source file:run1.bin --replay-speed 4

Ctrl-C autosaves the map to --save (default room_map.npz + .png).

ROS 2 migration: SlamMapper.process_scan(angles_deg, ranges_m) is the whole
algorithm and touches nothing else. To make this a node, feed it from a
sensor_msgs/LaserScan callback and publish grid.to_ros_occupancy() as a
nav_msgs/OccupancyGrid. Nothing above that boundary needs to change.
"""

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
import zlib
from collections import deque

import numpy as np

# ---------------------------------------------------------------- constants

LOG_ODDS_MIN = -4.0
LOG_ODDS_MAX = 6.0

# Wire protocol to the laptop viewer.
WIRE_MAGIC = b'CNR1'
MSG_MAP = 1
MSG_STATE = 2
HDR = struct.Struct('<4sBI')
MAP_HDR = struct.Struct('<dddII')      # ox, oy, res, w, h
STATE_HDR = struct.Struct('<ddddI')    # t, x, y, theta, n_points

# RPLiDAR legacy SCAN response descriptor: A5 5A 05 00 00 40 81
SCAN_DESCRIPTOR = bytes.fromhex('a55a0500004081')
CMD_STOP = bytes([0xA5, 0x25])
CMD_SCAN = bytes([0xA5, 0x20])
CMD_RESET = bytes([0xA5, 0x40])


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def blur2d(a, kernel):
    """Separable convolution with a small 1-D kernel. Avoids a scipy dep."""
    pad = len(kernel) // 2
    out = np.zeros_like(a)
    ap = np.pad(a, ((0, 0), (pad, pad)), mode='constant')
    for i, w in enumerate(kernel):
        out += w * ap[:, i:i + a.shape[1]]
    tmp = out
    out = np.zeros_like(a)
    ap = np.pad(tmp, ((pad, pad), (0, 0)), mode='constant')
    for i, w in enumerate(kernel):
        out += w * ap[i:i + a.shape[0], :]
    return out


# ------------------------------------------------------------- byte sources

class SerialSource:
    """Raw bytes straight off the RPLiDAR's USB serial port."""

    def __init__(self, port, baud, reset=False, motor_dtr=False):
        import serial  # pyserial; only needed on this path
        self.ser = serial.Serial(port, baud, timeout=1.0)
        # DTR is deliberately left alone. On an A1 it gates the motor, but the
        # existing rplidar_server.py never touched it and the unit streamed
        # fine, so the line is already in the right state on this hardware.
        # Forcing it here would change a known-good setup. --motor-dtr opts in
        # if a different unit turns out to need it.
        if motor_dtr:
            try:
                self.ser.dtr = False
            except (OSError, IOError):
                pass
        self.ser.write(CMD_STOP)
        self.ser.flush()
        time.sleep(0.1)
        if reset:
            self.ser.write(CMD_RESET)
            self.ser.flush()
            time.sleep(0.8)          # unit emits a text banner after reset
        self.ser.reset_input_buffer()
        self.ser.write(CMD_SCAN)
        self.ser.flush()

    def read(self, n):
        return self.ser.read(n)

    def close(self):
        try:
            self.ser.write(CMD_STOP)
            self.ser.flush()
        except Exception:
            pass
        self.ser.close()


class TcpSource:
    """Raw RPLiDAR bytes forwarded over TCP (the existing rplidar_server.py)."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=10.0)
        self.sock.settimeout(5.0)

    def read(self, n):
        try:
            return self.sock.recv(n)
        except socket.timeout:
            return b''

    def close(self):
        self.sock.close()


class FileSource:
    """Replay a recorded raw byte stream, so SLAM can be tuned without hardware."""

    def __init__(self, path, speed=1.0):
        self.f = open(path, 'rb')
        self.speed = max(0.0, float(speed))

    def read(self, n):
        data = self.f.read(n)
        if not data:
            return b''
        if self.speed > 0:
            # ~8 kB/s of node data at 10 Hz; pace the replay roughly to real time.
            time.sleep(len(data) / (8000.0 * self.speed))
        return data

    def close(self):
        self.f.close()


def open_source(spec, baud, reset, replay_speed, motor_dtr=False):
    if spec.startswith('tcp://'):
        hostport = spec[6:]
        host, _, port = hostport.partition(':')
        return TcpSource(host, int(port or 9000))
    if spec.startswith('file:'):
        return FileSource(spec[5:], replay_speed)
    if spec.startswith('serial:'):
        return SerialSource(spec[7:], baud, reset, motor_dtr)
    return SerialSource(spec, baud, reset, motor_dtr)


# -------------------------------------------------------- RPLiDAR protocol

class RPLidarParser:
    """Legacy 5-byte measurement nodes -> complete 360-degree revolutions.

    Each node carries a start flag in bit 0 of byte 0, which is what actually
    delimits one revolution from the next. Slicing on a fixed point count
    instead drifts out of phase within seconds.
    """

    def __init__(self, min_quality=5, min_range=0.15, max_range=12.0):
        self.buf = bytearray()
        self.synced = False
        self.min_quality = min_quality
        self.min_range = min_range
        self.max_range = max_range
        self.cur_ang = []
        self.cur_rng = []
        self.dropped = 0

    def feed(self, data):
        """Push raw bytes in, get a list of (angles_deg, ranges_m) scans out."""
        self.buf.extend(data)
        scans = []

        if not self.synced:
            pos = self.buf.find(SCAN_DESCRIPTOR)
            if pos < 0:
                # Keep a tail in case the descriptor straddles two reads.
                if len(self.buf) > 4096:
                    del self.buf[:-len(SCAN_DESCRIPTOR)]
                return scans
            del self.buf[:pos + len(SCAN_DESCRIPTOR)]
            self.synced = True

        while True:
            n = len(self.buf) // 5
            if n == 0:
                break
            raw = np.frombuffer(bytes(self.buf[:n * 5]), dtype=np.uint8).reshape(n, 5)
            b0, b1 = raw[:, 0], raw[:, 1]
            # S and !S must disagree, and the check bit in byte 1 must be set.
            start = (b0 & 1).astype(bool)
            good = (start != ((b0 >> 1) & 1).astype(bool)) & ((b1 & 1) == 1)

            if not good.all():
                # Consume the valid prefix, then resync one byte at a time.
                bad = int(np.argmax(~good))
                if bad > 0:
                    self._emit(raw[:bad], start[:bad], scans)
                    del self.buf[:bad * 5]
                else:
                    del self.buf[:1]
                    self.dropped += 1
                continue

            self._emit(raw, start, scans)
            del self.buf[:n * 5]
            break

        return scans

    def _emit(self, raw, start, scans):
        b0 = raw[:, 0].astype(np.uint16)
        b1 = raw[:, 1].astype(np.uint16)
        b2 = raw[:, 2].astype(np.uint16)
        b3 = raw[:, 3].astype(np.uint16)
        b4 = raw[:, 4].astype(np.uint16)

        quality = (b0 >> 2).astype(np.int32)
        angle = (((b1 >> 1) | (b2 << 7)).astype(np.float32)) / 64.0
        rng = (((b3 | (b4 << 8)).astype(np.float32)) / 4000.0)  # q2 mm -> m

        valid = ((quality >= self.min_quality)
                 & (rng >= self.min_range)
                 & (rng <= self.max_range))

        # Split at every start flag; each split closes the current revolution.
        breaks = np.flatnonzero(start)
        lo = 0
        for b in list(breaks) + [len(raw)]:
            if b > lo:
                seg = valid[lo:b]
                if seg.any():
                    self.cur_ang.append(angle[lo:b][seg])
                    self.cur_rng.append(rng[lo:b][seg])
            if b < len(raw):
                self._close(scans)
            lo = b

    def _close(self, scans):
        if self.cur_ang:
            a = np.concatenate(self.cur_ang)
            r = np.concatenate(self.cur_rng)
            if a.size >= 40:
                scans.append((a, r))
        self.cur_ang = []
        self.cur_rng = []


# ------------------------------------------------------------ occupancy map

class OccGrid:
    """Log-odds occupancy grid that grows on demand as the room opens up."""

    def __init__(self, res, initial_size_m, l_free, l_occ):
        self.res = float(res)
        n = max(64, int(round(initial_size_m / self.res)))
        self.h = self.w = n
        # World coordinate of the lower-left corner of cell (0, 0).
        self.ox = -0.5 * self.w * self.res
        self.oy = -0.5 * self.h * self.res
        self.log_odds = np.zeros((self.h, self.w), dtype=np.float32)
        self.l_free = float(l_free)
        self.l_occ = float(l_occ)
        self._field = None
        self._dirty = True

    def world_to_cell(self, wx, wy):
        ix = np.floor((wx - self.ox) / self.res).astype(np.int32)
        iy = np.floor((wy - self.oy) / self.res).astype(np.int32)
        return ix, iy

    def in_bounds(self, ix, iy):
        return (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)

    def ensure_capacity(self, wx, wy, margin_m=2.0):
        """Grow the grid so every (wx, wy) sits at least margin_m inside it."""
        if wx.size == 0:
            return
        pad = int(math.ceil(max(margin_m, 4.0) / self.res))
        left = pad if float(np.min(wx)) < self.ox + margin_m else 0
        right = pad if float(np.max(wx)) > self.ox + self.w * self.res - margin_m else 0
        down = pad if float(np.min(wy)) < self.oy + margin_m else 0
        up = pad if float(np.max(wy)) > self.oy + self.h * self.res - margin_m else 0
        if not (left or right or down or up):
            return
        self.log_odds = np.pad(self.log_odds, ((down, up), (left, right)),
                               mode='constant', constant_values=0.0)
        self.h, self.w = self.log_odds.shape
        self.ox -= left * self.res
        self.oy -= down * self.res
        self._dirty = True

    def integrate(self, pose, pts):
        """Ray-cast one scan into the grid. pts is Nx2 in the sensor frame."""
        px, py, pth = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = math.cos(pth), math.sin(pth)
        wx = px + c * pts[:, 0] - s * pts[:, 1]
        wy = py + s * pts[:, 0] + c * pts[:, 1]
        self.ensure_capacity(np.append(wx, px), np.append(wy, py))

        ncell = self.w * self.h
        dx, dy = wx - px, wy - py
        dist = np.hypot(dx, dy)

        # Free space: sample every ray at grid resolution rather than running a
        # Python-level Bresenham, so the whole update stays vectorised.
        free = np.zeros(ncell, dtype=bool)
        good = dist > self.res
        if np.any(good):
            ux, uy = dx[good] / dist[good], dy[good] / dist[good]
            dg = dist[good]
            steps = int(math.ceil(float(np.max(dg)) / self.res))
            t = (np.arange(steps, dtype=np.float32) * self.res)[:, None]
            keep = t < (dg[None, :] - self.res)      # stop short of the hit cell
            ix, iy = self.world_to_cell(px + ux[None, :] * t, py + uy[None, :] * t)
            ok = keep & self.in_bounds(ix, iy)
            if np.any(ok):
                free[iy[ok].astype(np.int64) * self.w + ix[ok].astype(np.int64)] = True

        # Occupied endpoints. A cell hit by any beam is occupied regardless of
        # how many other beams passed through it this scan.
        hit = np.zeros(ncell, dtype=bool)
        ix, iy = self.world_to_cell(wx, wy)
        ok = self.in_bounds(ix, iy)
        if np.any(ok):
            hit[iy[ok].astype(np.int64) * self.w + ix[ok].astype(np.int64)] = True

        # One vote per cell per scan. Without this the cells around the sensor
        # take a free-space hit from every one of ~700 beams and saturate
        # instantly, so later evidence could never flip them back.
        delta = np.where(hit, self.l_occ, np.where(free, self.l_free, 0.0))
        self.log_odds += delta.reshape(self.h, self.w).astype(np.float32)
        np.clip(self.log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=self.log_odds)
        self._dirty = True

    def prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def likelihood_field(self):
        if self._dirty or self._field is None:
            occ = np.clip((self.prob() - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
            # Narrow kernel on purpose: a wide blur flattens the score surface
            # and the matcher stops being able to localise precisely.
            # Normalised so scores land in 0..1 and --min-score means something
            # independent of how dense the map has become.
            k = np.array([0.05, 0.5, 1.0, 0.5, 0.05], dtype=np.float32)
            self._field = blur2d(occ, k / k.sum())
            self._dirty = False
        return self._field

    def _bilinear(self, wx, wy, want_grad=False):
        """Interpolate the likelihood field, optionally with its gradient.

        Nearest-cell lookup would cap the matcher's precision at one cell and
        make the sub-centimetre refinement stage a no-op.
        """
        field = self.likelihood_field()
        fx = (wx - self.ox) / self.res - 0.5
        fy = (wy - self.oy) / self.res - 0.5
        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        tx = (fx - x0).astype(np.float32)
        ty = (fy - y0).astype(np.float32)
        ok = (x0 >= 0) & (x0 < self.w - 1) & (y0 >= 0) & (y0 < self.h - 1)
        np.clip(x0, 0, self.w - 2, out=x0)
        np.clip(y0, 0, self.h - 2, out=y0)
        x1, y1 = x0 + 1, y0 + 1
        p00 = field[y0, x0]
        p10 = field[y0, x1]
        p01 = field[y1, x0]
        p11 = field[y1, x1]
        val = (p00 * (1 - tx) * (1 - ty) + p10 * tx * (1 - ty)
               + p01 * (1 - tx) * ty + p11 * tx * ty)
        val = np.where(ok, val, 0.0)
        if not want_grad:
            return val, None, None
        gx = ((1 - ty) * (p10 - p00) + ty * (p11 - p01)) / self.res
        gy = ((1 - tx) * (p01 - p00) + tx * (p11 - p10)) / self.res
        return val, np.where(ok, gx, 0.0), np.where(ok, gy, 0.0)

    def score(self, poses, pts):
        """Mean likelihood-field value for each candidate pose. poses is Nx3."""
        c = np.cos(poses[:, 2])[:, None]
        s = np.sin(poses[:, 2])[:, None]
        wx = poses[:, 0:1] + c * pts[None, :, 0] - s * pts[None, :, 1]
        wy = poses[:, 1:2] + s * pts[None, :, 0] + c * pts[None, :, 1]
        val, _, _ = self._bilinear(wx, wy)
        return val.sum(axis=1) / max(1, pts.shape[0])

    def to_ros_occupancy(self):
        """int8 grid in the ROS convention: -1 unknown, 0..100 occupied."""
        p = self.prob()
        out = np.full(p.shape, -1, dtype=np.int8)
        known = np.abs(self.log_odds) > 1e-3
        out[known] = np.round(p[known] * 100.0).astype(np.int8)
        return out

    def to_uint8(self):
        """0 = free .. 127 = unknown .. 255 = occupied, for the wire and PNGs."""
        return np.round(self.prob() * 255.0).astype(np.uint8)


# --------------------------------------------------------------- the mapper

class SlamMapper:
    """Scan-to-map SLAM with no odometry and no IMU.

    process_scan() is the entire algorithm and has no I/O in it, which is what
    makes the ROS 2 port a thin wrapper rather than a rewrite.
    """

    def __init__(self, args):
        self.grid = OccGrid(args.map_res, args.map_size, args.l_free, args.l_occ)
        self.pose = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)   # body frame, per scan
        self.trajectory = [self.pose.copy()]
        self.args = args
        self.n_scans = 0
        self.n_keyframes = 0
        self.n_rejected = 0
        self.n_consec_reject = 0
        self.lost = False
        self.still_anchor = None
        self.last_kf_pose = np.zeros(3, dtype=np.float64)
        self.last_score = 0.0
        self.initialised = False

    # -- geometry helpers -------------------------------------------------
    @staticmethod
    def polar_to_xy(angles_deg, ranges_m):
        th = np.deg2rad(angles_deg.astype(np.float64))
        return np.stack([ranges_m * np.cos(th), ranges_m * np.sin(th)], axis=1)

    def deskew(self, pts):
        """Undo the smear from the sensor moving during one revolution.

        The head takes ~100 ms to sweep 360 degrees. Carried at walking pace
        that is ~10 cm of translation between the first beam and the last, which
        shows up as doubled or thickened walls. Points are re-expressed in the
        frame at the end of the sweep, assuming last frame's velocity holds.
        """
        n = pts.shape[0]
        if n < 2 or not self.initialised:
            return pts
        vx, vy, vth = self.velocity
        if abs(vx) < 1e-4 and abs(vy) < 1e-4 and abs(vth) < 1e-4:
            return pts
        f = (np.arange(n, dtype=np.float64) / (n - 1)) - 1.0   # -1 .. 0
        a = f * vth
        ca, sa = np.cos(a), np.sin(a)
        out = np.empty_like(pts)
        out[:, 0] = ca * pts[:, 0] - sa * pts[:, 1] + f * vx
        out[:, 1] = sa * pts[:, 0] + ca * pts[:, 1] + f * vy
        return out

    # -- matching ---------------------------------------------------------
    def match(self, pts, seed, widen=1.0):
        """Coarse-to-fine correlative search, then a Gauss-Newton polish.

        The correlative stage is what survives the jerky, unpredictable motion
        of a hand-carried sensor: it needs no gradient and no good initial
        guess. Gauss-Newton alone would fall into a local minimum on any sharp
        movement. The polish then buys back sub-cell precision that a discrete
        search can never reach.

        widen scales the coarse stage's span *and* step together, so a lost
        tracker can search a much bigger area at the same candidate count
        (same compute cost) instead of endlessly retrying the same tiny
        window around a stale pose. The fine stage is left alone -- it is
        only ever a local polish around whatever the coarse stage finds.
        """
        # Subsample for the brute-force stage; it costs poses x points.
        step = max(1, pts.shape[0] // self.args.match_points)
        sub = pts[::step]

        best = np.asarray(seed, dtype=np.float64).copy()
        best_score = 0.0
        # (linear half-span, linear step, angular half-span, angular step).
        # The step has to stay near the blur width of the likelihood field.
        # That peak is only ~1.5 cells wide, so a coarser search steps clean
        # over it and silently returns whichever wrong pose scored least badly.
        stages = ((self.args.search_lin * widen, 0.08 * widen,
                   self.args.search_ang * widen, math.radians(2.5) * widen),
                  (0.10, 0.025, math.radians(3.0), math.radians(0.6)))
        for span_lin, step_lin, span_ang, step_ang in stages:
            nl = max(1, int(round(span_lin / step_lin)))
            na = max(1, int(round(span_ang / step_ang)))
            offs = np.arange(-nl, nl + 1) * step_lin
            angs = np.arange(-na, na + 1) * step_ang
            gx, gy, ga = np.meshgrid(offs, offs, angs, indexing='ij')
            cand = np.stack([best[0] + gx.ravel(),
                             best[1] + gy.ravel(),
                             best[2] + ga.ravel()], axis=1)
            scores = self.grid.score(cand, sub)
            k = int(np.argmax(scores))
            best = cand[k].copy()
            best_score = float(scores[k])

        best = self.gauss_newton(pts, best)
        best[2] = wrap_angle(best[2])
        return best, best_score

    def gauss_newton(self, pts, seed, iters=6):
        """Hector's objective: minimise sum (1 - M(S_i(xi)))^2 over the scan."""
        xi = np.asarray(seed, dtype=np.float64).copy()
        sx, sy = pts[:, 0], pts[:, 1]
        for _ in range(iters):
            c, s = math.cos(xi[2]), math.sin(xi[2])
            wx = xi[0] + c * sx - s * sy
            wy = xi[1] + s * sx + c * sy
            m, gx, gy = self.grid._bilinear(wx, wy, want_grad=True)
            # d(point)/d(theta)
            dsx = -s * sx - c * sy
            dsy = c * sx - s * sy
            J = np.stack([gx, gy, gx * dsx + gy * dsy], axis=1)
            r = 1.0 - m
            H = J.T @ J
            b = J.T @ r
            # Levenberg damping. H is singular in a featureless corridor,
            # where the along-wall direction is genuinely unobservable.
            H[0, 0] += 1e-3
            H[1, 1] += 1e-3
            H[2, 2] += 1e-4
            try:
                d = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break
            # A single step should never be large; if it is, the match is bad.
            d[0] = float(np.clip(d[0], -0.10, 0.10))
            d[1] = float(np.clip(d[1], -0.10, 0.10))
            d[2] = float(np.clip(d[2], -0.05, 0.05))
            xi += d
            if abs(d[0]) < 1e-4 and abs(d[1]) < 1e-4 and abs(d[2]) < 1e-5:
                break
        return xi

    # -- main entry point -------------------------------------------------
    def process_scan(self, angles_deg, ranges_m):
        pts = self.polar_to_xy(angles_deg, ranges_m)
        if pts.shape[0] < 40:
            return False

        if not self.initialised:
            # Nothing to match against yet: declare this pose the origin.
            self.grid.integrate(self.pose, pts)
            self.initialised = True
            self.n_scans = 1
            self.n_keyframes = 1
            return True

        if self.args.deskew:
            pts = self.deskew(pts)

        # Constant-velocity seed. Handheld motion is jerky, so damp it rather
        # than trusting the last frame outright.
        pred = self.pose.copy()
        if self.args.predict:
            vx, vy, vth = self.velocity * 0.7
            c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
            pred[0] += c * vx - s * vy
            pred[1] += s * vx + c * vy
            pred[2] = wrap_angle(pred[2] + vth)

        # A run of rejects means the last confirmed pose is stale and the
        # normal search window is centred on the wrong place -- widening it
        # (and the jump tolerance to match) is what actually gives the
        # matcher a chance to find where the sensor really is again, instead
        # of retrying the same too-small window forever.
        widen = 1.0
        if self.n_consec_reject >= self.args.relocalize_after:
            steps = self.n_consec_reject // self.args.relocalize_after
            widen = min(self.args.relocalize_max,
                       1.0 + steps * (self.args.relocalize_scale - 1.0))
            if not self.lost:
                self.lost = True
                print('[slam] lost tracking (%d rejects in a row) -- widening search'
                      % self.n_consec_reject)

        new_pose, score = self.match(pts, pred, widen)
        self.last_score = score
        self.n_scans += 1

        # Reject implausible jumps and low-confidence matches instead of
        # writing them into the map, where they would be unrecoverable.
        # max_jump scales with widen too, or a legitimate find from the wider
        # search would just get thrown away by this check anyway.
        jump = math.hypot(new_pose[0] - self.pose[0], new_pose[1] - self.pose[1])
        if score < self.args.min_score or jump > self.args.max_jump * widen:
            self.n_rejected += 1
            self.n_consec_reject += 1
            self.velocity[:] = 0.0
            return False

        if self.lost:
            self.lost = False
            print('[slam] tracking recovered after %d rejects' % self.n_consec_reject)
        self.n_consec_reject = 0

        # A match within noise distance of the previous pose is indistinguishable
        # from the sensor genuinely being still. Comparing against a fixed anchor
        # (rather than the last accepted pose) matters: noise that happened to
        # drift the same direction for a few scans in a row would otherwise walk
        # the "previous pose" reference along with it, so each new noisy sample
        # would compare against an already-drifted point and pass the check
        # again -- a slow, unbounded random walk while genuinely stationary. An
        # anchor that only moves once real motion clears the deadband can't do
        # that: noise always gets measured against, and snapped back to, the
        # same fixed point.
        if self.still_anchor is None:
            self.still_anchor = self.pose.copy()
        d = new_pose - self.still_anchor
        d[2] = wrap_angle(d[2])
        if (math.hypot(d[0], d[1]) < self.args.still_lin
                and abs(d[2]) < math.radians(self.args.still_ang)):
            new_pose = self.still_anchor.copy()
        else:
            self.still_anchor = new_pose.copy()

        # Body-frame delta, kept for the next frame's prediction and deskew.
        d = new_pose - self.pose
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.velocity = np.array([c * d[0] + s * d[1],
                                  -s * d[0] + c * d[1],
                                  wrap_angle(d[2])])
        self.pose = new_pose
        self.trajectory.append(self.pose.copy())

        # Only fold a scan into the map once the sensor has actually moved.
        # Integrating every scan while standing still just blurs the walls.
        md = math.hypot(self.pose[0] - self.last_kf_pose[0],
                        self.pose[1] - self.last_kf_pose[1])
        ma = abs(wrap_angle(self.pose[2] - self.last_kf_pose[2]))
        if md > self.args.kf_dist or ma > math.radians(self.args.kf_angle):
            self.grid.integrate(self.pose, pts)
            self.last_kf_pose = self.pose.copy()
            self.n_keyframes += 1
        return True

    # -- output -----------------------------------------------------------
    def save(self, path):
        base = os.path.splitext(path)[0]
        traj = np.asarray(self.trajectory)
        np.savez_compressed(base + '.npz',
                            prob=self.grid.prob().astype(np.float32),
                            log_odds=self.grid.log_odds,
                            ros_grid=self.grid.to_ros_occupancy(),
                            trajectory=traj,
                            resolution=self.grid.res,
                            origin=np.array([self.grid.ox, self.grid.oy]))
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            ext = [self.grid.ox, self.grid.ox + self.grid.w * self.grid.res,
                   self.grid.oy, self.grid.oy + self.grid.h * self.grid.res]
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(1.0 - self.grid.prob(), cmap='gray', origin='lower',
                      extent=ext, vmin=0.0, vmax=1.0)
            ax.plot(traj[:, 0], traj[:, 1], '-', color='#e5484d', lw=1.2)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title('canary-rover room map  (%d scans)' % self.n_scans)
            ax.set_aspect('equal')
            fig.savefig(base + '.png', dpi=140, bbox_inches='tight')
            plt.close(fig)
        except Exception as exc:
            print('[map] PNG export skipped: %s' % exc)
        print('[map] saved %s.npz / %s.png  (%dx%d @ %.3f m)'
              % (base, base, self.grid.w, self.grid.h, self.grid.res))


# --------------------------------------------------------------- TCP server

class MapServer(threading.Thread):
    """Pushes map + pose + live scan to any connected viewer.

    The map is the expensive message so it goes out at a slower cadence than
    the pose. It compresses to a few kB because an occupancy grid is mostly
    long runs of 'unknown'.
    """

    daemon = True

    def __init__(self, host, port, map_hz):
        super().__init__()
        self.addr = (host, port)
        self.map_interval = 1.0 / max(0.1, map_hz)
        self.lock = threading.Lock()
        self.state = None
        self.map_blob = None
        self.map_version = 0
        self.running = True
        self.clients = 0

    def publish_state(self, t, pose, world_pts):
        payload = STATE_HDR.pack(t, float(pose[0]), float(pose[1]),
                                 float(pose[2]), world_pts.shape[0])
        payload += world_pts.astype(np.float32).tobytes()
        with self.lock:
            self.state = payload

    def publish_map(self, grid):
        body = zlib.compress(grid.to_uint8().tobytes(), 6)
        payload = MAP_HDR.pack(grid.ox, grid.oy, grid.res, grid.w, grid.h) + body
        with self.lock:
            self.map_blob = payload
            self.map_version += 1

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.addr)
        srv.listen(4)
        srv.settimeout(1.0)
        print('[net] viewer server on %s:%d' % self.addr)
        while self.running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn, addr),
                             daemon=True).start()
        srv.close()

    def _serve(self, conn, addr):
        print('[net] viewer connected: %s' % (addr,))
        self.clients += 1
        conn.settimeout(10.0)
        sent_map = -1
        last_map = 0.0
        try:
            while self.running:
                now = time.time()
                with self.lock:
                    state = self.state
                    blob = self.map_blob
                    ver = self.map_version
                    if state is not None:
                        self.state = None
                # A fresh viewer needs the whole map before anything else on
                # screen makes sense, so the first send ignores the cadence.
                if blob is not None and (sent_map < 0
                                         or (ver != sent_map
                                             and now - last_map >= self.map_interval)):
                    conn.sendall(HDR.pack(WIRE_MAGIC, MSG_MAP, len(blob)) + blob)
                    sent_map = ver
                    last_map = now
                if state is not None:
                    conn.sendall(HDR.pack(WIRE_MAGIC, MSG_STATE, len(state)) + state)
                time.sleep(0.02)
        except (OSError, socket.timeout) as exc:
            print('[net] viewer %s dropped: %s' % (addr, exc))
        finally:
            self.clients -= 1
            conn.close()


# --------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(
        description='2D LiDAR SLAM for canary-rover (no ROS, no IMU).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group('scan source')
    src.add_argument('--source', default='serial:/dev/ttyUSB0',
                     help='serial:/dev/ttyUSB0 | tcp://host:9000 | file:raw.bin')
    src.add_argument('--baud', type=int, default=460800)
    src.add_argument('--reset', action='store_true',
                     help='send RESET before SCAN (recovers a wedged unit)')
    src.add_argument('--motor-dtr', action='store_true',
                     help='pull DTR low to spin the motor (some A1 units)')
    src.add_argument('--replay-speed', type=float, default=1.0)
    src.add_argument('--record', default=None, help='tee raw bytes to this file')

    f = p.add_argument_group('scan filtering')
    f.add_argument('--min-quality', type=int, default=5)
    f.add_argument('--min-range', type=float, default=0.15)
    f.add_argument('--max-range', type=float, default=12.0)

    m = p.add_argument_group('map')
    m.add_argument('--map-res', type=float, default=0.05, help='metres per cell')
    m.add_argument('--map-size', type=float, default=20.0,
                   help='initial extent in metres; grows automatically')
    m.add_argument('--l-occ', type=float, default=0.85)
    m.add_argument('--l-free', type=float, default=-0.4)

    s = p.add_argument_group('slam')
    s.add_argument('--search-lin', type=float, default=0.35,
                   help='half-width of the coarse position search, metres')
    s.add_argument('--search-ang', type=float, default=math.radians(15.0),
                   help='half-width of the coarse heading search, radians')
    s.add_argument('--match-points', type=int, default=240,
                   help='points used in the brute-force stage')
    s.add_argument('--min-score', type=float, default=0.10,
                   help='reject matches below this mean field score. Note: '
                        'with only one scan integrated, even a pixel-perfect '
                        'self-match tops out around 0.14-0.15 (log-odds from a '
                        'single hit is weak), so anything much above that is '
                        'unreachable until a few overlapping scans have built '
                        'up confidence in the same cells -- set too high, no '
                        'scan after the first can ever be accepted')
    s.add_argument('--max-jump', type=float, default=1.0,
                   help='reject frame-to-frame jumps larger than this, metres')
    s.add_argument('--kf-dist', type=float, default=0.05)
    s.add_argument('--kf-angle', type=float, default=2.5, help='degrees')
    s.add_argument('--still-lin', type=float, default=0.15,
                   help='matches closer than this to the anchor pose are '
                        'treated as noise, not motion, metres')
    s.add_argument('--still-ang', type=float, default=7.0,
                   help='matches closer than this to the anchor heading are '
                        'treated as noise, not motion, degrees')
    s.add_argument('--relocalize-after', type=int, default=8,
                   help='consecutive rejected scans before widening the '
                        'search to try to reacquire tracking')
    s.add_argument('--relocalize-scale', type=float, default=1.6,
                   help='how much wider each relocalization step makes the search')
    s.add_argument('--relocalize-max', type=float, default=6.0,
                   help='cap on how wide the relocalization search can grow')
    # Off by default: the correction is real but unverified against hardware,
    # and it shifts the pose reference from the start of the sweep to the end.
    # Turn it on if walls come out doubled or thick when walking quickly.
    s.add_argument('--deskew', action='store_true',
                   help='compensate sensor motion during a revolution')
    s.add_argument('--no-predict', dest='predict', action='store_false')

    o = p.add_argument_group('output')
    o.add_argument('--listen', default='0.0.0.0')
    o.add_argument('--port', type=int, default=9100)
    o.add_argument('--map-hz', type=float, default=1.5)
    o.add_argument('--no-server', action='store_true')
    o.add_argument('--save', default='room_map')
    o.add_argument('--quiet', action='store_true')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    print('[src] opening %s' % args.source)
    try:
        source = open_source(args.source, args.baud, args.reset,
                             args.replay_speed, args.motor_dtr)
    except Exception as exc:
        print('[src] could not open source: %s' % exc)
        return 1

    rec = open(args.record, 'wb') if args.record else None
    parser = RPLidarParser(args.min_quality, args.min_range, args.max_range)
    mapper = SlamMapper(args)

    server = None
    if not args.no_server:
        server = MapServer(args.listen, args.port, args.map_hz)
        server.start()

    last_map_push = 0.0
    last_log = time.time()
    rates = deque(maxlen=30)
    print('[slam] running -- Ctrl-C to stop and save')

    try:
        while True:
            data = source.read(4096)
            if not data:
                if isinstance(source, FileSource):
                    print('[src] replay finished')
                    break
                continue
            if rec:
                rec.write(data)

            for angles, ranges in parser.feed(data):
                t0 = time.time()
                ok = mapper.process_scan(angles, ranges)
                rates.append(time.time() - t0)
                if not ok or server is None:
                    continue

                pose = mapper.pose
                pts = SlamMapper.polar_to_xy(angles, ranges)
                c, s = math.cos(pose[2]), math.sin(pose[2])
                world = np.stack([pose[0] + c * pts[:, 0] - s * pts[:, 1],
                                  pose[1] + s * pts[:, 0] + c * pts[:, 1]], axis=1)
                server.publish_state(time.time(), pose, world)
                now = time.time()
                if now - last_map_push >= 1.0 / max(0.1, args.map_hz):
                    server.publish_map(mapper.grid)
                    last_map_push = now

            if not args.quiet and time.time() - last_log > 2.0:
                last_log = time.time()
                ms = 1000.0 * (sum(rates) / len(rates)) if rates else 0.0
                print('[slam] scans=%d kf=%d rej=%d%s  pose=(%.2f, %.2f, %.1f deg)'
                      '  score=%.3f  %.1f ms/scan  grid=%dx%d  viewers=%d'
                      % (mapper.n_scans, mapper.n_keyframes, mapper.n_rejected,
                         ' LOST(x%d)' % mapper.n_consec_reject if mapper.lost else '',
                         mapper.pose[0], mapper.pose[1], math.degrees(mapper.pose[2]),
                         mapper.last_score, ms, mapper.grid.w, mapper.grid.h,
                         server.clients if server else 0))
    except KeyboardInterrupt:
        print('\n[slam] stopping')
    finally:
        if server:
            server.running = False
        if rec:
            rec.close()
        try:
            source.close()
        except Exception:
            pass
        if mapper.n_scans > 1:
            mapper.save(args.save)
        else:
            print('[map] nothing to save -- no scans were matched')
    return 0


if __name__ == '__main__':
    sys.exit(main())

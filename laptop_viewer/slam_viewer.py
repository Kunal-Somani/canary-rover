#!/usr/bin/env python3
"""Live viewer for the canary-rover SLAM mapper. Runs on the laptop.

Connects to rplidar_slam.py over TCP and draws the occupancy grid, the walked
trajectory and the current scan. Only needs numpy and matplotlib -- no ROS, no
rviz, and nothing that has to be installed on the Jetson.

    python slam_viewer.py --host 192.168.55.1

The link to the Jetson over USB device mode drops packets in bursts, so the
viewer reconnects on its own rather than dying. A drop costs you a few seconds
of display, never the map -- the map lives in the mapper process.

Keys:
    s   save what is on screen to slam_view.png
    t   toggle the trajectory
    p   toggle the live scan points
    f   re-fit the view to the explored area
    q   quit
"""

import argparse
import socket
import struct
import sys
import threading
import time
import zlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Must match rplidar_slam.py.
WIRE_MAGIC = b'CNR1'
MSG_MAP = 1
MSG_STATE = 2
HDR = struct.Struct('<4sBI')
MAP_HDR = struct.Struct('<dddII')
STATE_HDR = struct.Struct('<ddddI')


class MapClient(threading.Thread):
    """Receives map/pose messages and holds the latest of each.

    Runs on its own thread so a stalled socket never freezes the GUI, which is
    the usual failure mode when you recv() straight from a matplotlib timer.
    """

    daemon = True

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.map = None            # (grid_uint8, ox, oy, res)
        self.map_seq = 0
        self.pose = None
        self.points = None
        self.trajectory = []
        self.connected = False
        self.status = 'connecting'
        self.running = True
        self.n_msgs = 0

    def _recv_exact(self, sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(min(65536, n - len(buf)))
            if not chunk:
                raise ConnectionError('mapper closed the connection')
            buf.extend(chunk)
        return bytes(buf)

    def run(self):
        backoff = 1.0
        while self.running:
            sock = None
            try:
                self.status = 'connecting to %s:%d' % (self.host, self.port)
                sock = socket.create_connection((self.host, self.port), timeout=8.0)
                sock.settimeout(15.0)
                self.connected = True
                self.status = 'connected'
                backoff = 1.0
                while self.running:
                    head = self._recv_exact(sock, HDR.size)
                    magic, mtype, length = HDR.unpack(head)
                    if magic != WIRE_MAGIC:
                        raise ConnectionError('bad framing from mapper')
                    payload = self._recv_exact(sock, length)
                    self._handle(mtype, payload)
                    self.n_msgs += 1
            except Exception as exc:
                self.connected = False
                self.status = 'disconnected (%s) -- retrying' % exc
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if not self.running:
                break
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.6)

    def _handle(self, mtype, payload):
        if mtype == MSG_MAP:
            ox, oy, res, w, h = MAP_HDR.unpack_from(payload, 0)
            raw = zlib.decompress(payload[MAP_HDR.size:])
            grid = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
            with self.lock:
                self.map = (grid, ox, oy, res)
                self.map_seq += 1
        elif mtype == MSG_STATE:
            _t, x, y, th, n = STATE_HDR.unpack_from(payload, 0)
            pts = np.frombuffer(payload, dtype=np.float32,
                                count=n * 2, offset=STATE_HDR.size).reshape(n, 2)
            with self.lock:
                self.pose = (x, y, th)
                self.points = pts
                self.trajectory.append((x, y))

    def snapshot(self):
        with self.lock:
            traj = np.asarray(self.trajectory) if self.trajectory else None
            return self.map, self.map_seq, self.pose, self.points, traj


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Live SLAM map viewer for canary-rover.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--host', default='192.168.55.1',
                    help='Jetson address running rplidar_slam.py')
    ap.add_argument('--port', type=int, default=9100)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--save', default='slam_view.png')
    args = ap.parse_args(argv)

    client = MapClient(args.host, args.port)
    client.start()

    plt.rcParams['figure.facecolor'] = '#f7f7f8'
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title('canary-rover :: live SLAM map')

    # Placeholder image; extent and data are replaced on the first map message.
    im = ax.imshow(np.full((2, 2), 0.5), cmap='gray_r', origin='lower',
                   vmin=0.0, vmax=1.0, extent=[-1, 1, -1, 1], interpolation='nearest')
    (traj_line,) = ax.plot([], [], '-', color='#e5484d', lw=1.4, label='path')
    scan = ax.scatter([], [], s=2.0, c='#0090ff', alpha=0.7, label='live scan')
    heading = ax.plot([], [], '-', color='#f76b15', lw=2.2)[0]
    robot = Circle((0, 0), 0.12, facecolor='#f76b15', edgecolor='white',
                   lw=1.2, zorder=5)
    ax.add_patch(robot)

    status = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
                     family='monospace', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.75, edgecolor='#d0d0d4'))

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)

    state = {'seq': -1, 'traj': True, 'points': True, 'fitted': False, 'stop': False}

    def fit_view():
        snap = client.snapshot()
        gm = snap[0]
        if gm is None:
            return
        grid, ox, oy, res = gm
        # Frame the explored area, not the whole allocated grid -- most of it
        # is unknown padding and would squash the room into a few pixels.
        known = np.abs(grid.astype(np.int16) - 127) > 12
        if not known.any():
            return
        ys, xs = np.nonzero(known)
        pad = 1.0
        ax.set_xlim(ox + xs.min() * res - pad, ox + (xs.max() + 1) * res + pad)
        ax.set_ylim(oy + ys.min() * res - pad, oy + (ys.max() + 1) * res + pad)

    def on_key(event):
        if event.key == 'q':
            state['stop'] = True
            plt.close(fig)
        elif event.key == 's':
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print('[view] saved %s' % args.save)
        elif event.key == 't':
            state['traj'] = not state['traj']
            traj_line.set_visible(state['traj'])
        elif event.key == 'p':
            state['points'] = not state['points']
            scan.set_visible(state['points'])
        elif event.key == 'f':
            fit_view()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_frame):
        gm, seq, pose, pts, traj = client.snapshot()

        if gm is not None and seq != state['seq']:
            grid, ox, oy, res = gm
            h, w = grid.shape
            # Wire values are 0=free .. 127=unknown .. 255=occupied. gray_r
            # renders high values dark, so occupied walls come out black and
            # free space white, with unknown a flat mid grey.
            im.set_data(grid.astype(np.float32) / 255.0)
            im.set_extent([ox, ox + w * res, oy, oy + h * res])
            state['seq'] = seq
            if not state['fitted']:
                fit_view()
                state['fitted'] = True

        if pose is not None:
            x, y, th = pose
            robot.center = (x, y)
            heading.set_data([x, x + 0.45 * np.cos(th)], [y, y + 0.45 * np.sin(th)])

        if pts is not None and state['points']:
            scan.set_offsets(pts)

        if traj is not None and traj.shape[0] > 1 and state['traj']:
            traj_line.set_data(traj[:, 0], traj[:, 1])

        dist = 0.0
        if traj is not None and traj.shape[0] > 1:
            dist = float(np.sum(np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]))))
        px, py, pth = pose if pose else (0.0, 0.0, 0.0)
        status.set_text(
            'link   %s\n'
            'pose   x=%+.2f m  y=%+.2f m  th=%+.1f deg\n'
            'walked %.1f m   msgs %d   [s]ave [t]rail [p]oints [f]it [q]uit'
            % (client.status, px, py, np.degrees(pth), dist, client.n_msgs))
        return im, traj_line, scan, heading, robot, status

    timer = fig.canvas.new_timer(interval=int(1000.0 / max(1.0, args.fps)))
    timer.add_callback(lambda: (update(0), fig.canvas.draw_idle()))
    timer.start()

    print('[view] connecting to %s:%d -- close the window or press q to quit'
          % (args.host, args.port))
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        client.running = False
    return 0


if __name__ == '__main__':
    sys.exit(main())

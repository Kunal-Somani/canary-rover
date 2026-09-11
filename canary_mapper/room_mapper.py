#!/usr/bin/env python3
"""2D occupancy-grid mapping from a single 360-degree LiDAR.

Runs on the Jetson, next to the RPLiDAR driver. Subscribes to a
sensor_msgs/LaserScan and publishes a nav_msgs/OccupancyGrid of the room.

There is no wheel-odometry dependency: the robot pose comes from
correlative scan-to-map matching (a coarse-to-fine grid search over
x/y/theta against a blurred likelihood field), the same idea Cartographer
uses for its real-time local matcher. That means this works on a rover
with no encoders. It also means it drifts in a featureless corridor --
see NOTES at the bottom.

Dependencies: rclpy, numpy, tf2_ros. Deliberately nothing else, so the
same file runs under Humble, Foxy, or a ROS 2 container on JetPack 4.

Typical use:
    ros2 run rplidar_ros rplidar_composition --ros-args \
        -p serial_port:=/dev/ttyUSB0 -p frame_id:=laser
    python3 room_mapper.py --ros-args -p max_range:=12.0

Save the finished map at any time (Ctrl-C also autosaves):
    ros2 topic pub --once /mapper/save_map std_msgs/msg/Empty
"""

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

LOG_ODDS_MIN = -6.0
LOG_ODDS_MAX = 8.0
UNKNOWN_EPS = 1e-3


def yaw_to_quat(yaw):
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


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


class OccupancyGrid2D:
    """Log-odds occupancy grid that grows on demand."""

    def __init__(self, resolution, initial_size_m, l_free, l_occ):
        self.res = float(resolution)
        n = max(32, int(round(initial_size_m / self.res)))
        self.h = n
        self.w = n
        # World coordinate of the lower-left corner of cell (0, 0).
        self.ox = -0.5 * self.w * self.res
        self.oy = -0.5 * self.h * self.res
        self.log_odds = np.zeros((self.h, self.w), dtype=np.float32)
        self.l_free = float(l_free)
        self.l_occ = float(l_occ)
        self._field = None
        self._field_dirty = True

    # -- geometry -----------------------------------------------------
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
        pad = int(math.ceil(margin_m / self.res))
        need_left = float(np.min(wx)) < self.ox + margin_m
        need_right = float(np.max(wx)) > self.ox + self.w * self.res - margin_m
        need_down = float(np.min(wy)) < self.oy + margin_m
        need_up = float(np.max(wy)) > self.oy + self.h * self.res - margin_m
        if not (need_left or need_right or need_down or need_up):
            return
        # Grow a whole margin at a time so we are not reallocating per scan.
        left = pad if need_left else 0
        right = pad if need_right else 0
        down = pad if need_down else 0
        up = pad if need_up else 0
        self.log_odds = np.pad(self.log_odds, ((down, up), (left, right)),
                               mode='constant', constant_values=0.0)
        self.h, self.w = self.log_odds.shape
        self.ox -= left * self.res
        self.oy -= down * self.res
        self._field_dirty = True

    # -- mapping ------------------------------------------------------
    def integrate(self, pose, points):
        """Ray-cast one scan into the grid. points is Nx2 in the sensor frame."""
        px, py, pth = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = math.cos(pth), math.sin(pth)
        wx = px + c * points[:, 0] - s * points[:, 1]
        wy = py + s * points[:, 0] + c * points[:, 1]
        self.ensure_capacity(np.append(wx, px), np.append(wy, py))

        # Free space: sample each ray at grid resolution rather than running a
        # Python-level Bresenham, so the whole update stays vectorised.
        dx = wx - px
        dy = wy - py
        dist = np.hypot(dx, dy)
        good = dist > self.res
        if np.any(good):
            dxg, dyg, dg = dx[good], dy[good], dist[good]
            ux, uy = dxg / dg, dyg / dg
            steps = int(math.ceil(float(np.max(dg)) / self.res))
            t = (np.arange(steps, dtype=np.float32) * self.res)[:, None]
            sx = px + ux[None, :] * t
            sy = py + uy[None, :] * t
            keep = t < (dg[None, :] - self.res)
            ix, iy = self.world_to_cell(sx[keep], sy[keep])
            ok = self.in_bounds(ix, iy)
            if np.any(ok):
                flat = iy[ok].astype(np.int64) * self.w + ix[ok].astype(np.int64)
                counts = np.bincount(flat, minlength=self.w * self.h)
                self.log_odds += counts.reshape(self.h, self.w).astype(np.float32) * self.l_free

        # Occupied endpoints, applied after the free pass so they win.
        ix, iy = self.world_to_cell(wx, wy)
        ok = self.in_bounds(ix, iy)
        if np.any(ok):
            flat = iy[ok].astype(np.int64) * self.w + ix[ok].astype(np.int64)
            counts = np.bincount(flat, minlength=self.w * self.h)
            self.log_odds += counts.reshape(self.h, self.w).astype(np.float32) * self.l_occ

        np.clip(self.log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=self.log_odds)
        self._field_dirty = True

    # -- scan matching ------------------------------------------------
    def likelihood_field(self):
        if self._field_dirty or self._field is None:
            p = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
            occ = np.clip((p - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
            # Narrow kernel on purpose: a wide blur flattens the score surface
            # and the matcher stops being able to localise precisely.
            self._field = blur2d(occ, np.array([0.05, 0.5, 1.0, 0.5, 0.05],
                                               dtype=np.float32))
            self._field_dirty = False
        return self._field

    def score(self, poses, points):
        """Score candidate poses. poses is Nx3, points Mx2 -> N scores.

        Bilinearly interpolates the likelihood field. Nearest-cell lookup
        would cap the matcher's precision at one cell (5 cm by default) and
        make the sub-centimetre refinement stage a no-op.
        """
        field = self.likelihood_field()
        c = np.cos(poses[:, 2])[:, None]
        s = np.sin(poses[:, 2])[:, None]
        wx = poses[:, 0:1] + c * points[None, :, 0] - s * points[None, :, 1]
        wy = poses[:, 1:2] + s * points[None, :, 0] + c * points[None, :, 1]

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
        vals = (field[y0, x0] * (1.0 - tx) * (1.0 - ty)
                + field[y0, x1] * tx * (1.0 - ty)
                + field[y1, x0] * (1.0 - tx) * ty
                + field[y1, x1] * tx * ty)
        vals[~ok] = 0.0
        return vals.sum(axis=1) / max(1, points.shape[0])


class RoomMapper(Node):

    def __init__(self):
        super().__init__('room_mapper')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('initial_size_m', 24.0)
        self.declare_parameter('min_range', 0.15)
        self.declare_parameter('max_range', 12.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('laser_z', 0.4)
        self.declare_parameter('publish_period', 1.0)
        self.declare_parameter('match_points', 180)
        self.declare_parameter('search_linear', 0.30)
        self.declare_parameter('search_angular', 0.35)
        self.declare_parameter('keyframe_dist', 0.05)
        self.declare_parameter('keyframe_angle', 0.05)
        self.declare_parameter('min_match_score', 0.10)
        self.declare_parameter('map_save_path',
                               os.path.expanduser('~/canary_maps/room'))

        def g(name):
            return self.get_parameter(name).value

        self.res = float(g('resolution'))
        self.min_range = float(g('min_range'))
        self.max_range = float(g('max_range'))
        self.map_frame = g('map_frame')
        self.base_frame = g('base_frame')
        self.laser_z = float(g('laser_z'))
        self.match_points = int(g('match_points'))
        self.search_linear = float(g('search_linear'))
        self.search_angular = float(g('search_angular'))
        self.kf_dist = float(g('keyframe_dist'))
        self.kf_angle = float(g('keyframe_angle'))
        self.min_score = float(g('min_match_score'))
        self.save_path = g('map_save_path')

        # Log-odds of one free / occupied observation (p = 0.4 / p = 0.75).
        self.grid = OccupancyGrid2D(self.res, float(g('initial_size_m')),
                                    l_free=-0.405, l_occ=1.1)

        self.pose = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.last_kf_pose = None
        self.last_stamp = None
        self.scan_frame = 'laser'
        self.scan_count = 0
        self.trajectory = []
        self._static_sent = False

        # A latched map, so a laptop that connects late still gets one
        # immediately -- which matters over a link that drops.
        map_qos = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST)
        scan_qos = QoSProfile(depth=10,
                              reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              history=QoSHistoryPolicy.KEEP_LAST)

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/mapper/pose', 10)
        self.path_pub = self.create_publisher(Path, '/mapper/path', map_qos)
        self.create_subscription(LaserScan, g('scan_topic'), self.on_scan, scan_qos)
        self.create_subscription(Empty, '/mapper/save_map',
                                 lambda _msg: self.save_map(), 1)

        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.create_timer(float(g('publish_period')), self.publish_map)

        self.get_logger().info(
            'room_mapper up: {} -> /map at {:.0f} mm/cell, range {:.2f}-{:.1f} m'
            .format(g('scan_topic'), self.res * 1000.0, self.min_range, self.max_range))

    # -- scan handling -------------------------------------------------
    def scan_to_points(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = ranges.shape[0]
        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        hi = min(self.max_range, msg.range_max if msg.range_max > 0 else self.max_range)
        lo = max(self.min_range, msg.range_min)
        valid = np.isfinite(ranges) & (ranges >= lo) & (ranges <= hi)
        r = ranges[valid]
        a = angles[valid]
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)

    def on_scan(self, msg):
        self.scan_frame = msg.header.frame_id or 'laser'
        if not self._static_sent:
            self.send_static_tf()
            self._static_sent = True

        points = self.scan_to_points(msg)
        if points.shape[0] < 20:
            self.get_logger().warn(
                'only {} usable returns, skipping scan'.format(points.shape[0]))
            return

        stamp = msg.header.stamp
        now = stamp.sec + stamp.nanosec * 1e-9

        if self.scan_count == 0:
            self.grid.integrate(self.pose, points)
            self.last_kf_pose = self.pose.copy()
            self.last_stamp = now
            self.scan_count = 1
            self.publish_pose(stamp)
            return

        # Constant-velocity seed, so the search window is centred on where
        # the rover has probably already got to.
        dt = max(1e-3, min(0.5, now - (self.last_stamp or now)))
        seed = self.pose + self.velocity * dt
        seed[2] = wrap_angle(seed[2])

        sub = points
        if points.shape[0] > self.match_points:
            idx = np.linspace(0, points.shape[0] - 1, self.match_points).astype(np.int32)
            sub = points[idx]

        matched, score = self.match(sub, seed)

        if score < self.min_score:
            self.get_logger().warn(
                'weak scan match ({:.3f} < {:.3f}); holding pose. Driving too '
                'fast, or the room is too featureless?'.format(score, self.min_score))
            self.velocity[:] = 0.0
        else:
            delta = matched - self.pose
            delta[2] = wrap_angle(delta[2])
            self.velocity = delta / dt
            self.pose = matched

        self.last_stamp = now
        self.scan_count += 1
        self.publish_pose(stamp)

        # Only fold a scan into the map once the rover has actually moved,
        # otherwise a parked robot smears the walls with its own noise.
        d = self.pose[:2] - self.last_kf_pose[:2]
        moved = math.hypot(float(d[0]), float(d[1]))
        turned = abs(wrap_angle(float(self.pose[2] - self.last_kf_pose[2])))
        if moved > self.kf_dist or turned > self.kf_angle or self.scan_count < 10:
            self.grid.integrate(self.pose, points)
            self.last_kf_pose = self.pose.copy()
            self.trajectory.append((self.pose.copy(), stamp))

    def match(self, points, seed):
        """Coarse-to-fine correlative scan matching against the map."""
        stages = (
            (0.08, math.radians(4.0), self.search_linear, self.search_angular),
            (0.02, math.radians(1.0), 0.10, math.radians(5.0)),
            (0.005, math.radians(0.25), 0.025, math.radians(1.25)),
        )
        best = seed.copy()
        best_score = 0.0
        for lin_step, ang_step, lin_span, ang_span in stages:
            nx = max(1, int(lin_span / lin_step))
            na = max(1, int(ang_span / ang_step))
            offs = np.arange(-nx, nx + 1) * lin_step
            angs = np.arange(-na, na + 1) * ang_step
            gx, gy, ga = np.meshgrid(offs, offs, angs, indexing='ij')
            cand = np.stack([best[0] + gx.ravel(),
                             best[1] + gy.ravel(),
                             best[2] + ga.ravel()], axis=1)
            scores = self.grid.score(cand, points)
            k = int(np.argmax(scores))
            best = cand[k].copy()
            best_score = float(scores[k])
        best[2] = wrap_angle(best[2])
        return best, best_score

    # -- outputs -------------------------------------------------------
    def send_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.scan_frame
        t.transform.translation.z = self.laser_z
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)

    def publish_pose(self, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(self.pose[0])
        t.transform.translation.y = float(self.pose[1])
        t.transform.rotation = yaw_to_quat(float(self.pose[2]))
        self.tf.sendTransform(t)

        p = PoseStamped()
        p.header.stamp = stamp
        p.header.frame_id = self.map_frame
        p.pose.position.x = float(self.pose[0])
        p.pose.position.y = float(self.pose[1])
        p.pose.orientation = yaw_to_quat(float(self.pose[2]))
        self.pose_pub.publish(p)

    def occupancy_bytes(self):
        lo = self.grid.log_odds
        p = 1.0 - 1.0 / (1.0 + np.exp(lo))
        data = np.full(lo.shape, -1, dtype=np.int8)
        known = np.abs(lo) > UNKNOWN_EPS
        data[known] = np.clip(np.round(p[known] * 100.0), 0, 100).astype(np.int8)
        return data

    def publish_map(self):
        if self.scan_count == 0:
            return
        data = self.occupancy_bytes()
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.res
        msg.info.width = self.grid.w
        msg.info.height = self.grid.h
        msg.info.origin.position.x = self.grid.ox
        msg.info.origin.position.y = self.grid.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = data.ravel().tolist()
        self.map_pub.publish(msg)

        path = Path()
        path.header.stamp = msg.header.stamp
        path.header.frame_id = self.map_frame
        for pose, stamp in self.trajectory:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = float(pose[0])
            ps.pose.position.y = float(pose[1])
            ps.pose.orientation = yaw_to_quat(float(pose[2]))
            path.poses.append(ps)
        self.path_pub.publish(path)

    def save_map(self):
        """Write map.pgm + map.yaml in the format nav2's map_server reads."""
        if self.scan_count == 0:
            self.get_logger().warn('nothing mapped yet, not saving')
            return
        base = self.save_path
        parent = os.path.dirname(base)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = self.occupancy_bytes()
        img = np.full(data.shape, 205, dtype=np.uint8)   # unknown
        img[(data >= 0) & (data < 65)] = 254             # free
        img[data >= 65] = 0                              # occupied
        img = np.flipud(img)                             # PGM row 0 is the top
        with open(base + '.pgm', 'wb') as f:
            f.write(b'P5\n# canary-rover room_mapper\n')
            f.write('{} {}\n255\n'.format(img.shape[1], img.shape[0]).encode())
            f.write(img.tobytes())
        with open(base + '.yaml', 'w') as f:
            f.write('image: {}\n'.format(os.path.basename(base) + '.pgm'))
            f.write('resolution: {}\n'.format(self.res))
            f.write('origin: [{}, {}, 0.0]\n'.format(self.grid.ox, self.grid.oy))
            f.write('negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
        self.get_logger().info('saved {}.pgm / .yaml  ({}x{} cells)'
                               .format(base, self.grid.w, self.grid.h))


def main(args=None):
    rclpy.init(args=args)
    node = RoomMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.save_map()
        except Exception as exc:                          # noqa: BLE001
            node.get_logger().error('map save failed: {}'.format(exc))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

# NOTES
# -----
# * No loop closure. Scan-to-map matching against an accumulating grid is
#   solid for a single room, which is the stated scope. Drive a long loop
#   through several rooms and the two ends will not line up.
# * Drive slowly. The search window is +/-0.30 m and +/-20 deg per scan, so
#   an RPLiDAR A1 at 5.5 Hz gives ~1.6 m/s of headroom in theory and much
#   less in practice. If you see "weak scan match", slow down.
# * A 360-degree scan taken while moving is skewed -- every beam is sampled
#   at a different pose. This does not de-skew, which costs accuracy above
#   roughly 0.5 m/s.

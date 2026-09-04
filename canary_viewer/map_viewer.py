#!/usr/bin/env python3
"""Live viewer for the room map, to run on the laptop.

Subscribes to the /map, /scan and /mapper/pose that room_mapper.py puts on
the wire and draws them with matplotlib. Intentionally not RViz: this needs
only rclpy + numpy + matplotlib, so it runs anywhere ROS 2 does, including
a Windows install, and it survives the DDS link dropping for a while.

    python3 map_viewer.py
    python3 map_viewer.py --ros-args -p scan_topic:=/scan

Keys:  s  save a PNG        c  clear the trail        q  quit

Both machines must agree on the domain, so on the laptop and the Jetson:
    export ROS_DOMAIN_ID=0        (set / setx ROS_DOMAIN_ID 0 on Windows)
"""

import math
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Colours for the occupancy raster, as RGB triples in 0..1.
C_UNKNOWN = (0.60, 0.60, 0.62)
C_FREE = (0.99, 0.99, 0.99)
C_OCCUPIED = (0.08, 0.09, 0.11)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MapViewer(Node):

    def __init__(self):
        super().__init__('map_viewer')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('pose_topic', '/mapper/pose')
        self.declare_parameter('occupied_thresh', 65)
        self.declare_parameter('save_dir', os.path.expanduser('~/canary_maps'))

        self.occ_thresh = int(self.get_parameter('occupied_thresh').value)
        self.save_dir = self.get_parameter('save_dir').value

        self.lock = threading.Lock()
        self.map_msg = None
        self.scan_pts = None          # Nx2, sensor frame
        self.pose = None              # (x, y, yaw)
        self.trail = []
        self.t_map = 0.0
        self.t_scan = 0.0
        self.t_pose = 0.0
        self.map_seq = 0
        self._drawn_seq = -1

        # Must mirror the publisher side or the subscription silently never
        # matches -- the single most common way a ROS 2 viewer shows nothing.
        map_qos = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST)
        scan_qos = QoSProfile(depth=5,
                              reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              history=QoSHistoryPolicy.KEEP_LAST)

        self.create_subscription(OccupancyGrid,
                                 self.get_parameter('map_topic').value,
                                 self.on_map, map_qos)
        self.create_subscription(LaserScan,
                                 self.get_parameter('scan_topic').value,
                                 self.on_scan, scan_qos)
        self.create_subscription(PoseStamped,
                                 self.get_parameter('pose_topic').value,
                                 self.on_pose, 10)

        self.get_logger().info('map_viewer waiting for {} ...'
                               .format(self.get_parameter('map_topic').value))

    def on_map(self, msg):
        with self.lock:
            self.map_msg = msg
            self.map_seq += 1
            self.t_map = time.time()

    def on_scan(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(ranges.shape[0], dtype=np.float32) * msg.angle_increment
        hi = msg.range_max if msg.range_max > 0 else 30.0
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < hi)
        r, a = ranges[valid], angles[valid]
        pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        with self.lock:
            self.scan_pts = pts
            self.t_scan = time.time()

    def on_pose(self, msg):
        yaw = yaw_from_quat(msg.pose.orientation)
        with self.lock:
            self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)
            self.t_pose = time.time()
            if not self.trail or math.hypot(self.pose[0] - self.trail[-1][0],
                                            self.pose[1] - self.trail[-1][1]) > 0.05:
                self.trail.append((self.pose[0], self.pose[1]))

    def snapshot(self):
        with self.lock:
            return (self.map_msg, self.map_seq,
                    None if self.scan_pts is None else self.scan_pts.copy(),
                    self.pose, list(self.trail),
                    self.t_map, self.t_scan, self.t_pose)


def grid_to_rgb(msg, occ_thresh):
    h, w = msg.info.height, msg.info.width
    data = np.asarray(msg.data, dtype=np.int8).reshape(h, w)
    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[...] = C_UNKNOWN
    free = (data >= 0) & (data < occ_thresh)
    occupied = data >= occ_thresh
    rgb[free] = C_FREE
    rgb[occupied] = C_OCCUPIED
    return rgb


def main(args=None):
    rclpy.init(args=args)
    node = MapViewer()

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.canvas.manager.set_window_title('canary-rover  |  room map')
    fig.patch.set_facecolor('#101114')
    ax.set_facecolor('#101114')
    ax.tick_params(colors='#8a8f98', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a2d33')
    ax.set_xlabel('x (m)', color='#8a8f98', fontsize=9)
    ax.set_ylabel('y (m)', color='#8a8f98', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, color='#22252b', linewidth=0.5)

    im = ax.imshow(np.zeros((1, 1, 3)), origin='lower',
                   extent=(-1, 1, -1, 1), interpolation='nearest', zorder=1)
    trail_line, = ax.plot([], [], '-', color='#4da3ff', linewidth=1.4,
                          alpha=0.9, zorder=3)
    scan_dots, = ax.plot([], [], '.', color='#ff5f56', markersize=1.8,
                         alpha=0.85, zorder=4)
    robot_dot, = ax.plot([], [], 'o', color='#ffd23f', markersize=8,
                         markeredgecolor='#101114', zorder=6)
    heading = ax.annotate('', xy=(0, 0), xytext=(0, 0), zorder=5,
                          arrowprops=dict(arrowstyle='-|>', color='#ffd23f',
                                          linewidth=1.6))
    status = ax.set_title('waiting for /map ...', color='#c8ccd4',
                          fontsize=10, loc='left', pad=12)

    state = {'saved': 0}

    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
        elif event.key == 'c':
            with node.lock:
                node.trail.clear()
        elif event.key == 's':
            os.makedirs(node.save_dir, exist_ok=True)
            path = os.path.join(node.save_dir,
                                'room_{}.png'.format(time.strftime('%Y%m%d_%H%M%S')))
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            state['saved'] += 1
            node.get_logger().info('saved {}'.format(path))

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_frame):
        msg, seq, pts, pose, trail, t_map, t_scan, t_pose = node.snapshot()
        now = time.time()

        if msg is not None and seq != node._drawn_seq:
            node._drawn_seq = seq
            rgb = grid_to_rgb(msg, node.occ_thresh)
            ox = msg.info.origin.position.x
            oy = msg.info.origin.position.y
            w = msg.info.width * msg.info.resolution
            h = msg.info.height * msg.info.resolution
            im.set_data(rgb)
            im.set_extent((ox, ox + w, oy, oy + h))
            ax.set_xlim(ox, ox + w)
            ax.set_ylim(oy, oy + h)

        if trail:
            arr = np.asarray(trail)
            trail_line.set_data(arr[:, 0], arr[:, 1])

        if pose is not None:
            x, y, yaw = pose
            robot_dot.set_data([x], [y])
            heading.set_position((x, y))
            heading.xy = (x + 0.45 * math.cos(yaw), y + 0.45 * math.sin(yaw))
            if pts is not None and pts.shape[0]:
                c, s = math.cos(yaw), math.sin(yaw)
                wx = x + c * pts[:, 0] - s * pts[:, 1]
                wy = y + s * pts[:, 0] + c * pts[:, 1]
                scan_dots.set_data(wx, wy)

        # The link to the rover is the thing most likely to be wrong, so put
        # the age of every stream on screen rather than making them guess.
        def age(t):
            return 'never' if t == 0.0 else '{:.1f}s'.format(now - t)

        if msg is None:
            status.set_text('waiting for /map ...   is the mapper running, '
                            'and does ROS_DOMAIN_ID match?')
            status.set_color('#ff9f43')
        else:
            stale = (now - t_map) > 5.0
            status.set_color('#ff9f43' if stale else '#c8ccd4')
            status.set_text(
                'map {}x{} @ {:.0f} mm   |   map {}   scan {}   pose {}   |   '
                's save  c clear  q quit'.format(
                    msg.info.width, msg.info.height,
                    msg.info.resolution * 1000.0,
                    age(t_map), age(t_scan), age(t_pose)))

        return im, trail_line, scan_dots, robot_dot

    # cache_frame_data=False keeps FuncAnimation from retaining every frame.
    anim = FuncAnimation(fig, update, interval=200, blit=False,
                         cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        del anim
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

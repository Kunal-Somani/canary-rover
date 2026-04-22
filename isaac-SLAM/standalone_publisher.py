#!/usr/bin/env python3
"""
standalone_publisher.py — Canary Rover sensor publisher (NO Isaac Sim required)
════════════════════════════════════════════════════════════════════════════════
Simulates the mine tunnel geometry from slam_ros2.py and publishes:
  /scan    → sensor_msgs/LaserScan   (10 Hz, frame_id = "laser")  ← FIXED
  /odom    → nav_msgs/Odometry       (20 Hz)
  /tf      → odom → base_link        (20 Hz)

Run (in a sourced ROS 2 Jazzy shell):
  Terminal 1:  python3 standalone_publisher.py
  Terminal 2:  ros2 launch canary_slam slam_toolbox.launch.py
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np

# ── Tunnel geometry (mirrors slam_ros2.py) ────────────────────────────────────
TUNNEL_R   = 1.6          # tunnel radius (m)
_TUNNEL_X0 = -2.0
_TUNNEL_X1 = 58.0
FINISH_X   = 55.0

# RPLiDAR A1M8 parameters
SCAN_RAYS      = 360
SCAN_MAX_RANGE = 3.5
SCAN_NOISE_STD = 0.02
SCAN_HZ        = 10.0

# Rover motion
ROVER_SPEED = 0.3         # m/s — forward along tunnel X-axis
ODOM_HZ     = 20.0        # odometry / TF publish rate

# Rock obstacles  (x, y, half-x, half-y)
_ROCK_XY = [
    ( 3.0,  0.6, 0.06, 0.05),
    (10.0,  0.5, 0.06, 0.05),
    (16.0, -0.6, 0.06, 0.05),
    (23.0,  0.8, 0.06, 0.05),
    (31.0,  0.3, 0.06, 0.05),
    (38.0, -0.8, 0.06, 0.05),
    (45.0,  0.6, 0.06, 0.05),
    (48.0, -0.5, 0.06, 0.05),
]

_rng = np.random.default_rng(seed=0)

# ── Ray casting (identical to slam_ros2.py) ───────────────────────────────────
def _cast_ray_2d(ox: float, oy: float, angle: float) -> float:
    """2-D horizontal ray → range (m), including tunnel walls + rocks."""
    dx  = math.cos(angle)
    dy  = math.sin(angle)
    t   = SCAN_MAX_RANGE
    eps = 1e-9

    # Tunnel side walls (cylinder at y=±TUNNEL_R)
    if dy > eps:
        tc = (TUNNEL_R - oy) / dy
        if 0 < tc < t and _TUNNEL_X0 <= ox + dx*tc <= _TUNNEL_X1:
            t = tc
    if dy < -eps:
        tc = (-TUNNEL_R - oy) / dy
        if 0 < tc < t and _TUNNEL_X0 <= ox + dx*tc <= _TUNNEL_X1:
            t = tc

    # Tunnel end caps
    if dx > eps:
        tc = (_TUNNEL_X1 - ox) / dx
        if 0 < tc < t and -TUNNEL_R <= oy + dy*tc <= TUNNEL_R:
            t = tc
    if dx < -eps:
        tc = (_TUNNEL_X0 - ox) / dx
        if 0 < tc < t and -TUNNEL_R <= oy + dy*tc <= TUNNEL_R:
            t = tc

    # Rock obstacles (2-D AABB)
    for (rx, ry, rhx, rhy) in _ROCK_XY:
        x0, x1 = rx - rhx, rx + rhx
        y0, y1 = ry - rhy, ry + rhy
        tx0 = (x0 - ox)/dx if abs(dx) > eps else (-1e9 if x0 <= ox <= x1 else 1e9)
        tx1 = (x1 - ox)/dx if abs(dx) > eps else tx0
        ty0 = (y0 - oy)/dy if abs(dy) > eps else (-1e9 if y0 <= oy <= y1 else 1e9)
        ty1 = (y1 - oy)/dy if abs(dy) > eps else ty0
        if tx0 > tx1: tx0, tx1 = tx1, tx0
        if ty0 > ty1: ty0, ty1 = ty1, ty0
        te = max(tx0, ty0);  tx = min(tx1, ty1)
        if te < tx and 0 < te < t:
            t = te

    return float(np.clip(t + _rng.normal(0, SCAN_NOISE_STD), 0.01, SCAN_MAX_RANGE))


def simulate_laserscan(x_pos: float, y_pos: float = 0.0) -> list:
    step = 2 * math.pi / SCAN_RAYS
    return [_cast_ray_2d(x_pos, y_pos, i * step) for i in range(SCAN_RAYS)]


# ── ROS 2 node ────────────────────────────────────────────────────────────────
class StandalonePublisher(Node):

    def __init__(self):
        super().__init__("canary_standalone")

        # QoS profiles (match slam_ros2.py)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pub_scan = self.create_publisher(LaserScan, "/scan", sensor_qos)
        self.pub_odom = self.create_publisher(Odometry,  "/odom", reliable_qos)
        self.tf_bcast = TransformBroadcaster(self)

        self.x_pos = 0.0                   # rover position along tunnel
        self._dt_odom = 1.0 / ODOM_HZ
        self._dt_scan = 1.0 / SCAN_HZ

        # Odometry + TF at ODOM_HZ
        self.create_timer(self._dt_odom, self._publish_odom_tf)
        # LaserScan at SCAN_HZ
        self.create_timer(self._dt_scan, self._publish_scan)

        self.get_logger().info(
            "=== Canary Standalone Publisher ===\n"
            "  Topics: /scan (laser, 10 Hz)  /odom (20 Hz)  /tf (odom→base_link)\n"
            "  Rover speed: %.1f m/s  Tunnel length: %.0f m\n"
            "  Ready — start slam_toolbox launch now." % (ROVER_SPEED, FINISH_X)
        )

    # ── Odometry + TF (20 Hz) ─────────────────────────────────────────────────
    def _publish_odom_tf(self):
        self.x_pos += ROVER_SPEED * self._dt_odom
        if self.x_pos >= FINISH_X:
            self.x_pos = 0.0          # loop back to start
            self.get_logger().info("Rover looped back to start.")

        stamp = self.get_clock().now().to_msg()

        # Odometry message
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x = self.x_pos
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.w = 1.0
        odom.pose.covariance[0]  = 0.01   # x variance
        odom.pose.covariance[7]  = 0.01   # y variance
        odom.pose.covariance[35] = 0.001  # yaw variance
        odom.twist.twist.linear.x = ROVER_SPEED
        odom.twist.covariance[0]  = 0.01
        self.pub_odom.publish(odom)

        # TF: odom → base_link  (required by slam_toolbox)
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id  = "base_link"
        tf.transform.translation.x = self.x_pos
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w    = 1.0
        self.tf_bcast.sendTransform(tf)

    # ── LaserScan (10 Hz) ─────────────────────────────────────────────────────
    def _publish_scan(self):
        ranges = simulate_laserscan(self.x_pos)

        stamp = self.get_clock().now().to_msg()
        msg = LaserScan()
        msg.header.stamp     = stamp
        # ✅ FIX: use "laser" to match the static TF (base_link→laser)
        #    published by slam_toolbox.launch.py
        msg.header.frame_id  = "laser"
        msg.angle_min        = 0.0
        msg.angle_max        = 2*math.pi - (2*math.pi / SCAN_RAYS)
        msg.angle_increment  = 2*math.pi / SCAN_RAYS
        msg.time_increment   = (1.0/SCAN_HZ) / SCAN_RAYS
        msg.scan_time        = 1.0 / SCAN_HZ
        msg.range_min        = 0.10
        msg.range_max        = SCAN_MAX_RANGE
        msg.ranges           = [float(r) for r in ranges]
        msg.intensities      = [100.0] * SCAN_RAYS
        self.pub_scan.publish(msg)


def main():
    print("\n" + "="*60)
    print("  Canary Rover — Standalone ROS 2 Publisher (no Isaac Sim)")
    print("  Terminal 1 (this):  python3 standalone_publisher.py")
    print("  Terminal 2:         ros2 launch canary_slam slam_toolbox.launch.py")
    print("="*60 + "\n")

    rclpy.init()
    node = StandalonePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

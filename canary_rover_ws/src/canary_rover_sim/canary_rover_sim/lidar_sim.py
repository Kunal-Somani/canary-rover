"""
RPLiDAR Simulation Node — Canary Rover
Simulates a 360° RPLiDAR A1M8 scan inside a mine tunnel.
Laser-based → works in both DAY and NIGHT (no ambient light needed).

Publishes : /scan  (sensor_msgs/LaserScan)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import numpy as np

class LidarSimNode(Node):
    def __init__(self):
        super().__init__('lidar_sim_node')
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)

        # RPLiDAR A1M8 hardware parameters
        self.NUM_RAYS       = 360
        self.ANGLE_MIN      = 0.0
        self.ANGLE_MAX      = 2 * math.pi
        self.ANGLE_INC      = (2 * math.pi) / self.NUM_RAYS
        self.RANGE_MIN      = 0.15
        self.RANGE_MAX      = 12.0
        self.SCAN_RATE_HZ   = 10.0

        # Simulated static tunnel geometry (angle_start°, angle_end°, dist m)
        self.tunnel_walls = [
            (0,   40,  2.5),   # Front wall / entry
            (40,  80,  3.2),   # Front-right open
            (80,  130, 1.5),   # RIGHT tunnel wall
            (130, 170, 2.0),   # Rear-right
            (170, 220, 3.0),   # Rear open
            (220, 280, 1.5),   # LEFT tunnel wall
            (280, 330, 2.2),   # Front-left
            (330, 360, 2.5),   # Front (wrap)
        ]

        self.scan_count = 0
        self.timer = self.create_timer(1.0 / self.SCAN_RATE_HZ, self.publish_scan)

        self.get_logger().info('------------')
        self.get_logger().info(' RPLiDAR Simulation Node: STARTED')
        self.get_logger().info(f' Rays={self.NUM_RAYS} | Range={self.RANGE_MIN}–{self.RANGE_MAX} m | {self.SCAN_RATE_HZ} Hz')
        self.get_logger().info(' Laser-based: Day AND Night capable')
        self.get_logger().info('------------')

    def publish_scan(self):
        msg                 = LaserScan()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'
        msg.angle_min       = self.ANGLE_MIN
        msg.angle_max       = self.ANGLE_MAX
        msg.angle_increment = self.ANGLE_INC
        msg.time_increment  = 0.0
        msg.scan_time       = 1.0 / self.SCAN_RATE_HZ
        msg.range_min       = self.RANGE_MIN
        msg.range_max       = self.RANGE_MAX

        ranges      = []
        intensities = []
        t           = self.scan_count * (1.0 / self.SCAN_RATE_HZ)

        for deg in range(self.NUM_RAYS):
            base = self.RANGE_MAX  # default: open space

            # Static tunnel walls
            for (a_start, a_end, dist) in self.tunnel_walls:
                if a_start <= deg < a_end:
                    base = dist
                    break

            # Dynamic obstacle: person/gas cloud oscillating near 90°
            if 80 <= deg <= 100:
                dyn  = 1.2 + 0.6 * math.sin(t * 0.8)
                base = min(base, dyn)

            # Simulated rock debris in 200–240° zone
            if 200 <= deg <= 240:
                rock = base - 0.4 * abs(math.sin(deg * 0.5 + t * 2))
                base = max(self.RANGE_MIN, rock)

            # Sensor noise: RPLiDAR A1 ≈ ±1.5 cm RMS
            noise = np.random.normal(0.0, 0.015)
            r     = float(np.clip(base + noise, self.RANGE_MIN, self.RANGE_MAX))
            ranges.append(r)
            intensities.append(float(max(0.0, 5000.0 / (r ** 2))))

        msg.ranges      = ranges
        msg.intensities = intensities
        self.pub_scan.publish(msg)
        self.scan_count += 1

        # Console summary every 2 seconds
        if self.scan_count % (int(self.SCAN_RATE_HZ) * 2) == 0:
            min_r   = min(ranges)
            min_deg = ranges.index(min_r)
            self.get_logger().info(
                f'[LiDAR #{self.scan_count:04d}] '
                f'Closest: {min_r:.2f} m @ {min_deg}° | '
                f'Dynamic obstacle @ 90°: {ranges[90]:.2f} m | '
                f'L-wall @ 250°: {ranges[250]:.2f} m | '
                f'R-wall @ 100°: {ranges[100]:.2f} m'
            )


def main(args=None):
    rclpy.init(args=args)
    node = LidarSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
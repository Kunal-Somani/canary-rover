"""
IMU Simulation Node — Canary Rover
Simulates MPU6050 / BNO055 IMU on rover chassis.
Cycles through mine terrain phases: Flat → Slope → Rocky → Ramp

Publishes:
  /imu/data          (sensor_msgs/Imu)
  /imu/terrain_slope (std_msgs/Float32)  — slope angle in degrees
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
import math
import numpy as np


class ImuSimNode(Node):

    # (duration_s, pitch_deg, roll_deg, label)
    TERRAIN = [
        (3.0,   0.0,   0.0,  'Flat ground — mine entry'),
        (2.5,  12.0,   2.0,  'Ascending slope 12 deg'),
        (2.0,  26.0,   5.0,  'Steep incline 26 deg (near design limit 25-30 deg)'),
        (1.5,  10.0,  -8.0,  'Rocky bump — sharp RIGHT tilt'),
        (1.5,  10.0,   9.0,  'Rocky bump — sharp LEFT tilt'),
        (2.0, -10.0,   1.0,  'Descending slope after obstacle'),
        (2.5,   0.0,   0.0,  'Flat gallery section'),
        (2.5,  20.0,   0.0,  'Ramp climb 20 deg'),
        (1.5,  20.0,  -6.0,  'Ramp + side camber'),
        (2.0,   0.0,   0.0,  'Flat — inspection zone'),
    ]

    def __init__(self):
        super().__init__('imu_sim_node')

        self.pub_imu   = self.create_publisher(Imu,     '/imu/data',          10)
        self.pub_slope = self.create_publisher(Float32, '/imu/terrain_slope', 10)

        # MPU6050 parameters
        self.IMU_RATE_HZ  = 50.0
        self.DT           = 1.0 / self.IMU_RATE_HZ
        self.GYRO_NOISE   = 0.005   # rad/s RMS
        self.ACCEL_NOISE  = 0.02    # m/s² RMS
        self.G            = 9.81

        self.time_s    = 0.0
        self.phase_idx = 0
        self.phase_t   = 0.0

        self.timer = self.create_timer(self.DT, self.publish_imu)

        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info(' IMU Simulation Node  ▶  STARTED')
        self.get_logger().info(f' Rate={self.IMU_RATE_HZ} Hz | Gyro noise={self.GYRO_NOISE} rad/s | Accel noise={self.ACCEL_NOISE} m/s2')
        self.get_logger().info(f' Terrain phases: {len(self.TERRAIN)} phases loaded')
        self.get_logger().info('═══════════════════════════════════════════')

    def step_terrain(self):
        dur, pitch_d, roll_d, label = self.TERRAIN[self.phase_idx]
        self.phase_t += self.DT
        if self.phase_t >= dur:
            self.phase_t  = 0.0
            self.phase_idx = (self.phase_idx + 1) % len(self.TERRAIN)
            nxt = self.TERRAIN[self.phase_idx]
            self.get_logger().info(
                f'[IMU] Phase --> "{nxt[3]}"  pitch={nxt[1]:.1f} deg  roll={nxt[2]:.1f} deg'
            )
        return math.radians(pitch_d), math.radians(roll_d)

    @staticmethod
    def rpy_to_quat(roll, pitch, yaw):
        cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def publish_imu(self):
        pitch, roll = self.step_terrain()
        yaw = 0.05 * self.time_s

        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)

        ax = self.G * math.sin(pitch)                           + np.random.normal(0, self.ACCEL_NOISE)
        ay = -self.G * math.sin(roll)                           + np.random.normal(0, self.ACCEL_NOISE)
        az = self.G * math.cos(pitch) * math.cos(roll)         + np.random.normal(0, self.ACCEL_NOISE)

        gx = np.random.normal(0,           self.GYRO_NOISE)
        gy = np.random.normal(pitch * 0.1, self.GYRO_NOISE)
        gz = np.random.normal(0.05,        self.GYRO_NOISE)

        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_frame'

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.orientation_covariance        = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]

        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz
        msg.angular_velocity_covariance   = [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4]

        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        msg.linear_acceleration_covariance = [1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1e-3]

        self.pub_imu.publish(msg)

        slope_deg       = math.degrees(math.atan2(math.sqrt(ax**2 + ay**2), az))
        slope_msg       = Float32()
        slope_msg.data  = float(slope_deg)
        self.pub_slope.publish(slope_msg)

        self.time_s += self.DT

        # Console log every 1 second
        if abs(self.time_s % 1.0) < self.DT:
            self.get_logger().info(
                f'[IMU] pitch={math.degrees(pitch):+6.1f} deg  '
                f'roll={math.degrees(roll):+6.1f} deg  '
                f'slope={slope_deg:5.1f} deg  '
                f'az={az:5.2f} m/s2  '
                f'gy={gy:+.4f} rad/s'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ImuSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
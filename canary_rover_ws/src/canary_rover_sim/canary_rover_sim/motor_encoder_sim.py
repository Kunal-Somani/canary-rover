"""
Encoded Motor + Torque Controller — Canary Rover
4x BLDC motors with quadrature encoders.
Reads IMU terrain slope and adapts torque per wheel automatically.

Subscribes:
  /cmd_vel           (geometry_msgs/Twist)
  /imu/terrain_slope (std_msgs/Float32)

Publishes:
  /motor/encoder_data    (std_msgs/Int32MultiArray)    ticks [FL,FR,RL,RR]
  /motor/torque_feedback (std_msgs/Float32MultiArray)  Nm   [FL,FR,RL,RR]
  /motor/wheel_velocity  (std_msgs/Float32MultiArray)  rad/s [FL,FR,RL,RR]
  /motor/status          (std_msgs/String)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Float32, String
import math
import numpy as np


class MotorEncoderSimNode(Node):

    # Demo command sequence — runs automatically
    # (linear_x m/s, angular_z rad/s, label, hold_steps)
    DEMO_SEQ = [
        (0.30,  0.00, 'Straight forward — flat terrain',      25),
        (0.25,  0.40, 'Right turn',                           15),
        (0.25, -0.40, 'Left turn',                            15),
        (0.20,  0.00, 'Slow forward — ascending slope',       25),
        (0.10,  0.00, 'Crawl — steep rocky terrain',          20),
        (0.30,  0.00, 'Resume speed — flat gallery',          20),
        (0.00,  0.00, 'STOP',                                 10),
    ]

    def __init__(self):
        super().__init__('motor_encoder_sim_node')

        # Subscribers
        self.sub_cmd   = self.create_subscription(Twist,   '/cmd_vel',           self.cb_cmd,   10)
        self.sub_slope = self.create_subscription(Float32, '/imu/terrain_slope', self.cb_slope, 10)

        # Publishers
        self.pub_enc    = self.create_publisher(Int32MultiArray,   '/motor/encoder_data',    10)
        self.pub_torque = self.create_publisher(Float32MultiArray, '/motor/torque_feedback', 10)
        self.pub_vel    = self.create_publisher(Float32MultiArray, '/motor/wheel_velocity',  10)
        self.pub_status = self.create_publisher(String,            '/motor/status',          10)

        # Motor / rover parameters
        self.TICKS_PER_REV = 1000
        self.WHEEL_RADIUS  = 0.06    # m
        self.TRACK_WIDTH   = 0.35    # m
        self.ROVER_MASS    = 8.0     # kg
        self.MAX_TORQUE    = 5.0     # Nm per motor
        self.MAX_RPM       = 150.0
        self.MAX_RAD_S     = (self.MAX_RPM / 60.0) * 2 * math.pi
        self.MOTOR_TAU     = 0.15    # first-order lag
        self.CTRL_HZ       = 50.0
        self.DT            = 1.0 / self.CTRL_HZ

        # State
        self.ticks     = [0, 0, 0, 0]
        self.vel_act   = [0.0, 0.0, 0.0, 0.0]
        self.cmd_lin   = 0.0
        self.cmd_ang   = 0.0
        self.slope_deg = 0.0
        self.log_ctr   = 0

        # Demo state
        self.demo_idx   = 0
        self.demo_ticks = 0

        self.timer_ctrl = self.create_timer(self.DT,  self.control_loop)
        self.timer_demo = self.create_timer(0.4,      self.demo_step)

        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info(' Encoded Motor Simulation Node  ▶  STARTED')
        self.get_logger().info(f' 4x BLDC | Encoder={self.TICKS_PER_REV} ticks/rev')
        self.get_logger().info(f' MaxTorque={self.MAX_TORQUE} Nm | MaxRPM={self.MAX_RPM}')
        self.get_logger().info(' Torque auto-compensates for IMU slope in real-time')
        self.get_logger().info('═══════════════════════════════════════════')

    def cb_cmd(self, msg: Twist):
        self.cmd_lin = msg.linear.x
        self.cmd_ang = msg.angular.z

    def cb_slope(self, msg: Float32):
        self.slope_deg = msg.data

    def demo_step(self):
        lin, ang, label, hold = self.DEMO_SEQ[self.demo_idx]
        self.cmd_lin    = lin
        self.cmd_ang    = ang
        self.demo_ticks += 1
        if self.demo_ticks == 1:
            self.get_logger().info(f'[DEMO CMD] --> "{label}"  lin={lin} m/s  ang={ang} rad/s')
        if self.demo_ticks >= hold:
            self.demo_ticks = 0
            self.demo_idx   = (self.demo_idx + 1) % len(self.DEMO_SEQ)

    def kinematics(self, v, omega):
        v_l = (v - omega * self.TRACK_WIDTH / 2.0) / self.WHEEL_RADIUS
        v_r = (v + omega * self.TRACK_WIDTH / 2.0) / self.WHEEL_RADIUS
        return [v_l, v_r, v_l, v_r]   # FL, FR, RL, RR

    def required_torque(self, wheel_rad_s: float, slope_deg: float) -> float:
        g       = 9.81
        slope_r = math.radians(abs(slope_deg))
        mu      = 0.70

        N        = (self.ROVER_MASS * g * math.cos(slope_r)) / 4.0
        tau_trac = mu * N * self.WHEEL_RADIUS * 0.3
        tau_grav = (self.ROVER_MASS * g * math.sin(slope_r) * self.WHEEL_RADIUS) / 4.0

        vib = 1.0
        if abs(slope_deg) > 10.0:
            t   = self.get_clock().now().nanoseconds * 1e-9
            vib = 1.0 + 0.25 * abs(math.sin(t * 4.0))

        base  = (tau_trac + tau_grav) * max(0.1, abs(wheel_rad_s)) * vib
        noise = np.random.normal(0, 0.04)
        return float(np.clip(base + noise, 0.0, self.MAX_TORQUE))

    def control_loop(self):
        desired = self.kinematics(self.cmd_lin, self.cmd_ang)
        torques = []

        for i, (des, act) in enumerate(zip(desired, self.vel_act)):
            new_vel         = self.MOTOR_TAU * des + (1 - self.MOTOR_TAU) * act
            new_vel         = float(np.clip(new_vel, -self.MAX_RAD_S, self.MAX_RAD_S))
            self.vel_act[i] = new_vel

            delta           = int((new_vel * self.DT / (2 * math.pi)) * self.TICKS_PER_REV)
            self.ticks[i]  += delta

            torques.append(self.required_torque(new_vel, self.slope_deg))

        enc_msg      = Int32MultiArray()
        enc_msg.data = list(self.ticks)
        self.pub_enc.publish(enc_msg)

        torq_msg      = Float32MultiArray()
        torq_msg.data = [float(t) for t in torques]
        self.pub_torque.publish(torq_msg)

        vel_msg      = Float32MultiArray()
        vel_msg.data = [float(v) for v in self.vel_act]
        self.pub_vel.publish(vel_msg)

        labels = ['FL', 'FR', 'RL', 'RR']
        status = (
            f"slope={self.slope_deg:5.1f} deg | "
            + " | ".join(
                f"{lb}: vel={v:+5.2f} rad/s  torque={t:.2f} Nm  ticks={tk}"
                for lb, v, t, tk in zip(labels, self.vel_act, torques, self.ticks)
            )
        )
        s_msg      = String()
        s_msg.data = status
        self.pub_status.publish(s_msg)

        self.log_ctr += 1
        if self.log_ctr % int(self.CTRL_HZ) == 0:
            avg_torque = sum(torques) / 4
            self.get_logger().info(
                f'[Motor] slope={self.slope_deg:5.1f} deg | '
                f'cmd=({self.cmd_lin:.2f} m/s, {self.cmd_ang:.2f} rad/s) | '
                f'avg_torque={avg_torque:.2f} Nm | '
                f'ticks={self.ticks}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = MotorEncoderSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
# Copyright 2024 Kunal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
full_sim.launch.py — Canary Rover Simulation
Launches all three sensor/actuator simulation nodes together:
  1. lidar_sim_node       — RPLiDAR A1M8  → /scan
  2. imu_sim_node         — MPU6050 IMU   → /imu/data, /imu/terrain_slope
  3. motor_encoder_sim    — 4x BLDC motor → /motor/encoder_data, /motor/torque_feedback
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='canary_rover_sim',
            executable='lidar_sim',
            name='lidar_sim_node',
            output='screen',
            emulate_tty=True,
        ),

        Node(
            package='canary_rover_sim',
            executable='imu_sim',
            name='imu_sim_node',
            output='screen',
            emulate_tty=True,
        ),

        Node(
            package='canary_rover_sim',
            executable='motor_encoder_sim',
            name='motor_encoder_sim_node',
            output='screen',
            emulate_tty=True,
        ),

    ])
#!/bin/bash

source /opt/ros/humble/setup.bash
source ~/Desktop/Kunal_Personal/Capstone/isaac_ws/install/setup.bash 2>/dev/null || true

ros2 topic list
ros2 topic hz /scan --window 5
ros2 topic hz /tf --window 5
ros2 topic hz /odom --window 5
ros2 topic echo /imu/data --once
echo "--- TF tree ---"
ros2 run tf2_tools view_frames

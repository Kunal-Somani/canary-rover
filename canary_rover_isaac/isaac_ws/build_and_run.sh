#!/bin/bash

# STEP 1 — Install missing deps
echo "Installing missing dependencies..."
sudo apt install -y ros-humble-slam-toolbox \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-rviz2 \
  ros-humble-nav2-map-server

# STEP 2 — Source ROS2 and build
echo "Sourcing ROS2 and building workspace..."
source /opt/ros/humble/setup.bash
cd ~/Desktop/Kunal_Personal/Capstone/isaac_ws
colcon build --symlink-install 2>&1 | tee build.log

# Check for errors
if grep -q "^CMake Error\|^error:" build.log; then
  echo "BUILD FAILED — check build.log"; exit 1
fi
source install/setup.bash

# STEP 3 — Verify topics will exist (pre-check)
echo "Checking ROS2 environment..."
ros2 pkg list | grep canary || { echo "Packages not found!"; exit 1; }

# STEP 4 — Print launch instructions clearly
echo ""
echo "============================================================"
echo " CANARY ROVER LAUNCH SEQUENCE"
echo "============================================================"
echo ""
echo " TERMINAL 1 (Isaac Sim):"
echo "   cd ~/Desktop/Kunal_Personal/Capstone/isaac_sim_demo"
echo "   ~/.local/share/ov/pkg/isaac-sim-5.1.0/python.sh canary_demo.py"
echo ""
echo " TERMINAL 2 (ROS2 stack — wait for Isaac to publish /scan):"
echo "   source ~/Desktop/Kunal_Personal/Capstone/isaac_ws/install/setup.bash"
echo "   ros2 launch canary_bringup canary_full.launch.py"
echo ""
echo " TERMINAL 3 (diagnostics):"
echo "   source ~/Desktop/Kunal_Personal/Capstone/isaac_ws/install/setup.bash"
echo "   ros2 topic hz /scan   # should be ~10Hz"
echo "   ros2 topic hz /imu/data  # should be ~1Hz (every 60 steps)"
echo "   ros2 topic echo /slam_path --once"
echo "============================================================"

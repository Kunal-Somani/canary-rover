"""
2-D SLAM using slam_toolbox (online async mode)
Run with:
  ros2 launch canary_slam slam_toolbox.launch.py

FIXES applied:
  1. All slam_toolbox parameters inline — no yaml path dependency.
  2. Lifecycle manager added — slam_toolbox in ROS 2 Jazzy is a lifecycle node.
     Without auto-activating it, the node sits in 'unconfigured' state forever
     with Subscription count=0 on /scan and no /map published.

Subscribes to:
  /scan   (sensor_msgs/LaserScan)  — standalone_publisher.py or slam_ros2.py
  /tf     (odom → base_link)       — standalone_publisher.py or slam_ros2.py

Publishes:
  /map    (nav_msgs/OccupancyGrid)
  /tf     (map → odom)
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── static TF: base_link → laser ─────────────────────────────────────
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_laser",
            arguments=[
                "0", "0", "0.32",   # x y z — sensor height above base_link
                "0", "0", "0",      # roll pitch yaw
                "base_link", "laser",
            ],
        ),

        # ── slam_toolbox async (lifecycle node) ───────────────────────────────
        # FIX 1: All parameters inline — avoids yaml path resolution failure.
        # FIX 2: In ROS 2 Jazzy, async_slam_toolbox_node is a lifecycle node.
        #        It starts in 'unconfigured' state and subscribes to NOTHING
        #        until configured+activated. The lifecycle_manager below handles
        #        this automatically via autostart=True.
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[{
                # Frames
                "odom_frame":   "odom",
                "map_frame":    "map",
                "base_frame":   "base_link",
                "scan_topic":   "/scan",
                "use_sim_time": False,

                # Mode
                "mode": "mapping",

                # Solver
                "solver_plugin":        "solver_plugins::CeresSolver",
                "ceres_linear_solver":  "SPARSE_NORMAL_CHOLESKY",
                "ceres_preconditioner": "SCHUR_JACOBI",
                "ceres_trust_strategy": "LEVENBERG_MARQUARDT",
                "ceres_dogleg_type":    "TRADITIONAL_DOGLEG",

                # Scan buffer
                "scan_buffer_size":                  10,
                "scan_buffer_maximum_scan_distance": 10.0,

                # Link matching — tuned for narrow mine tunnel
                "link_match_minimum_response_fine":   0.1,
                "link_scan_maximum_distance":         2.0,
                "loop_search_maximum_distance":       5.0,
                "do_loop_closing":                    True,
                "loop_match_minimum_chain_size":      3,
                "loop_match_maximum_variance_coarse": 3.0,
                "loop_match_minimum_response_coarse": 0.35,
                "loop_match_minimum_response_fine":   0.45,

                # Correlation
                "correlation_search_space_dimension":       0.5,
                "correlation_search_space_resolution":      0.01,
                "correlation_search_space_smear_deviation": 0.1,

                # Motion thresholds — rover moves slowly in mine
                "minimum_travel_distance": 0.15,
                "minimum_travel_heading":  0.05,

                # Map
                "map_update_interval": 2.0,
                "resolution":          0.05,

                # Performance
                "throttle_scans":           1,
                "transform_publish_period": 0.02,
                "transform_timeout":        0.5,
                "tf_buffer_duration":       30.0,
            }],
        ),

        # ── Lifecycle manager — CRITICAL for ROS 2 Jazzy ─────────────────────
        # slam_toolbox is a lifecycle node. Without this it sits forever in
        # 'unconfigured' state and never subscribes to /scan.
        # autostart=True drives it: unconfigured → configured → active
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_slam",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "autostart":    True,
                "bond_timeout": 0.0,
                "node_names":   ["slam_toolbox"],
            }],
        ),

        # ── RViz2 ─────────────────────────────────────────────────────────────
        # PathJoinSubstitution removed — avoids crash if slam_2d.rviz is
        # missing from install/. Add Map manually: Add → By topic → /map → Map
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
        ),
    ])

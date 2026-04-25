"""
3-D SLAM using RTAB-Map (real-time appearance-based mapping)
Run with:
  ros2 launch canary_slam rtabmap.launch.py

Subscribes to:
  /velodyne_points  (sensor_msgs/PointCloud2) — VLP-16 sim from slam_ros2.py
  /imu/data         (sensor_msgs/Imu)
  /odom             (nav_msgs/Odometry)

Publishes:
  /rtabmap/map        (nav_msgs/OccupancyGrid)  — 2-D projection
  /rtabmap/cloud_map  (sensor_msgs/PointCloud2) — 3-D coloured map
  /rtabmap/grid_map   (nav_msgs/OccupancyGrid)  — 2.5-D elevation
  /tf                 (map → odom)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pkg = FindPackageShare("canary_slam")

    return LaunchDescription([

        # ── static TF: base_link → velodyne ───────────────────────────────────
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_velodyne",
            arguments=["0", "0", "0.32",   # sensor height above base
                       "0", "0", "0",
                       "base_link", "velodyne"],
        ),

        # ── RTAB-Map LiDAR-only 3-D SLAM ─────────────────────────────────────
        # Docs: https://github.com/introlab/rtabmap_ros
        Node(
            package="rtabmap_odom",
            executable="icp_odometry",
            name="icp_odometry",
            output="screen",
            parameters=[{
                "use_sim_time":          False,
                "frame_id":              "base_link",
                "odom_frame_id":         "odom",
                "wait_for_transform":    0.2,
                "expected_update_rate":  10.0,
                # ICP settings tuned for a mine tunnel (long, thin corridor)
                "Icp/MaxCorrespondenceDistance": "0.5",
                "Icp/PointToPlane":      "true",
                "Icp/Iterations":        "20",
                "OdomF2M/MaxSize":       "5000",
            }],
            remappings=[
                ("scan_cloud", "/velodyne_points"),
                ("odom",       "/odom_icp"),        # renamed to avoid conflict
            ],
        ),

        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[
                PathJoinSubstitution([pkg, "config", "rtabmap.yaml"]),
                {
                    "use_sim_time":    False,
                    "frame_id":        "base_link",
                    "odom_frame_id":   "odom",
                    "map_frame_id":    "map",
                    # Use ICP odom so we don't need a camera
                    "subscribe_scan_cloud": True,
                    "subscribe_odom":       True,
                    "subscribe_imu":        True,
                    # Mapping params tuned for mine tunnel
                    "Grid/RangeMax":        "8.0",
                    "Grid/CellSize":        "0.10",
                    "Grid/3D":              "true",
                    "Mem/IncrementalMemory":"true",
                    "Rtabmap/DetectionRate":"1.0",
                },
            ],
            remappings=[
                ("scan_cloud",   "/velodyne_points"),
                ("odom",         "/odom_icp"),
                ("imu",          "/imu/data"),
            ],
        ),

        # ── RViz2 with 3-D config ─────────────────────────────────────────────
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", PathJoinSubstitution([pkg, "rviz", "slam_3d.rviz"])],
        ),
    ])

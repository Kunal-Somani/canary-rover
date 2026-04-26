import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    canary_slam_dir = get_package_share_directory('canary_slam')
    canary_bringup_dir = get_package_share_directory('canary_bringup')
    rviz_config_file = os.path.join(canary_bringup_dir, 'config', 'canary_rviz.rviz')

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(canary_slam_dir, 'launch', 'slam.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    slam_monitor_node = Node(
        package='canary_slam',
        executable='slam_monitor',
        name='slam_monitor',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    path_painter_node = Node(
        package='canary_visualizer',
        executable='path_painter',
        name='path_painter',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        slam_launch,
        slam_monitor_node,
        path_painter_node,
        rviz_node
    ])

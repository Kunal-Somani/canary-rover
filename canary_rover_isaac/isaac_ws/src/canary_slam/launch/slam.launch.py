import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('canary_slam')
    config_file = os.path.join(pkg_dir, 'config', 'slam_params.yaml')

    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_laser',
        arguments=['0', '0', '0.4', '0', '0', '0', 'base_link', 'laser']
    )

    urdf_path = os.path.expanduser('~/Desktop/Kunal_Personal/Capstone/canary_rover_isaac/isaac_sim_demo/vleg_rover.urdf')
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': urdf_content}]
    )

    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    return LaunchDescription([
        tf_node,
        rsp_node,
        slam_node
    ])

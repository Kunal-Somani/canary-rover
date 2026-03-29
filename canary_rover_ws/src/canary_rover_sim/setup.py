from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'canary_rover_sim'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kunal',
    maintainer_email='kunal120222@gmail.com',
    description='Canary Rover — RPLiDAR, IMU, Motor Encoder Simulation in ROS2 Humble',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lidar_sim         = canary_rover_sim.lidar_sim:main',
            'imu_sim           = canary_rover_sim.imu_sim:main',
            'motor_encoder_sim = canary_rover_sim.motor_encoder_sim:main',
        ],
    },
)
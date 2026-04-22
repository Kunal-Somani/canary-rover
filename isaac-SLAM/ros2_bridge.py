"""
Runs under system Python 3.12 with ROS 2 Jazzy.
Reads sensor data written by Isaac Sim and publishes to ROS 2 topics.
Run with:  python3 ~/ros2_bridge.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
import json, math, time, os
import numpy as np

SHARED_FILE = "/tmp/canary_sensor_data.json"

class BridgeNode(Node):
    def __init__(self):
        super().__init__("canary_bridge")
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pub_scan  = self.create_publisher(LaserScan,   "/scan",            sensor_qos)
        self.pub_imu   = self.create_publisher(Imu,         "/imu/data",        sensor_qos)
        self.pub_odom  = self.create_publisher(Odometry,    "/odom",            reliable_qos)
        self.tf_bcast  = TransformBroadcaster(self)
        self.timer     = self.create_timer(0.05, self.publish_from_file)
        self.last_mtime = 0
        self.get_logger().info("Bridge ready — reading from " + SHARED_FILE)

    def publish_from_file(self):
        if not os.path.exists(SHARED_FILE):
            return
        try:
            mtime = os.path.getmtime(SHARED_FILE)
            if mtime == self.last_mtime:
                return
            self.last_mtime = mtime
            with open(SHARED_FILE, "r") as f:
                d = json.load(f)
        except Exception:
            return

        stamp = self.get_clock().now().to_msg()

        # Odometry + TF
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x = d["x"]
        odom.pose.pose.position.y = d["y"]
        odom.pose.pose.position.z = d["z"]
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = d["vx"]
        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id  = "base_link"
        tf.transform.translation.x = d["x"]
        tf.transform.translation.y = d["y"]
        tf.transform.translation.z = d["z"]
        tf.transform.rotation.w = 1.0
        self.tf_bcast.sendTransform(tf)

        # IMU
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "base_link"
        imu.linear_acceleration.x = d["ax"]
        imu.linear_acceleration.y = d["ay"]
        imu.linear_acceleration.z = d["az"]
        imu.angular_velocity.x = d["gx"]
        imu.angular_velocity.y = d["gy"]
        imu.angular_velocity.z = d["gz"]
        imu.orientation.w = 1.0
        imu.orientation_covariance[0] = -1.0
        self.pub_imu.publish(imu)

        # LaserScan
        if "ranges" in d:
            scan = LaserScan()
            scan.header.stamp = stamp
            # FIX: use "laser" to match the static TF (base_link→laser) in launch file
            scan.header.frame_id = "laser"
            scan.angle_min       = 0.0
            scan.angle_max       = 2 * math.pi - (2 * math.pi / len(d["ranges"]))
            scan.angle_increment = 2 * math.pi / len(d["ranges"])
            scan.range_min       = 0.10
            scan.range_max       = 3.5
            scan.scan_time       = 0.1
            scan.time_increment  = 0.1 / len(d["ranges"])
            scan.ranges          = [float(r) for r in d["ranges"]]
            scan.intensities     = [100.0] * len(d["ranges"])
            self.pub_scan.publish(scan)

def main():
    rclpy.init()
    node = BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

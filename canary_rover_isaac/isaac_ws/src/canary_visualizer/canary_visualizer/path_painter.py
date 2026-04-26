import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return 0.0, 0.0, math.atan2(siny_cosp, cosy_cosp)

class PathPainter(Node):
    def __init__(self):
        super().__init__('path_painter')
        
        self.declare_parameter('junction_x_threshold', 17.0)
        self.declare_parameter('dead_end_range_threshold', 0.6)
        self.declare_parameter('dead_end_cone_half_angle', 0.26)
        
        self.junction_x_threshold = self.get_parameter('junction_x_threshold').value
        self.dead_end_range_threshold = self.get_parameter('dead_end_range_threshold').value
        self.dead_end_cone_half_angle = self.get_parameter('dead_end_cone_half_angle').value
        
        self.state = 0
        self.branches_explored = 0
        self.dead_end_detected = False
        self.state_1_start_time = None
        
        self.visited_red_cells = set()
        self.red_points = []
        self.green_points = []
        
        self.path_pub = self.create_publisher(MarkerArray, '/slam_path', 10)
        self.arrow_pub = self.create_publisher(MarkerArray, '/slam_arrows', 10)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
    def add_red_point(self, x, y):
        if not self.red_points or math.hypot(x - self.red_points[-1].x, y - self.red_points[-1].y) > 0.05:
            rx, ry = round(x, 1), round(y, 1)
            self.visited_red_cells.add((rx, ry))
            p = Point()
            p.x, p.y, p.z = x, y, 0.0
            self.red_points.append(p)

    def add_green_point(self, odom_x, odom_y, yaw):
        green_x = odom_x + 0.10 * math.sin(yaw)
        green_y = odom_y - 0.10 * math.cos(yaw)
        rx, ry = round(green_x, 1), round(green_y, 1)
        if (rx, ry) not in self.visited_red_cells:
            if not self.green_points or math.hypot(green_x - self.green_points[-1].x, green_y - self.green_points[-1].y) > 0.05:
                p = Point()
                p.x, p.y, p.z = green_x, green_y, 0.0
                self.green_points.append(p)

    def publish_path(self):
        ma = MarkerArray()
        
        if self.red_points:
            m_red = Marker()
            m_red.header.frame_id = "map"
            m_red.header.stamp = self.get_clock().now().to_msg()
            m_red.ns = "canary_slam"
            m_red.id = 0
            m_red.type = Marker.LINE_STRIP
            m_red.action = Marker.ADD
            m_red.scale.x = 0.04
            m_red.color.r = 1.0
            m_red.color.g = 0.0
            m_red.color.b = 0.0
            m_red.color.a = 1.0
            m_red.points = self.red_points
            ma.markers.append(m_red)
            
        if self.green_points:
            m_green = Marker()
            m_green.header.frame_id = "map"
            m_green.header.stamp = self.get_clock().now().to_msg()
            m_green.ns = "canary_slam"
            m_green.id = 1000
            m_green.type = Marker.LINE_STRIP
            m_green.action = Marker.ADD
            m_green.scale.x = 0.04
            m_green.color.r = 0.0
            m_green.color.g = 0.8
            m_green.color.b = 0.2
            m_green.color.a = 1.0
            m_green.points = self.green_points
            ma.markers.append(m_green)
            
        if ma.markers:
            self.path_pub.publish(ma)

    def publish_arrows(self):
        ma = MarkerArray()
        directions = [
            (1, 1.0, 0.0), # +X
            (2, 0.0, 1.0), # +Y
            (3, 0.0, -1.0) # -Y
        ]
        for id_, dx, dy in directions:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "canary_slam"
            m.id = 2000 + id_
            m.type = Marker.ARROW
            m.action = Marker.ADD
            
            yaw = math.atan2(dy, dx)
            
            m.pose.position.x = 17.0
            m.pose.position.y = 0.0
            m.pose.position.z = 0.0
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = math.sin(yaw * 0.5)
            m.pose.orientation.w = math.cos(yaw * 0.5)
            
            m.scale.x = 0.3
            m.scale.y = 0.06
            m.scale.z = 0.06
            
            m.color.r = 1.0
            m.color.g = 0.8
            m.color.b = 0.0
            m.color.a = 1.0
            
            ma.markers.append(m)
            
        self.arrow_pub.publish(ma)

    def scan_callback(self, msg):
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        
        min_dist = float('inf')
        cone = self.dead_end_cone_half_angle
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_inc
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if -cone <= angle <= cone:
                if not math.isinf(r) and not math.isnan(r):
                    if r < min_dist:
                        min_dist = r
        
        if min_dist < self.dead_end_range_threshold:
            self.dead_end_detected = True

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        _, _, yaw = euler_from_quaternion(msg.pose.pose.orientation)
        
        if self.state == 0:
            self.add_red_point(x, y)
            if x > self.junction_x_threshold and abs(y) < 1.0:
                self.state = 1
                self.state_1_start_time = self.get_clock().now()
                self.publish_arrows()
                
        elif self.state == 1:
            self.add_red_point(x, y)
            self.publish_arrows()
            if (self.get_clock().now() - self.state_1_start_time).nanoseconds > 3e9:
                self.state = 2
                self.branches_explored = 0
                self.dead_end_detected = False
                
        elif self.state == 2:
            self.add_red_point(x, y)
            if self.dead_end_detected:
                self.state = 3
                
        elif self.state == 3:
            self.add_green_point(x, y, yaw)
            dist_to_junction = math.hypot(x - self.junction_x_threshold, y - 0.0)
            if dist_to_junction < 1.5:
                self.branches_explored += 1
                if self.branches_explored >= 3:
                    self.state = 4
                else:
                    self.state = 2
                self.dead_end_detected = False
                
        elif self.state == 4:
            self.add_green_point(x, y, yaw)
            
        self.publish_path()

def main(args=None):
    rclpy.init(args=args)
    node = PathPainter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

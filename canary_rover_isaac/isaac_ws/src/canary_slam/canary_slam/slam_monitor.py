import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry

class SlamMonitor(Node):
    def __init__(self):
        super().__init__('slam_monitor')
        self.map_count = 0
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

    def map_callback(self, msg):
        self.map_count = sum(1 for cell in msg.data if cell != -1)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        print(f"[SLAM] x={x:.2f} y={y:.2f}  map_cells={self.map_count}")

def main(args=None):
    rclpy.init(args=args)
    node = SlamMonitor()
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

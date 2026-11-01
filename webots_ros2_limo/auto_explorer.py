import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class AutoExplorer(Node):
    def __init__(self):
        super().__init__('auto_explorer')
        
        # LIMO의 /scan 토픽은 BEST_EFFORT QoS를 사용하므로 이에 맞춰 설정합니다.
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('자율 탐색 주행 노드가 시작되었습니다!')

    def scan_callback(self, msg):
        # /scan 데이터(총 248개 빔) 중 정면 영역(중앙부 약 60개 빔) 추출
        num_beams = len(msg.ranges)
        center_idx = num_beams // 2
        front_ranges = msg.ranges[center_idx - 30 : center_idx + 30]
        
        # 센서 특성상 감지되지 않은 값(0.0)이나 무한대 값을 제외하고 유효 거리만 필터링
        valid_ranges = [r for r in front_ranges if r > 0.05 and r < 10.0]
        min_front_dist = min(valid_ranges) if valid_ranges else 10.0

        twist = Twist()
        # 전방 0.5m 이내에 장애물이 감지되면 제자리 회전하여 회피
        if min_front_dist < 0.5:
            twist.linear.x = 0.0
            twist.angular.z = 0.6  # 좌회전 (필요시 좌우 비교 로직으로 고도화 가능)
        else:
            twist.linear.x = 0.2   # 전진 속도 (m/s)
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AutoExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 로봇 정지
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
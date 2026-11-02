import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('auto_explorer')
        
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, scan_qos)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.target_frontier = None
        
        # 끼임 방지용 상태 변수 (회전 방향 유지 및 후진 타이머)
        self.turn_dir = 0.0
        self.escape_steps = 0
        
        self.get_logger().info('개선된 프론티어 탐색 노드(끼임 탈출 포함)가 시작되었습니다!')

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def map_cb(self, msg):
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))

        free_mask = (grid == 0)
        unknown_mask = (grid == -1)
        obstacle_mask = (grid > 50)

        # 벽(장애물) 바로 옆 3칸(약 15cm) 이내의 픽셀은 목표에서 제외 (벽 끼임 방지)
        obs_near = obstacle_mask.copy()
        for _ in range(3):
            obs_near |= (
                np.roll(obs_near, 1, axis=0) | np.roll(obs_near, -1, axis=0) |
                np.roll(obs_near, 1, axis=1) | np.roll(obs_near, -1, axis=1)
            )

        up = np.roll(unknown_mask, -1, axis=0)
        down = np.roll(unknown_mask, 1, axis=0)
        left = np.roll(unknown_mask, -1, axis=1)
        right = np.roll(unknown_mask, 1, axis=1)
        
        # 벽에서 충분히 떨어진 안전한 빈 공간의 경계선만 추출
        frontier_mask = free_mask & (~obs_near) & (up | down | left | right)
        fy, fx = np.where(frontier_mask)

        if len(fx) == 0:
            self.target_frontier = None
            return

        wx = ox + (fx + 0.5) * res
        wy = oy + (fy + 0.5) * res
        dists = np.hypot(wx - self.robot_x, wy - self.robot_y)

        # 0.8m 이상 떨어진 경계선 중 가장 가까운 곳 선택
        valid_idx = np.where(dists > 0.8)[0]
        if len(valid_idx) > 0:
            best_i = valid_idx[np.argmin(dists[valid_idx])]
            self.target_frontier = (wx[best_i], wy[best_i])
        else:
            self.target_frontier = None

    def scan_cb(self, msg):
        n = len(msg.ranges)
        c = n // 2
        
        def get_min(arr):
            # 0.0(미감지) 제외, 실제 물리적 거리만 필터링
            valid = [r for r in arr if 0.02 < r < 10.0]
            return min(valid) if valid else 10.0

        # 정면 시야각을 넓히고(±40빔), 좌우 측면 구역까지 꼼꼼히 확인
        front_dist = get_min(msg.ranges[c - 40 : c + 40])
        left_dist = get_min(msg.ranges[c + 40 : c + 105])
        right_dist = get_min(msg.ranges[c - 105 : c - 40])

        twist = Twist()

        # [상태 1] 강제 후진 탈출 루틴 진행 중일 때
        if self.escape_steps > 0:
            self.escape_steps -= 1
            twist.linear.x = -0.18  # 뒤로 후진하며 공간 확보
            twist.angular.z = self.turn_dir * 0.8
            self.cmd_pub.publish(twist)
            return

        # [상태 2] 벽이나 코너에 너무 바짝 끼었을 때 (전방 0.35m 또는 측면 0.24m 이내)
        if front_dist < 0.35 or left_dist < 0.24 or right_dist < 0.24:
            self.turn_dir = 1.0 if left_dist >= right_dist else -1.0
            self.escape_steps = 12  # 약 1.2초간 후진+회전 실행 (/scan 10Hz 기준)
            self.get_logger().info('벽 끼임 감지! 후진하여 탈출합니다.', throttle_duration_sec=2.0)
            return

        # [상태 3] 전방 장애물 감지 시 (0.65m 이내) -> 한쪽 방향으로 회전 유지
        if front_dist < 0.65:
            if self.turn_dir == 0.0:
                self.turn_dir = 1.0 if left_dist >= right_dist else -1.0
            twist.linear.x = 0.0
            twist.angular.z = self.turn_dir * 0.85
        else:
            # 전방이 뚫리면 회전 고정 해제
            self.turn_dir = 0.0
            twist.linear.x = 0.22

            # [상태 4] 프론티어 목표 방향으로 조향
            if self.target_frontier is not None:
                tx, ty = self.target_frontier
                target_yaw = math.atan2(ty - self.robot_y, tx - self.robot_x)
                angle_err = math.atan2(math.sin(target_yaw - self.robot_yaw), math.cos(target_yaw - self.robot_yaw))
                twist.angular.z = max(-0.7, min(0.7, 0.8 * angle_err))
            else:
                twist.angular.z = 0.0

            # [상태 5] 측면 벽 밀어내기 (오른쪽 벽이 가까우면 왼쪽으로, 왼쪽이 가까우면 오른쪽으로 조향 보정)
            if right_dist < 0.45:
                twist.angular.z += 0.45 * (0.45 - right_dist) / 0.45
            if left_dist < 0.45:
                twist.angular.z -= 0.45 * (0.45 - left_dist) / 0.45

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
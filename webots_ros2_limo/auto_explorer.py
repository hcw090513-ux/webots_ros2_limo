import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist

class CoveragePatrolExplorer(Node):
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
        
        # 탐색 상태 관리 ('EXPLORE_FRONTIER' -> 'PATROL_UNCOVERED')
        self.mode = 'EXPLORE_FRONTIER'
        self.patrol_cycle = 1
        self.sensor_radius = 1.5  # 가스 센서 유효 감지 반경 (1.5m)
        
        self.target_point = None
        self.target_lock_time = 0.0
        self.blacklist = []
        self.visited_breadcrumbs = []  # 로봇이 실제로 밟고 지나간 (x, y) 궤적
        
        # 끼임 감지 및 2단계 탈출 변수
        self.last_check_x = 0.0
        self.last_check_y = 0.0
        self.stuck_counter = 0
        self.backup_steps = 0
        self.turn_steps = 0
        self.turn_dir = 1.0
        
        self.get_logger().info('[Phase 1] 미지 공간 프론티어 탐색을 시작합니다!')

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        # 로봇의 실제 이동 궤적을 0.6m 간격으로 기록 (가스 센서 커버리지 확인용)
        if not self.visited_breadcrumbs or math.hypot(
            self.robot_x - self.visited_breadcrumbs[-1][0],
            self.robot_y - self.visited_breadcrumbs[-1][1]
        ) > 0.6:
            self.visited_breadcrumbs.append((self.robot_x, self.robot_y))

    def is_blacklisted(self, x, y):
        for bx, by in self.blacklist:
            if math.hypot(x - bx, y - by) < 1.0:
                return True
        return False

    def is_covered_by_sensor(self, x, y):
        # 해당 좌표가 로봇이 지나갔던 궤적으로부터 sensor_radius(1.5m) 이내인지 확인
        for vx, vy in self.visited_breadcrumbs:
            if math.hypot(x - vx, y - vy) < self.sensor_radius:
                return True
        return False

    def map_cb(self, msg):
        now = time.time()

        # 1. 현재 목표점이 있다면 도착 여부 및 타임아웃(8초) 검사
        if self.target_point is not None:
            dist_to_target = math.hypot(
                self.target_point[0] - self.robot_x,
                self.target_point[1] - self.robot_y
            )
            if dist_to_target < 0.75:
                self.get_logger().info(f'목표 지점 도달 완료! ({self.mode})')
                self.target_point = None
            elif now - self.target_lock_time > 8.0:
                self.blacklist.append(self.target_point)
                self.target_point = None
            else:
                return

        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))

        free_mask = (grid == 0)
        unknown_mask = (grid == -1)
        obstacle_mask = (grid > 50)

        # 벽에서 6칸(약 30cm) 이내 픽셀은 안전을 위해 목표에서 제외
        obs_near = obstacle_mask.copy()
        for _ in range(6):
            obs_near |= (
                np.roll(obs_near, 1, axis=0) | np.roll(obs_near, -1, axis=0) |
                np.roll(obs_near, 1, axis=1) | np.roll(obs_near, -1, axis=1)
            )
        safe_free_mask = free_mask & (~obs_near)

        # 2. [Phase 1] 프론티어(미지의 경계선) 검출
        up = np.roll(unknown_mask, -1, axis=0)
        down = np.roll(unknown_mask, 1, axis=0)
        left = np.roll(unknown_mask, -1, axis=1)
        right = np.roll(unknown_mask, 1, axis=1)
        
        frontier_mask = safe_free_mask & (up | down | left | right)
        fy, fx = np.where(frontier_mask)

        best_target = None
        best_score = -999.0

        # 미탐색 경계선이 충분히 남아있다면 -> Phase 1 (프론티어 탐색) 수행
        if len(fx) > 10:
            self.mode = 'EXPLORE_FRONTIER'
            wx = ox + (fx + 0.5) * res
            wy = oy + (fy + 0.5) * res
            dists = np.hypot(wx - self.robot_x, wy - self.robot_y)

            for i in range(0, len(wx), 5):
                d = dists[i]
                if d < 0.9 or d > 6.5 or self.is_blacklisted(wx[i], wy[i]):
                    continue
                angle_to_p = math.atan2(wy[i] - self.robot_y, wx[i] - self.robot_x)
                angle_diff = abs(math.atan2(math.sin(angle_to_p - self.robot_yaw), math.cos(angle_to_p - self.robot_yaw)))
                score = -d - (1.2 * angle_diff)
                if score > best_score:
                    best_score = score
                    best_target = (wx[i], wy[i])

        # 3. [Phase 2] 프론티어가 없다면 -> 가스 센서가 닿지 않은 '사각지대 빈 공간' 탐색
        if best_target is None:
            if self.mode == 'EXPLORE_FRONTIER':
                self.mode = 'PATROL_UNCOVERED'
                self.blacklist.clear()
                self.get_logger().info(
                    f'[Phase 2 전환] 맵핑 완료! 가스 센서 미탐지 사각지대({self.patrol_cycle}회차) 탐색을 시작합니다.'
                )

            sy, sx = np.where(safe_free_mask)
            if len(sx) == 0:
                return

            # 연산량을 줄이기 위해 안전한 빈 공간을 15픽셀 간격으로 샘플링하여 미방문 구역 확인
            for i in range(0, len(sx), 15):
                px = ox + (sx[i] + 0.5) * res
                py = oy + (sy[i] + 0.5) * res
                d = math.hypot(px - self.robot_x, py - self.robot_y)

                if d < 0.8 or self.is_blacklisted(px, py):
                    continue
                # 이미 가스 센서 반경(1.5m)으로 훑고 지나간 곳은 패스!
                if self.is_covered_by_sensor(px, py):
                    continue

                angle_to_p = math.atan2(py - self.robot_y, px - self.robot_x)
                angle_diff = abs(math.atan2(math.sin(angle_to_p - self.robot_yaw), math.cos(angle_to_p - self.robot_yaw)))
                score = -d - (0.8 * angle_diff)
                if score > best_score:
                    best_score = score
                    best_target = (px, py)

            # 4. [Phase 3 - 방안 1 핵심] 모든 사각지대까지 100% 다 돌았다면 -> 방문 기록 리셋 후 다음 회차 순찰!
            if best_target is None:
                self.patrol_cycle += 1
                self.visited_breadcrumbs.clear()
                self.visited_breadcrumbs.append((self.robot_x, self.robot_y))
                self.blacklist.clear()
                self.get_logger().info(
                    f'[전 구역 이상 없음] 방문 기록을 리셋하고 {self.patrol_cycle}회차 정기 순찰을 새로 시작합니다!'
                )
                return

        self.target_point = best_target
        self.target_lock_time = now
        self.get_logger().info(f'[{self.mode}] 새 목표 설정: ({best_target[0]:.2f}, {best_target[1]:.2f})')

    def scan_cb(self, msg):
        n = len(msg.ranges)
        c = n // 2
        
        def get_clean(arr):
            valid = [r for r in arr if not math.isnan(r) and not math.isinf(r) and r > 0.02]
            return valid if valid else [10.0]

        front_dist = min(get_clean(msg.ranges[c - 38 : c + 38]))
        left_dist = min(get_clean(msg.ranges[c + 38 : c + 110]))
        right_dist = min(get_clean(msg.ranges[c - 110 : c - 38]))

        sectors = [
            np.mean(get_clean(msg.ranges[c - 100 : c - 60])),
            np.mean(get_clean(msg.ranges[c - 60 : c - 20])),
            np.mean(get_clean(msg.ranges[c - 20 : c + 20])),
            np.mean(get_clean(msg.ranges[c + 20 : c + 60])),
            np.mean(get_clean(msg.ranges[c + 60 : c + 100]))
        ]
        sector_angles = [-0.9, -0.45, 0.0, 0.45, 0.9]
        open_space_angle = sector_angles[int(np.argmax(sectors))]

        twist = Twist()

        # [탈출 1단계] 직선 후진
        if self.backup_steps > 0:
            self.backup_steps -= 1
            twist.linear.x = -0.25
            twist.linear.y = 0.15 if self.turn_dir > 0 else -0.15
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return

        # [탈출 2단계] 제자리 회전
        if self.turn_steps > 0:
            self.turn_steps -= 1
            twist.linear.x = 0.0
            twist.angular.z = self.turn_dir * 1.0
            self.cmd_pub.publish(twist)
            return

        # [물리적 끼임 감지] 1.5초간 5cm 미만 이동 시 현재 목표 블랙리스트 등록 후 탈출
        self.stuck_counter += 1
        if self.stuck_counter >= 15:
            moved_dist = math.hypot(self.robot_x - self.last_check_x, self.robot_y - self.last_check_y)
            self.last_check_x = self.robot_x
            self.last_check_y = self.robot_y
            self.stuck_counter = 0
            
            if moved_dist < 0.05:
                if self.target_point is not None:
                    self.blacklist.append(self.target_point)
                    self.target_point = None
                self.turn_dir = 1.0 if left_dist >= right_dist else -1.0
                self.backup_steps = 14
                self.turn_steps = 12
                return

        if front_dist < 0.40 or left_dist < 0.26 or right_dist < 0.26:
            self.turn_dir = 1.0 if left_dist >= right_dist else -1.0
            self.backup_steps = 9
            self.turn_steps = 10
            return

        if front_dist < 0.75:
            twist.linear.x = 0.06
            twist.angular.z = 0.9 if left_dist >= right_dist else -0.9
        else:
            twist.linear.x = 0.26
            if self.target_point is not None:
                tx, ty = self.target_point
                target_yaw = math.atan2(ty - self.robot_y, tx - self.robot_x)
                angle_err = math.atan2(math.sin(target_yaw - self.robot_yaw), math.cos(target_yaw - self.robot_yaw))
                steer = (0.65 * angle_err) + (0.35 * open_space_angle)
                twist.angular.z = max(-0.7, min(0.7, steer))
            else:
                twist.angular.z = 0.6 * open_space_angle

            if right_dist < 0.60:
                twist.angular.z += 0.65 * (0.60 - right_dist)
            if left_dist < 0.60:
                twist.angular.z -= 0.65 * (0.60 - left_dist)

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = CoveragePatrolExplorer()
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
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray

class SmoothCoverageExplorer(Node):
    def __init__(self):
        super().__init__('auto_explorer')
        
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, scan_qos)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # RViz2 가시화를 위한 MarkerArray 퍼블리셔 추가
        self.marker_pub = self.create_publisher(MarkerArray, '/coverage_markers', 10)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        self.mode = 'EXPLORE_FRONTIER'
        self.patrol_cycle = 1
        self.sensor_radius = 1.5  # 가스 센서 감지 반경 (1.5m)
        
        self.target_point = None
        self.target_lock_time = 0.0
        self.blacklist = []
        self.visited_breadcrumbs = []
        
        self.prev_angular_z = 0.0
        self.last_check_x = 0.0
        self.last_check_y = 0.0
        self.stuck_counter = 0
        self.backup_steps = 0
        self.turn_steps = 0
        self.turn_dir = 1.0
        
        # 0.5초마다 RViz 마커 업데이트
        self.marker_timer = self.create_timer(0.5, self.publish_markers)
        
        self.get_logger().info('[가시화 포함] 프론티어 탐색 및 사각지대 순찰 노드를 시작합니다!')

    def publish_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # 1. 지나온 궤적(가스 센서 감지 반경)을 초록색 반투명 원판으로 표시
        for i, (vx, vy) in enumerate(self.visited_breadcrumbs):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'sensor_coverage'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(vx)
            m.pose.position.y = float(vy)
            m.pose.position.z = 0.01
            m.pose.orientation.w = 1.0
            # 지름 = 센서 반경(1.5m) * 2 = 3.0m
            m.scale.x = self.sensor_radius * 2.0
            m.scale.y = self.sensor_radius * 2.0
            m.scale.z = 0.02
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 0.3
            m.color.a = 0.18  # 겹쳐도 지도가 잘 보이도록 반투명 설정
            marker_array.markers.append(m)

        # 2. 현재 향하고 있는 목표 지점을 노란색 구(Sphere)로 표시
        if self.target_point is not None:
            tm = Marker()
            tm.header.frame_id = 'map'
            tm.header.stamp = now
            tm.ns = 'current_target'
            tm.id = 9999
            tm.type = Marker.SPHERE
            tm.action = Marker.ADD
            tm.pose.position.x = float(self.target_point[0])
            tm.pose.position.y = float(self.target_point[1])
            tm.pose.position.z = 0.15
            tm.pose.orientation.w = 1.0
            tm.scale.x = 0.35
            tm.scale.y = 0.35
            tm.scale.z = 0.35
            tm.color.r = 1.0
            tm.color.g = 0.85
            tm.color.b = 0.0
            tm.color.a = 0.9
            marker_array.markers.append(tm)

        self.marker_pub.publish(marker_array)

    def clear_all_markers(self):
        # 순찰 회차 리셋 시 RViz 화면의 기존 마커 전체 삭제
        marker_array = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.action = Marker.DELETEALL
        marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.visited_breadcrumbs or math.hypot(
            self.robot_x - self.visited_breadcrumbs[-1][0],
            self.robot_y - self.visited_breadcrumbs[-1][1]
        ) > 0.5:
            self.visited_breadcrumbs.append((self.robot_x, self.robot_y))

    def is_blacklisted(self, x, y):
        for bx, by in self.blacklist:
            if math.hypot(x - bx, y - by) < 1.1:
                return True
        return False

    def is_covered_by_sensor(self, x, y):
        for vx, vy in self.visited_breadcrumbs:
            if math.hypot(x - vx, y - vy) < self.sensor_radius:
                return True
        return False

    def map_cb(self, msg):
        now = time.time()

        if self.target_point is not None:
            dist_to_target = math.hypot(
                self.target_point[0] - self.robot_x,
                self.target_point[1] - self.robot_y
            )
            if dist_to_target < 0.8:
                self.get_logger().info(f'목표 도달 완료! 다음 구역으로 이동합니다. ({self.mode})')
                self.target_point = None
            elif now - self.target_lock_time > 7.0:
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

        obs_near = obstacle_mask.copy()
        for _ in range(6):
            obs_near |= (
                np.roll(obs_near, 1, axis=0) | np.roll(obs_near, -1, axis=0) |
                np.roll(obs_near, 1, axis=1) | np.roll(obs_near, -1, axis=1)
            )
        safe_free_mask = free_mask & (~obs_near)

        up = np.roll(unknown_mask, -1, axis=0)
        down = np.roll(unknown_mask, 1, axis=0)
        left = np.roll(unknown_mask, -1, axis=1)
        right = np.roll(unknown_mask, 1, axis=1)
        
        frontier_mask = safe_free_mask & (up | down | left | right)
        fy, fx = np.where(frontier_mask)

        best_target = None
        best_score = -999.0

        if len(fx) > 12:
            self.mode = 'EXPLORE_FRONTIER'
            wx = ox + (fx + 0.5) * res
            wy = oy + (fy + 0.5) * res
            dists = np.hypot(wx - self.robot_x, wy - self.robot_y)

            for i in range(0, len(wx), 5):
                d = dists[i]
                if d < 1.2 or d > 7.0 or self.is_blacklisted(wx[i], wy[i]):
                    continue
                angle_to_p = math.atan2(wy[i] - self.robot_y, wx[i] - self.robot_x)
                angle_diff = abs(math.atan2(math.sin(angle_to_p - self.robot_yaw), math.cos(angle_to_p - self.robot_yaw)))
                score = -abs(d - 2.5) - (1.5 * angle_diff)
                if score > best_score:
                    best_score = score
                    best_target = (wx[i], wy[i])

        if best_target is None:
            if self.mode == 'EXPLORE_FRONTIER':
                self.mode = 'PATROL_UNCOVERED'
                self.blacklist.clear()
                self.get_logger().info(
                    f'[Phase 2 전환] 1차 맵핑 완료! 초록색 마커가 닿지 않은 사각지대({self.patrol_cycle}회차)를 탐색합니다.'
                )

            sy, sx = np.where(safe_free_mask)
            if len(sx) == 0:
                return

            for i in range(0, len(sx), 15):
                px = ox + (sx[i] + 0.5) * res
                py = oy + (sy[i] + 0.5) * res
                d = math.hypot(px - self.robot_x, py - self.robot_y)

                if d < 1.0 or self.is_blacklisted(px, py) or self.is_covered_by_sensor(px, py):
                    continue

                angle_to_p = math.atan2(py - self.robot_y, px - self.robot_x)
                angle_diff = abs(math.atan2(math.sin(angle_to_p - self.robot_yaw), math.cos(angle_to_p - self.robot_yaw)))
                score = -d - (1.0 * angle_diff)
                if score > best_score:
                    best_score = score
                    best_target = (px, py)

            if best_target is None:
                self.patrol_cycle += 1
                self.clear_all_markers()
                self.visited_breadcrumbs.clear()
                self.visited_breadcrumbs.append((self.robot_x, self.robot_y))
                self.blacklist.clear()
                self.get_logger().info(
                    f'[전 구역 스캔 완료] 마커를 초기화하고 {self.patrol_cycle}회차 정기 순찰을 시작합니다!'
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

        front_dist = min(get_clean(msg.ranges[c - 35 : c + 35]))
        left_dist = min(get_clean(msg.ranges[c + 35 : c + 105]))
        right_dist = min(get_clean(msg.ranges[c - 105 : c - 35]))

        sectors = [
            np.mean(get_clean(msg.ranges[c - 100 : c - 60])),
            np.mean(get_clean(msg.ranges[c - 60 : c - 20])),
            np.mean(get_clean(msg.ranges[c - 20 : c + 20])),
            np.mean(get_clean(msg.ranges[c + 20 : c + 60])),
            np.mean(get_clean(msg.ranges[c + 60 : c + 100]))
        ]
        sector_angles = [-0.85, -0.42, 0.0, 0.42, 0.85]
        open_space_angle = sector_angles[int(np.argmax(sectors))]

        twist = Twist()

        if self.backup_steps > 0:
            self.backup_steps -= 1
            twist.linear.x = -0.24
            twist.linear.y = 0.14 if self.turn_dir > 0 else -0.14
            twist.angular.z = 0.0
            self.prev_angular_z = 0.0
            self.cmd_pub.publish(twist)
            return

        if self.turn_steps > 0:
            self.turn_steps -= 1
            twist.linear.x = 0.0
            twist.angular.z = self.turn_dir * 0.85
            self.prev_angular_z = twist.angular.z
            self.cmd_pub.publish(twist)
            return

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
                self.backup_steps = 12
                self.turn_steps = 12
                return

        if front_dist < 0.38 or left_dist < 0.24 or right_dist < 0.24:
            self.turn_dir = 1.0 if left_dist >= right_dist else -1.0
            self.backup_steps = 8
            self.turn_steps = 10
            return

        if front_dist < 0.75:
            twist.linear.x = 0.08
            raw_angular = 0.85 if left_dist >= right_dist else -0.85
        else:
            twist.linear.x = 0.26
            if self.target_point is not None:
                tx, ty = self.target_point
                target_yaw = math.atan2(ty - self.robot_y, tx - self.robot_x)
                angle_err = math.atan2(math.sin(target_yaw - self.robot_yaw), math.cos(target_yaw - self.robot_yaw))
                raw_angular = (0.60 * angle_err) + (0.40 * open_space_angle)
            else:
                raw_angular = 0.65 * open_space_angle

            if right_dist < 0.55:
                raw_angular += 0.55 * (0.55 - right_dist)
            if left_dist < 0.55:
                raw_angular -= 0.55 * (0.55 - left_dist)

        smoothed_angular = (0.55 * self.prev_angular_z) + (0.45 * raw_angular)
        twist.angular.z = max(-0.75, min(0.75, smoothed_angular))
        self.prev_angular_z = twist.angular.z

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SmoothCoverageExplorer()
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
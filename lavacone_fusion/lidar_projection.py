"""라이다 점을 카메라 원본 이미지에 투영해 /projection/image 로 발행한다.

tune:=true 이면 base_link -> camera_link 외부 파라미터 6개를 ROS 파라미터로 받아
정적 TF로 다시 발행한다. rqt_reconfigure 에서 바꾸면 바로 반영되고,
~/save 서비스를 부르면 extrinsics.yaml 의 camera_link 값을 덮어쓴다.

    ros2 run rqt_reconfigure rqt_reconfigure
    ros2 service call /lidar_projection/save std_srvs/srv/Trigger
"""
import datetime
import math
import os
import re
from collections import deque

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from lavacone_fusion.common import (LATEST_ONLY, CameraModel, apply, default_extrinsics, distance_colors,
                                    draw_colorbar, put_lines, rpy_to_quat, scan_to_points, transform_to_matrix)

EXTRINSIC_KEYS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
RANGES = {'x': (-1.0, 3.0, 0.001), 'y': (-1.0, 1.0, 0.001), 'z': (0.0, 2.0, 0.001),
          'roll': (-0.5, 0.5, 0.0005), 'pitch': (-0.5, 1.2, 0.0005), 'yaw': (-0.5, 0.5, 0.0005)}


class LidarProjection(Node):
    def __init__(self):
        super().__init__('lidar_projection')
        default_file = default_extrinsics()
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('info_topic', '/camera_info')
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('max_dist', 10.0)
        self.declare_parameter('point_radius', 3)
        self.declare_parameter('max_rate', 10.0)
        self.declare_parameter('scan_sync', 'nearest')     # nearest | latest
        self.declare_parameter('fallback_dfov_deg', 87.0)   # 캘리브레이션 전 임시 화각. 콘·소실점으로 추정(f 약 770)
        self.declare_parameter('tune', False)
        self.declare_parameter('extrinsics_file', default_file)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera = None
        self.scans = deque(maxlen=20)
        self.last_pub = 0.0
        self.warned = set()

        self.tune = self.get_parameter('tune').value
        if self.tune:
            self.ext_file = os.path.realpath(self.get_parameter('extrinsics_file').value)
            with open(self.ext_file) as f:
                cam = yaml.safe_load(f)['camera_link']
            self.ext_parent, self.ext_child = cam['parent'], cam['child']
            for k in EXTRINSIC_KEYS:
                lo, hi, step = RANGES[k]
                desc = ParameterDescriptor(
                    description=f'base_link -> camera_link {k} ({"m" if len(k) == 1 else "rad"})',
                    floating_point_range=[FloatingPointRange(from_value=lo, to_value=hi, step=step)])
                self.declare_parameter(k, float(cam[k]), desc)
            self.tf_pub = StaticTransformBroadcaster(self)
            self.add_on_set_parameters_callback(self.on_params)
            self.publish_extrinsic({k: self.get_parameter(k).value for k in EXTRINSIC_KEYS})
            self.create_service(Trigger, '~/save', self.on_save)
            self.get_logger().info(f'외부 파라미터 조정 모드: {self.ext_file}')

        self.pub = self.create_publisher(Image, '/projection/image', 2)
        self.create_subscription(CameraInfo, self.get_parameter('info_topic').value, self.on_info, 2)
        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value,
                                 lambda m: self.scans.append(m), qos_profile_sensor_data)
        # 원본 이미지는 크므로(1280x720 RGB 2.7 MB) 바이트로 받아, 그릴 프레임만 메시지로 풀어 쓴다
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_image_bytes,
                                 LATEST_ONLY, raw=True)

    # ------------------------------------------------------------ 외부 파라미터

    def current_extrinsic(self):
        return {k: self.get_parameter(k).value for k in EXTRINSIC_KEYS}

    def publish_extrinsic(self, e):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.ext_parent
        t.child_frame_id = self.ext_child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = e['x'], e['y'], e['z']
        q = rpy_to_quat(e['roll'], e['pitch'], e['yaw'])
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_pub.sendTransform(t)

    def on_params(self, params):
        e = self.current_extrinsic()
        for p in params:
            if p.name in EXTRINSIC_KEYS:
                e[p.name] = float(p.value)
        self.publish_extrinsic(e)
        return SetParametersResult(successful=True)

    def on_save(self, request, response):
        e = self.current_extrinsic()
        with open(self.ext_file) as f:
            lines = f.read().split('\n')
        out, in_block = [], False
        stamp = datetime.date.today().isoformat()
        for line in lines:
            if re.match(r'^\S', line):
                in_block = line.startswith('camera_link:')
            m = re.match(r'^(\s+)(x|y|z|roll|pitch|yaw):\s*[-+0-9.eE]+(.*)$', line) if in_block else None
            if m:
                key = m.group(2)
                note = f'   # {math.degrees(e[key]):.2f} deg, 4단계 조정 {stamp}' if len(key) > 1 \
                    else f'   # 4단계 조정 {stamp}'
                line = f'{m.group(1)}{key}: {e[key]:.4f}{note}'
            out.append(line)
        with open(self.ext_file, 'w') as f:
            f.write('\n'.join(out))
        response.success = True
        response.message = f'{self.ext_file} 저장: ' + ', '.join(f'{k}={e[k]:.4f}' for k in EXTRINSIC_KEYS)
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------ 투영

    def on_info(self, msg):
        if self.camera is None or (msg.width, msg.height) != (self.camera.width, self.camera.height) \
                or list(msg.k) != list(self.camera_info_k):
            self.camera = CameraModel(msg, self.get_parameter('fallback_dfov_deg').value)
            self.camera_info_k = list(msg.k)
            if self.camera.provisional:
                self.get_logger().warn('camera_info 가 비어 있어 공칭 화각 임시 내부 파라미터를 씁니다 '
                                       '(캘리브레이션 전에는 정합 정확도를 판단하지 마세요)')

    def pick_scan(self, stamp):
        if not self.scans:
            return None
        if self.get_parameter('scan_sync').value == 'latest':
            return self.scans[-1]
        t = Time.from_msg(stamp).nanoseconds
        return min(self.scans, key=lambda s: abs(Time.from_msg(s.header.stamp).nanoseconds - t))

    def warn_once(self, key, text):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warn(text)

    def on_image_bytes(self, data):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_pub < 1.0 / self.get_parameter('max_rate').value:
            return
        self.on_image(deserialize_message(data, Image), now)

    def on_image(self, msg, now):
        if self.camera is None:
            return self.warn_once('info', 'camera_info 를 기다리는 중')
        scan = self.pick_scan(msg.header.stamp)
        if scan is None:
            return self.warn_once('scan', '스캔을 기다리는 중')
        cam_frame = msg.header.frame_id or 'camera_optical_frame'
        try:
            tf = self.tf_buffer.lookup_transform(cam_frame, scan.header.frame_id, Time())
        except Exception as ex:  # noqa: BLE001  TF 가 아직 없을 때
            return self.warn_once('tf', f'TF {scan.header.frame_id} -> {cam_frame} 없음: {ex}')
        self.last_pub = now

        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        pts, _, rng = scan_to_points(scan)
        uv, inside = self.camera.project(apply(transform_to_matrix(tf), pts))
        max_dist = self.get_parameter('max_dist').value
        rad = int(self.get_parameter('point_radius').value)
        order = np.argsort(-rng[inside])        # 먼 점부터 그려 가까운 점이 위에 보이게
        for (u, v), c in zip(uv[inside][order], distance_colors(rng[inside][order], max_dist)):
            cv2.circle(img, (int(round(u)), int(round(v))), rad, tuple(int(x) for x in c), -1, cv2.LINE_AA)

        dt = (Time.from_msg(msg.header.stamp).nanoseconds - Time.from_msg(scan.header.stamp).nanoseconds) * 1e-6
        lines = [f'points {int(inside.sum())}/{len(pts)}   image-scan dt {dt:+.0f} ms']
        if self.tune:
            e = self.current_extrinsic()
            lines.append('x {x:.3f} y {y:.3f} z {z:.3f} m'.format(**e))
            lines.append('roll {:.2f} pitch {:.2f} yaw {:.2f} deg'.format(
                *(math.degrees(e[k]) for k in ('roll', 'pitch', 'yaw'))))
        if self.camera.provisional:
            lines.append('UNCALIBRATED intrinsics (nominal FOV)')
        put_lines(img, lines, color=(255, 255, 255) if not self.camera.provisional else (0, 200, 255))
        draw_colorbar(img, max_dist)

        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main():
    rclpy.init()
    node = LidarProjection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

"""YOLO 박스와 라이다 군집 정합.

입력: /cone/detections (Detection2DArray, stamp = 이미지 시각), /scan_filtered, /camera_info, TF
출력: /cone/positions (Detection3DArray, base_link 기준 콘 위치)
      /cone/markers   (RViz MarkerArray)
      /cone/fusion_image (디버그 이미지, publish_image:=true 일 때)

정합 절차 (프레임마다)
 1. 검출 시각과 가장 가까운 스캔을 고른다 (max_dt 를 넘으면 버린다).
 2. 스캔을 각도 순서로 군집화하고, 폭이 콘 크기인 군집만 남긴다.
 3. 군집 중심(원 맞춤)을 카메라 광학 프레임으로 옮겨 이미지에 투영한다.
 4. 박스-군집 비용을 계산한다.
      - 투영점이 박스 안(가로·세로, margin_px 여유)에 있어야 후보가 된다.
      - 가로 비용: |u - 박스 중심| / 박스 폭
      - 거리 비용: 박스 아랫변을 지면(z=0)에 투영해 얻은 거리와 라이다 거리의 상대 차이
        (ground_tol 을 넘으면 후보에서 뺀다. 차량이 기울면 커지므로 여유 있게 둔다)
 5. 헝가리안 알고리즘으로 1:1 배정한다.
 6. 짝지어진 콘은 라이다 위치(정확한 거리)를, 짝이 없는 박스는 지면 추정 위치를 낸다.

추적 이동 보정: TF odom -> base_link (wheel_odometry) 가 있으면 직전 스캔 이후 차량 이동만큼 추적 위치를
옮기고, 카메라로 확인된 콘은 라이다에서 안 보여도 track_memory 초 동안 예측 위치로 유지한다.
TF 가 없으면 보정 없이 동작한다. 영상-스캔 사이(max_dt 이내)의 이동은 보정하지 않는다.
"""
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from lavacone_fusion.cone_tracker import ConeTracker, relative_motion
from lavacone_fusion.common import (RECENT_FRAMES, CameraModel, apply, cluster_scan, distance_colors, draw_colorbar,
                                    fit_circle_center, put_lines, scan_to_points, transform_to_matrix)

BIG = 1e6


def stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


class ConeFusion(Node):
    def __init__(self):
        super().__init__('cone_fusion')
        p = self.declare_parameter
        p('detections_topic', '/cone/detections')
        p('scan_topic', '/scan_filtered')
        p('info_topic', '/camera_info')
        # 검출 노드가 검출한 프레임에 박스를 그려 내는 이미지 (검출 주기로만 들어와 가볍다)
        p('image_topic', '/cone/debug_image')
        p('target_frame', 'base_link')
        p('max_dt', 0.12)                # 검출-스캔 시각 차 허용 (s). 스캔 주기 0.1 s
        p('max_wait', 0.3)               # 영상 시각 뒤의 스캔을 기다리는 최대 시간 (s)
        # 이미지 stamp 는 프레임이 PC 에 도착한 시각이다(v4l2_camera). 실제 노출은 그보다 앞서므로
        # 스캔을 고를 때 이 값(s)만큼 앞당긴다. 측정 전이라 0. 주행 중 정합 오차로 정한다
        p('image_delay', 0.0)
        # 군집화
        p('jump_dist', 0.10)
        p('jump_k', 3.0)
        p('min_points', 3)
        p('min_width', 0.02)
        p('max_width', 0.45)
        p('max_range', 15.0)
        # 콘 형상 (스캔 높이에서의 반지름. 0 이면 점 평균 사용)
        p('cone_radius_at_scan', 0.0)
        p('cone_height', 0.70)
        # 정합
        p('margin_px', 12.0)
        p('ground_tol', 0.35)            # 지면 추정 거리와 라이다 거리의 상대 차 허용
        p('w_ground', 1.0)
        p('max_cost', 1.5)
        # 출력
        p('publish_image', True)
        p('publish_unmatched', True)
        p('draw_scan_points', True)      # 디버그 이미지에 스캔 점 전체를 거리 색으로 투영 (카메라-라이다 정렬 확인)
        # 추적: 카메라로 확인된 라이다 군집을 스캔마다 따라가 콘이 프레임마다 깜빡이지 않게 한다
        p('track', True)
        p('track_gate', 0.5)             # 스캔 사이 이동 허용 (m). 10 Hz 에서 약 5 m/s
        p('track_max_misses', 5)         # 라이다에서 이만큼 연속으로 안 보이면 삭제 (스캔 수)
        p('min_cam_hits', 2)             # 카메라 확인이 이 횟수 이상이면 콘으로 출력
        p('camera_only_hold', 0.3)       # 카메라 전용 추정을 추적 출력에 덧붙이는 유효 시간 (s)
        p('odom_frame', 'odom')          # 이동 보정용 TF 프레임. 빈 값이면 보정 끔
        p('track_memory', 3.0)           # 이동 보정 중 안 보이는 확인 콘을 유지하는 시간 (s, 10 Hz 기준 스캔 수로 바꿈)
        p('odom_max_age', 0.2)           # 스캔 시각의 TF 가 없을 때 쓰는 최신 TF 의 허용 나이 (s)
        p('fallback_dfov_deg', 87.0)   # 캘리브레이션 전 임시 화각. 콘·소실점으로 추정(f 약 770)

        self.g = lambda n: self.get_parameter(n).value
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera = None
        self.camera_k = None
        self.scans = deque(maxlen=20)
        self.records = deque(maxlen=20)     # 스캔별 군집과 추적 id (검출과 같은 스캔 결과를 쓰려고)
        self.images = deque(maxlen=10)
        self.T_base_laser = None
        self.tracker = ConeTracker(self.g('track_gate'), self.g('track_max_misses'), self.g('min_cam_hits'),
                                   memory_misses=int(round(self.g('track_memory') * 10)))
        self.prev_pose = None               # 직전 스캔 시각의 odom -> base_link (4x4)
        self.odom_ok = None                 # 이동 보정 상태 (로그용)
        self.camera_only = (0, [])          # (stamp_ns, [(x, y, score, det_index)])
        self.pending = deque(maxlen=10)     # 맞는 스캔을 기다리는 검출
        self.warned = set()

        self.pub = self.create_publisher(Detection3DArray, '/cone/positions', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cone/markers', 10)
        self.image_pub = self.create_publisher(Image, '/cone/fusion_image', 2)
        self.create_subscription(CameraInfo, self.g('info_topic'), self.on_info, 2)
        self.create_subscription(LaserScan, self.g('scan_topic'), self.on_scan, qos_profile_sensor_data)
        if self.g('publish_image'):
            self.create_subscription(Image, self.g('image_topic'), lambda m: self.images.append(m),
                                     RECENT_FRAMES)
        self.create_subscription(Detection2DArray, self.g('detections_topic'), self.on_detections, 10)

    # ------------------------------------------------------------ 보조

    def warn_once(self, key, text):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warn(text)

    def on_info(self, msg):
        if self.camera is None or list(msg.k) != self.camera_k or msg.width != self.camera.width:
            self.camera = CameraModel(msg, self.g('fallback_dfov_deg'))
            self.camera_k = list(msg.k)
            if self.camera.provisional:
                self.get_logger().warn('camera_info 가 비어 있어 공칭 화각 임시 값으로 정합합니다 (정확도 낮음)')

    def nearest(self, buf, t_ns):
        if not buf:
            return None, None
        m = min(buf, key=lambda s: abs(stamp_ns(s.header.stamp) - t_ns))
        return m, (stamp_ns(m.header.stamp) - t_ns) * 1e-9

    def lookup(self, target, source):
        return transform_to_matrix(self.tf_buffer.lookup_transform(target, source, Time()))

    # ------------------------------------------------------------ 정합

    def lidar_clusters(self, scan, T_base_laser):
        r_cone = self.g('cone_radius_at_scan')
        laser_origin = np.zeros(2)
        out = []
        for pts, rng in cluster_scan(scan, self.g('jump_dist'), self.g('jump_k'),
                                     min_points=self.g('min_points')):
            width = float(np.linalg.norm(pts[0, :2] - pts[-1, :2]))
            if not (self.g('min_width') <= width <= self.g('max_width')):
                continue
            if rng.min() > self.g('max_range'):
                continue
            if r_cone > 0:
                c2 = fit_circle_center(pts, r_cone, laser_origin)
            else:
                c2 = pts[:, :2].mean(axis=0)
            center = np.array([c2[0], c2[1], 0.0])
            out.append({'laser': center, 'base': apply(T_base_laser, center[None])[0],
                        'pts': pts, 'width': width, 'range': float(np.linalg.norm(c2))})
        return out

    def odom_pose(self, stamp):
        """스캔 시각의 odom -> base_link (4x4). 없으면 최신 TF (odom_max_age 이내), 그것도 없으면 None."""
        odom, base = self.g('odom_frame'), self.g('target_frame')
        if not odom:
            return None
        try:
            return transform_to_matrix(self.tf_buffer.lookup_transform(odom, base, Time.from_msg(stamp)))
        except Exception:  # noqa: BLE001
            pass
        try:
            tf = self.tf_buffer.lookup_transform(odom, base, Time())
        except Exception:  # noqa: BLE001
            return None
        age = (stamp_ns(stamp) - stamp_ns(tf.header.stamp)) * 1e-9
        return transform_to_matrix(tf) if abs(age) <= self.g('odom_max_age') else None

    def compensate(self, stamp):
        pose = self.odom_pose(stamp)
        if pose is not None and self.prev_pose is not None:
            rot, trans = relative_motion(self.prev_pose, pose)
            self.tracker.move(rot, trans)
        ok = pose is not None
        if ok != self.odom_ok:
            self.get_logger().info('이동 보정 켜짐 (odom TF 수신)' if ok else
                                   f'이동 보정 꺼짐 ({self.g("odom_frame") or "odom_frame 없음"} TF 없음)')
            self.odom_ok = ok
        self.prev_pose = pose

    def on_scan(self, scan):
        self.scans.append(scan)
        if self.T_base_laser is None:
            try:
                self.T_base_laser = self.lookup(self.g('target_frame'), scan.header.frame_id)
            except Exception as ex:  # noqa: BLE001
                return self.warn_once('tf_laser', f'TF 없음: {ex}')
        clusters = self.lidar_clusters(scan, self.T_base_laser)
        ids = []
        if self.g('track'):
            self.compensate(scan.header.stamp)
            ids = self.tracker.update([c['base'][:2] for c in clusters])
        self.records.append({'stamp': stamp_ns(scan.header.stamp), 'clusters': clusters, 'ids': ids})
        self.process_pending()
        if self.g('track'):
            self.publish_tracked(scan.header.stamp, clusters, ids)
            self.tracker.end_scan()

    def ground_point(self, u, v, T_base_opt):
        """픽셀 -> 지면(base_link z=0) 교점. 지평선 위면 None."""
        ray = apply(T_base_opt, self.camera.pixel_ray([[u, v]]))[0] - T_base_opt[:3, 3]
        origin = T_base_opt[:3, 3]
        if ray[2] >= -1e-6:
            return None
        s = -origin[2] / ray[2]
        return origin + s * ray

    def on_detections(self, msg):
        # 스캔은 한 바퀴(0.1 s)를 다 돈 뒤 도착하므로 검출 시점에는 영상 시각의 스캔이 아직 없다.
        # 바로 처리하면 120 ms 넘게 지난 스캔과 짝지어져 버려지므로, 영상 시각 뒤의 스캔이 올 때까지 기다린다.
        self.pending.append((msg, self.get_clock().now().nanoseconds))
        self.process_pending()

    def process_pending(self):
        newest = self.records[-1]['stamp'] if self.records else None
        now = self.get_clock().now().nanoseconds
        while self.pending:
            msg, arrived = self.pending[0]
            t_ns = stamp_ns(msg.header.stamp) - int(self.g('image_delay') * 1e9)
            waited = (now - arrived) * 1e-9
            if (newest is None or newest < t_ns) and waited < self.g('max_wait'):
                break
            self.pending.popleft()
            self.match(msg)

    def match(self, msg):
        if self.camera is None:
            return self.warn_once('info', 'camera_info 를 기다리는 중')
        t_ns = stamp_ns(msg.header.stamp) - int(self.g('image_delay') * 1e9)
        if not self.records:
            return self.warn_once('scan', '스캔을 기다리는 중')
        rec = min(self.records, key=lambda r: abs(r['stamp'] - t_ns))
        dt = (rec['stamp'] - t_ns) * 1e-9
        if abs(dt) > self.g('max_dt'):
            self.get_logger().warn(f'검출과 스캔 시각 차 {dt * 1e3:+.0f} ms > max_dt, 이 프레임은 건너뜀',
                                   throttle_duration_sec=5.0)
            return
        base = self.g('target_frame')
        opt_frame = msg.header.frame_id or 'camera_optical_frame'
        try:
            T_opt_base = self.lookup(opt_frame, base)
        except Exception as ex:  # noqa: BLE001
            return self.warn_once('tf', f'TF 없음: {ex}')
        T_base_opt = np.linalg.inv(T_opt_base)

        clusters = rec['clusters']
        if clusters:
            uv, inside = self.camera.project(apply(T_opt_base, np.array([c['base'] for c in clusters])))
        else:
            uv, inside = np.zeros((0, 2)), np.zeros(0, bool)

        dets = msg.detections
        cost = np.full((len(dets), len(clusters)), BIG)
        ground = []
        m = self.g('margin_px')
        for i, d in enumerate(dets):
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            x1, x2, y1, y2 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
            gp = self.ground_point(cx, y2, T_base_opt)
            ground.append(gp)
            for j, c in enumerate(clusters):
                if not inside[j]:
                    continue
                u, v = uv[j]
                if not (x1 - m <= u <= x2 + m and y1 - m <= v <= y2 + m):
                    continue
                cst = abs(u - cx) / max(w, 1.0)
                if gp is not None:
                    d_lidar = math.hypot(*c['base'][:2])
                    d_ground = math.hypot(*gp[:2])
                    rel = abs(d_ground - d_lidar) / max(d_lidar, 0.5)
                    if rel > self.g('ground_tol'):
                        continue
                    cst += self.g('w_ground') * rel
                cost[i, j] = cst

        pairs = []
        if len(dets) and len(clusters):
            rows, cols = linear_sum_assignment(cost)
            pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < self.g('max_cost')]
        matched_d = {i for i, _ in pairs}
        matched_c = {j for _, j in pairs}

        if self.g('track'):
            for i, j in pairs:
                score = dets[i].results[0].hypothesis.score if dets[i].results else 1.0
                self.tracker.confirm(rec['ids'][j], score)
            self.camera_only = (t_ns, [(float(gp[0]), float(gp[1]),
                                        dets[i].results[0].hypothesis.score if dets[i].results else 1.0, i)
                                       for i, gp in enumerate(ground) if i not in matched_d and gp is not None])
        else:
            self.publish_positions(msg, dets, clusters, pairs, ground)
            self.publish_markers(msg, clusters, pairs, ground, matched_c, matched_d)
        if self.g('publish_image') and self.image_pub.get_subscription_count():
            scan = next((sc for sc in self.scans if stamp_ns(sc.header.stamp) == rec['stamp']), None)
            self.publish_image(msg, t_ns, dets, clusters, uv, inside, pairs, matched_d, dt, scan, T_opt_base)

    # ------------------------------------------------------------ 출력

    def make_det3d(self, header, x, y, score, det_id, source):
        h_cone = self.g('cone_height')
        d = Detection3D()
        d.header = header
        d.id = f'{source}:{det_id}'
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = x, y, h_cone / 2
        d.bbox.center.orientation.w = 1.0
        diam = 2 * max(self.g('cone_radius_at_scan'), 0.05)
        d.bbox.size.x = d.bbox.size.y = diam
        d.bbox.size.z = h_cone
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = 'cone'
        hyp.hypothesis.score = float(score)
        hyp.pose.pose = d.bbox.center
        d.results.append(hyp)
        return d

    def publish_positions(self, msg, dets, clusters, pairs, ground):
        out = Detection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.g('target_frame')
        for i, j in pairs:
            c = clusters[j]['base']
            score = dets[i].results[0].hypothesis.score if dets[i].results else 1.0
            out.detections.append(self.make_det3d(out.header, float(c[0]), float(c[1]), score, i, 'fused'))
        if self.g('publish_unmatched'):
            matched = {i for i, _ in pairs}
            for i, gp in enumerate(ground):
                if i in matched or gp is None:
                    continue
                score = dets[i].results[0].hypothesis.score if dets[i].results else 1.0
                # 카메라만으로 얻은 위치는 거리 오차가 크므로 점수를 절반으로 낮춰 구분한다
                out.detections.append(self.make_det3d(out.header, float(gp[0]), float(gp[1]), 0.5 * score,
                                                      i, 'camera'))
        self.pub.publish(out)

    def publish_tracked(self, stamp, clusters, ids):
        """카메라로 확인된 추적 콘(+ 최근 카메라 전용 추정)을 스캔마다 발행한다."""
        out = Detection3DArray()
        out.header.stamp = stamp
        out.header.frame_id = self.g('target_frame')
        confirmed = self.tracker.confirmed()
        for t in confirmed:
            out.detections.append(self.make_det3d(out.header, float(t.xy[0]), float(t.xy[1]), t.score, t.id, 'fused'))
        cam_t, cam = self.camera_only
        fresh = (stamp_ns(stamp) - cam_t) * 1e-9 <= self.g('camera_only_hold')
        if self.g('publish_unmatched') and fresh:
            for x, y, score, i in cam:
                # 카메라만으로 얻은 위치는 거리 오차가 크므로 점수를 절반으로 낮춰 구분한다
                out.detections.append(self.make_det3d(out.header, x, y, 0.5 * score, i, 'camera'))
        self.pub.publish(out)

        conf_ids = {t.id for t in confirmed}
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        h_cone = self.g('cone_height')

        def mk(ns, mid, mtype, x, y, z, sc, rgba):
            mm = Marker()
            mm.header = out.header
            mm.ns, mm.id, mm.type, mm.action = ns, mid, mtype, Marker.ADD
            mm.pose.position.x, mm.pose.position.y, mm.pose.position.z = float(x), float(y), float(z)
            mm.pose.orientation.w = 1.0
            mm.scale.x, mm.scale.y, mm.scale.z = (float(v) for v in sc)
            mm.color.r, mm.color.g, mm.color.b, mm.color.a = (float(v) for v in rgba)
            return mm

        for t in confirmed:
            seen = t.misses == 0             # 이번 스캔에서 라이다로 보임. 아니면 이동 보정으로 기억한 위치
            arr.markers.append(mk('tracked', t.id, Marker.CYLINDER, t.xy[0], t.xy[1], h_cone / 2,
                                  (0.2, 0.2, h_cone), (1.0, 0.5, 0.0, 0.9) if seen else (0.6, 0.4, 0.3, 0.4)))
            txt = mk('tracked_text', t.id, Marker.TEXT_VIEW_FACING, t.xy[0], t.xy[1], h_cone + 0.25,
                     (0, 0, 0.2), (1, 1, 1, 1))
            txt.text = f'#{t.id} {math.hypot(*t.xy):.2f} m' + ('' if seen else ' (기억)')
            arr.markers.append(txt)
        if fresh:
            for x, y, _, i in cam:
                arr.markers.append(mk('camera_only', i, Marker.SPHERE, x, y, 0.1, (0.25, 0.25, 0.25),
                                      (1.0, 1.0, 0.0, 0.5)))
        lst = mk('clusters', 0, Marker.SPHERE_LIST, 0, 0, 0, (0.06, 0.06, 0.06), (0.6, 0.6, 0.6, 0.8))
        for c, tid in zip(clusters, ids):
            if tid not in conf_ids:
                lst.points.append(Point(x=float(c['base'][0]), y=float(c['base'][1]), z=0.0))
        arr.markers.append(lst)
        self.marker_pub.publish(arr)

    def publish_markers(self, msg, clusters, pairs, ground, matched_c, matched_d):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        frame = self.g('target_frame')
        h_cone = self.g('cone_height')

        def mk(ns, mid, mtype, x, y, z, sx, sy, sz, rgba):
            mm = Marker()
            mm.header.frame_id = frame
            mm.header.stamp = msg.header.stamp
            mm.ns, mm.id, mm.type, mm.action = ns, mid, mtype, Marker.ADD
            mm.pose.position.x, mm.pose.position.y, mm.pose.position.z = float(x), float(y), float(z)
            mm.pose.orientation.w = 1.0
            mm.scale.x, mm.scale.y, mm.scale.z = float(sx), float(sy), float(sz)
            mm.color.r, mm.color.g, mm.color.b, mm.color.a = (float(v) for v in rgba)
            return mm

        for k, (i, j) in enumerate(pairs):
            c = clusters[j]['base']
            arr.markers.append(mk('fused', k, Marker.CYLINDER, c[0], c[1], h_cone / 2, 0.2, 0.2, h_cone,
                                  (1.0, 0.5, 0.0, 0.9)))
            t = mk('fused_text', k, Marker.TEXT_VIEW_FACING, c[0], c[1], h_cone + 0.25, 0, 0, 0.2,
                   (1.0, 1.0, 1.0, 1.0))
            t.text = f'{math.hypot(c[0], c[1]):.2f} m'
            arr.markers.append(t)
        for i, gp in enumerate(ground):
            if i in matched_d or gp is None:
                continue
            arr.markers.append(mk('camera_only', i, Marker.SPHERE, gp[0], gp[1], 0.1, 0.25, 0.25, 0.25,
                                  (1.0, 1.0, 0.0, 0.5)))
        lst = mk('clusters', 0, Marker.SPHERE_LIST, 0, 0, 0, 0.06, 0.06, 0.06, (0.6, 0.6, 0.6, 0.8))
        for j, c in enumerate(clusters):
            if j not in matched_c:
                lst.points.append(Point(x=float(c['base'][0]), y=float(c['base'][1]), z=float(c['base'][2])))
        arr.markers.append(lst)
        self.marker_pub.publish(arr)

    def publish_image(self, msg, t_ns, dets, clusters, uv, inside, pairs, matched_d, dt, scan=None, T_opt_base=None):
        img_msg, _ = self.nearest(self.images, t_ns)
        if img_msg is None:
            return
        img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        if self.g('draw_scan_points') and scan is not None and T_opt_base is not None:
            # 스캔 점 전체 (라이다 높이 0.48 m 평면): 콘·벽·사람 위를 가로지르는 한 줄로 보여야 정렬이 맞다
            pts, _, rng = scan_to_points(scan)
            puv, pin = self.camera.project(apply(T_opt_base @ self.T_base_laser, pts))
            for (u, v), col in zip(puv[pin].astype(int), distance_colors(rng[pin], 10.0)):
                cv2.circle(img, (int(u), int(v)), 2, tuple(int(x) for x in col), -1)
            draw_colorbar(img, 10.0)
        pair_of = {i: j for i, j in pairs}
        cols = distance_colors([c['range'] for c in clusters], 10.0) if clusters else []
        for j, c in enumerate(clusters):
            if inside[j]:
                u, v = map(int, np.round(uv[j]))
                col = tuple(int(x) for x in cols[j])
                cv2.drawMarker(img, (u, v), col, cv2.MARKER_CROSS, 14, 2)
        for i, d in enumerate(dets):
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            p1 = (int(cx - w / 2), int(cy - h / 2))
            p2 = (int(cx + w / 2), int(cy + h / 2))
            ok = i in matched_d
            color = (0, 220, 0) if ok else (0, 0, 255)
            cv2.rectangle(img, p1, p2, color, 2)
            label = '' if ok else 'no lidar'
            if ok:
                j = pair_of[i]
                c = clusters[j]['base']
                label = f'{math.hypot(c[0], c[1]):.2f} m'
                u, v = map(int, np.round(uv[j]))
                cv2.line(img, (int(cx), p2[1]), (u, v), color, 1, cv2.LINE_AA)
            # 검출 노드가 박스 위에 점수를 쓰므로 정합 결과는 박스 아래에 쓴다
            cv2.putText(img, label, (p1[0], p2[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
        lines = [f'boxes {len(dets)}  clusters {int(inside.sum())}/{len(clusters)}  matched {len(pairs)}',
                 f'det-scan dt {dt * 1e3:+.0f} ms']
        if self.camera.provisional:
            lines.append('UNCALIBRATED intrinsics (nominal FOV)')
        put_lines(img, lines)
        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = msg.header
        self.image_pub.publish(out)


def main():
    rclpy.init()
    node = ConeFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

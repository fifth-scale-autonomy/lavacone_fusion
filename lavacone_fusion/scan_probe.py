"""라이다 점검 도구.

occlusion: 일정 시간 스캔을 모아 항상 가까운 값이 찍히는 각도 구간(laser 프레임)을 찾는다.
    ros2 run lavacone_fusion scan_probe --mode occlusion --topic /scan

locate: 기대 위치(base_link 기준) 근처의 물체 점을 모아 중심을 구하고 오차를 낸다.
        RViz 에 기대 위치(빈 원), 측정 중심(채운 원), 오차 글자를 /scan_probe/markers 로 띄운다.
    ros2 run lavacone_fusion scan_probe --mode locate --expect 1.95 0.0 --object-radius 0.05
    ros2 run lavacone_fusion scan_probe --mode locate --expect 1.95 0.0 --continuous   # Ctrl+C 로 종료
"""
import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lavacone_fusion.common import apply, fit_circle_center, scan_to_points, transform_to_matrix

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Collector:
    def __init__(self, node, topic):
        self.node = node
        self.scans = []
        node.create_subscription(LaserScan, topic, self.scans.append, qos_profile_sensor_data)

    def collect(self, seconds):
        self.scans.clear()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return list(self.scans)


def occlusion(scans, near, ratio):
    n = len(scans[0].ranges)
    r = np.array([s.ranges for s in scans if len(s.ranges) == n], dtype=float)
    ang = np.degrees(scans[0].angle_min + np.arange(n) * scans[0].angle_increment)
    hit = (np.isfinite(r) & (r < near)).mean(axis=0) >= ratio
    idx = np.flatnonzero(hit)
    print(f'스캔 {r.shape[0]}개, 점 {n}개, 기준: {near} m 미만이 {ratio:.0%} 이상')
    if idx.size == 0:
        print('가려지는 구간 없음')
        return
    # 반사가 불안정해 3도 이하로 끊긴 곳은 한 구간으로 본다
    gap = max(1, int(round(3.0 / math.degrees(scans[0].angle_increment))))
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1):
        vals = r[:, seg]
        print(f'  laser {ang[seg[0]]:7.2f} ~ {ang[seg[-1]]:7.2f} deg  '
              f'중앙값 {np.nanmedian(vals):.3f} m, 최소 {np.nanmin(vals):.3f} m')


def lookup(node, buf, target, source):
    deadline = time.monotonic() + 5.0
    while not buf.can_transform(target, source, Time()) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    return transform_to_matrix(buf.lookup_transform(target, source, Time(), timeout=Duration(seconds=1.0)))


def measure(scans, T, expect, radius, obj_radius):
    """반환: (중심 평균, 표준편차, 검출 스캔 수, 스캔당 점 수 중앙값) 또는 None"""
    sensor = T[:2, 3]
    centers, counts = [], []
    for s in scans:
        pts, _, _ = scan_to_points(s)
        p = apply(T, pts)[:, :2]
        sel = np.linalg.norm(p - expect, axis=1) < radius
        if sel.sum() >= 3:
            q = p[sel]
            centers.append(fit_circle_center(q, obj_radius, sensor) if obj_radius > 0 else q.mean(axis=0))
            counts.append(sel.sum())
    if not centers:
        return None
    c = np.array(centers)
    return c.mean(axis=0), c.std(axis=0), len(c), float(np.median(counts))


def markers(frame, expect, center, radius, text):
    arr = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)

    def mk(mid, mtype, x, y, z, s, rgba):
        m = Marker()
        m.header.frame_id = frame
        m.ns, m.id, m.type, m.action = 'scan_probe', mid, mtype, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in s)
        m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
        return m

    # 기대 위치: 반투명 원판 + 허용 오차 5 cm 원
    arr.markers.append(mk(0, Marker.CYLINDER, expect[0], expect[1], 0.0, (0.10, 0.10, 0.01), (0.2, 0.6, 1.0, 0.9)))
    arr.markers.append(mk(1, Marker.CYLINDER, expect[0], expect[1], -0.005, (0.10 + 2 * 0.05, 0.10 + 2 * 0.05, 0.005),
                          (0.2, 0.6, 1.0, 0.25)))
    if center is not None:
        d = max(2 * radius, 0.04)
        arr.markers.append(mk(2, Marker.CYLINDER, center[0], center[1], 0.02, (d, d, 0.04), (1.0, 0.2, 0.2, 0.9)))
    t = mk(3, Marker.TEXT_VIEW_FACING, expect[0], expect[1], 0.35, (0, 0, 0.12), (1, 1, 1, 1))
    t.text = text
    arr.markers.append(t)
    return arr


def run_locate(node, col, args):
    buf = Buffer()
    TransformListener(buf, node)
    pub = node.create_publisher(MarkerArray, '/scan_probe/markers', LATCHED)
    expect = np.array(args.expect)
    T = None
    while True:
        scans = col.collect(args.seconds)
        if not scans:
            print(f'{args.topic} 에서 스캔을 받지 못했습니다.')
            return 1
        if T is None:
            T = lookup(node, buf, args.frame, scans[0].header.frame_id)
        r = measure(scans, T, expect, args.radius, args.object_radius)
        print(f'{args.frame} 기준 기대 위치 ({expect[0]:.3f}, {expect[1]:.3f}), 탐색 반경 {args.radius} m')
        if r is None:
            print('  반경 안에 점이 없음. 물체 위치나 방향(TF)을 확인하세요.')
            pub.publish(markers(args.frame, expect, None, args.object_radius, '점 없음'))
        else:
            m, sd, n, cnt = r
            err = m - expect
            dist = float(np.linalg.norm(err))
            verdict = 'PASS' if dist <= 0.05 else 'FAIL'
            print(f'  스캔 {n}/{len(scans)}개에서 검출, 스캔당 점 {cnt:.0f}개')
            print(f'  측정 중심 ({m[0]:.3f}, {m[1]:.3f})  표준편차 ({sd[0]:.3f}, {sd[1]:.3f})')
            print(f'  오차 dx={err[0]:+.3f} dy={err[1]:+.3f}  거리 {dist:.3f} m  -> {verdict} (기준 0.05 m)')
            print(f'  ({"반지름 %.3f m 원으로 맞춘 중심" % args.object_radius if args.object_radius > 0 else "점 평균. --object-radius 로 표면 치우침 보정"})')
            pub.publish(markers(args.frame, expect, m, args.object_radius,
                                f'err {dist * 100:.1f} cm {verdict}'))
        if not args.continuous:
            # 마커가 RViz 에 전달될 시간을 준다
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
            return 0
        print('-' * 60)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['occlusion', 'locate'], required=True)
    ap.add_argument('--topic', default='/scan_filtered')
    ap.add_argument('--seconds', type=float, default=5.0)
    ap.add_argument('--near', type=float, default=0.5, help='occlusion: 이 거리(m) 미만을 가림으로 봄')
    ap.add_argument('--ratio', type=float, default=0.5, help='occlusion: 이 비율 이상 가까우면 가림')
    ap.add_argument('--frame', default='base_link')
    ap.add_argument('--expect', type=float, nargs=2, metavar=('X', 'Y'))
    ap.add_argument('--radius', type=float, default=0.3, help='locate: 기대 위치 주변 탐색 반경(m)')
    ap.add_argument('--object-radius', type=float, default=0.0,
                    help='locate: 스캔 높이에서 물체 반지름(m). 주면 원을 맞춰 중심을 구함')
    ap.add_argument('--continuous', action='store_true', help='locate: Ctrl+C 까지 반복 측정')
    args = ap.parse_args(rclpy.utilities.remove_ros_args(argv if argv is not None else sys.argv)[1:])
    if args.mode == 'locate' and args.expect is None:
        ap.error('--mode locate 에는 --expect X Y 가 필요합니다')

    rclpy.init()
    node = rclpy.create_node('scan_probe')
    col = Collector(node, args.topic)
    try:
        if args.mode == 'occlusion':
            scans = col.collect(args.seconds)
            if not scans:
                print(f'{args.topic} 에서 스캔을 받지 못했습니다. 라이다가 실행 중인지 확인하세요.')
                return 1
            occlusion(scans, args.near, args.ratio)
            return 0
        return run_locate(node, col, args)
    except KeyboardInterrupt:
        return 0
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())

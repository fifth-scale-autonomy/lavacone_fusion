"""주행 bag 으로 카메라 숙인 각(pitch)과 yaw 를 추정한다 (카메라-라이다 정렬 확인).

    ros2 run lavacone_fusion extrinsic_check ~/bags/outdoor_1458
    ros2 run lavacone_fusion extrinsic_check ~/bags/outdoor_1458 --apply     # extrinsics.yaml 에 기록

방법 (2026-10-12 야외 bag 으로 만듦)
 1. 검출 박스마다 같은 시각 스캔의 콘 크기 군집과 짝짓는다. 기울기와 무관한 것만 쓴다:
    가로 위치(박스 안) + 박스 높이로 잰 거리 (f x 콘 높이 / 박스 높이) 와 라이다 거리의 차 30% 이내.
    아래가 잘린 박스, 30 px 미만 박스, 점수 0.5 미만은 뺀다. 후보 군집이 하나일 때만 짝으로 쓴다.
 2. 라이다 평면(높이 z_laser)이 콘의 같은 높이(박스 아랫변에서 z_laser / 콘 높이 만큼 위)에 오는 pitch,
    박스 가운데에 오는 yaw 를 잔차 중앙값이 0 이 되게 찾는다.
 3. 5 초 구간별 pitch 도 낸다. 구간마다 같으면 카메라 고정은 정상이고, bag 끼리 다르면 카메라가 움직인 것이다.
    (지면 기울기나 차체 기울기는 거리에 따라 다르게 나타나므로 가까운/먼 콘 추정값도 비교한다.)
 4. cone_fusion 과 같은 기준(박스 +-12 px 안, 지면 거리 35% 이내)으로 지금 값과 추정값의 짝지어지는 박스 비율.
"""
import argparse
import math
import os
import re

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.optimize import linear_sum_assignment

from lavacone_fusion.common import CameraModel, cluster_scan, default_extrinsics, rpy_to_matrix

OPT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])   # camera_link -> 광학 (z 전방, x 우, y 아래)


def read_bag(path, T_base_laser):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=path, storage_id=''), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in rd.get_all_topics_and_types()}
    rd.set_filter(rosbag2_py.StorageFilter(topics=['/cone/detections', '/scan_filtered', '/camera_info']))
    dets, scans, info = [], [], None
    while rd.has_next():
        topic, data, _ = rd.read_next()
        m = deserialize_message(data, get_message(types[topic]))
        ts = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if topic == '/cone/detections':
            boxes = [(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y,
                      d.results[0].hypothesis.score if d.results else 1.0) for d in m.detections]
            if boxes:
                dets.append((ts, np.array(boxes)))
        elif topic == '/scan_filtered':
            cl = []
            for pts, rng in cluster_scan(m, 0.10, 3.0, min_points=3):
                if 0.02 <= np.linalg.norm(pts[0, :2] - pts[-1, :2]) <= 0.45 and rng.min() <= 15.0:
                    c = pts[:, :2].mean(axis=0)
                    cl.append(T_base_laser[:3, :3] @ [c[0], c[1], 0.0] + T_base_laser[:3, 3])
            scans.append((ts, np.array(cl).reshape(-1, 3)))
        elif info is None:
            info = m
    st = np.array([s[0] for s in scans])
    frames = []
    for ts, boxes in dets:
        if len(st):
            k = int(np.argmin(np.abs(st - ts)))
            if abs(st[k] - ts) <= 0.12 and len(scans[k][1]):
                frames.append((ts, boxes, scans[k][1]))
    return frames, info


class Rig:
    def __init__(self, ext, cam):
        c = ext['camera_link']
        self.pos = np.array([c['x'], c['y'], c['z']])
        self.roll = c['roll']
        self.cam = cam

    def T_opt_base(self, pitch, yaw):
        R = rpy_to_matrix(self.roll, pitch, yaw)                       # base -> camera_link 회전
        Rob = OPT @ R.T
        return Rob, -Rob @ self.pos

    def project(self, P, pitch, yaw):
        Rob, t = self.T_opt_base(pitch, yaw)
        q = P @ Rob.T + t
        uv, ok = self.cam.project(q)
        return uv, ok, q[:, 2]

    def ground(self, u, v, pitch, yaw):
        Rob, t = self.T_opt_base(pitch, yaw)
        K = self.cam.K
        ray = Rob.T @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        if ray[2] >= -1e-6:
            return None
        return self.pos[:2] + (-self.pos[2] / ray[2]) * ray[:2]


def pairs_for(frames, rig, pitch0, yaw0, cone_h, z_laser):
    f = rig.cam.K[1, 1]
    out = []
    for ts, boxes, cl in frames:
        uv, ok, Z = rig.project(cl, pitch0, yaw0)
        for cx, cy, w, h, sc in boxes:
            y2 = cy + h / 2
            if y2 > rig.cam.height - 8 or h < 30 or sc < 0.5:
                continue
            d_box = f * cone_h / h
            cand = ok & (Z > 0.5) & (np.abs(uv[:, 0] - cx) <= w / 2 + 12) & (np.abs(Z - d_box) / Z <= 0.3)
            if cand.sum() == 1:
                j = int(np.flatnonzero(cand)[0])
                out.append((ts, cl[j], cx, y2 - (z_laser / cone_h) * h, Z[j], d_box))
    return out


def fit(pairs, rig, yaw, axis):
    """axis 1: pitch (세로 잔차 0), axis 0: yaw (가로 잔차 0). 도 단위 격자 탐색."""
    P = np.array([p[1] for p in pairs])
    target = np.array([p[3] if axis == 1 else p[2] for p in pairs])
    best = None
    for deg in np.arange(-10.0, 15.0001, 0.05):
        a = math.radians(deg)
        uv, ok, _ = rig.project(P, a, yaw) if axis == 1 else rig.project(P, yaw, a)
        if ok.sum() < 5:
            continue
        m = np.median(uv[ok, axis] - target[ok])
        if best is None or abs(m) < abs(best[1]):
            best = (deg, m)
    return best[0] if best else float('nan')


def match_rate(frames, rig, pitch, yaw):
    n = tot = 0
    for ts, boxes, cl in frames:
        tot += len(boxes)
        uv, ok, _ = rig.project(cl, pitch, yaw)
        cost = np.full((len(boxes), len(cl)), 1e6)
        for i, (cx, cy, w, h, sc) in enumerate(boxes):
            g = rig.ground(cx, cy + h / 2, pitch, yaw)
            for j in np.flatnonzero(ok):
                u, v = uv[j]
                if abs(u - cx) > w / 2 + 12 or abs(v - cy) > h / 2 + 12:
                    continue
                c = abs(u - cx) / max(w, 1.0)
                if g is not None:
                    dl = math.hypot(*cl[j, :2])
                    rel = abs(math.hypot(*g) - dl) / max(dl, 0.5)
                    if rel > 0.35:
                        continue
                    c += rel
                cost[i, j] = c
        r, c = linear_sum_assignment(cost)
        n += int(np.sum(cost[r, c] < 1.5))
    return n / max(tot, 1)


def apply_to_yaml(path, pitch_deg, yaw_deg, note):
    """camera_link 의 pitch / yaw 줄만 바꾼다 (주석은 그대로)."""
    text = open(path).read()
    head, sep, rest = text.partition('camera_link:')
    block_end = re.search(r'\n\S', rest)
    block, tail = (rest[:block_end.start()], rest[block_end.start():]) if block_end else (rest, '')
    for key, deg in (('pitch', pitch_deg), ('yaw', yaw_deg)):
        block = re.sub(rf'(\n\s*{key}:\s*)[-0-9.eE]+[^\n]*',
                       lambda m, d=deg: f'{m.group(1)}{math.radians(d):.4f}   # {d:.2f} deg ({note})', block)
    open(path, 'w').write(head + sep + block + tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--extrinsics', default=default_extrinsics(), help='기본: fifth_scale_bringup 의 extrinsics.yaml')
    ap.add_argument('--cone-height', type=float, default=0.70)
    ap.add_argument('--fallback-dfov', type=float, default=87.0, help='camera_info 가 비었을 때 공칭 대각 화각')
    ap.add_argument('--apply', action='store_true', help='추정값을 extrinsics.yaml 의 camera_link 에 기록')
    a = ap.parse_args()
    ext = yaml.safe_load(open(a.extrinsics))
    la = ext['laser']
    T_base_laser = np.eye(4)
    T_base_laser[:3, :3] = rpy_to_matrix(la['roll'], la['pitch'], la['yaw'])
    T_base_laser[:3, 3] = [la['x'], la['y'], la['z']]
    pitch0, yaw0 = ext['camera_link']['pitch'], ext['camera_link']['yaw']
    frames, info = [], None
    per_bag = {}
    for b in a.bags:
        fr, inf = read_bag(os.path.expanduser(b), T_base_laser)
        per_bag[b] = fr
        frames += fr
        info = info or inf
    rig = Rig(ext, CameraModel(info, a.fallback_dfov))
    pairs = pairs_for(frames, rig, pitch0, yaw0, a.cone_height, la['z'])
    if len(pairs) < 20:
        print(f'짝이 {len(pairs)}개뿐이라 추정할 수 없습니다 (콘이 2~8 m 에 보이는 주행 bag 이 필요)')
        return
    pitch = fit(pairs, rig, yaw0, 1)
    yaw = fit(pairs, rig, math.radians(pitch), 0)
    pitch = fit(pairs, rig, math.radians(yaw), 1)
    ratio = np.median([p[4] / p[5] for p in pairs])
    print(f'카메라 짝 {len(pairs)}개 ({"보정 전 공칭 화각" if rig.cam.provisional else "내부 캘리브레이션 값"} 사용). '
          f'라이다 거리 / 박스 높이 거리 중앙 {ratio:.2f} (1 에 가까우면 초점거리·콘 높이 가정이 맞음)')
    print(f'지금 값 : pitch {math.degrees(pitch0):6.2f} 도, yaw {math.degrees(yaw0):6.2f} 도 '
          f'-> 정합 {100 * match_rate(frames, rig, pitch0, yaw0):.0f}%')
    print(f'추정 값 : pitch {pitch:6.2f} 도, yaw {yaw:6.2f} 도 '
          f'-> 정합 {100 * match_rate(frames, rig, math.radians(pitch), math.radians(yaw)):.0f}%')
    for b, fr in per_bag.items():
        pb = pairs_for(fr, rig, pitch0, yaw0, a.cone_height, la['z'])
        if len(pb) < 20:
            continue
        t0 = pb[0][0]
        win = []
        for w0 in range(0, int(pb[-1][0] - t0) + 1, 5):
            s = [p for p in pb if w0 <= p[0] - t0 < w0 + 5]
            if len(s) >= 15:
                win.append(f'{fit(s, rig, math.radians(yaw), 1):+.1f}')
        near = [p for p in pb if p[4] < 4.0]
        far = [p for p in pb if p[4] >= 4.0]
        nf = ' / '.join(f'{fit(s, rig, math.radians(yaw), 1):+.1f}' if len(s) >= 15 else '-' for s in (near, far))
        print(f'  {os.path.basename(b.rstrip("/"))}: pitch {fit(pb, rig, math.radians(yaw), 1):+.2f} 도 (짝 {len(pb)}) | '
              f'5 초 구간 {" ".join(win)} | 4 m 안/밖 {nf}')
    if len(per_bag) > 1:
        print('  bag 끼리 pitch 가 1 도 넘게 다르면 카메라 고정부가 움직인 것입니다. 고정한 뒤 다시 재 주세요.')
    if a.apply:
        apply_to_yaml(os.path.realpath(a.extrinsics), pitch, yaw, 'extrinsic_check 추정')
        print(f'{os.path.realpath(a.extrinsics)} 에 기록했습니다 (다음 실행부터 적용)')


if __name__ == '__main__':
    main()

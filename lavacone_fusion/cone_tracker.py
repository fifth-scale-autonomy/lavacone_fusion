"""라이다 군집 기반 콘 추적 (ROS 와 무관한 순수 계산).

매 프레임 짝지어진 콘만 쓰면 콘 수가 프레임마다 흔들린다 (정지 상태에서도 0~7개).
라이다는 360도를 10 Hz 로 보므로, 한 번 카메라로 콘임이 확인된 군집은 카메라 시야를 벗어나거나
검출이 잠깐 빠져도 라이다로 계속 따라간다.

- update(): 스캔마다 콘 크기 군집 중심(base_link 2D)을 기존 추적과 이어 붙인다 (헝가리안, gate 이내)
- confirm(): YOLO 박스와 짝지어진 군집의 추적에 카메라 확인 횟수를 더한다
- confirmed(): 카메라 확인이 min_cam_hits 번 이상이고 최근 스캔에서 보인 추적

- move(): 주행 거리계로 구한 직전 스캔 -> 이번 스캔 사이 차량 이동만큼 추적 위치를 옮긴다 (이동 보정)

이동 보정이 있으면 카메라로 확인된 콘은 라이다에서 안 보여도 memory_misses 스캔 동안 위치를 예측하며 유지한다.
이동 보정이 없으면 위치 예측 없이 가장 가까운 군집과 잇고, output_max_misses 만큼만 출력한다.
10 Hz 에서 gate 0.5 m 는 보정 없이 약 5 m/s 까지 따라갈 수 있다 (콘 간격이 gate 보다 넓어야 한다).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def relative_motion(pose_prev, pose_now):
    """odom 기준 차량 자세 두 개 (4x4 또는 3x3 동차 행렬) -> 직전 base_link 좌표를 이번 base_link 좌표로 옮기는
    (2x2 회전, 2 이동). 차량이 앞으로 가면 정지한 콘은 뒤로(x 감소) 온다."""
    a, b = np.asarray(pose_prev, dtype=float), np.asarray(pose_now, dtype=float)
    n = a.shape[0] - 1
    m = np.linalg.inv(b) @ a
    return m[:2, :2], m[:2, n]


@dataclass
class Track:
    id: int
    xy: np.ndarray
    hits: int = 1           # 라이다로 이어진 스캔 수
    misses: int = 0         # 연속으로 안 보인 스캔 수
    cam_hits: int = 0       # 카메라(YOLO)로 확인된 횟수
    score: float = 0.0      # 마지막 카메라 확인 점수


class ConeTracker:
    def __init__(self, gate=0.5, max_misses=5, min_cam_hits=2, output_max_misses=1, memory_misses=30):
        self.gate = gate
        self.max_misses = max_misses
        self.min_cam_hits = min_cam_hits
        self.output_max_misses = output_max_misses
        self.memory_misses = memory_misses     # 이동 보정 중 확인된 콘을 유지하는 스캔 수 (10 Hz 에서 3 s)
        self.tracks = []
        self._next_id = 0
        self.compensated = False               # 마지막 update 전에 move() 가 불렸는지

    def move(self, rot, trans):
        """직전 base_link 좌표 -> 이번 base_link 좌표 (2x2 회전, 2 이동) 를 모든 추적에 적용한다."""
        rot = np.asarray(rot, dtype=float)
        trans = np.asarray(trans, dtype=float)
        for t in self.tracks:
            t.xy = rot @ t.xy + trans
        self.compensated = True

    def _keep(self, t):
        if self.compensated and t.cam_hits >= self.min_cam_hits:
            return t.misses <= self.memory_misses
        return t.misses <= self.max_misses

    def update(self, centers):
        """centers (N,2) -> 각 군집이 이어진 추적 id 목록 (길이 N)."""
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        ids = [-1] * len(centers)
        matched_t = set()
        if self.tracks and len(centers):
            prev = np.array([t.xy for t in self.tracks])
            cost = np.linalg.norm(prev[:, None, :] - centers[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(np.where(cost <= self.gate, cost, 1e6))
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.gate:
                    t = self.tracks[r]
                    t.xy = centers[c].copy()
                    t.hits += 1
                    t.misses = 0
                    ids[c] = t.id
                    matched_t.add(r)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if self._keep(t)]
        for c, tid in enumerate(ids):
            if tid < 0:
                t = Track(self._next_id, centers[c].copy())
                self._next_id += 1
                self.tracks.append(t)
                ids[c] = t.id
        return ids

    def confirm(self, track_id, score=1.0):
        for t in self.tracks:
            if t.id == track_id:
                t.cam_hits += 1
                t.score = max(score, 0.9 * t.score)
                return True
        return False

    def confirmed(self):
        limit = self.memory_misses if self.compensated else self.output_max_misses
        out = [t for t in self.tracks if t.cam_hits >= self.min_cam_hits and t.misses <= limit]
        return out

    def end_scan(self):
        """스캔 처리 후 호출: 다음 스캔에 이동 보정이 없으면 기억을 쓰지 않는다."""
        self.compensated = False

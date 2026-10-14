"""여러 노드가 같이 쓰는 좌표 변환, 카메라 모델, 스캔 처리 함수."""
import math
import os

import cv2
import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# 처리가 느린 노드는 대기열에 쌓인 옛 프레임을 처리하지 않도록 최신 1장만 받는다.
# 이미지(1280x720 YUYV 1.8 MB)는 여러 조각으로 나뉘어 전송되는데, BEST_EFFORT 로 받으면 조각 하나만
# 빠져도 프레임 전체를 버린다. 측정 결과 카메라 10 fps 에서 BEST_EFFORT 0~5 Hz, RELIABLE 10 Hz.
LATEST_ONLY = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
# 최근 몇 장을 모아 두고 시각으로 고르는 용도 (정합 노드의 디버그 이미지)
RECENT_FRAMES = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)


# ---------------------------------------------------------------- 변환

def default_extrinsics():
    """센서 위치 파일 기본 경로: 통합 패키지 fifth_scale_bringup 의 config/extrinsics.yaml (없으면 '').
    이 패키지는 통합 패키지에 의존하지 않으므로 (통합 패키지가 이 패키지를 실행한다) 실행 때 찾는다."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('fifth_scale_bringup'), 'config', 'extrinsics.yaml')
    except Exception:  # noqa: BLE001
        return ''


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def transform_to_matrix(tf_stamped):
    """geometry_msgs/TransformStamped -> 4x4 (source 좌표를 target 좌표로 옮기는 행렬)."""
    t = tf_stamped.transform
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(t.rotation)
    m[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return m


def apply(m, pts):
    return pts @ m[:3, :3].T + m[:3, 3]


# ---------------------------------------------------------------- 카메라

class CameraModel:
    """sensor_msgs/CameraInfo 의 내부 파라미터와 왜곡 계수로 투영한다.

    camera_info 가 비어 있으면(캘리브레이션 전) 공칭 대각 화각으로 만든 임시 값을 쓰고
    provisional=True 로 표시한다. 임시 값으로는 정합 정확도를 판단하면 안 된다.
    """

    def __init__(self, info, fallback_dfov_deg=87.0):
        self.width, self.height = info.width, info.height
        k = np.array(info.k, dtype=float).reshape(3, 3)
        self.provisional = not np.any(k)
        if self.provisional:
            f = 0.5 * math.hypot(self.width, self.height) / math.tan(math.radians(fallback_dfov_deg) / 2)
            k = np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1]])
            d = np.zeros(5)
            model = 'plumb_bob'
        else:
            d = np.array(info.d, dtype=float)
            model = info.distortion_model
        self.K, self.D = k, d
        self.fisheye = model == 'equidistant'
        # 화각 밖 점은 왜곡 다항식이 엉뚱한 곳으로 접어 버리므로 미리 거른다
        self.max_tan_x = 1.15 * max(self.width - k[0, 2], k[0, 2]) / k[0, 0]
        self.max_tan_y = 1.15 * max(self.height - k[1, 2], k[1, 2]) / k[1, 1]

    def project(self, pts):
        """광학 프레임 점 (N,3) -> 픽셀 (N,2), 이미지 안에 들어오는 점의 마스크."""
        uv = np.full((len(pts), 2), np.nan)
        if len(pts) == 0:
            return uv, np.zeros(0, bool)
        z = pts[:, 2]
        ok = z > 0.1
        ok &= np.abs(pts[:, 0]) < self.max_tan_x * np.maximum(z, 1e-6)
        ok &= np.abs(pts[:, 1]) < self.max_tan_y * np.maximum(z, 1e-6)
        if ok.any():
            p = pts[ok].reshape(-1, 1, 3).astype(np.float64)
            zero = np.zeros(3)
            if self.fisheye:
                out, _ = cv2.fisheye.projectPoints(p, zero, zero, self.K, self.D[:4])
            else:
                out, _ = cv2.projectPoints(p, zero, zero, self.K, self.D)
            uv[ok] = out.reshape(-1, 2)
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        return uv, inside

    def pixel_ray(self, uv):
        """픽셀 (N,2) -> 광학 프레임 방향 벡터 (N,3), z=1."""
        p = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            n = cv2.fisheye.undistortPoints(p, self.K, self.D[:4])
        else:
            n = cv2.undistortPoints(p, self.K, self.D)
        n = n.reshape(-1, 2)
        return np.hstack([n, np.ones((len(n), 1))])


# ---------------------------------------------------------------- 스캔

def scan_to_points(scan):
    """LaserScan -> (점 (N,3) laser 프레임, 원래 인덱스 (N,), 거리 (N,))."""
    r = np.asarray(scan.ranges, dtype=float)
    a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
    ok = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max)
    idx = np.flatnonzero(ok)
    r, a = r[ok], a[ok]
    pts = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], axis=1)
    return pts, idx, r


def cluster_scan(scan, jump_dist=0.10, jump_k=3.0, max_gap=3, min_points=3):
    """각도 순서로 이웃한 점끼리 묶는다.

    이웃 점 사이 거리가 max(jump_dist, jump_k * r * dtheta) 를 넘거나
    무효 값이 max_gap 칸보다 길게 이어지면 새 군집으로 나눈다.
    반환: [(점 (M,3), 거리 (M,))]
    """
    pts, idx, r = scan_to_points(scan)
    if len(pts) == 0:
        return []
    dth = abs(scan.angle_increment)
    step = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    thr = np.maximum(jump_dist, jump_k * np.minimum(r[:-1], r[1:]) * dth * np.diff(idx))
    cut = (step > thr) | (np.diff(idx) > max_gap + 1)
    groups = np.split(np.arange(len(pts)), np.flatnonzero(cut) + 1)
    # 360도 스캔이므로 처음과 끝이 이어질 수 있다
    n = len(scan.ranges)
    if len(groups) > 1 and idx[0] + (n - idx[-1]) <= max_gap + 1:
        if np.linalg.norm(pts[0, :2] - pts[-1, :2]) <= max(jump_dist, jump_k * min(r[0], r[-1]) * dth):
            groups[0] = np.concatenate([groups[-1], groups[0]])
            groups.pop()
    return [(pts[g], r[g]) for g in groups if len(g) >= min_points]


def fit_circle_center(pts, r, sensor):
    """반지름 r 을 아는 원의 중심을 표면 점에서 구한다 (Gauss-Newton, 2D).

    라이다는 물체의 센서 쪽 면만 보므로 점 평균은 중심보다 센서 쪽으로 치우친다.
    """
    pts = np.asarray(pts, dtype=float)[:, :2]
    sensor = np.asarray(sensor, dtype=float)[:2]
    m = pts.mean(axis=0)
    d = m - sensor
    c = m + r * d / max(np.linalg.norm(d), 1e-9)
    for _ in range(20):
        v = pts - c
        dist = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        res = dist - r
        jac = -v / dist[:, None]
        step, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    # 점이 적거나 거의 일직선이면 발산할 수 있다. 그러면 단순 보정으로 돌아간다
    if np.linalg.norm(c - m) > 2 * r:
        c = m + r * d / max(np.linalg.norm(d), 1e-9)
    return c


def distance_colors(dist, max_dist):
    """거리 -> BGR 색 (가까우면 빨강, 멀면 파랑)."""
    v = np.clip(np.asarray(dist) / max_dist, 0, 1)
    v = (255 * (1 - v)).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(v, cv2.COLORMAP_JET).reshape(-1, 3)


def draw_colorbar(img, max_dist, x=10, y=None, w=220, h=12):
    y = img.shape[0] - 30 if y is None else y
    bar = distance_colors(np.linspace(0, max_dist, w), max_dist).reshape(1, w, 3)
    img[y:y + h, x:x + w] = np.repeat(bar, h, axis=0)
    cv2.putText(img, '0m', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f'{max_dist:.0f}m', (x + w - 30, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)


def put_lines(img, lines, org=(10, 24), color=(255, 255, 255), scale=0.55):
    x, y = org
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(26 * scale / 0.55)

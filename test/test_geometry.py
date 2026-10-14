"""정합에 쓰는 기하 계산 단위 시험 (센서 없이 실행).

    cd ~/ros2_ws/src/lavacone_fusion && python3 -m pytest test -q
"""
import math

import numpy as np
from sensor_msgs.msg import CameraInfo, LaserScan

from lavacone_fusion.common import (CameraModel, apply, cluster_scan, fit_circle_center, rpy_to_matrix,
                                    rpy_to_quat)


def synthetic_scan(circles, n=3240, wall=None):
    """laser 프레임에 원(cx, cy, r)들을 둔 360도 스캔."""
    s = LaserScan()
    s.angle_min, s.angle_max = -math.pi, math.pi
    s.angle_increment = 2 * math.pi / (n - 1)
    s.range_min, s.range_max = 0.05, 30.0
    ranges = np.full(n, np.inf)
    a = s.angle_min + np.arange(n) * s.angle_increment
    u = np.stack([np.cos(a), np.sin(a)], 1)
    if wall is not None:                       # x = wall 인 벽
        with np.errstate(divide='ignore'):
            t = wall / u[:, 0]
        ranges = np.where(t > 0, t, ranges)
    for cx, cy, r in circles:
        c = np.array([cx, cy])
        b = u @ c
        disc = b * b - (c @ c - r * r)
        t = np.where(disc > 0, b - np.sqrt(np.maximum(disc, 0)), np.inf)
        ranges = np.minimum(ranges, np.where(t > 0, t, np.inf))
    s.ranges = ranges.astype(float).tolist()
    return s


def test_cluster_finds_cones_in_front_of_wall():
    cones = [(2.0, 0.0, 0.05), (4.0, 1.0, 0.05), (6.0, -1.5, 0.05)]
    clusters = cluster_scan(synthetic_scan(cones, wall=8.0))
    small = [(p, r) for p, r in clusters if np.linalg.norm(p[0, :2] - p[-1, :2]) < 0.2]
    assert len(small) == 3
    for (cx, cy, rad), (pts, _) in zip(sorted(cones), sorted(small, key=lambda c: c[0][:, 0].mean())):
        c = fit_circle_center(pts, rad, (0, 0))
        assert np.linalg.norm(c - [cx, cy]) < 0.01


def test_cluster_wraps_around_pi():
    # 뒤쪽(+-180도 경계)에 걸친 물체가 한 군집으로 묶이는지
    clusters = cluster_scan(synthetic_scan([(-2.0, 0.0, 0.1)]))
    assert len(clusters) == 1


def make_info(w=1280, h=720, f=700.0, d=(0.0, 0.0, 0.0, 0.0, 0.0)):
    info = CameraInfo()
    info.width, info.height = w, h
    info.k = [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0]
    info.d = list(d)
    info.distortion_model = 'plumb_bob'
    return info


def test_project_and_back_with_distortion():
    cam = CameraModel(make_info(d=(-0.2, 0.05, 0.001, -0.001, 0.0)))
    pts = np.array([[0.5, 0.2, 3.0], [-1.0, -0.3, 5.0], [0.0, 0.0, 10.0]])
    uv, inside = cam.project(pts)
    assert inside.all()
    ray = cam.pixel_ray(uv)
    assert np.allclose(ray[:, :2], pts[:, :2] / pts[:, 2:], atol=1e-4)


def test_points_behind_or_outside_are_rejected():
    cam = CameraModel(make_info())
    _, inside = cam.project(np.array([[0, 0, -1.0], [50.0, 0, 1.0]]))
    assert not inside.any()


def test_provisional_intrinsics_from_fov():
    info = make_info()
    info.k = [0.0] * 9
    cam = CameraModel(info, fallback_dfov_deg=90.0)
    assert cam.provisional
    assert abs(cam.K[0, 0] - math.hypot(1280, 720) / 2) < 1e-6


def test_ground_ray_matches_known_geometry():
    # 카메라: 높이 0.92 m, 22도 숙임. 광학 프레임 회전 포함
    R_base_cam = rpy_to_matrix(0, math.radians(22), 0)
    R_cam_opt = rpy_to_matrix(-math.pi / 2, 0, -math.pi / 2)
    T = np.eye(4)
    T[:3, :3] = R_base_cam @ R_cam_opt
    T[:3, 3] = [0.53, 0, 0.92]
    cam = CameraModel(make_info())
    # 지면 위 점 (4, 0.5, 0) 을 투영했다가 다시 지면과 교차시키면 같은 점이어야 한다
    g = np.array([[4.0, 0.5, 0.0]])
    uv, inside = cam.project(apply(np.linalg.inv(T), g))
    assert inside.all()
    ray = apply(T, cam.pixel_ray(uv))[0] - T[:3, 3]
    s = -T[2, 3] / ray[2]
    assert np.allclose(T[:3, 3] + s * ray, g[0], atol=1e-3)


def test_rpy_quat_consistent():
    r, p, y = 0.1, -0.2, 0.3
    x, yy, z, w = rpy_to_quat(r, p, y)

    class Q:
        pass
    q = Q()
    q.x, q.y, q.z, q.w = x, yy, z, w
    from lavacone_fusion.common import quat_to_matrix
    assert np.allclose(quat_to_matrix(q), rpy_to_matrix(r, p, y), atol=1e-9)

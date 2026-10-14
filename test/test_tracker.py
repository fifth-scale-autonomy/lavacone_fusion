"""콘 추적 단위 시험 (센서 없이 실행)."""
import numpy as np

from lavacone_fusion.cone_tracker import ConeTracker, relative_motion

CONES = np.array([[3.0, 1.5], [3.0, -1.5], [5.0, 1.5], [5.0, -1.5], [7.0, 1.5], [7.0, -1.5]])


def test_ids_stay_stable_while_driving_forward():
    tr = ConeTracker()
    first = tr.update(CONES)
    for k in range(1, 15):                       # 2 m/s, 10 Hz -> 스캔마다 0.2 m 씩 다가옴
        ids = tr.update(CONES - [0.2 * k, 0.0])
        assert ids == first


def test_cone_confirmed_once_stays_through_camera_dropouts():
    tr = ConeTracker(min_cam_hits=2)
    ids = tr.update(CONES)
    for tid in ids:                               # 카메라로 두 번 확인
        tr.confirm(tid)
        tr.confirm(tid)
    for k in range(1, 10):                        # 이후 카메라 확인 없이 라이다만
        tr.update(CONES - [0.2 * k, 0.0])
        assert len(tr.confirmed()) == 6


def test_unconfirmed_clusters_are_not_output():
    tr = ConeTracker(min_cam_hits=2)
    ids = tr.update(np.vstack([CONES, [[4.0, 0.0]]]))      # 벽 조각 같은 군집 하나 추가
    for tid in ids[:6]:
        tr.confirm(tid)
        tr.confirm(tid)
    tr.update(np.vstack([CONES, [[4.0, 0.0]]]))
    out = {tuple(np.round(t.xy, 1)) for t in tr.confirmed()}
    assert (4.0, 0.0) not in out and len(out) == 6


def test_short_lidar_miss_keeps_track_long_miss_drops():
    tr = ConeTracker(max_misses=5, output_max_misses=1)
    ids = tr.update(CONES)
    for tid in ids:
        tr.confirm(tid)
        tr.confirm(tid)
    tr.update(CONES[1:])                          # 첫 콘이 한 스캔 빠짐
    assert len(tr.confirmed()) == 6               # 한 번 빠진 것은 출력 유지
    tr.update(CONES)
    assert tr.update(CONES)[0] == ids[0]          # 같은 id 로 이어짐
    for _ in range(6):
        tr.update(CONES[1:])
    assert ids[0] not in [t.id for t in tr.tracks]   # 0.6 s 넘게 안 보이면 삭제


def test_far_jump_creates_new_track():
    tr = ConeTracker(gate=0.5)
    a = tr.update([[3.0, 0.0]])
    b = tr.update([[3.0, 1.0]])                   # 1 m 튐 -> 다른 물체로 봄
    assert a != b


def _pose(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def _to_base(world, pose):
    """월드(odom) 좌표 콘 -> 차량 자세 pose 에서 본 base_link 좌표."""
    inv = np.linalg.inv(pose)
    return world @ inv[:2, :2].T + inv[:2, 2]


def test_relative_motion_moves_static_cones_backward():
    rot, trans = relative_motion(_pose(0, 0, 0), _pose(0.5, 0, 0))
    assert np.allclose(rot @ [3.0, 1.0] + trans, [2.5, 1.0])
    rot, trans = relative_motion(_pose(0, 0, 0), _pose(1.0, 1.0, np.pi / 2))   # 좌회전 90도
    assert np.allclose(rot @ [3.0, 1.0] + trans, [0.0, -2.0])


def test_compensation_keeps_ids_on_fast_turn():
    """3 m/s 로 반경 4 m 좌회전: 보정 없으면 측면 콘이 gate(0.5 m)를 넘어 새 id 가 된다."""
    world = np.array([[4.0, 2.0], [4.0, -2.0], [6.0, 3.0], [6.0, -1.0]])

    def run(compensate):
        tr = ConeTracker(gate=0.5)
        prev, first = _pose(0, 0, 0), None
        for k in range(10):
            yaw = 0.075 * k * 1.0                # 0.1 s 마다 0.3 m, yaw 율 3/4 rad/s
            pose = _pose(4 * np.sin(yaw), 4 * (1 - np.cos(yaw)), yaw)
            if compensate and k:
                tr.move(*relative_motion(prev, pose))
            ids = tr.update(_to_base(world, pose))
            tr.end_scan()
            first = first or ids
            if ids != first:
                return False
            prev = pose
        return True
    assert run(True)
    assert not run(False)


def test_memory_holds_confirmed_cone_out_of_view_only_with_compensation():
    world = np.array([[3.0, 1.5], [3.0, -1.5]])
    tr = ConeTracker(max_misses=5, output_max_misses=1, memory_misses=30)
    ids = tr.update(world)
    for tid in ids:
        tr.confirm(tid)
        tr.confirm(tid)
    tr.end_scan()
    prev = _pose(0, 0, 0)
    for k in range(1, 21):                       # 1 m/s 로 2 s 직진, 라이다에 아무것도 안 보임
        pose = _pose(0.1 * k, 0, 0)
        tr.move(*relative_motion(prev, pose))
        tr.update(np.zeros((0, 2)))
        out = tr.confirmed()
        tr.end_scan()
        prev = pose
    assert len(out) == 2                         # 기억으로 유지
    assert np.allclose(sorted(t.xy.tolist() for t in out), [[1.0, -1.5], [1.0, 1.5]])

    tr2 = ConeTracker(max_misses=5, output_max_misses=1, memory_misses=30)
    for tid in tr2.update(world):
        tr2.confirm(tid)
        tr2.confirm(tid)
    for _ in range(3):                           # 보정 없음: 2 스캔째부터 출력 안 함
        tr2.update(np.zeros((0, 2)))
    assert tr2.confirmed() == []


def test_memory_expires():
    tr = ConeTracker(memory_misses=30)
    for tid in tr.update([[3.0, 1.5]]):
        tr.confirm(tid)
        tr.confirm(tid)
    for _ in range(31):
        tr.move(np.eye(2), [0.0, 0.0])
        tr.update(np.zeros((0, 2)))
        tr.end_scan()
    assert tr.tracks == []


def test_unconfirmed_cluster_is_not_remembered():
    tr = ConeTracker(max_misses=5, memory_misses=30)
    tr.update([[3.0, 0.0]])                      # 카메라 확인 없는 군집
    for _ in range(6):
        tr.move(np.eye(2), [0.0, 0.0])
        tr.update(np.zeros((0, 2)))
        tr.end_scan()
    assert tr.tracks == []

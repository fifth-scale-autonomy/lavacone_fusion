"""콘 검출 모델 처리 시간 측정 (3단계).

1) 실제 카메라 프레임 저장 (카메라 launch 가 떠 있어야 함)
    ~/ros2_ws/venv/bin/python -m lavacone_fusion.bench_models capture --out ~/ros2_ws/bench/frames_1280 --n 40

2) 모델별 측정 (ROS 없이 같은 프레임으로 반복)
    ~/ros2_ws/venv/bin/python -m lavacone_fusion.bench_models run \
        --frames ~/ros2_ws/bench/frames_1280 --models ~/ros2_ws/fsoco_yolo/cone_640_*.pt --imgsz 640 \
        --backends openvino pytorch --csv ~/ros2_ws/bench/bench.csv

전처리·추론·후처리는 Ultralytics 가 재는 값(res.speed), total 은 predict 호출 전체의 벽시계 시간이다.
"""
import argparse
import csv
import glob
import os
import platform
import time

import cv2
import numpy as np


def capture(args):
    import rclpy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('bench_capture')
    bridge = CvBridge()
    saved = []
    gate = {'next': 0.0}

    def cb(msg):
        now = time.monotonic()
        if len(saved) >= args.n or now < gate['next']:
            return
        gate['next'] = now + 1.0 / args.rate
        img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        path = os.path.join(os.path.expanduser(args.out), f'frame_{len(saved):03d}.png')
        cv2.imwrite(path, img)
        saved.append(path)

    node.create_subscription(Image, args.topic, cb, 10)
    end = time.monotonic() + args.n / args.rate + 10
    while len(saved) < args.n and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    print(f'{len(saved)}장 저장: {args.out}')
    node.destroy_node()
    rclpy.shutdown()


def stats(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), np.percentile(v, 50), np.percentile(v, 95), v.max()


def run(args):
    from ultralytics import YOLO

    from lavacone_fusion.cone_detector import imgsz_tag, normalize_imgsz, prepare_model

    args.imgsz = normalize_imgsz(args.imgsz)
    frames = sorted(glob.glob(os.path.join(os.path.expanduser(args.frames), '*.png')))
    if not frames:
        raise SystemExit(f'프레임 없음: {args.frames}')
    imgs = [cv2.imread(f) for f in frames]
    h, w = imgs[0].shape[:2]

    class Log:
        def info(self, m):
            print(m)

    rows = []
    for model_path in args.models:
        for backend in args.backends:
            if backend == 'pytorch':
                model, loaded = YOLO(model_path, task='detect'), model_path
            else:
                model, loaded = prepare_model(model_path, backend, args.imgsz, args.export_dir, Log())
            kw = dict(imgsz=args.imgsz, conf=args.conf, verbose=False)
            if backend == 'pytorch':
                kw['device'] = 'cpu'
            for i in range(args.warmup):
                model.predict(imgs[i % len(imgs)], **kw)
            pre, inf, post, tot, ndet = [], [], [], [], []
            for i in range(args.n):
                t0 = time.perf_counter()
                r = model.predict(imgs[i % len(imgs)], **kw)[0]
                tot.append((time.perf_counter() - t0) * 1e3)
                pre.append(r.speed['preprocess'])
                inf.append(r.speed['inference'])
                post.append(r.speed['postprocess'])
                ndet.append(len(r.boxes))
            row = {
                'model': os.path.basename(model_path), 'backend': backend, 'imgsz': imgsz_tag(args.imgsz),
                'frame': f'{w}x{h}', 'n': args.n,
            }
            for name, v in (('pre', pre), ('inf', inf), ('post', post), ('total', tot)):
                m, p50, p95, mx = stats(v)
                row.update({f'{name}_mean': round(m, 2), f'{name}_p50': round(p50, 2),
                            f'{name}_p95': round(p95, 2), f'{name}_max': round(mx, 2)})
            row['max_hz'] = round(1000.0 / row['total_mean'], 2)
            row['det_mean'] = round(float(np.mean(ndet)), 2)
            rows.append(row)
            print(f"{row['model']:22s} {backend:9s} imgsz={row['imgsz']} {w}x{h}  "
                  f"전처리 {row['pre_mean']:6.1f}  추론 {row['inf_mean']:6.1f} (p95 {row['inf_p95']:6.1f})  "
                  f"후처리 {row['post_mean']:5.1f}  합계 {row['total_mean']:6.1f} ms  -> 최대 {row['max_hz']:.1f} Hz  "
                  f"검출 {row['det_mean']:.1f}개", flush=True)
            del model
    if args.csv:
        path = os.path.expanduser(args.csv)
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=['time', 'host'] + list(rows[0].keys()))
            if new:
                wr.writeheader()
            for r in rows:
                wr.writerow({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'host': platform.node(), **r})
        print(f'CSV 추가: {path}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--out', required=True)
    c.add_argument('--n', type=int, default=40)
    c.add_argument('--rate', type=float, default=2.0, help='초당 저장 장수 (장면 변화를 담으려고 띄엄띄엄)')
    c.add_argument('--topic', default='/image_raw')
    r = sub.add_parser('run')
    r.add_argument('--frames', required=True)
    r.add_argument('--models', nargs='+', required=True)
    r.add_argument('--imgsz', type=int, nargs='+', required=True, help='정수 하나 또는 높이 너비')
    r.add_argument('--backends', nargs='+', default=['openvino'], choices=['onnx', 'openvino', 'pytorch'])
    r.add_argument('--conf', type=float, default=0.25)
    r.add_argument('--n', type=int, default=60)
    r.add_argument('--warmup', type=int, default=5)
    r.add_argument('--export-dir', default=os.path.expanduser('~/ros2_ws/models'))
    r.add_argument('--csv', default='')
    args = ap.parse_args()
    capture(args) if args.cmd == 'capture' else run(args)


if __name__ == '__main__':
    main()

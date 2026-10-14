"""YOLO 콘 검출 노드.  입력: 원본 이미지, 출력: vision_msgs/Detection2DArray

- header.stamp 는 입력 이미지의 stamp 를 그대로 쓴다 (라이다와 시각 맞춤용).
- 박스는 원본 해상도 픽셀 좌표.
- backend:=onnx / openvino 이면 .pt 를 처음 한 번 ONNX / OpenVINO(FP32)로 변환해 export_dir 에 둔다.
  .pt 옆에 같은 이름의 변환본 (<이름>_imgsz<크기>.onnx 등) 이 있으면 변환 없이 그것을 쓴다 (패키지 동봉 모델 models/).
- imgsz 는 정수(정사각형) 또는 [높이, 너비]. 1280x720 카메라에 [384, 640] 을 쓰면 위아래 여백이
  거의 없다 (640x640 은 입력의 44% 가 여백).
- 단계별(형식 변환/전처리/추론/후처리/발행/디버그) 경과 시간과 CPU 시간을 이동 평균으로 로그에 남기고,
  timing_csv 를 주면 프레임별로 기록한다. CPU 시간은 프로세스 전체(모든 스레드) 기준이다.

venv 파이썬으로 실행해야 한다 (bringup.launch.py 가 처리).
"""
import csv
import os
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from lavacone_fusion.common import LATEST_ONLY

EXPORTS = {
    # backend: (ultralytics export format, 결과 경로 접미사)
    'openvino': ('openvino', '_openvino_model'),   # 폴더
    'onnx': ('onnx', '.onnx'),                      # 파일, onnxruntime(CPU)으로 실행
}
STAGES = ('convert', 'pre', 'inf', 'post', 'predict_other', 'publish', 'debug')


def normalize_imgsz(v):
    """정수 또는 [h, w] -> int 또는 [h, w] (ultralytics 형식)."""
    if isinstance(v, (list, tuple)) or hasattr(v, '__len__'):
        v = [int(x) for x in v]
        if len(v) == 1:
            return v[0]
        if len(v) != 2:
            raise ValueError(f'imgsz 는 정수 또는 [높이, 너비]: {v}')
        return v
    return int(v)


def imgsz_tag(imgsz):
    return f'{imgsz[0]}x{imgsz[1]}' if isinstance(imgsz, list) else str(imgsz)


def prepare_model(path, backend, imgsz, export_dir, logger):
    """backend 에 맞는 모델을 준비한다. .pt 를 주면 처음 한 번 변환해 export_dir 에 둔다."""
    from ultralytics import YOLO

    path = os.path.expanduser(path)
    if backend == 'pytorch' or not path.endswith('.pt'):
        return YOLO(path, task='detect'), path
    if backend not in EXPORTS:
        raise ValueError(f'backend 는 {["pytorch", *EXPORTS]} 중 하나: {backend}')
    fmt, suffix = EXPORTS[backend]
    stem = os.path.splitext(os.path.basename(path))[0]
    tag = imgsz_tag(imgsz)
    export_dir = os.path.expanduser(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    out = os.path.join(export_dir, f'{stem}_imgsz{tag}{suffix}')
    bundled = os.path.join(os.path.dirname(path), f'{stem}_imgsz{tag}{suffix}')
    if os.path.exists(bundled):                 # .pt 옆에 같은 이름의 변환본이 있으면 (패키지 동봉 모델) 그것을 쓴다
        return YOLO(bundled, task='detect'), bundled
    if not os.path.exists(out):
        # 원본 폴더(fsoco_yolo)에 파일을 만들지 않도록 링크를 거쳐 export_dir 에 내보낸다
        link = os.path.join(export_dir, f'{stem}_imgsz{tag}.pt')
        if not os.path.exists(link):
            os.symlink(path, link)
        logger.info(f'{fmt} 변환 중 (처음 한 번): {path} imgsz={imgsz}')
        kw = dict(format=fmt, imgsz=imgsz, dynamic=False)   # FP32 (기본값)
        if fmt == 'onnx':
            kw['simplify'] = True
        YOLO(link, task='detect').export(**kw)
    return YOLO(out, task='detect'), out


def configure_onnxruntime(model, onnx_path, spinning, threads, logger):
    """ultralytics 가 기본 옵션으로 만든 onnxruntime 세션을 옵션을 바꾼 세션으로 교체한다.

    ultralytics(8.4)는 세션 옵션을 넘길 방법이 없다. 기본값은 작업 스레드가 추론 뒤에도 바쁘게 대기(spin)해
    10 Hz 에서 프로세스 CPU 가 약 2배가 된다 (측정: spin 켬 204%, 끔 112%, 추론은 +9 ms).
    """
    import onnxruntime as ort

    backend = getattr(model.predictor.model, 'backend', None)
    session = getattr(backend, 'session', None)
    if not isinstance(session, ort.InferenceSession):
        logger.warn('onnxruntime 세션을 찾지 못해 spin/스레드 설정을 적용하지 못했습니다')
        return
    so = ort.SessionOptions()
    flag = '1' if spinning else '0'
    so.add_session_config_entry('session.intra_op.allow_spinning', flag)
    so.add_session_config_entry('session.inter_op.allow_spinning', flag)
    if threads > 0:
        so.intra_op_num_threads = threads
    backend.session = ort.InferenceSession(onnx_path, so, providers=session.get_providers())
    logger.info(f'onnxruntime 세션 교체: spin={"켬" if spinning else "끔"}, '
                f'스레드={threads if threads > 0 else "자동"}')


class StageTimer:
    """with 블록의 경과 시간(ms)과 프로세스 CPU 시간(ms)을 stage 이름으로 모은다."""

    def __init__(self):
        self.wall, self.cpu = {}, {}

    def reset(self):
        self.wall = {s: 0.0 for s in STAGES}
        self.cpu = {s: 0.0 for s in STAGES}

    def wrap(self, name, fn):
        def inner(*a, **kw):
            w0, c0 = time.perf_counter(), time.process_time()
            try:
                return fn(*a, **kw)
            finally:
                self.wall[name] += (time.perf_counter() - w0) * 1e3
                self.cpu[name] += (time.process_time() - c0) * 1e3
        return inner

    def run(self, name, fn, *a, **kw):
        return self.wrap(name, fn)(*a, **kw)


class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        self.declare_parameter('model', os.path.join(get_package_share_directory('lavacone_fusion'), 'models',
                                                     'cone_n_640_base.pt'))   # 패키지 동봉 모델
        self.declare_parameter('backend', 'onnx')            # onnx | openvino | pytorch
        self.declare_parameter('imgsz', [384, 640], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.7)
        self.declare_parameter('max_det', 100)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('publish_debug', False)
        self.declare_parameter('export_dir', os.path.expanduser('~/ros2_ws/models'))
        self.declare_parameter('timing_csv', '')
        self.declare_parameter('log_period', 5.0)
        # onnxruntime 대기 중 spin. 끄면 CPU 가 약 절반, 추론은 약 9 ms 느려진다
        self.declare_parameter('ort_spinning', False)
        self.declare_parameter('ort_threads', 0)             # 0 = 자동(코어 수)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.imgsz = normalize_imgsz(g('imgsz'))
        self.conf, self.iou, self.max_det = g('conf'), g('iou'), g('max_det')
        self.model, self.model_path = prepare_model(g('model'), g('backend'), self.imgsz, g('export_dir'),
                                                    self.get_logger())
        self.names = self.model.names if hasattr(self.model, 'names') else {0: 'cone'}
        # 첫 추론은 초기화 때문에 느리므로 미리 한 번 돌린다 (이때 predictor 가 만들어진다)
        warm = np.zeros((720, 1280, 3), np.uint8)
        self.model.predict(warm, imgsz=self.imgsz, verbose=False)
        if self.model_path.endswith('.onnx'):
            configure_onnxruntime(self.model, self.model_path, g('ort_spinning'), g('ort_threads'),
                                  self.get_logger())
            self.model.predict(warm, imgsz=self.imgsz, verbose=False)
        # predictor 의 단계 함수를 감싸 단계별 시간을 잰다
        self.timer = StageTimer()
        p = self.model.predictor
        p.preprocess = self.timer.wrap('pre', p.preprocess)
        p.inference = self.timer.wrap('inf', p.inference)
        p.postprocess = self.timer.wrap('post', p.postprocess)
        self.get_logger().info(f'모델 준비: {self.model_path} imgsz={self.imgsz} conf={self.conf}')

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Detection2DArray, '/cone/detections', 10)
        self.debug_pub = self.create_publisher(Image, '/cone/debug_image', 2)
        self.create_subscription(Image, g('image_topic'), self.on_image, LATEST_ONLY)
        self.add_on_set_parameters_callback(self.on_params)

        self.stats = deque(maxlen=100)
        self.last_log = time.monotonic()
        self.csv = None
        if g('timing_csv'):
            path = os.path.expanduser(g('timing_csv'))
            new = not os.path.exists(path)
            self.csv_file = open(path, 'a', newline='')
            self.csv = csv.writer(self.csv_file)
            if new:
                self.csv.writerow(['wall_time', 'model', 'imgsz', 'width', 'height', 'n_det',
                                   'preprocess_ms', 'inference_ms', 'postprocess_ms', 'callback_ms', 'latency_ms',
                                   'callback_cpu_ms']
                                  + [f'{s}_ms' for s in STAGES] + [f'{s}_cpu_ms' for s in STAGES])

    def on_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'conf':
                self.conf = float(p.value)
        return SetParametersResult(successful=True)

    def detections_msg(self, msg, xyxy, scores, classes):
        out = Detection2DArray()
        out.header = msg.header
        for i, ((x1, y1, x2, y2), s, c) in enumerate(zip(xyxy, scores, classes)):
            d = Detection2D()
            d.header = msg.header
            d.id = str(i)
            d.bbox.center.position.x = float((x1 + x2) / 2)
            d.bbox.center.position.y = float((y1 + y2) / 2)
            d.bbox.size_x = float(x2 - x1)
            d.bbox.size_y = float(y2 - y1)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = str(self.names.get(int(c), 'cone'))
            h.hypothesis.score = float(s)
            d.results.append(h)
            out.detections.append(d)
        self.pub.publish(out)
        return out

    def debug_image(self, msg, img, xyxy, scores):
        dbg = img.copy()
        for (x1, y1, x2, y2), s in zip(xyxy.astype(int), scores):
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(dbg, f'{s:.2f}', (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 165, 255), 1, cv2.LINE_AA)
        m = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        m.header = msg.header
        self.debug_pub.publish(m)

    def on_image(self, msg):
        t0, c0 = time.perf_counter(), time.process_time()
        tm = self.timer
        tm.reset()
        img = tm.run('convert', self.bridge.imgmsg_to_cv2, msg, 'bgr8')

        w1, c1 = time.perf_counter(), time.process_time()
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 max_det=self.max_det, verbose=False)[0]
        # predict 안에서 전/추론/후처리를 뺀 나머지 (입력 정리, Results 생성 등)
        tm.wall['predict_other'] = (time.perf_counter() - w1) * 1e3 - tm.wall['pre'] - tm.wall['inf'] - tm.wall['post']
        tm.cpu['predict_other'] = (time.process_time() - c1) * 1e3 - tm.cpu['pre'] - tm.cpu['inf'] - tm.cpu['post']

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        scores = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        classes = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)
        out = tm.run('publish', self.detections_msg, msg, xyxy, scores, classes)

        if self.get_parameter('publish_debug').value and self.debug_pub.get_subscription_count():
            tm.run('debug', self.debug_image, msg, img, xyxy, scores)

        cb_ms = (time.perf_counter() - t0) * 1e3
        cb_cpu = (time.process_time() - c0) * 1e3
        latency_ms = (self.get_clock().now().nanoseconds
                      - (msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)) * 1e-6
        sp = res.speed
        self.stats.append((time.monotonic(), sp['preprocess'], sp['inference'], sp['postprocess'], cb_ms,
                           latency_ms, cb_cpu))
        if self.csv:
            self.csv.writerow([f'{time.time():.3f}', os.path.basename(self.model_path), imgsz_tag(self.imgsz),
                               msg.width, msg.height, len(out.detections), f'{sp["preprocess"]:.2f}',
                               f'{sp["inference"]:.2f}', f'{sp["postprocess"]:.2f}', f'{cb_ms:.2f}',
                               f'{latency_ms:.2f}', f'{cb_cpu:.2f}']
                              + [f'{tm.wall[s]:.2f}' for s in STAGES] + [f'{tm.cpu[s]:.2f}' for s in STAGES])
        if time.monotonic() - self.last_log > self.get_parameter('log_period').value and len(self.stats) > 1:
            a = np.array(self.stats)
            hz = (len(a) - 1) / max(a[-1, 0] - a[0, 0], 1e-6)
            self.get_logger().info(
                f'{hz:.1f} Hz | 전처리 {a[:, 1].mean():.1f} 추론 {a[:, 2].mean():.1f} '
                f'후처리 {a[:, 3].mean():.1f} 콜백 {a[:, 4].mean():.1f} (CPU {a[:, 6].mean():.0f}) '
                f'지연 {a[:, 5].mean():.0f} ms | 검출 {len(out.detections)}개')
            self.last_log = time.monotonic()
            if self.csv:
                self.csv_file.flush()


def main():
    rclpy.init()
    node = ConeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.csv:
            node.csv_file.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

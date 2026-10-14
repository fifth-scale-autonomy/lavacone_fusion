# lavacone_fusion
#### YOLO 라바콘 검출과 2D 라이다 정합 — 카메라 박스와 라이다 군집을 짝지어 base_link 기준 콘 위치를 내고, 오도메트리로 콘을 추적·기억
#### Traffic-cone detection (YOLO) fused with a 2D lidar — matches camera boxes to lidar clusters, publishes cone positions in base_link and tracks them with odometry.

> **대회 목적**: 1/5 스케일 자율주행 차량 대회 (콘·흰 띠 트랙) 참가를 위해 개발한 패키지입니다.
> 구성: [lavacone_fusion](https://github.com/fifth-scale-autonomy/lavacone_fusion) (라바콘 정합),
> [band_track_detection](https://github.com/fifth-scale-autonomy/band_track_detection) (트랙 검출),
> [dtr_controller](https://github.com/fifth-scale-autonomy/dtr_controller) (경로·조향),
> [fifth_scale_bringup](https://github.com/fifth-scale-autonomy/fifth_scale_bringup) (통합 실행), 차량 드라이버 [henes_t870_driver](https://github.com/fifth-scale-autonomy/henes_t870_driver).

## 노드

| 실행 파일 | 입력 | 출력 |
|---|---|---|
| `cone_detector` | `/image_raw` | `/cone/detections` (Detection2DArray, 이미지 stamp 그대로, 원본 해상도 픽셀) |
| `cone_fusion` | `/cone/detections`, `/scan_filtered`, `/camera_info`, TF (odom 이 있으면 추적 이동 보정) | `/cone/positions` (Detection3DArray, base_link), `/cone/markers`, `/cone/fusion_image` |
| `lidar_projection` | `/scan_filtered`, `/image_raw`, TF | `/projection/image` (라이다 점을 영상에 투영). 조정 모드에서 `~/save` 로 extrinsics.yaml 에 기록 |
| `scan_probe` | `/scan` | 라이다 가림 각도·물체 위치 측정 (설치 확인용) |
| `extrinsic_check` | bag | YOLO 박스-라이다 짝으로 카메라 pitch / yaw 추정 |

`bench_models` (모델 추론 지연 측정) 은 `python -m lavacone_fusion.bench_models` 로 실행한다.
`common.py` (카메라 모델, 좌표 변환, 라이다 군집화) 는 `band_track_detection` 도 쓴다.

## 사용 조건과 채울 값

- **TF**: base_link -> laser, base_link -> camera_link -> camera_optical_frame. 통합 패키지 `fifth_scale_bringup` 이
  `config/extrinsics.yaml` 로 발행한다. 카메라 pitch 가 틀리면 박스와 라이다 군집이 짝지어지지 않는다
  (6.5° 설정에 실제 장착이 달랐을 때 라이다 투영이 약 46 px 위로 어긋나 정합률이 43% 로 떨어졌다).
- **`/camera_info`**: 카메라 내부 파라미터. 비어 있으면 공칭 대각 화각 (`fallback_dfov_deg` 87°) 임시 값을 쓴다.
- **`/scan_filtered`**: 차체에 가린 각도를 뺀 스캔 (`fifth_scale_bringup` 의 `laser_filter.yaml`).
- **콘 크기** (`cone_fusion` 매개변수): `cone_height` 0.70 m, `cone_radius_at_scan` (스캔 높이에서 콘 반지름, 0 이면 점 평균).
  군집 폭 범위 (`min_width` / `max_width`) 는 스캔 높이의 콘 지름에 맞춘다.
- **영상**: 1280x720 기준. 다른 해상도는 `imgsz` 를 맞춘다 (`profile:=960` 은 1920x1080).

## YOLO 모델 (동봉)

`models/` 에 기본 모델이 들어 있다. 설치하면 `share/lavacone_fusion/models` 로 복사되고 `cone_detector` 와
`bringup.launch.py` 의 기본 경로가 이곳이다.

| 파일 | 내용 |
|---|---|
| `cone_n_640_base.pt` | YOLO26n 을 FSOCO 로 학습한 1 라벨 모델 (`cone`, 학습 해상도 640) |
| `cone_n_640_base_imgsz384x640.onnx` | 위 모델의 ONNX 변환본 (입력 384x640, FP32, opset 18). `rect:=true` (기본) 에서 바로 쓴다 |
| `cone_n_640_base_args.yaml` | 학습 설정 (Ultralytics args.yaml, 학습 PC 경로만 파일 이름으로 줄임) |
| `sha256sums.txt` | 체크섬 (`sha256sum -c sha256sums.txt`) |

- **라바콘 색을 구분하지 않는 1 라벨 모델이다.** FSOCO 의 콘 클래스 5 개 (파랑, 노랑, 작은 주황, 큰 주황, 규정 외 콘) 를
  하나의 `cone` 라벨로 합쳐 콘의 형태만 학습했다. 출력 클래스는 `cone` 하나이고 색 정보는 내지 않는다.
  그래서 트랙 좌우 구분은 콘 색이 아니라 위치 (`dtr_controller` 의 좌우 경계 열) 로 한다.
- 학습: Ultralytics `yolo26n.pt` (v8.4.0 릴리스) 에서 시작, 100 에폭 (최고 46 에폭), seed 0, 나머지 기본값.
  데이터는 FSOCO 바운딩 박스 데이터셋을 단일 클래스로 바꾸고 워터마크 테두리를 지운 것. 출처 (팀) 단위로 나눠
  train 9,252 / val 1,166 / test 1,151 장.
- test 성능: P 0.884, R 0.727, mAP50 0.801, mAP50-95 0.475. 같은 조건의 YOLO26s (20 MB, 동봉 안 함) 는 mAP50 0.836.
- FSOCO 는 일반 차량 높이 카메라 시점이다. 1/5 차량의 낮은 시점 성능은 따로 재지 않았다.
  콘이 없는 실내에서 갈색 나무 책상 다리를 conf 약 0.5 로 콘으로 검출한 적이 있다.
- 다른 모델: `cone_{n_}{profile}_{variant}.pt` 이름으로 한 폴더에 두고 `model_dir:=<폴더>`
  (`size:=s`, `profile:=960`, `variant:=aug`), 또는 `model:=<경로>`. `.pt` 옆에 `<이름>_imgsz<크기>.onnx` 가 없으면
  처음 한 번 변환해 `export_dir` (기본 `~/ros2_ws/models`) 에 둔다 (venv 에 onnx, onnxslim 필요).
### 출처와 라이선스

| 구성 요소 | 출처 | 라이선스 / 조건 |
|---|---|---|
| 학습·추론 프레임워크 | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 (또는 Ultralytics Enterprise) |
| 기반 모델 | Ultralytics YOLO26n (`yolo26n.pt`, v8.4.0 릴리스) | AGPL-3.0 |
| 동봉 모델 (`models/`) | 위 기반 모델을 FSOCO 로 미세 조정 (`cone_n_640_base_args.yaml` 이 학습 설정) | AGPL-3.0 (모델 파일 안에도 표기) |
| 학습 데이터 | [FSOCO dataset](https://github.com/fsoco/fsoco-dataset) (Formula Student Objects in Context) | 아래 |

- Ultralytics 는 학습된 모델도 기본적으로 AGPL-3.0 을 따른다고 명시한다 (https://www.ultralytics.com/license).
  이 패키지와 동봉 모델은 그에 맞춰 AGPL-3.0 으로 공개한다.
- **FSOCO**: 출처 https://github.com/fsoco/fsoco-dataset. 데이터 (영상·라벨) 는 이 레포에 포함하지 않는다.
  FSOCO 는 Formula Student Driverless 관련 학술·대회 목적의 사용을 전제로 공개된 데이터셋이며, 이 모델도
  대회 참가 목적으로 학습했다. 사용할 때 다음 논문을 인용한다.

  Niclas Vödisch, David Dodel, Michael Schötz, "FSOCO: The Formula Student Objects in Context Dataset",
  SAE International Journal of Connected and Automated Vehicles, vol. 5, no. 12-05-01-0003, 2022.
  [arXiv:2012.07139](https://arxiv.org/abs/2012.07139)

  ```bibtex
  @article{fsoco_2022,
    title={FSOCO: The Formula Student Objects in Context Dataset},
    author={V{\"o}disch, Niclas and Dodel, David and Sch{\"o}tz, Michael},
    journal={SAE International Journal of Connected and Automated Vehicles},
    volume={5},
    number={12-05-01-0003},
    year={2022}
  }
  ```

## 실행 환경

- `cone_detector` 는 ultralytics, onnxruntime (OpenVINO 를 쓰면 openvino) 가 설치된 **venv 파이썬**으로 실행한다.
  `fifth_scale_bringup` 의 `bringup.launch.py` 가 `venv:=` (기본 `~/ros2_ws/venv`) 로 실행해 준다.
- `backend:=onnx` (기본) / `openvino` / `pytorch`. 입력 크기는 `rect:=true` (기본) 면 16:9 입력 (640 -> 384x640).
  정사각형 640x640 은 입력의 44% 가 여백이다.
- 측정 (AMD Ryzen 3 4300U, GPU 없음, 1280x720 10 fps): nano + ONNX 로 카메라 프레임을 거의 다 처리한다
  (전체 파이프라인 9.7 Hz, 영상 도착부터 검출 발행까지 중앙 132 ms, 640x640 기준). 검출 노드 혼자 약 3.4 코어를
  쓰므로 `detector_cpus:=0-2` 처럼 코어를 묶으면 다른 노드가 덜 밀린다.
- 영상 구독은 RELIABLE + 최신 1 장 (`common.LATEST_ONLY`). 1.8 MB 영상을 BEST_EFFORT 로 받으면 조각 하나만 빠져도
  프레임이 버려져 수신이 거의 0 Hz 가 된다.

## 정합 방식 (`cone_fusion`)

1. 검출 시각과 가장 가까운 스캔을 고른다 (`max_dt` 0.12 s 넘으면 버림).
2. 스캔을 각도 순서로 군집화하고 폭이 콘 크기인 군집만 남긴다.
3. 군집 중심을 카메라로 투영해 박스 안에 든 것만 후보로, 가로 위치와 거리 (박스 아랫변을 지면에 투영한 거리 vs
   라이다 거리) 로 비용을 매겨 헝가리안으로 1:1 배정한다.
4. 짝지어진 콘은 라이다 위치를 낸다 (id `fused:<n>`).
5. 카메라로 2 번 이상 확인된 군집은 라이다로 계속 추적하고, TF odom -> base_link 가 있으면 차량 이동만큼 옮겨
   안 보여도 `track_memory` (기본 3 s) 동안 유지한다.

## 카메라-라이다 위치 맞추기

센서 위치는 통합 패키지 `fifth_scale_bringup` 의 `config/extrinsics.yaml` 에 있고, 이 패키지는 실행할 때 그 파일을
찾는다 (없으면 `--extrinsics` / `extrinsics_file` 로 준다).

```bash
ros2 launch fifth_scale_bringup bringup.launch.py tune_extrinsics:=true   # 투영 영상을 보며 조정, ~/save 로 기록
ros2 run lavacone_fusion extrinsic_check ~/bags/<bag>                     # 주행 bag 으로 pitch / yaw 추정
ros2 run lavacone_fusion extrinsic_check ~/bags/<bag> --apply             # 추정값을 extrinsics.yaml 에 기록
ros2 run lavacone_fusion scan_probe --mode occlusion --topic /scan        # 라이다 가림 각도 (laser_filter.yaml 용)
```

`extrinsic_check` 는 bag 마다, 5 s 구간마다 pitch 를 따로 낸다. 구간끼리는 같은데 bag 끼리 다르면 카메라 고정이
흔들린 것이다.

## 시험

```bash
cd ~/ros2_ws/src/lavacone_fusion && python3 -m pytest test -q
```

## 라이선스

GNU Affero General Public License v3.0 (`AGPL-3.0-only`, `LICENSE`). 검출에 Ultralytics (AGPL-3.0) 를 쓰고
YOLO26 에서 학습한 모델을 동봉하므로 Ultralytics 라이선스에 맞춰 AGPL-3.0 으로 공개한다.
동봉 모델의 출처와 학습 데이터 조건은 위 "출처와 라이선스".

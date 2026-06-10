# 기말 프로젝트 결과 보고서

## 1. 프로젝트 목표

본 프로젝트의 목표는 손글씨 숫자 동영상을 입력으로 받아 YOLO 기반 객체 검출을 수행하고, 숫자가 검출된 시점의 영역을 MNIST 입력 형식과 동일한 28x28 PGM 파일로 저장한 뒤, MNIST 숫자 인식 예제의 정확도를 개선하는 것이다. 기존 예제에 포함된 숫자 1, 3, 5의 인식 성능은 유지하면서 새로 수집한 손글씨 6, 8 샘플의 인식 성공률을 측정하였다. 수행시간은 프로그램 출력으로 확인되는 전체 시간 기준이며, 영상 입출력, YOLO 추론, crop 전처리, PGM 저장, MNIST 추론 시간을 모두 포함하도록 구성하였다.

## 2. 문제 해결 접근법

전체 처리는 세 단계로 구성하였다. 첫 번째 단계에서는 손글씨 영상에서 숫자 영역을 안정적으로 찾기 위해 YOLO 모델을 학습한다. 학습 데이터는 영상에서 추출한 프레임 또는 별도 촬영 이미지에 대해 숫자 영역 bounding box를 라벨링하고, YOLO 형식의 `images/train`, `labels/train`, `images/val`, `labels/val` 구조로 배치한다. 숫자 종류 자체는 MNIST 분류기가 담당하므로 YOLO는 숫자 객체 존재 여부와 위치 검출에 집중하도록 단일 클래스 `digit`로 구성하였다.

두 번째 단계에서는 학습된 YOLO 모델에 손글씨 동영상을 입력한다. 각 프레임에 대해 숫자 bounding box를 검출하고, 검출 영역에 padding을 추가한 뒤 grayscale 변환, 배경/전경 반전, Otsu 이진화, digit 중심 정렬을 수행한다. 최종 이미지는 MNIST 입력과 동일한 28x28 크기의 PGM 파일로 저장한다. 이 과정에서 전체 수행시간을 `time.perf_counter()`로 측정하여 영상 읽기, 모델 추론, 전처리, 파일 저장 시간을 모두 포함하였다.

세 번째 단계에서는 MNIST 분류기를 재학습한다. 기본 입력 크기와 추론 순서는 유지하되, CNN 내부의 convolution channel 수, dropout, AdamW optimizer, affine augmentation을 적용하여 일반화 성능을 높였다. 학습 대상은 1, 3, 5, 6, 8로 제한하고, MNIST 기본 데이터에 새로 수집한 6, 8 PGM/이미지를 추가로 결합하였다. 평가 시에는 전체 정확도뿐 아니라 숫자별 정확도와 평균 latency를 함께 출력하도록 구성하였다.

## 3. 코드 수정 및 구현 내용

YOLO 학습 코드는 `src/yolo/train_yolo.py`에 구현하였다. `ultralytics` 패키지의 YOLO 인터페이스를 사용하며, 학습 종료 후 가장 성능이 좋은 `best.pt`를 `models/yolo_digits.pt`로 복사한다. 출력에는 전체 학습 수행시간이 포함된다.

영상에서 PGM을 생성하는 코드는 `src/yolo/video_to_pgm.py`에 구현하였다. 입력 동영상의 각 프레임을 읽고 YOLO 추론을 수행한 뒤, confidence threshold를 통과한 detection만 저장한다. 저장된 파일명은 `frame_000000_det_00.pgm` 형식이며, detection 정보는 `results/video_detections.csv`에 함께 기록된다. 이 CSV에는 프레임 번호, timestamp, confidence, bounding box 좌표, PGM 경로가 포함된다.

MNIST 모델은 `src/mnist/model.py`의 `DigitCNN`으로 분리하였다. 학습 및 MNIST test set 평가는 `src/mnist/train_mnist_13568.py`에서 수행한다. 새 손글씨 데이터는 `data/custom_digits/6`, `data/custom_digits/8`에 넣으면 자동으로 학습 데이터에 포함된다. PGM 평가 코드는 `src/mnist/eval_pgm.py`에 구현하였고, 라벨별 디렉터리 구조가 있는 경우 digit별 정확도를 계산한다.

## 4. 실험 방법

YOLO 학습은 다음 명령으로 수행한다.

```bash
python src/yolo/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --out models/yolo_digits.pt
```

동영상에서 PGM 파일을 생성할 때는 다음 명령을 사용한다.

```bash
python src/yolo/video_to_pgm.py --weights models/yolo_digits.pt --video data/videos/test.mp4 --out-dir data/pgm --conf 0.35
```

MNIST와 새 손글씨 샘플을 결합한 학습 및 평가는 다음 명령으로 수행한다.

```bash
python src/mnist/train_mnist_13568.py --custom-root data/custom_digits --epochs 8 --model-out models/mnist_13568.pt --metrics-out results/mnist_metrics.json
```

YOLO로 생성된 PGM 파일에 대한 최종 평가는 다음 명령으로 수행한다.

```bash
python src/mnist/eval_pgm.py --weights models/mnist_13568.pt --pgm-root data/pgm --metrics-out results/pgm_eval.json
```

## 5. 결과

현재 저장소에는 실제 촬영 영상, YOLO 라벨, 새 손글씨 6/8 샘플이 아직 포함되어 있지 않으므로 아래 표는 실험 후 채워야 한다.

| 항목 | 결과 |
|---|---:|
| YOLO 영상 처리 전체 시간 | TBD sec |
| 저장된 PGM 수 | TBD |
| MNIST 학습 및 평가 전체 시간 | TBD sec |
| PGM 최종 평가 전체 시간 | TBD sec |
| 평균 추론 latency | TBD ms/sample |

| 숫자 | 정확도 | 정답 수 / 전체 수 |
|---:|---:|---:|
| 1 | TBD | TBD |
| 3 | TBD | TBD |
| 5 | TBD | TBD |
| 6 | TBD | TBD |
| 8 | TBD | TBD |

## 6. 결론

구현은 YOLO 검출과 MNIST 분류를 분리하여 구성하였다. YOLO는 동영상에서 숫자 위치를 찾는 역할만 수행하고, MNIST 모델은 28x28 PGM으로 정규화된 숫자 이미지를 분류한다. 이 구조는 기존 MNIST 예제의 입력 형식을 유지하면서 새로 수집한 손글씨 6, 8 샘플을 학습에 추가할 수 있다는 장점이 있다. 최종 제출 전에는 실제 80개 손글씨 파일 기반 영상으로 YOLO 라벨링 및 학습을 완료하고, `results/mnist_metrics.json`, `results/pgm_eval.json`의 숫자별 정확도와 latency 값을 본 보고서의 결과 표에 반영해야 한다.


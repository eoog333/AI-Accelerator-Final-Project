# AI 가속기 기말 프로젝트

손글씨 동영상에서 숫자를 YOLO로 검출하고, 검출된 숫자 영역을 28x28 PGM으로 저장한 뒤 MNIST 분류기의 1, 3, 5 성능을 유지하면서 새 손글씨 6, 8 인식률을 측정하는 프로젝트입니다.

## 폴더 구조

```text
data/
  videos/              # 입력 손글씨 동영상
  yolo/images/         # YOLO 학습 이미지
  yolo/labels/         # YOLO txt 라벨
  pgm/                 # 검출된 28x28 PGM 출력
  custom_digits/6/     # 수집한 6 손글씨 샘플
  custom_digits/8/     # 수집한 8 손글씨 샘플
models/                # 학습된 YOLO/MNIST 가중치
results/               # csv/json 결과 파일
src/yolo/              # YOLO 학습 및 동영상-to-PGM 코드
src/mnist/             # MNIST 학습/평가 코드
report/                # 보고서 초안
```

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 연구실 PC 실행 방향

최종 MNIST 재학습과 YOLO 자체 학습이 필요할 경우의 기준 환경은 연구실 PC의 CUDA GPU입니다.
현재 이 저장소를 편집하는 로컬 환경은 연구실 PC가 아니므로, 여기서 생성한 시간/성능 수치는 최종 실험값으로 사용하지 않습니다.

연구실 PC에서 저장소를 받은 뒤 환경을 준비합니다.

```bash
git clone https://github.com/eoog333/AI-Accelerator-Final-Project.git
cd AI-Accelerator-Final-Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

먼저 CUDA와 필요한 파일 상태를 확인합니다.

```bash
scripts/run_lab_pc.sh check
```

MNIST + 커스텀 6/8 모델을 연구실 PC에서 학습합니다.

```bash
scripts/run_lab_pc.sh train-mnist
```

기본값은 `DEVICE=cuda`, `EPOCHS=10`, `CUSTOM_REPEAT=20`, `BATCH_SIZE=256`, `NUM_WORKERS=4`, `AMP=1`입니다.
출력 모델과 결과 파일은 Colab/로컬 산출물과 섞이지 않도록 아래 경로로 저장됩니다.

```text
models/mnist_13568_lab.pt
results/mnist_metrics_lab.json
```

GPU 메모리가 부족하면 배치 크기를 줄입니다.

```bash
BATCH_SIZE=128 scripts/run_lab_pc.sh train-mnist
```

MNIST 원본 데이터 다운로드가 막혀 있으면 다른 PC에서 받은 IDX gzip 파일 4개를 `data/mnist/raw/`에 복사한 뒤 같은 명령을 다시 실행합니다.

이미 만들어진 PGM 파일만 평가하려면:

```bash
PGM_ROOT=data/pgm MAX_IMAGES=15 scripts/run_lab_pc.sh eval-pgm
```

영상에서 YOLO로 PGM을 만들고 이어서 MNIST 평가까지 실행하려면:

```bash
YOLO_WEIGHTS=models/yolo_digits.pt \
MNIST_WEIGHTS=models/mnist_13568_lab.pt \
VIDEO=data/videos/test.mp4 \
MAX_EVENTS=15 \
MAX_IMAGES=15 \
scripts/run_lab_pc.sh video-pipeline
```

YOLO 공개 가중치가 실제 영상에서 숫자를 잘 검출하지 못할 때만 연구실 PC에서 자체 YOLO 학습을 실행합니다.

```bash
scripts/run_lab_pc.sh yolo-train
```

## 로컬/개인 PC 실행

Colab 외에 로컬 PC, 연구실 서버, 다른 CUDA GPU 머신, Jetson Nano에서도 같은 코드로 실행할 수 있습니다.
먼저 환경과 필요한 파일이 있는지 확인하세요.

```bash
scripts/run_local.sh check
```

로컬에서 MNIST 모델을 다시 학습하려면:

```bash
scripts/run_local.sh train-mnist
```

기본 설정은 `EPOCHS=10`, `CUSTOM_REPEAT=20`, 출력 모델은 `models/mnist_13568_local.pt`입니다.
다른 장치를 강제로 쓰려면 `DEVICE`를 지정합니다.

```bash
DEVICE=cuda scripts/run_local.sh train-mnist
DEVICE=mps scripts/run_local.sh train-mnist
DEVICE=cpu scripts/run_local.sh train-mnist
```

이미 만들어진 PGM 파일만 평가하려면:

```bash
MNIST_WEIGHTS=models/mnist_13568_local.pt \
PGM_ROOT=data/pgm \
MAX_IMAGES=15 \
scripts/run_local.sh eval-pgm
```

영상에서 YOLO로 PGM을 만들고 이어서 MNIST 평가까지 실행하려면:

```bash
YOLO_WEIGHTS=models/yolo_digits.pt \
MNIST_WEIGHTS=models/mnist_13568_local.pt \
VIDEO=data/videos/test.mp4 \
MAX_EVENTS=15 \
MAX_IMAGES=15 \
scripts/run_local.sh video-pipeline
```

주의: `models/*.pt`, `results/*.json`, `data/pgm/*.pgm`은 생성 산출물이므로 기본적으로 GitHub에 올라가지 않습니다. 연구실 PC에서 만든 최종 모델은 Jetson이나 발표용 PC로 별도 복사해야 합니다.

## 구글 Colab

[Colab에서 열기](https://colab.research.google.com/github/eoog333/AI-Accelerator-Final-Project/blob/main/notebooks/colab_mnist_13568.ipynb)

위 링크로 열면 노트북의 `REPO_URL`이 `https://github.com/eoog333/AI-Accelerator-Final-Project.git`로 설정되어 있어
프로젝트를 `/content/AI_accelerator_final_project`에 clone해서 실행합니다.
Colab 메뉴에서 `Runtime > Change runtime type > GPU`를 선택한 뒤 셀을 순서대로 실행하세요.
처음 학습 셀은 `--download` 옵션으로 MNIST 원본 파일을 `data/mnist/raw`에 내려받습니다.

## 1. YOLO 모델

YOLO 단계는 먼저 공개된 손글씨 숫자 검출 모델이나 숫자 검출 모델을 사용하는 것을 우선합니다.
공개 가중치가 손글씨 숫자를 잘 검출하면 해당 파일을 아래 경로로 저장하세요.

```text
models/yolo_digits.pt
```

검색 키워드 예시는 아래와 같습니다.

```text
handwritten digit detection YOLO weights
digit detection YOLO pretrained
MNIST YOLO detector
Roboflow handwritten digit detection YOLO
```

공개 가중치를 구한 뒤 바로 `2. 동영상에서 28x28 PGM 생성` 단계를 실행합니다.
실제 손글씨 영상에서 `events_saved`가 영상 속 숫자 개수와 맞고 PGM crop 품질이 좋으면 별도 YOLO 학습은 필요 없습니다.

공개 모델이 우리 영상에서 숫자를 잘 못 잡는 경우에만 자체 학습으로 전환합니다.
자체 학습을 할 때는 라벨링된 이미지를 `data/yolo/images/{train,val}`에, YOLO txt 라벨을 `data/yolo/labels/{train,val}`에 넣습니다.

```bash
python src/yolo/train_yolo.py \
  --data data/yolo/dataset.yaml \
  --model yolo11n.pt \
  --epochs 80 \
  --imgsz 640 \
  --out models/yolo_digits.pt
```

## 2. 동영상에서 28x28 PGM 생성

```bash
python src/yolo/video_to_pgm.py \
  --weights models/yolo_digits.pt \
  --video data/videos/test.mp4 \
  --out-dir data/pgm \
  --clean-out-dir \
  --max-events 15 \
  --event-gap-frames 8 \
  --conf 0.35 \
  --sample-stride 1
```

이 스크립트는 인접 프레임의 검출 결과를 하나의 등장 이벤트로 묶고, 손글씨가 등장한 횟수마다 28x28 PGM을 1개만 저장합니다.
예를 들어 어떤 숫자가 2초 동안 보이면 프레임마다 저장하지 않고 한 번만 저장합니다. 저장되는 이미지는 해당 등장 구간에서 confidence가 가장 높은 검출 결과입니다.

## 3. MNIST + 커스텀 6/8 학습

최종 학습은 연구실 PC에서 아래 래퍼 스크립트를 사용한다.

```bash
scripts/run_lab_pc.sh train-mnist
```

동일한 작업을 직접 실행하면 다음과 같다.

```bash
python src/mnist/train_mnist_13568.py \
  --download \
  --custom-root data/custom_digits \
  --custom-repeat 20 \
  --epochs 10 \
  --batch-size 256 \
  --num-workers 4 \
  --device cuda \
  --amp \
  --model-out models/mnist_13568_lab.pt \
  --metrics-out results/mnist_metrics_lab.json
```

## 4. PGM 인식 평가

```bash
python src/mnist/eval_pgm.py \
  --weights models/mnist_13568_lab.pt \
  --pgm-root data/pgm \
  --max-images 15 \
  --device cuda \
  --metrics-out results/pgm_eval_lab.json
```

`data/pgm` 아래에 `data/pgm/6/*.pgm`, `data/pgm/8/*.pgm`처럼 라벨 폴더가 있으면 숫자별 정확도도 계산됩니다.
프로젝트 동영상 평가에서는 `--max-images`를 영상에 등장한 손글씨 개수인 10~15개로 맞추세요. 평가 스크립트는 입력 PGM, 예측 숫자, 전체 이미지 수, MNIST만의 추론 시간, 0~9 숫자별 예측 개수를 출력합니다.

# AI Accelerator Final Project

손글씨 동영상에서 숫자를 YOLO로 검출하고, 검출된 숫자 영역을 28x28 PGM으로 저장한 뒤 MNIST 분류기의 1, 3, 5 성능을 유지하면서 새 손글씨 6, 8 인식률을 측정하는 프로젝트입니다.

## Directory

```text
data/
  videos/              # input handwriting videos
  yolo/images/         # YOLO training images
  yolo/labels/         # YOLO txt labels
  pgm/                 # detected 28x28 pgm output
  custom_digits/6/     # collected 6 samples
  custom_digits/8/     # collected 8 samples
models/                # trained YOLO and MNIST weights
results/               # csv/json metrics
src/yolo/              # YOLO training and video-to-PGM code
src/mnist/             # MNIST training/evaluation code
report/                # report draft
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Execution

Colab 외에 로컬 PC, 연구실 서버, 다른 CUDA GPU 머신, Jetson Nano에서도 같은 코드로 실행할 수 있습니다.
먼저 환경과 필요한 파일이 있는지 확인하세요.

```bash
scripts/run_local.sh check
```

로컬에서 MNIST 모델을 다시 학습하려면:

```bash
scripts/run_local.sh train-mnist
```

기본 설정은 `EPOCHS=10`, `CUSTOM_REPEAT=20`, 출력 모델은 `models/mnist_13568_colab.pt`입니다.
다른 장치를 강제로 쓰려면 `DEVICE`를 지정합니다.

```bash
DEVICE=cuda scripts/run_local.sh train-mnist
DEVICE=mps scripts/run_local.sh train-mnist
DEVICE=cpu scripts/run_local.sh train-mnist
```

이미 만들어진 PGM 파일만 평가하려면:

```bash
MNIST_WEIGHTS=models/mnist_13568_colab.pt \
PGM_ROOT=data/pgm \
MAX_IMAGES=15 \
scripts/run_local.sh eval-pgm
```

영상에서 YOLO로 PGM을 만들고 이어서 MNIST 평가까지 실행하려면:

```bash
YOLO_WEIGHTS=models/yolo_digits.pt \
MNIST_WEIGHTS=models/mnist_13568_colab.pt \
VIDEO=data/videos/test.mp4 \
MAX_EVENTS=15 \
MAX_IMAGES=15 \
scripts/run_local.sh video-pipeline
```

주의: `models/*.pt`, `results/*.json`, `data/pgm/*.pgm`은 생성 산출물이므로 기본적으로 GitHub에 올라가지 않습니다. Colab에서 만든 최종 모델은 로컬이나 Jetson으로 별도 복사해야 합니다.

## Google Colab

[Open in Colab](https://colab.research.google.com/github/eoog333/AI-Accelerator-Final-Project/blob/main/notebooks/colab_mnist_13568.ipynb)

위 링크로 열면 노트북의 `REPO_URL`이 `https://github.com/eoog333/AI-Accelerator-Final-Project.git`로 설정되어 있어
프로젝트를 `/content/AI_accelerator_final_project`에 clone해서 실행합니다.
Colab 메뉴에서 `Runtime > Change runtime type > GPU`를 선택한 뒤 셀을 순서대로 실행하세요.
처음 학습 셀은 `--download` 옵션으로 MNIST 원본 파일을 `data/mnist/raw`에 내려받습니다.

## 1. YOLO training

Put labeled images in `data/yolo/images/{train,val}` and labels in `data/yolo/labels/{train,val}`.

```bash
python src/yolo/train_yolo.py \
  --data data/yolo/dataset.yaml \
  --model yolo11n.pt \
  --epochs 80 \
  --imgsz 640 \
  --out models/yolo_digits.pt
```

## 2. Video to 28x28 PGM

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

The script groups detections across adjacent frames and saves one 28x28 PGM for each handwriting appearance.
For example, a digit visible for 2 seconds is saved once, not once per frame. The saved frame is the highest-confidence detection inside that appearance event.

## 3. MNIST + custom 6/8 training

```bash
python src/mnist/train_mnist_13568.py \
  --custom-root data/custom_digits \
  --custom-repeat 20 \
  --epochs 8 \
  --model-out models/mnist_13568.pt \
  --metrics-out results/mnist_metrics.json
```

## 4. Evaluate PGM recognition

```bash
python src/mnist/eval_pgm.py \
  --weights models/mnist_13568.pt \
  --pgm-root data/pgm \
  --max-images 15 \
  --metrics-out results/pgm_eval.json
```

If `data/pgm` has label subdirectories such as `data/pgm/6/*.pgm` and `data/pgm/8/*.pgm`, the evaluator reports per-digit accuracy.
For project videos, set `--max-images` to the number of handwritten digits in the video, such as 10 to 15. The evaluator prints each input PGM, the predicted digit, total image count, MNIST-only inference time, and prediction counts for digits 0 through 9.

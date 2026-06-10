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
  --conf 0.35 \
  --sample-stride 1
```

The script prints total elapsed time including video I/O, YOLO inference, crop preprocessing, and PGM writes.

## 3. MNIST + custom 6/8 training

```bash
python src/mnist/train_mnist_13568.py \
  --custom-root data/custom_digits \
  --epochs 8 \
  --model-out models/mnist_13568.pt \
  --metrics-out results/mnist_metrics.json
```

## 4. Evaluate PGM recognition

```bash
python src/mnist/eval_pgm.py \
  --weights models/mnist_13568.pt \
  --pgm-root data/pgm \
  --metrics-out results/pgm_eval.json
```

If `data/pgm` has label subdirectories such as `data/pgm/6/*.pgm` and `data/pgm/8/*.pgm`, the evaluator reports per-digit accuracy.

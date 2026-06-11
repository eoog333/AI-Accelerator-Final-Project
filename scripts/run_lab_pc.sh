#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_lab_pc.sh check
  scripts/run_lab_pc.sh train-mnist
  scripts/run_lab_pc.sh eval-pgm
  scripts/run_lab_pc.sh video-to-pgm
  scripts/run_lab_pc.sh yolo-train
  scripts/run_lab_pc.sh video-pipeline

Defaults are for the lab PC CUDA environment. Override with environment variables if needed.

Common variables:
  PYTHON=python3
  DEVICE=cuda
  DATA_DIR=data/mnist
  CUSTOM_ROOT=data/custom_digits
  PGM_ROOT=data/pgm
  MNIST_WEIGHTS=models/mnist_13568_lab.pt
  YOLO_WEIGHTS=models/yolo_digits.pt
  VIDEO=data/videos/test.mp4

MNIST training variables:
  EPOCHS=10
  CUSTOM_REPEAT=20
  BATCH_SIZE=256
  NUM_WORKERS=4
  LR=0.001
  AMP=1

YOLO variables:
  YOLO_BASE_MODEL=yolo11n.pt
  YOLO_EPOCHS=80
  YOLO_BATCH=16
  IMGSZ=640
  CONF=0.35
  MAX_EVENTS=15
  MAX_IMAGES=15
  EVENT_GAP_FRAMES=8
EOF
}

device_args=()
if [[ -n "$DEVICE" ]]; then
  device_args=(--device "$DEVICE")
fi

cmd="${1:-}"
case "$cmd" in
  check)
    "$PYTHON" scripts/check_environment.py
    ;;

  train-mnist)
    train_args=(
      --download
      --data-dir "${DATA_DIR:-data/mnist}"
      --custom-root "${CUSTOM_ROOT:-data/custom_digits}"
      --custom-repeat "${CUSTOM_REPEAT:-20}"
      --epochs "${EPOCHS:-10}"
      --batch-size "${BATCH_SIZE:-256}"
      --num-workers "${NUM_WORKERS:-4}"
      --lr "${LR:-0.001}"
      --model-out "${MNIST_WEIGHTS:-models/mnist_13568_lab.pt}"
      --metrics-out "${MNIST_METRICS:-results/mnist_metrics_lab.json}"
    )
    if [[ "${AMP:-1}" == "1" ]]; then
      train_args+=(--amp)
    fi
    "$PYTHON" src/mnist/train_mnist_13568.py "${train_args[@]}" "${device_args[@]}"
    ;;

  eval-pgm)
    eval_args=(
      --weights "${MNIST_WEIGHTS:-models/mnist_13568_lab.pt}"
      --pgm-root "${PGM_ROOT:-data/pgm}"
      --metrics-out "${PGM_METRICS:-results/pgm_eval_lab.json}"
    )
    if [[ "${MAX_IMAGES:-15}" -gt 0 ]]; then
      eval_args+=(--max-images "${MAX_IMAGES:-15}")
    fi
    "$PYTHON" src/mnist/eval_pgm.py "${eval_args[@]}" "${device_args[@]}"
    ;;

  video-to-pgm)
    yolo_args=(
      --weights "${YOLO_WEIGHTS:-models/yolo_digits.pt}"
      --video "${VIDEO:-data/videos/test.mp4}"
      --out-dir "${PGM_ROOT:-data/pgm}"
      --csv-out "${YOLO_CSV:-results/video_detections_lab.csv}"
      --clean-out-dir
      --event-gap-frames "${EVENT_GAP_FRAMES:-8}"
      --conf "${CONF:-0.35}"
      --imgsz "${IMGSZ:-640}"
    )
    if [[ "${MAX_EVENTS:-15}" -gt 0 ]]; then
      yolo_args+=(--max-events "${MAX_EVENTS:-15}")
    fi
    "$PYTHON" src/yolo/video_to_pgm.py "${yolo_args[@]}" "${device_args[@]}"
    ;;

  yolo-train)
    "$PYTHON" src/yolo/train_yolo.py \
      --data "${YOLO_DATA:-data/yolo/dataset.yaml}" \
      --model "${YOLO_BASE_MODEL:-yolo11n.pt}" \
      --epochs "${YOLO_EPOCHS:-80}" \
      --batch "${YOLO_BATCH:-16}" \
      --imgsz "${IMGSZ:-640}" \
      --out "${YOLO_WEIGHTS:-models/yolo_digits.pt}" \
      "${device_args[@]}"
    ;;

  video-pipeline)
    "$0" video-to-pgm
    "$0" eval-pgm
    ;;

  -h|--help|help|"")
    usage
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac

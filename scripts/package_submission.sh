#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-submission_final_project.zip}"

zip -r "$OUT" \
  README.md \
  requirements.txt \
  report \
  src \
  scripts \
  data/yolo/dataset.yaml \
  results \
  models \
  -x "**/__pycache__/*" "*.DS_Store"

echo "created=$OUT"


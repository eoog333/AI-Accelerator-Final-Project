# Project Handoff

이 문서는 다음 작업자가 현재 프로젝트 상태를 빠르게 이어받기 위한 인수인계 문서입니다.

## 1. 프로젝트 목표

손글씨가 촬영된 동영상을 YOLO로 감지하고, 감지된 숫자 영역을 28x28 PGM 파일로 저장한 뒤, MNIST 기반 분류기로 숫자를 인식한다.

요구사항 핵심:

- 손글씨 동영상에는 무작위 손글씨 숫자 10~15개가 등장한다.
- YOLO는 숫자 영역을 감지하고 PGM 파일 저장까지 한 번에 수행해야 한다.
- 같은 숫자가 여러 프레임 동안 계속 보이더라도, 해당 등장 구간에서는 PGM 파일을 1개만 저장해야 한다.
- MNIST 분류기는 기존 1, 3, 5 성능을 유지하면서 새 손글씨 6, 8도 인식해야 한다.
- 최종 출력은 입력 PGM, 분류 결과, 총 이미지 수, MNIST 수행시간, 숫자별 예측 개수를 포함해야 한다.

## 2. 현재 완료된 단계

완료:

- GitHub 저장소 연동
- Colab GPU 실행 노트북 작성
- MNIST + custom 6/8 학습 완료
- PGM 평가 완료
- 정확도 개선용 `--custom-repeat` 옵션 추가
- YOLO 결과를 프레임별 저장이 아니라 등장 이벤트별 저장으로 수정
- MNIST PGM 평가 출력 형식을 과제 예시 형식으로 수정
- Jetson Nano에서 실행할 수 있는 코드 구조 정리

GitHub 저장소:

```text
https://github.com/eoog333/AI-Accelerator-Final-Project
```

Colab 노트북:

```text
notebooks/colab_mnist_13568.ipynb
```

Colab 바로 열기:

```text
https://colab.research.google.com/github/eoog333/AI-Accelerator-Final-Project/blob/main/notebooks/colab_mnist_13568.ipynb
```

## 3. 주요 파일 설명

```text
README.md
```

기본 프로젝트 설명과 실행 명령어가 들어 있다.

```text
HANDOFF.md
```

현재 문서. 다음 작업자가 볼 인수인계 문서이다.

```text
requirements.txt
```

Python 의존성 목록이다.

```text
notebooks/colab_mnist_13568.ipynb
```

Colab GPU에서 MNIST 학습, PGM 평가를 실행하는 노트북이다. 첫 셀은 GitHub 저장소를 clone하거나 이미 clone된 경우 최신 `origin/main`으로 갱신한다.

```text
src/mnist/train_mnist_13568.py
```

MNIST 1/3/5와 custom 6/8 데이터를 함께 학습하는 스크립트이다.

중요 옵션:

```bash
--custom-root data/custom_digits
--custom-repeat 20
--epochs 10
--model-out models/mnist_13568_colab.pt
--metrics-out results/mnist_metrics_colab.json
```

```text
src/mnist/eval_pgm.py
```

PGM 파일을 MNIST 모델로 분류하는 스크립트이다. 과제 요구 출력 형식에 맞게 `INPUT`, `Result of classification`, `Total Images`, `Total Time`, `Digit 0~9`를 출력한다.

중요 옵션:

```bash
--weights models/mnist_13568_colab.pt
--pgm-root data/pgm
--max-images 15
--metrics-out results/pgm_eval_video.json
```

```text
src/yolo/video_to_pgm.py
```

YOLO로 동영상에서 숫자 영역을 감지하고 PGM으로 저장하는 스크립트이다. 같은 숫자가 여러 프레임 동안 계속 감지되어도 이벤트로 묶어서 1개만 저장한다.

중요 옵션:

```bash
--weights models/yolo_digits.pt
--video data/videos/test.mp4
--out-dir data/pgm
--clean-out-dir
--max-events 15
--event-gap-frames 8
--conf 0.35
```

```text
scripts/run_pipeline.sh
```

YOLO video-to-PGM, MNIST 학습, PGM 평가를 순서대로 실행하는 스크립트이다. Jetson에서는 학습보다 추론만 수행하는 것이 일반적이므로 필요에 따라 개별 명령어 실행을 권장한다.

## 4. Colab 학습 결과

학습 설정:

```text
Device: CUDA GPU
Epochs: 10
custom-repeat: 20
Model output: models/mnist_13568_colab.pt
Metrics output: results/mnist_metrics_colab.json
```

학습 결과:

```text
Total training time: 1740.10 sec
MNIST test accuracy: 99.72%
MNIST digit 1 accuracy: 99.82% (1133 / 1135)
MNIST digit 3 accuracy: 99.80% (1008 / 1010)
MNIST digit 5 accuracy: 99.33% (886 / 892)
MNIST digit 6 accuracy: 99.58% (954 / 958)
MNIST digit 8 accuracy: 100.00% (974 / 974)
Custom 6/8 validation accuracy: 92.86%
Custom digit 6 accuracy: 92.86% (13 / 14)
Custom digit 8 accuracy: 92.86% (13 / 14)
MNIST average latency: 0.028924 ms/sample
```

PGM 평가 결과:

```text
PGM samples: 139
Correct / Total: 137 / 139
Overall PGM accuracy: 98.56%
Digit 6 accuracy: 98.57% (69 / 70)
Digit 8 accuracy: 98.55% (68 / 69)
Average latency: 1.571643 ms/sample
Total evaluation time: 0.515309 sec
```

이전 모델 대비 개선:

```text
Overall PGM accuracy: 94.96% -> 98.56%
Digit 8 accuracy: 91.30% -> 98.55%
Correct count: 132 / 139 -> 137 / 139
```

## 5. 중요한 산출물

Colab에서 생성된 산출물:

```text
models/mnist_13568_colab.pt
results/mnist_metrics_colab.json
results/pgm_eval_colab.json
```

주의:

이 파일들은 `.gitignore`에 의해 GitHub에는 올라가지 않는다.

```text
models/*.pt
results/*.json
results/*.csv
```

따라서 Jetson Nano로 옮기려면 Colab에서 직접 다운로드하거나 Google Drive, scp, USB 등을 통해 복사해야 한다.

## 6. Jetson Nano로 옮길 파일

전체 프로젝트를 모두 옮길 필요는 없다. 추론 실행에는 아래 파일만 있으면 된다.

MNIST PGM 평가만 할 경우:

```text
src/mnist/
models/mnist_13568_colab.pt
data/pgm/ 또는 data/custom_digits_pgm/
requirements.txt
```

동영상에서 PGM 생성까지 Jetson에서 할 경우:

```text
src/
scripts/
requirements.txt
models/mnist_13568_colab.pt
models/yolo_digits.pt
data/videos/test.mp4
data/pgm/
results/
```

Jetson에 불필요한 것:

```text
.venv/
data/mnist/
notebooks/
data/custom_digits/      # 추론만 할 경우 불필요
```

## 7. Jetson Nano에서 실행할 명령

1. 저장소 받기:

```bash
git clone https://github.com/eoog333/AI-Accelerator-Final-Project.git
cd AI-Accelerator-Final-Project
```

2. 필요한 산출물 복사:

```text
models/mnist_13568_colab.pt
models/yolo_digits.pt
data/videos/test.mp4
```

3. YOLO로 동영상에서 PGM 생성:

```bash
python src/yolo/video_to_pgm.py \
  --weights models/yolo_digits.pt \
  --video data/videos/test.mp4 \
  --out-dir data/pgm \
  --clean-out-dir \
  --max-events 15 \
  --event-gap-frames 8 \
  --conf 0.35
```

출력 예:

```text
frames_read=...
events_saved=15
detection_csv=results/video_detections.csv
total_time_sec=...
```

4. MNIST로 PGM 분류:

```bash
python src/mnist/eval_pgm.py \
  --weights models/mnist_13568_colab.pt \
  --pgm-root data/pgm \
  --max-images 15 \
  --metrics-out results/pgm_eval_jetson.json
```

출력 예:

```text
INPUT: frame_000015_digit_6_conf_0.88_0001.pgm
Result of classification: 6

INPUT: frame_000020_digit_8_conf_0.94_0002.pgm
Result of classification: 8

Total Images : 15
Total Time   : 2.713 sec

Digit 0 : 0
Digit 1 : 0
Digit 2 : 0
Digit 3 : 0
Digit 4 : 0
Digit 5 : 0
Digit 6 : 7
Digit 7 : 0
Digit 8 : 8
Digit 9 : 0
```

## 8. 아직 남은 작업

남은 핵심 작업:

1. 실제 손글씨 동영상 준비
   - 10~15개 손글씨 숫자가 등장하는 영상
   - 예: `data/videos/test.mp4`

2. YOLO 모델 준비
   - `models/yolo_digits.pt` 필요
   - 아직 없다면 `src/yolo/train_yolo.py`로 학습해야 한다.
   - YOLO 학습 데이터는 `data/yolo/images/{train,val}`와 `data/yolo/labels/{train,val}` 형식이어야 한다.

3. 영상에서 PGM 생성 확인
   - `events_saved`가 영상 속 손글씨 개수와 맞아야 한다.
   - 너무 많이 저장되면 `--event-gap-frames`를 키우거나 `--max-events`를 손글씨 개수로 제한한다.
   - 너무 적게 저장되면 `--conf`를 낮추거나 `--event-gap-frames`를 줄인다.

4. Jetson Nano에서 최종 실행
   - Colab에서 만든 `models/mnist_13568_colab.pt`를 Jetson으로 복사
   - YOLO weight와 test video도 Jetson으로 복사
   - `video_to_pgm.py` 실행
   - `eval_pgm.py` 실행
   - Jetson 실행 시간과 출력 결과를 보고서에 기록

## 9. 현재 상태 요약

현재까지는 MNIST 분류 성능 개선과 PGM 평가까지 완료됐다.

최종 MNIST/PGM 성능은 충분히 좋다:

```text
MNIST test accuracy: 99.72%
PGM final accuracy: 98.56%
Digit 6 PGM accuracy: 98.57%
Digit 8 PGM accuracy: 98.55%
```

아직 완전히 끝난 것은 아니다. 남은 것은 실제 영상과 YOLO weight를 사용해서 다음 전체 흐름을 Jetson Nano에서 실행하는 것이다.

```text
video -> YOLO detection -> one PGM per handwriting appearance -> MNIST classification -> final printed result
```

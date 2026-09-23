# Gesture-Guided Object Filter

손으로 화면의 사각형 영역(ROI)을 지정하고, 그 안에서 탐지된 객체에만 실시간 필터를 적용하는 Python/OpenCV 프로젝트입니다.

## 실행

현재 PC의 Windows conda-prefix 환경은 `.venv/python.exe`입니다. 프로젝트 디렉터리에서 실행합니다.

```powershell
.\.venv\python.exe main.py
```

모델은 `.cache/models/`에 저장하며 Git에 포함하지 않습니다. 새 환경에서는 먼저 다음을 실행합니다.

```powershell
python -m pip install -r requirements.txt
python main.py --download-models
python main.py
```

`requirements.txt`는 현재 Windows/CUDA 환경의 고정 버전입니다. 다른 OS나 CPU 환경은 해당 플랫폼용 PyTorch 설치가 필요합니다. 실행 시 `--device cpu`로 CPU 추론을 선택할 수 있습니다.

## 조작

1. 엄지와 검지를 모아 약 0.1초 유지하면 사각형 그리기가 시작됩니다.
2. 손가락을 모은 채 손을 움직여 반대편 꼭짓점으로 이동합니다.
3. 손가락을 펴면 ROI가 고정되고, 그 안의 `person` 객체에 필터가 적용됩니다.
4. `R`을 누르면 선택을 초기화합니다. 마우스 왼쪽 드래그로도 ROI를 만들 수 있습니다.

| 입력 | 기능 |
| --- | --- |
| `1` / `2` / `3` / `4` | 블러 / 모자이크 / 흑백 / 엣지 |
| `C` | 시작 클래스 → person → cup → bottle → all 순환 |
| `R` | ROI 초기화 |
| `F` | 전체 화면 선택 |
| `H` | ROI·탐지 박스·HUD 표시 전환 |
| `Q` / `Esc` | 종료 |

카메라 영상은 기본적으로 좌우 반전됩니다. `--no-mirror`로 끌 수 있습니다. ROI는 실제 입력 프레임의 픽셀 좌표이며, 영상 파일은 반전하지 않습니다.

```powershell
# 카메라 변경, 손 추적 없이 마우스로 사용
.\.venv\python.exe main.py --source 1 --no-hands

# 대상 클래스와 필터 지정
.\.venv\python.exe main.py --classes person,cup --filter pixelate

# 파일 기반 재현 가능한 실행 및 결과 저장
.\.venv\python.exe main.py --source input.mp4 --no-hands --headless --roi 100 100 600 450 --output runs/demo.mp4
```

`--width`/`--height`는 카메라에 요청하는 해상도이며 장치가 다른 크기를 반환할 수 있습니다. `--roi`는 반환된 프레임 안에서 최소 40×40 픽셀이어야 합니다. 카메라 접근 실패 시 Windows 카메라 권한과 다른 앱의 카메라 점유 여부를 확인하세요.

`--output`은 기존 파일을 덮어쓰지 않으므로 새 경로를 지정해야 합니다.

## 구현 범위

- MediaPipe Tasks HandLandmarker, 손 크기에 정규화한 pinch 거리, 히스테리시스·debounce·좌표 평활화
- `IDLE → DRAWING → LOCKED` 상태 관리. 추적이 0.4초 이상 끊기면 진행 중인 선택 취소
- 고정된 ROI만 YOLO11n pretrained 모델에 입력하고 좌표를 전체 화면으로 변환
- 클래스 및 confidence 조건에 맞는 **모든 객체**에 bounding-box 필터 적용
- CUDA 자동 선택 / CPU 옵션, FPS·추론 시간·탐지 개수 표시, 선택적 MP4 저장

현재는 동일 클래스 중 특정 개체를 고정하는 기능이나 객체 추적·실루엣 분할을 구현하지 않았습니다. 박스 내부 배경도 함께 처리되고, ROI 경계에 걸친 객체는 잘린 영상에서 탐지됩니다. ROI는 화면에 고정되며 움직이는 객체를 따라가지 않습니다. 필터 픽셀은 박스 안에서만 변경하지만 HUD와 박스 선은 화면 위에 별도로 그려집니다.

FPS는 캡처·처리·출력을 포함한 루프 속도이며 HUD는 직전 프레임까지의 이동평균을 표시합니다. 추론 시간은 ROI 탐지 호출의 경과 시간입니다. 첫 추론에는 모델 준비 비용이 포함됩니다. ROI crop도 모델 입력 크기로 리사이즈되므로 영역을 작게 잡는다고 반드시 비례해서 빨라지지는 않습니다. 저장 영상은 입력 FPS로 기록하며 실제 처리 지연을 시간축에 반영하지 않습니다.

## 구조와 검증

`main.py` → `gesture_filter/app.py`(입력/UI) → `roi.py`(상태/좌표), `vision.py`(모델), `filters.py`(합성).

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
.\.venv\python.exe smoke_test.py
```

단위 테스트는 제스처 전환·추적 소실·좌표 복원·필터 외부 픽셀 보존을 검증합니다. `smoke_test.py`는 Ultralytics에 포함된 샘플 이미지로 실제 모델 추론과 MP4 입출력을 검증하며 웹캠을 켜지 않습니다. 결과는 `.cache/smoke/`에 저장됩니다. 실제 손동작 감도와 웹캠 지연은 카메라로 실행해 별도 확인해야 합니다.

추후 단계는 객체 선택/추적, instance segmentation, 웹 실행입니다. [시스템 설계도](docs/system-architecture.svg)는 이 확장 계획을 포함합니다.

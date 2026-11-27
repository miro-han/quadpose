# Computer Vision Term Project - Team 7: RGB 카메라로 상자의 자세 추적하기

움직이는 물체의 자세를 추적하는 솔루션 중 Optitrack이 있다. Optitrack은 매우 정확하지만 비용이 비싸다는 단점이 있다. 일반적인 RGB카메라를 여러 대 이용하여 움직이는 물체의 자세를 얼마나 정확하게 측정할 수 있을까? 본 프로젝트에서는 알려진 치수의 직육면체를 영상에 투영해 코너와 모서리에 물리 ID를 붙이고, 여러 시점에서 관측에 맞는 하나의 6DoF 자세를 추정한다.

---

## 빠른 시작 (Quick Start)

### 1. 환경 설치 (Windows, Python 3.10+)
처음 받은 PC에서는 루트의 [Setup_Project.cmd](Setup_Project.cmd)를 실행하거나, 기존 `.venv`에 패키지를 editable 모드로 설치한다:
```powershell
.\.venv\Scripts\python.exe -m pip install -e . -r capture\requirements.txt -r perception\segmentation\requirements.txt
.\.venv\Scripts\python.exe -m perception.segmentation.setup_sam2
```

### 2. 전체 단위 테스트 실행
```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s . -p "test_*.py" -v
```

### 3. GUI 실행기 (더블클릭 또는 터미널)
- **카메라 녹화기**: [01_Camera_Recorder.cmd](01_Camera_Recorder.cmd)
- **SAM 2.1 물체 추적기**: [02_Object_Tracking.cmd](02_Object_Tracking.cmd)

### 4. 주요 CLI 모듈 실행 예시
```powershell
# 실영상 단일 프레임 관측 추출 및 자세 적합 데모
.\.venv\Scripts\python.exe -m perception.observations.demo_single

# 보정·동기화 CLI 도움말 (인계 영역)
.\.venv\Scripts\python.exe -m rig.calibrate --help
.\.venv\Scripts\python.exe -m rig.synchronize --help

# 자세 추정 벤치마크 (smoke 실행)
.\.venv\Scripts\python.exe -m evaluation.pose_benchmark.benchmark --suite selection --smoke --output outputs/smoke_selection
```

---

## 디렉터리 구조

| 폴더 | 역할 | 담당 / 비고 |
| :--- | :--- | :--- |
| [capture/](capture/) | 다중 카메라 녹화 UI, 수신 스레드, AVI/CSV 기록 | **인계 영역** (촬영·장치 검증) |
| [rig/](rig/) | 카메라 내부/외부 보정, 점멸 시간 동기화, 프레임 추출, 마운트 검증 | **인계 영역** (캘리브레이션·동기화) |
| [perception/](perception/) | Canny/LSD 에지(`edges/`), SAM 2.1 분할(`segmentation/`), 관측 어댑터(`observations/`) | 영상 관측 파이프라인 |
| [pose/](pose/) | 6DoF 다중 시점 융합(`fusion.py`), 3-뷰 가설(`hypotheses.py`), 재추정(`refinement.py`), 복구(`recovery.py`) | 핵심 자세 추정 코어 |
| [evaluation/](evaluation/) | 배치 가시성/자세 평가, 동일 모서리 검증, 독립 벤치마크(`pose_benchmark/`), 자원 제한 실행기 | 실험·재현·감사 |
| [experiments/](experiments/) | 초기 PoC(`fusion/`, `mount_occlusion/`) 및 중단된 AMD 실험(`amd/`) | 참고/보관 |
| [tools/](tools/) | 회귀 테스트(`run_handoff_regression.py`), SAM 벤치마크, 패키징 스크립트 | 유틸리티 |
| [docs/](docs/) | [기술 가이드](docs/guide/INDEX.md), [환경 보고서](docs/ENVIRONMENT_REPORT.md), [과거 단계별 기록](docs/archive/) | 문서 일체 |

---

## 문서 안내

- **처음 프로젝트를 이해하려면**: [팀 요약과 요구사항](docs/guide/01_PROJECT_OVERVIEW.md) → [기술 가이드 목차](docs/guide/INDEX.md) → [시스템 구조](docs/guide/03_SYSTEM_ARCHITECTURE.md)
- **카메라 녹화·보정·실물 검증을 맡았다면**: [실물 검증 순서](docs/guide/15_PHYSICAL_VALIDATION.md) · [카메라 녹화 안내](capture/README.txt) · [보정·동기화 도구](rig/README.md) · [마운트 검증 절차](rig/MOUNT_VALIDATION.md)
- **자세 추정 알고리즘·결과를 보려면**: [관측 어댑터](perception/observations/README.md) · [6DoF 융합](pose/README.md) · [알고리즘 설계](evaluation/pose_benchmark/ALGORITHM_DESIGN.md) · [단계별 효과](evaluation/pose_benchmark/ABLATION_STUDY.md) · [실패 사례](evaluation/pose_benchmark/FAILURE_ANALYSIS.md)
- **과거 단계별 문서·원자료를 찾으려면**: [자료 목록](docs/guide/SOURCE_MAP.md) · [아카이브](docs/archive/)

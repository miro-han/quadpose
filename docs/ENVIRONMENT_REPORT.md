# SAM 2.1 실행 환경과 벤치마크

## 실행 환경

2026-11-25 확인 기준입니다.

| 항목 | 확인값 |
| --- | --- |
| 운영체제 | Windows 11 x64, 빌드 26200 |
| CPU | AMD Ryzen 7 6800H, 8코어 / 16스레드 |
| 메모리 | Windows에서 인식한 총 메모리 31.19 GiB |
| 전원 관리 모드 | 균형 조정(Balanced) |
| NVIDIA GPU | GeForce RTX 3060 Laptop GPU, VRAM 6,144 MiB, Compute Capability 8.6 |
| NVIDIA 드라이버 | 595.71, `nvidia-smi`의 CUDA 호환 버전 표시 13.2 |
| Python | 3.12.10 x64, 프로젝트 루트의 `.venv` |
| CUDA용 PyTorch | `torch==2.8.0+cu126`, `torchvision==0.23.0+cu126`, 휠에 포함된 CUDA 런타임 12.6 |
| OpenCV / NumPy / Pillow | `5.0.0.93` / `2.5.4` / `13.0.0` |

`.venv/pyvenv.cfg`에서 `include-system-site-packages = false`를 확인했습니다. [기존 환경 기록](environment_verified.json)에 적힌 Python 3.10.7 및 전역 패키지 공유 환경과 별개로, 프로젝트 전용 폴더에 설치했습니다. PyTorch CUDA 휠은 [공식 Windows 설치 조합](https://pytorch.org/get-started/previous-versions/#v280)을 사용합니다.

## 설치와 실행

프로젝트 루트에서 PowerShell로 실행합니다. `python`은 Windows x64용 Python 3.12를 가리켜야 합니다. 다른 버전이 기본이면 `-Python` 인수로 3.12의 `python.exe` 경로를 넘길 수 있습니다.

```powershell
.\tools\setup_cuda.ps1
.\02_Object_Tracking.cmd
```

설치 스크립트는 `.venv`에 [잠금 파일](../tools/requirements-win-py312.lock.txt)의 패키지와 지정된 CUDA 휠을 설치하고 `pip check`와 CUDA 텐서 연산을 실행합니다. 이 PC에서 의존성 검사, 카메라 패키지와 Tk 가져오기, MP4 쓰기·읽기, RTX 3060의 CUDA 텐서 연산을 확인했습니다. PyTorch 휠의 SHA-256은 공식 인덱스의 `f6c79eac0018f9d131479ee1b7a68edb030619a316bfbc69275043aa4f338e4c`와 일치했습니다. [PyTorch CUDA 12.6 휠 인덱스](https://download.pytorch.org/whl/cu126/torch/)

SAM 2.1 소스와 체크포인트는 배포본의 `third_party/sam2`와 `models`를 사용합니다. 파일이 없으면 아래 명령으로 준비합니다.

```powershell
.\.venv\Scripts\python.exe .\perception/segmentation\setup_sam2.py
```

기존 GUI의 처음 안내 문구에는 `CPU 모드`가 표시됩니다. 실제 장치는 모델을 불러올 때 PyTorch가 선택하며, 모델 준비 상태와 결과 `session.json`의 `device` 필드에서 확인할 수 있습니다.

## 벤치마크 방법

```powershell
.\.venv\Scripts\python.exe .\tools\benchmark_sam.py --device cuda
.\.venv\Scripts\python.exe .\tools\benchmark_sam.py --device cpu
```

- `WIN_20261124_20_05_34_Pro.mp4`의 프레임 0~29를 사용합니다. 입력은 1920×1080이고, OpenCV가 보고한 원본 프레임률은 약 29.98 fps입니다.
- 기존 30프레임 CPU 세션과 동일하게 첫 프레임에 박스 `(831.58, 272.92, 1237.45, 706.78)`만 적용합니다. 스크립트는 기록된 좌표의 전체 정밀도를 사용합니다.
- SAM 2.1 Hiera Tiny의 기존 `SamSession` 경로를 호출합니다. JPEG 품질 95와 공식 1024×1024 로더를 사용하고, 입력 프레임과 추적 상태는 CPU에 저장합니다. 선택적 SAM CUDA 후처리 확장은 비활성 상태입니다.
- 모델 로드, 입력 준비, 첫 프롬프트 적용, 30프레임 추적을 각각 측정합니다. 추적 시간에는 전파, 원본 디코딩, 마스크와 에지 PNG, 로그 및 미리보기 MP4 생성이 포함됩니다. GUI 상태 표시와 화면 그리기 콜백은 생략합니다. 단계별 합계는 Python 모듈 가져오기 이후부터 측정하며, CUDA 단계의 바깥쪽 측정은 `torch.cuda.synchronize()` 후 종료합니다.
- `--device cuda`는 CUDA 사용 불가 시 실패합니다. `--device cpu`는 같은 PyTorch 설치에서 CUDA 장치를 숨깁니다. 스크립트는 프레임 30개와 순서를 확인한 뒤 새 세션 폴더에 `benchmark.json`을 저장합니다.

소스 커밋은 `2b90b9f5ceec907a1c18123530e92e794ad901a4`, 체크포인트 SHA-256은 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`입니다. 입력 영상 SHA-256은 `85ec02f0fa57343d87045f72eb2950fc036f7ff33233ff743c2c9b01761c55c5`입니다.

## 측정 결과

| 실행 장치 | 모델 로드 | 입력 준비 | 프롬프트 | 추적·저장 | 추적 FPS | 단계별 합계 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Ryzen 7 6800H CPU](../outputs/segmentation/20261125_193749_786615/benchmark.json) | 1.456초 | 8.648초 | 0.244초 | 293.269초 | 0.102 | 303.628초 |
| [RTX 3060 CUDA](../outputs/segmentation/20261125_193720_260171/benchmark.json) | 2.907초 | 4.062초 | 0.216초 | **11.730초** | **2.558** | 18.922초 |

같은 PC에서 CUDA의 추적·저장은 CPU보다 **25.00배** 빨랐습니다. 각 장치에서 별도 프로세스로 한 번 실행하고, 두 실행에서 PyTorch의 CPU 스레드 수를 2개로 유지했습니다. 표의 추적·저장은 `session.json`의 `processing_seconds`를, 단계별 합계는 단계 호출 전체의 계측값을 사용합니다. CUDA로 기록한 추적 FPS는 원본 영상의 약 29.98 fps보다 낮습니다.

GPU 세션 모델 파라미터의 장치는 `cuda:0`, 두 세션의 데이터 타입은 `float32`입니다. CUDA 메모리의 PyTorch 최대 할당은 718.4 MiB, 최대 예약은 992.0 MiB였습니다.

두 세션에 각각 비어 있지 않은 마스크 30장, 에지 PNG 30장, 30프레임이 모두 디코딩되는 미리보기 MP4가 생성됐습니다. 같은 프레임의 CPU·CUDA 마스크 간 IoU는 평균 **0.999988**, 최솟값 **0.999955**(원본 프레임 17)였습니다. 이 수치는 두 장치의 출력 일치도를 나타냅니다.

## 기존 CPU 기록과 비교

| 2026-11-24 CPU 세션 | 원본 프레임 | 프롬프트 | 추적 시간 | 추적 FPS |
| --- | ---: | --- | ---: | ---: |
| [30프레임](../outputs/segmentation/20261124_215925_8f211c/session.json) | 0~29 | 박스 | 599.700초 | 0.050 |
| [3프레임](../outputs/segmentation/20261124_215508_e1cb28/session.json) | 11~13 | 박스 | 22.887초 | 0.131 |
| [GUI 2프레임](../outputs/segmentation/20261124_215749_db9b73/session.json) | 11~12 | 박스 + 포함·제외 클릭 | 12.172초 | 0.164 |

동봉 세션에는 `device: cpu`가 기록돼 있습니다. 동봉 환경 기록에서 Python 3.10.7 x64와 torch 2.14.1을 확인할 수 있습니다. CPU 모델, 코어 수, RAM 및 전원 모드는 기존 세션과 환경 기록에 기재돼 있지 않아 확인되지 않았습니다. 3프레임 세션의 최초 준비 및 마스크 확인 약 26초는 추적 시간과 별도로 기록됐습니다. [기존 환경 기록](environment_verified.json), [작업 인계](PROJECT_CONTEXT.txt)

동일한 범위·박스의 과거 30프레임 기록과 비교하면 현재 CUDA 실행은 **51.13배**, 현재 CPU 실행은 **2.04배** 빨랐습니다. 이 배수에는 장치, 하드웨어 및 소프트웨어 환경의 차이가 포함됩니다. 두 짧은 세션은 프레임 범위와 프롬프트 조건이 달라 별도 기록으로 남겼습니다.

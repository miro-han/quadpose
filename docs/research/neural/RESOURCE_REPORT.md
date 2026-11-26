# Stage 0 · 실행 환경과 자원 기록

## 현재 PC에서 확인한 값

2026-11-26 15:24–15:26 KST 읽기 전용 점검이다. CPU 경로로 저장소의 계약과 테스트를 확인했다. 이번 점검에서 CUDA·OpenCL·DirectML 연산, 학습, 라이브러리 설치를 실행하지 않았다.

| 항목 | 확인값 |
| --- | --- |
| OS | Windows 11 x64, 빌드 26200 |
| CPU | AMD Ryzen 7 6800H, 8코어 / 16스레드 |
| Windows 메모리 | 총 31.19 GiB, 조사 시 가용 15.83 GiB |
| 저장 공간 | 작업 드라이브 가용 약 366.1 GiB |
| 전원 정책 | Balanced |
| RTX GPU | GeForce RTX 3060 Laptop GPU, 6,144 MiB, 드라이버 595.71, compute capability 8.6 |
| RTX 관측 | `nvidia-smi` 스냅샷의 시스템 전체 1,432 MiB / 10%. 한 시점의 수치이며 프로세스별 예산이나 사용 허가를 나타내지 않는다. |
| 프로젝트 `.venv` | Python 3.12.10, 시스템 site-packages 미포함. `pip check` 통과 |
| 실제 설치 | PyTorch `2.8.0+cu126`, TorchVision `0.23.0+cu126`, NumPy `2.5.4`, OpenCV `5.0.0.93`, Pillow `13.0.0`, pip `25.0.1` |

`nvidia-smi`는 명칭·버전·메모리 카운터 조회에만 사용했다. CUDA 사용 가능 여부나 처리량의 별도 실측은 이번 점검에 포함하지 않았다. 과거 CUDA 추적 및 AMD 실험 수치는 각 실험 당시 장치와 프로토콜의 기록이다. [Phase I 실행 환경](../../../docs/ENVIRONMENT_REPORT.md) · [Phase III 자원 격리](../../../evaluation/RESOURCE_ISOLATION.md)

## CPU에서 바로 확인할 수 있는 작업

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m evaluation.resource_limits --threads 1 --memory-gib 3 pose.check_dataset pose/results
.\.venv\Scripts\python.exe -m evaluation.resource_limits --threads 1 --memory-gib 3 pose.test_hypotheses_recovery
```

위 데이터 검사와 5개 회귀 테스트가 통과했다. `evaluation.resource_limits`는 NumPy/OpenCV import 전 환경변수로 CUDA와 OpenCL을 비활성화하고 BLAS 계열 스레드 수를 지정한다. Windows Job Object는 이 프로세스의 commit 한도를 제한한다. 개별 라이브러리, OS, 드라이버의 스레드 생성까지 고정하는 방식은 아니므로 학습 진입점에서 PyTorch/OpenCV 자체 스레드 설정도 명시한다. 보고 시 wall time, 프로세스 peak RSS/commit, epoch/step 수, 배치와 워커를 남긴다. [CPU 한도 코드](../../../evaluation/resource_limits.py)

기존 3,072프레임의 합성 NPZ는 입력 4,516,345바이트, 타깃 627,989바이트다. allow-list의 8개 순수 패킷 배열의 비압축 합은 2,138,112바이트다. 따라서 데이터 크기로는 M01 토큰 기준선과 M04/M05 소형 학습을 기존 CPU 환경에서 시작할 수 있다. 데이터 크기와 관계없이 미분 최적화의 긴 그래프, 영상 백본, 시뮬레이터 반복 렌더링과 RL 재생은 별도 프로파일이 필요하다. 먼저 짧은 단일 설정에서 특징 생성, 학습, 추론, 기하 solver 시간을 각각 계측한 뒤 반복 규모를 정한다. [패킷 감사](CODEBASE_AUDIT.md) · [방법 비용 비교](METHODOLOGY_SURVEY.md)

## 브랜치 내 지도 잔차 모델의 CPU·CUDA 실측

읽기 전용 점검과 별도로 연구 브랜치에서 한 번 실행한 D0 소형 병진 보정 실험이다. 같은 시드 907, 훈련 156개 행의 100회 full-batch optimizer step, 조건을 만족한 개발 평가 148개 행을 사용했다. 공통 입력 생성·기하 처리에 **18.113초**가 걸렸다. [재현 코드](../../../experiments/neural/residual_study.py) · [집계와 원시 요약 SHA-256](stage0_resource_evidence.json)

| 항목 | CPU | RTX 3060 CUDA |
| --- | ---: | ---: |
| 학습 반복 구간 | 0.146초 | 0.566초 |
| 단일 프레임 병진 보정 구간 p95 | 0.380 ms | 1.210 ms |
| 개발 위치 RMSE | 5.7632 mm | 5.7632 mm |
| PyTorch 최대 CUDA 할당량 | 해당 없음 | 17.36 MiB |

학습 반복 시간은 모델/데이터 준비 후부터 재며, CUDA는 구간 시작과 끝을 동기화한다. 보정 p95는 한 행의 텐서 생성과 모델 호출, CUDA에서 필요한 전송·CPU 읽기를 포함한다. 전체 영상 추론이나 자세 추적 지연은 이 구간과 별개다. 이 작은 반복에서는 CPU 쪽 숫자가 낮았고 기하 기준선 위치 RMSE는 3.7964 mm였다. 단발 측정이므로 장치 일반 우열, 큰 배치, 영상 모델 또는 다른 부하의 처리량으로 외삽하지 않는다. CUDA 할당량은 PyTorch 메모리 카운터이며 시스템 전체 VRAM은 아니다.

## 실행 경로의 해석

- `.venv`는 CUDA 빌드가 설치된 기존 실행 환경이다. CPU 연구 코드는 `device=cpu`와 제한 실행기를 사용할 수 있다. 외부 프로세스의 RTX 점유율을 낮추거나 CUDA 기본 장치를 변경할 이유는 없다. GPU 실험이 나중에 승인되면 승인된 장치·메모리·실행시간에 맞춰 먼저 짧은 별도 벤치마크를 잡고 시스템 전체 점유와 연구 프로세스 점유를 분리해 기록한다. [기존 설치 기록](../../../docs/ENVIRONMENT_REPORT.md)
- 과거 AMD는 별도 환경에서 시험했다. Torch DirectML은 native 장치 열거의 안전 조건에서 중단됐고 학습을 시작하지 않았다. ORT DML1 추론과 AMD OpenCL은 지정 어댑터 및 프로세스 지표가 확인된 과거 제한 실험이다. AMD 추론 성공을 AMD 학습 호환성으로 해석할 수 없다. [Torch·ORT 기록](../../../experiments/amd/results/fusion_train.json) · [ORT 추론](../../../experiments/amd/results/fusion_inference.json) · [AMD OpenCL](../../../experiments/amd/opencl/README.md)
- 현재 목표 데이터는 수치 합성 패킷이다. SAM 모델 파일과 소스는 로컬에서 확인했으나, 영상 기반 학습을 위한 동기 검증·라벨 영상이 추적 대상 Git 자료에 포함되어 있지 않다. M02/M08 등은 RAM이나 VRAM보다 현장 영상·정답 확보가 먼저 필요한 경우다. [실물 자료 조건](CODEBASE_AUDIT.md) · [촬영 검증 안내](../../../docs/guide/15_PHYSICAL_VALIDATION.md)

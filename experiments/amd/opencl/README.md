# Radeon 680M으로 실행한 OpenCV

Windows 기본 Python 3.12와 OpenCV 5.0.0에서 동봉된 1920×1080 영상의 원본 프레임 11과 12를 읽었다. 렌즈·노출·배경에 대한 일반화 실험이 아니라 연산별 장치 선택과 결과 일치도를 확인하는 입력이다. 코어 연산은 CPU 스레드 1개와 비교했다. AMD 준비 비용과 최초 커널 컴파일은 워밍업에서 제외했다.

| 연산, 중앙값 ms | CPU | AMD에 입력 상주 | CPU→AMD→CPU 왕복 포함 | 출력 비교 |
|---|---:|---:|---:|---|
| Gaussian 5×5 | 3.41 | 1.51 | 3.79 | 픽셀 99.9633% 동일 |
| Canny 15/45 | 11.65 | 1.63 | 3.83 | 100% 동일 |
| 5×5 closing | 2.61 | 1.47 | 3.63 | 100% 동일 |
| 피라미드 LK | 25.53 | 4.67 | 7.04 | 양쪽 유효점의 좌표 차이 중앙값 0.00221 px |
| Gaussian → Canny → closing | 17.55 | 4.58 | 6.84 | 100% 동일 |

연산마다 AMD 단독 라이브러리의 플랫폼 이름·GPU 종류·장치 이름을 지정했다. OPENCV_OPENCL_RUNTIME은 Windows DriverStore의 amdocl64.dll 절대경로, OPENCV_OPENCL_DEVICE는 AMD:GPU:gfx1035이다. 장치가 없거나 AMD 검사에 실패하면 실행을 종료한다. 각 입력을 UMat으로 만들고 GPU 연산 뒤 finish를 호출해 큐 제출 시간만 재는 오류를 피했다. 상주 실험에서는 마지막 결과 읽기를 타이머 밖에, 왕복 실험에서는 입력 생성과 결과 읽기를 타이머 안에 뒀다. OpenCV의 투명 API는 지원되지 않는 분기를 CPU로 돌릴 수 있다. 이 입력에서는 각 연산 구간의 **해당 PID와 AMD LUID에 compute 엔진 활동**이 관측됐다(상주 기준 35~70%). 혼합된 CPU 제어 코드는 남을 수 있다. [OpenCV OpenCL 동작 및 환경변수](https://github.com/opencv/opencv/wiki/OpenCL-optimizations)

AMD LUID는 0x00000000_0x0001386d, PCI 식별자는 1002:1681이며 드라이버는 3570.0 (PAL,LC)이다. 별도 그래픽 장치를 만들지 않고 DXGI QueryVideoMemoryInfo로 해당 PID의 local/nonlocal 사용량을 측정했다. 시작과 각 연산 구간 직후 RTX 3060은 두 메모리 값이 모두 0바이트였고, 이 PID에서 NVIDIA 엔진 활동이 관측되지 않았다. AMD local은 최대 139.74 MiB였다. 이 값은 표본 간 찰나의 할당을 포착하는 연속 추적이나 다른 프로젝트의 메모리 총량 측정이 아니다. [실행 결과·P95·실제 사용 엔진](measured.json)

재현은 프로젝트 루트에서 수행한다. 기본 프로세스 메모리 상한 4 GiB는 Windows Job Object의 commit 상한이다. 이 실행의 최대 working set은 270.75 MiB였다. CPU-only 합성 및 융합에는 프로젝트 루트에서 별도 모듈 런처를 사용해 OpenCL 런타임도 비활성화할 수 있다.

```powershell
python -m experiments.amd.opencl.measure --threads 1 --memory-gib 4 --output experiments/amd/opencl/replay.json
python -m evaluation.resource_limits --help
```

시스템 정책이 WDDM 성능 카운터를 차단하면 engine_counter_issue에 이유가 기록된다. 그러한 실행은 커널 실행 확인 근거에서 제외했다. 처리 시간이 필요한 경로에서는 Canny 등 연결된 연산을 AMD에 두는 것이 유리했다. 한 단계만 왕복한 blur와 closing은 이 CPU 기준보다 느렸다.

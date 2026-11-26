# 장치 선택과 자원 기록

배치·관측 시뮬레이션, 모델 대응과 SE(3) 적합은 NumPy CPU 경로에서 실행한다. 실험 명령은 [기하 스윕](layout_geometry/README.md)과 [자세 스윕](layout_pose/README.md)에 있다. 관리 대상은 이 프로젝트의 프로세스다. Windows Job Object의 per-process commit 한도와 BLAS/OMP 스레드 환경변수를 Python 수치 라이브러리 import 전에 적용한다. Windows 전체 메모리나 OS/드라이버가 생성하는 모든 스레드를 고정한다는 뜻은 아니다. 실행 예:

```powershell
python -m evaluation.resource_limits --threads 1 --memory-gib 3 evaluation.layout_geometry.test_sweep
```

읽기 전용 DXGI 열거에서 이 부팅 세션의 인덱스 0은 NVIDIA(10de:2560), 1은 Radeon(1002:1681), 2는 소프트웨어 렌더러였다. 인덱스 0을 GPU 기본값으로 사용하는 코드는 AMD 확인 수단이 될 수 없다. AMD OpenCL은 AMD 전용 런타임과 이름 gfx1035로 지정한다. 연산·결과 전송 구간별 Windows GPU Engine 카운터를 PID 및 LUID로 나누고, QueryVideoMemoryInfo로 이 프로세스의 GPU별 local/nonlocal 메모리를 읽었다. 기존 NVIDIA CUDA 환경을 설치 대상으로 사용하지 않았다. [OpenCL 로그와 재현](../experiments/amd/opencl/README.md)

DirectML은 별도 Python 환경을 사용한다. ORT에서 어댑터 ID는 DXGI 어댑터 열거 순서를 따른다. 실행 제공자 이름만 보고 처리 위치를 판정하면 일부 노드의 CPU fallback을 놓칠 수 있으므로 노드별 프로파일과 CPU 비교가 필요하다. PyTorch DirectML의 공식 버전 제한도 CUDA 환경과 분리해야 하는 이유다. [Microsoft PyTorch DirectML 안내](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows) · [ONNX Runtime DirectML 실행 제공자](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)

실험 시작 때 읽은 **시스템 전체** RTX 메모리는 2,763 MiB, 사용률은 39%였다. 이 숫자는 우리 PID의 메모리와 다르다. [OpenCL 본 측정](../experiments/amd/opencl/measured.json)에서는 AMD 연산 엔진만 양의 활동을 보였고 우리 PID의 RTX 메모리는 관측한 모든 지점에서 0이었다. 작업 중 다른 프로세스의 RTX 총량이 바뀌더라도 여기서 그 원인이나 외부 학습 속도를 추정하지 않는다. 사용자 요청에 따라 외부 학습 처리량 비교는 조사 범위에서 제외했다.

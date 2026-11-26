# 어디서 시작해 무엇을 확인할까

PowerShell에서 프로젝트 루트로 이동한다. 기존 의존성이 있는 환경을 사용한다. Phase IV 실험에는 새 GPU 환경 구성이 필요하지 않다. 신규 연구 결과는 원래 저장된 실험을 덮어쓰지 않도록 새 경로로 재현한다. [설치 및 버전 기록](../ENVIRONMENT_REPORT.md)

**기존 기능을 재현하는 경로**

1. [01_Camera_Recorder.cmd](../../01_Camera_Recorder.cmd)로 슬롯과 장치 ID를 확인하고 AVI·CSV·session.json을 저장한다. 라이브 하드웨어가 없으면 기존 측정과 합성 테스트를 사용한다.
2. [02_Object_Tracking.cmd](../../02_Object_Tracking.cmd)에서 짧은 구간을 선택해 첫 프레임의 상자를 지정한다. 실제 파일에서 결과를 다시 읽을 때는 session 상태, PNG 수와 원본 프레임 인덱스를 확인한다.
3. [실영상 어댑터의 예제 명령](../../perception/observations/README.md)을 따라 Canny/LSD·마스크에서 관측 JSON과 오버레이를 만든다. 보정된 동시 그룹은 별도의 rig.json, 정렬 근거, 실측 모델과 초기 자세가 필요하다.
4. [Phase IV 재현 절차](../../evaluation/pose_benchmark/README.md)에서 작은 테스트, 개발 분할 비교, 독립 시험, 결과 재생성 순서로 확인한다. [고정 설정](../../evaluation/pose_benchmark/benchmark_config.json)과 함께 시드·카메라 부호·물체 치수·초기화 범위를 확인한다.

**파일 해석 예.** 픽셀 공간이 raw distorted인데 후단이 왜곡 없는 K 픽셀로 읽으면 가장자리의 선이 체계적으로 빗나간다. 세션 이름이 달라도 `camera_1.avi`는 반복될 수 있으므로 경로·장치·프레임·보정 ID를 함께 비교한다. 원본 샘플은 독립 촬영 파일이므로 다중 시점 입력처럼 시간을 맞추지 않는다.

기존 실행 환경·촬영 UI·Phase I–III의 구현과 결과는 그대로 접근할 수 있다. 루트 README 개정 전 바이트 사본과 시작 시점 해시는 [보존 기록](../../evaluation/pose_benchmark/provenance/)에 있다.

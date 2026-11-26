Team 7 멀티 카메라 녹화 UI

실행
- capture\Start_Camera_UI.vbs를 더블클릭하거나 바탕화면의 Team7 Camera Recorder 바로가기를 실행합니다.
- PowerShell: .\.venv\Scripts\python.exe capture\camera_recorder.py
- 프로그램 실행 자체로 녹화하지 않습니다.

사용 순서
1. [카메라 새로 찾기]로 장치를 확인합니다.
2. CAMERA 01~04에서 각각 다른 장치를 선택합니다. 남은 칸은 [사용 안 함]으로 둡니다.
   현재 노트북의 외장 장치는 [1] Web Camera, [2] Web Camera로 확인되었습니다.
   USB 재연결 후 번호가 바뀔 수 있으므로 실제 화면과 session.json의 장치 경로로 확인하세요.
   내장 ATIV 및 OBS 가상 카메라는 기본 자동 선택에서 제외합니다.
3. 기본 1280x720 / 30 FPS로 [선택 카메라 켜기]를 누릅니다.
4. 선택한 카메라 모두 영상이 들어오면 [전체 녹화 시작] 버튼이 활성화됩니다.
5. 같은 버튼의 [전체 녹화 종료]를 누르고 [저장 완료] 표시를 확인합니다.
6. [폴더 열기]로 결과를 확인합니다.

출력
recordings/한국시각_고유ID/
  camera_1.avi, camera_2.avi, ... : 카메라별 원본 해상도 MJPEG 영상
  camera_1_timestamps.csv, ...   : 영상 프레임과 1:1 대응하는 수신 시각
  session.json                 : 장치 경로, 해상도, FPS, 시작/종료, 저장 수, 누락, 오류
촬영 메모에 정면/회전/가림/조명/배경 등의 조건을 적으면 session.json에 저장됩니다.
녹화마다 새 폴더를 생성하며 기존 촬영을 덮어쓰지 않습니다.
영상에는 UI·문자·에지·오디오를 넣지 않습니다. 미리보기 크기와 저장 해상도는 별개입니다.
MJPEG AVI는 프레임별 압축이므로 분석하기 편하지만 파일 크기가 큽니다.

시간 정렬에 관한 중요한 구분
- 한 버튼은 모든 카메라에 같은 소프트웨어 녹화 시작 기준 시각을 적용합니다.
- USB 카메라 센서의 노출 시작을 하드웨어 동기화하는 기능은 아닙니다.
- host_monotonic_ns: VideoCapture.read() 직후의 PC 단조 시계. 같은 PC 내 비교용.
- host_unix_ns: 같은 시점의 PC wall clock. UTC Unix epoch 기반이며 OS 시간 보정의 영향을 받을 수 있습니다.
- seconds_from_record_start: 공통 녹화 시작부터의 수신 시각 차이.
- video_frame_index: 영상의 0부터 시작하는 프레임 번호.
- capture_sequence: 카메라 연결 이후 수신 순번. 녹화 중 누락을 확인할 때 사용합니다.
- 타임스탬프는 센서 노출 시각이 아니며, USB/드라이버 버퍼 지연을 포함합니다.
- AVI의 재생 FPS는 녹화 시작 시 미리보기에서 측정한 카메라별 수신 FPS로 정합니다(충분한 샘플이 없으면 요청 FPS).
  카메라를 켠 뒤 2~3초 기다리고 녹화하면 이 값이 안정됩니다. 값은 cameras.*.encoding_fps에 기록됩니다.
  AVI는 고정 FPS이므로 녹화 중 수신 속도가 바뀌거나 프레임이 누락되면 재생 길이가 실시간과 다를 수 있습니다.
  다중 시점 처리는 같은 영상 프레임 번호끼리 무조건 결합하지 말고 CSV 및 공통 점멸 신호로 정렬해야 합니다.
- 프레임 복제/보간은 하지 않습니다. 저장 대기열이 차면 새 프레임을 버리고 누락 수를 기록합니다.
- USB/센서 내부에서 PC에 도착하기 전에 손실된 프레임은 이 프로그램이 정확히 셀 수 없습니다.

문제 해결
- 다른 카메라 앱/화상회의 앱을 닫으세요. 장치 점유가 있으면 연결되지 않을 수 있습니다.
- 영상이 안 나오면 카메라를 끄고 DirectShow/MSMF를 바꾸어 다시 찾으세요.
- 여러 대 연결 시 느리면 640x480 또는 15 FPS로 낮추고, 다른 USB 포트/컨트롤러를 사용해 비교하세요.
- 화면에 표시되는 해상도와 FPS가 실제 수신 값입니다. 드라이버가 요청 설정을 거절할 수도 있습니다.
- 일부 장치 오류 시 전체 녹화를 종료하고 오류를 남깁니다. 실패한 카메라를 제외한 뒤 새 촬영을 시작할 수 있습니다.
- 창을 닫으면 남은 녹화 프레임을 저장한 후 장치를 해제합니다. 완료 전 강제 종료는 피하세요.
- 장치 드라이버의 read 자체가 멈추면 해제까지 지연될 수 있습니다.

다른 PC 설치
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r capture\requirements.txt
  .\.venv\Scripts\python.exe capture\camera_recorder.py
현재 PC의 .venv는 기존 OpenCV/NumPy/Pillow를 공유하는 --system-site-packages 환경이며,
cv2-enumerate-cameras 1.4.0만 프로젝트 환경에 추가 설치했습니다.

검증
  .\.venv\Scripts\python.exe capture\verify_recorder.py
    합성 4개 스트림의 프레임-CSV 대응, 시각 순서, 시작/종료 경계, 반복 녹화를 검증합니다.
  .\.venv\Scripts\python.exe capture\verify_recorder.py --hardware
    이름이 Web Camera인 외장 2대를 UI로 켜고 약 4초 녹화한 뒤 파일을 다시 읽어 검증합니다.
    검증 영상과 보고서는 outputs/camera/recorder_checks/에 저장됩니다.
4대 실제 동시 구동 성능은 4대가 연결된 환경에서 별도 검증해야 합니다.

2026-11-24 실제 검증 결과
- 외장 Web Camera 2대, DirectShow, 1280x720, 요청 30 FPS.
- 두 카메라 모두 실제 수신 약 10.8 FPS. 원인은 아직 분리 검증하지 않았습니다.
- 약 4.5초 녹화에서 각 49프레임을 저장하고 49개 CSV행과 1:1 대응을 확인했습니다.
- 저장 대기열 누락 0, 모든 출력 프레임 재독출 성공, 타임스탬프 순서/녹화 범위 검증 통과.
- outputs/camera/recorder_checks/hardware_report.json 및 ui_live_check.png에 증거를 남겼습니다.
- 검사 환경: Python 3.10, OpenCV 5.0.0, NumPy 2.2.6, Pillow 12.3.0, cv2-enumerate-cameras 1.4.0.
- 초기 테스트에서는 Windows 카메라 앱 실행 및 외장 장치 한 대 미인식 상태가 있었습니다.
  사용자가 카메라 앱을 종료하고 외장 장치를 재연결한 뒤 2대 검증에 성공했습니다.

외부 코드·AI 지원 범위
- OpenCV: 장치 영상 읽기, 색 변환, MJPEG 인코딩. 라이브러리 호출로 사용.
  https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
  https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
- Python tkinter/ttk: 데스크톱 UI와 이벤트 루프.
  https://docs.python.org/3/library/tkinter.html
- Pillow: 미리보기 리사이즈와 Tk 이미지 변환.
  https://pillow.readthedocs.io/en/stable/reference/ImageTk.html
- cv2_enumerate_cameras 1.4.0, lukehugh, MIT: 장치 이름·경로·OpenCV 번호 탐색.
  https://github.com/lukehugh/cv2_enumerate_cameras
- camera_core.py / capture\camera_recorder.py / capture\verify_recorder.py는 Codex가 이 프로젝트용으로 작성했습니다.
  카메라별 획득 스레드, 공통 녹화 시작 기준, 저장 큐, CSV, 오류 처리 및 UI 연결 부분입니다.
- 외부 완성 녹화 프로그램을 복사한 구현이나 3D 자세 추정 논문의 재현 코드가 아닙니다.

제공된 텀프로젝트 가이드라인과 연결
- PDF 11쪽(슬라이드 23): 전체 시스템, 실제 데이터, 문제 해결과 정량/정성 분석.
- PDF 13쪽(슬라이드 25): 외부 코드·사전학습 모델·AI 사용 공개 및 주요 구성 설명.
- PDF 16쪽(슬라이드 28): GitHub 소스/실행 안내, 자체 수집 샘플, 발표/시연 제출.
- PDF 17쪽(슬라이드 29): 팀원별 알고리즘/선택 이유/실패 사례/기여 설명.
이 UI는 자체 데이터 수집을 위한 도구이며, 최종 에지 융합·자세 추정과 그 평가를 대신하지 않습니다.
GitHub 공개 업로드는 아직 수행하지 않았습니다.


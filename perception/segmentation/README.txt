물체 지정 및 SAM 2 영상 추적

실행: 이 폴더의 Start_Object_Tracking.cmd를 더블클릭합니다.
현재 노트북에는 환경과 모델이 준비되어 있습니다.

1. 영상 선택 → 시작 프레임과 처리 길이 설정 → 선택 구간 불러오기.
2. 상자 전체를 둘러싸는 사각형을 드래그합니다.
3. 마스크 확인 / 수정 적용을 누릅니다. 초록색 영역이 선택된 물체입니다.
4. 빠진 물체 부분은 포함 클릭(+), 잘못 포함된 손/바닥은 제외 클릭(-) 후
   다시 수정 적용을 누릅니다. 클릭이 항상 원하는 결과를 보장하지는 않습니다.
5. 확인된 마스크로 추적 시작을 누릅니다. 결과 폴더 열기로 확인합니다.

처음에는 시작 프레임 11, 길이 3으로 원본 긴 영상을 시험해 보세요.
기본 길이는 30프레임, 최대 120프레임입니다. 30 FPS에서 30프레임은 약 1초입니다.
현재 CPU 실측: 최초 준비와 마스크 확인 약 26초, 3프레임 추적 약 23초.
실시간 프로그램이 아닙니다. 중단은 현재 모델 계산이 끝난 뒤 반영됩니다.
다른 구간을 처리하려면 구간을 불러오고 대상을 다시 지정합니다.
아직 추적 도중의 특정 프레임을 지정하여 재교정하는 UI는 없습니다.

결과: ../outputs/segmentation/날짜_시간_ID/
- initial_mask.png: 사용자가 확인한 첫 마스크
- masks/: 프레임별 예측 마스크
- selected_edges/: 마스크와 좁은 경계 띠 내 Canny 에지 후보
- tracking_preview.mp4: 마스크를 덧씌운 결과 영상
- session.json: 입력 영상, 원본 시작 프레임, 박스/클릭 이력, 모델, 처리 상태
- frames.jsonl: 프레임 대응, 마스크 면적, 처리 경과 시간
- input_frames/: 공식 입력 로더용 JPEG(품질 95)

이 단계는 목표 물체의 영상 영역을 선택합니다. 에지는 아직 물리적인 모서리의
정답이 아니며, 손에 가린 경계나 물체 표면 무늬가 포함될 수 있습니다.
SAM 2가 가려진 상자의 실제 모서리를 복원한다고 해석하면 안 됩니다.
마스크 밖 배경을 줄이는 효과와 모서리의 정확도는 별도로 평가해야 합니다.
캘리브레이션, 카메라 간 대응, 3D 융합과 자세 추정은 후속 단계입니다.

출처 및 자체 작성 범위
- Ravi et al., SAM 2: Segment Anything in Images and Videos (2024)
  https://arxiv.org/abs/2408.00714
- 공식 코드: https://github.com/facebookresearch/sam2
- 사용 버전: 2b90b9f5ceec907a1c18123530e92e794ad901a4
- 공식 예제: https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb
- 모델: 공식 SAM 2.1 Hiera Tiny. 원본 코드와 LICENSE는 third_party/sam2에 보존.
- CUDA 확장 후처리는 끄고 공식 비디오 predictor를 CPU에서 실행합니다.
- 자체 작성(AI 보조): Tkinter UI, 클릭 좌표 변환, 구간 추출, 공식 predictor 연결,
  에지 후보 선택, 파일 저장 및 실행 도구. SAM 모델 자체를 구현한 것이 아닙니다.
- Canny 구현: OpenCV. https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

다른 컴퓨터에서 설치 (프로젝트 루트에서 실행, Python 3.10 이상 필요):
  py -3 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r perception/segmentation\requirements.txt
  .venv\Scripts\python.exe perception/segmentation\setup_sam2.py
그 후 Start_Object_Tracking.cmd를 더블클릭합니다.
.venv는 다른 컴퓨터에 복사하여 사용하는 폴더가 아닙니다.
모델은 약 156 MB이며 PyTorch 설치에는 추가 다운로드와 디스크 공간이 필요합니다.
설치 도구는 공식 HTTPS 소스와 고정 커밋을 사용하고 모델 SHA256을 검증합니다.

카메라 녹화
capture/Start_Camera_UI.cmd에서 선택되어 연결된 카메라마다 AVI와 CSV를 별도 저장합니다.
1대 선택 시 영상 1개, 4대 선택 시 영상 4개가 같은 녹화 세션 폴더에 생깁니다.
연결을 중지한 뒤 사용하지 않을 슬롯을 '사용 안 함'으로 바꾸고 재연결하세요.
녹화 버튼 하나로 함께 시작하지만 하드웨어 동기 촬영은 아닙니다.

Team 7 - 1단계: 에지 확인

실행 (프로젝트 최상위 폴더에서 PowerShell):
  .\.venv\Scripts\python.exe perception/edges\check_edges.py

다른 임계값으로 실행:
  .\.venv\Scripts\python.exe perception/edges\check_edges.py --low 15 --high 45 --output outputs_sensitive

다른 영상으로 실행:
  .\.venv\Scripts\python.exe perception/edges\check_edges.py --video "영상의 전체 경로.mp4"

필요 패키지: numpy, opencv-python (현재 Python 환경에 설치되어 있음)

outputs/edges/edge_preview.mp4: 원본 / Canny 에지 / 초록색 에지 중첩 영상.
outputs/edges/threshold_comparison.jpg: 6개 시점에서 3가지 임계값 비교.
outputs/edges/frame_*: 선택한 시점의 원본 해상도 이미지.
outputs/edges/run_info.json: 입력 영상, 설정, 처리 프레임 수.

에지는 원본 해상도에서 계산하며, 미리보기만 축소합니다.
비교 이미지의 확대 영역은 이 영상에 맞춰 수동으로 지정한 표시 영역입니다.
판 자동 검출 결과가 아니며, 손·바닥·그림자의 경계도 검출됩니다.
카메라 보정 없이 원본 영상에 적용한 첫 기준 실험입니다.
출력 영상에는 오디오가 없고 재생 시간은 명목 FPS 기준입니다.
같은 출력 폴더로 재실행하면 생성 결과 파일을 덮어씁니다.
원본 영상은 수정하지 않습니다.


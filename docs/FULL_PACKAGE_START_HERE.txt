전체 인계 ZIP을 받은 팀원에게

이 파일이 들어 있는 Team7_Complete_Handoff.zip 하나를 전달받으면 됩니다.
이 ZIP은 기존 경량 Team7_Project_Handoff.zip보다 자료를 더 포함한 전체 인계본입니다.

포함: 실행 코드, 인계/노션용 문서, 원본 과제 자료, SAM 2 모델 가중치,
공식 소스와 라이선스, 기존 outputs 실험 결과, 프로젝트에 사용한 원본 영상 2개.
제외: 다른 PC에서 재사용할 수 없는 Python 가상환경, 캐시, 과거 문서 백업,
기존 ZIP, 프로젝트와 무관한 개인 사진/영상.

처음 실행
1. ZIP 전체를 쓰기 가능한 폴더에 압축 해제합니다. ZIP 안에서 직접 실행하지 마세요.
2. Windows에 Python 3.10 이상을 설치합니다.
3. 프로젝트 루트의 Setup_Project.cmd를 실행합니다.
   Python 라이브러리 설치에는 인터넷이 필요합니다. 모델과 소스는 동봉되어 있습니다.
4. 01_Camera_Recorder.cmd 또는 02_Object_Tracking.cmd를 더블클릭합니다.
5. SAM 화면은 동봉한 긴 샘플 영상을 기본으로 선택합니다.
   시작 프레임 11, 처리 길이 3 → 선택 구간 불러오기 → 상자를 드래그
   → 마스크 확인 → 필요하면 포함/제외 클릭 → 추적 시작 순서로 확인하세요.
   현재 CPU에서는 느리므로 짧은 구간부터 시험합니다.

자료 위치
- docs/PROJECT_CONTEXT.txt: 현재 상태·검증 근거·다음 작업·미확정 사항
- docs/NOTION_SUMMARY.txt: 노션에 붙여 넣을 프로젝트 요약
- docs/source/: 원본 제안서·과제 지침·초기 인계문서
- data/sample_videos/: 원본 긴 영상(530프레임)과 짧은 영상(50프레임)
- outputs/: 에지 실험, 녹화 검증, SAM 2 추적 결과
- models/ 및 third_party/: 공식 모델과 소스·라이선스
- FULL_PACKAGE_MANIFEST.json: 동봉 파일 크기와 SHA256

두 원본 영상은 동일 시각에 촬영된 다중 카메라 쌍으로 검증된 자료가 아닙니다.
동기화/융합 평가에는 별도 동시 촬영이 필요합니다.
결과 JSON에 남은 원본 PC의 절대 경로는 과거 실험 기록입니다. 새 PC에서는
data/sample_videos의 같은 파일을 선택하세요. 기존 JSON의 경로를 자동으로 바꾸지 않았습니다.
session.json의 tracking/error/cancelled 상태를 완료 결과로 해석하지 마세요.

현재 구현은 촬영 및 단일 영상의 대상 분할·에지 후보 추출까지입니다.
후속 개발은 조건별 품질 평가 → 고정 배치 보정·시간 정렬 → 관측 대응·3D 자세 융합입니다.
새 PC의 설치와 실제 4대 카메라 구동은 수령 환경에서 추가 확인해야 합니다.

문서 안내: README의 dist 경량 ZIP 및 FILE_ORGANIZATION의 용량 최적화 설명은
이전에 만든 경량 배포본에 대한 것입니다. 이번 전체 ZIP 구성은 이 안내가 기준입니다.

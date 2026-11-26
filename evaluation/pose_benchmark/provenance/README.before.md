# Team 7 다중 카메라 상자 추적

여러 RGB 카메라의 관측을 공통 3D 상자 구조와 위치·회전으로 결합하는 프로젝트입니다. 현재는 촬영 UI와 단일 영상의 물체 분할·에지 후보 추출을 구현했습니다. 카메라 간 대응과 3D 융합은 다음 단계입니다.

## 실행

- `01_Camera_Recorder.cmd`: 최대 4대 미리보기 및 한 버튼 녹화.
- `02_Object_Tracking.cmd`: 박스·포함/제외 클릭으로 물체 지정 후 SAM 2 추적.
- `Setup_Project.cmd`: 다른 Windows PC에서 처음 한 번 실행. Python 3.10 이상과 인터넷 필요.

현재 노트북은 설치 완료 상태입니다. 다른 PC에는 `.venv`를 복사하지 말고 설치 도구를 실행하세요.

## 문서

- [노션용 요약](docs/NOTION_SUMMARY.txt): 목표, 작업 내용, 설계 변경, 향후 계획과 출처.
- [현재 상태 및 인계](docs/PROJECT_CONTEXT.txt): 검증 근거, 다음 작업의 완료 기준, 미확정 사항.
- [파일 정리 내역](docs/FILE_ORGANIZATION.txt): 보존 및 전달 정책.
- 기능별 사용법: [촬영](camera/README.txt), [물체 지정](segmentation/README.txt).

| 폴더 | 내용 |
|---|---|
| camera | 촬영 UI·기록·검증 |
| segmentation | SAM 2 연결·대상 지정 UI·설치 도구 |
| edges | Canny/LSD 기준 실험 |
| docs | 최신 인계·연구 자료·원본 과제 자료·과거 문서 |
| models / third_party | 모델 가중치 / 공식 소스 및 라이선스 |
| recordings / outputs | 사용자 녹화 / 검증·실험 결과 |
| tools / dist | 전달 ZIP 생성 도구 / 전달 ZIP |

SAM 2는 대상 영역을 선택하고 Canny는 그 영역의 에지 후보를 추출합니다. 마스크 경계와 물리적인 모서리는 다르며, 카메라 간 동일 모서리 대응은 아직 구현하지 않았습니다.

실제 2대 녹화와 짧은 구간의 SAM 2 추적을 검증했습니다. 실제 4대 촬영, 장기 추적 정확도, 3D 자세 정확도는 미검증입니다. CPU 추적은 실시간이 아니며 기본 30프레임, 최대 120프레임 구간을 처리합니다.

전달 ZIP 재생성: `.venv\Scripts\python.exe tools\package_project.py`

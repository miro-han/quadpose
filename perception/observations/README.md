# 실제 프레임에서 기하 관측 만들기

프로젝트 루트에서 실행한다. 각 실행에는 새 출력 디렉터리를 사용한다.

```powershell
.\.venv\Scripts\python.exe perception/observations/adapter.py --video data/sample_videos/WIN_20261124_20_05_34_Pro.mp4 --session outputs/segmentation/20261124_215508_e1cb28 --lsd outputs/edges/lsd_baseline --frame 11 --camera-id camera_1 --output perception/observations/replay11
.\.venv\Scripts\python.exe perception/observations/demo_single.py --output perception/observations/replay_illustrative
.\.venv\Scripts\python.exe -m unittest perception.observations.test_adapter -v
```

원본 프레임 인덱스로 LSD JSONL과 SAM JSONL을 찾아 해당 로컬 마스크와 Canny PNG를 읽는다. 원본 해상도에서 마스크 경계를 3 px 확장하고 선분 위 21개 표본의 ROI 포함률과 에지 지지율을 검사한다. 임계값은 이 연결을 확인하기 위한 초기값이다. 참조 가능한 원본 경로와 영상 이름, 재인코딩한 해당 프레임과 보관 JPEG의 일치 여부를 확인한다. 과거 LSD 파일에는 독립 영상 해시가 없다. 다른 세션에 똑같은 이름의 영상이 있으면 새 녹화에서 해당 영상으로 LSD를 다시 생성해야 한다.

보정 파일을 주면 Brown 왜곡을 제거하고 원래 K를 사용하는 픽셀로 변환한다. 보정 전 원본 선을 표본화해 보정 후 직선과의 최대 편차가 2 px를 넘으면 제외한다. 가까운 유한 선분 사이에서 각도와 연장 거리 조건을 만족하는 교점만 남기고 4 px 내 중복은 합친다. 관측 JSON은 원래 LSD ID, 코너 생성에 기여한 선분 쌍, 제거 이유, 소스 프레임과 좌표계를 보존한다.

**실제 원본 11번 프레임**에서 선분 97개 중 37개와 교점 7개가 남았다. 보정과 초기값이 없는 실행은 물리 ID를 미할당으로 기록했다. [관측 JSON](sample11/observations.json) · [오버레이](sample11/overlay.jpg)

**시각화 예시**에서는 같은 영상에서 직접 지정한 여섯 코너, 가정한 초점거리 1400 px, 왜곡 0, 한 변 0.12 m 모델로 초기 투영을 만들었다. 모델의 보이는 점과 선에만 배타 할당하여 5개 점과 8개 선에 ID를 부여했다. 영상에 물리 ID와 px 잔차를 표시했다. 일부 코너 잔차는 36.9 px로 크다. 실제 상자 치수·카메라 보정·기준 자세가 없고 초기화에 같은 프레임을 썼으므로 이 파일을 추적 정확도나 실측 자세 근거로 사용할 수 없다. [가정과 클릭 좌표](illustrative11/prior.json) · [대응 및 잔차](illustrative11/frame/overlay.jpg)

### 실물 자료 연결

각 카메라에서 위 명령에 `--rig rig.json`을 추가한다. 한 시점의 투영 대응도 확인하려면 `--prior prior.json`을 추가한다. prior는 실측 물체 치수 size_m과 4×4 T_world_object를 포함한다. 전처리 결과는 원본 프레임 번호·카메라 장치·보정 ID로 촬영 자료에 연결한다.

공통 점멸 신호로 검증한 alignment.json을 생성한 후 `python -m perception.observations.fuse_group --config group.json --output pose.json`을 실행한다. group.json의 예시는 아래와 같다. 각 파일 경로는 설정 파일 기준이다.

```json
{"rig":"rig.json","alignment":"alignment.json","group_id":12,"camera_order":["front","side","top","diagonal"],"prior":"prior.json","observations":{"front":"front/observations.json","side":"side/observations.json","top":"top/observations.json","diagonal":"diagonal/observations.json"},"method":"reliable"}
```

시간 미검증 입력, 장치·보정·원본 동영상·프레임 불일치는 거부한다. 누락 슬롯은 빈 검출 배열로 전달한다. 실시간 미리보기나 무보정 정성 검토는 어댑터 단독으로 가능하다. 그룹 결합은 합성 기하·메타데이터로 API 경계를 검증했다. 동봉된 두 원본 영상을 묶어 실행한 실험은 없다.

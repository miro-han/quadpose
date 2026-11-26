# Stage 0 · 저장소 및 데이터 계약 감사

2026-11-26 `research/neural-network`의 공개 인계 기준 `1f8d65d`를 조사했다. 이 문서는 기존 모듈과 합성 산출물을 대상으로 한다. 새 브랜치의 병행 구현과 무관하게, 학습 전 입력 경계와 비교 기준에 대한 감사를 남긴다.

## 기존 코드의 재사용 경계

| 영역 | 실제 연결과 검증 | 신경망 연구에서 사용할 경계 |
| --- | --- | --- |
| `capture/`, `rig/` | 장치별 영상·CSV, 캘리브레이션·플래시 검증 시간 정렬. 실물 그룹 실행기는 검증된 시간 정렬, 장치·슬롯·카메라·캘리브레이션·프레임 일치를 검사한다. | 보정 및 동기화 입출력을 고정한다. 세션 ID는 분할과 추적에 사용한다. [실물 그룹 실행기](../../../perception/observations/fuse_group.py) · [리그 절차](../../../rig/README.md) |
| `perception/` | SAM 마스크와 Canny/LSD 후보, 교차점 기반 코너, 왜곡 제거 픽셀. SAM 2.1 체크포인트와 소스 마커 또는 실행 결과가 있으면 기존 단일 영상 처리를 재사용할 수 있다. | M02 영상 특징 교체의 기준선, M03의 원본 코너·선분 후보. 현재 모델 패킷은 개별 원본 영상과 관측별 정답을 싣지 않는다. [관측 어댑터](../../../perception/observations/adapter.py) · [분할 세션](../../../perception/segmentation/sam_session.py) |
| `pose/fusion.py` | Pose/Camera/BoxModel/Detections, 가시성·투영, 점과 선분의 배정, Huber 기반 수치 적합, All/Best/Reliable. Oracle 대응은 별도 함수다. | M04의 고정 대응 NumPy solver, M05의 공통 후보 생성, M06의 기하 초기값. 미분 가능한 solver는 새 구현이 필요하다. [기하 융합](../../../pose/fusion.py) |
| `pose/hypotheses.py`, `pose/refinement.py` | 3-of-4 행 및 고정 홀드아웃 열, 랭크·정보량, 유한 에지 검증, 재배정과 bounded alternating. `segment_pairs`는 `(모델 에지 ID, 원본 관측 인덱스)` 쌍을 반환한다. | 학습 가설 순위를 휴리스틱 Cross 및 Cross+Verify와 비교한다. 반환된 대응이 유지되는지를 함께 집계한다. [가설 평가](../../../pose/hypotheses.py) · [정제](../../../pose/refinement.py) |
| `pose/recovery.py` | 시간 순서를 강제하고 `measured`, `predicted`, `lost`를 구분한다. CV, 반경 확대, 다중 초기값, IPPE 기반 영상 PnP, 부분 카메라 관측 검사가 있다. | M10 및 M11 비교에서 기존 정책을 유지한다. 예측 자세를 측정 정답이나 유효 대응으로 승격하지 않는다. [복구](../../../pose/recovery.py) |
| `pose/contracts.py`, `pose/check_dataset.py` | `Evidence` → 이름 있는 8개 패킷 배열. NPZ 입력은 allow-list로 분리하며 `target_*` 입력을 거부한다. | 학습과 평가의 경계. 상대 순서를 바꿀 경우 카메라 열과 가설 행, 멤버십·유효 마스크도 함께 순열한다. [패킷 본체](../../../pose/contracts.py) · [검사기](../../../pose/check_dataset.py) |
| `pose/synthetic_benchmark.py`, `evaluation/pose_benchmark/` | Phase II 데이터 생성 및 독립 자율 롤아웃, Phase IV 결함·배정·선택·복구 및 지연 평가. | Stage 초기 소형 모델에는 Phase II 패킷을 이용한다. Phase IV 시나리오 비교는 축·선택기·선행 자세를 명시한 별도 자율 실험으로 본다. [데이터 생성기](../../../pose/synthetic_benchmark.py) · [평가 프로토콜](../../../evaluation/pose_benchmark/BENCHMARK_PROTOCOL.md) |

실물 그룹 실행기는 `analyze_frame`의 Phase II `Evidence`에서 패킷을 만들며, 이후 골라진 방법에 따라 Phase IV `evaluate_hypotheses`를 별도로 실행한다. 따라서 패킷의 후보 점수와 최종 결과의 새로 정제된 후보를 사용하려면 버전이 지정된 변환과 대응 표가 필요하다. 첫 Stage의 고정 후보 실험에서는 그 두 후보 집합을 섞지 않는다. 실제 패킷의 `Camera.image_size`는 필요하고, `T_camera_world`를 읽어 세계 좌표의 카메라 중심을 계산한다. 모델은 캘리브레이션의 Z-up, 미터, `undistorted_K_pixels` 규약을 유지한다. [그룹 코드](../../../perception/observations/fuse_group.py) · [스키마 설명](../../../pose/NEURAL_CONTRACTS.md)

## 확인한 데이터 크기와 순서

`check(Path('pose/results'))`를 기존 3 GiB CPU 한도에서 실행해 **3,072프레임**의 고유·동일 순서 프레임 키, 유효 마스크와 궤적 분리를 확인했다. 가설·복구 회귀 테스트 **5/5**도 통과했다. [입력 NPZ](../../../pose/results/tokens.npz) · [정답 NPZ](../../../pose/results/targets.npz) · [합성 매니페스트](../../../pose/results/manifest.json)

| 분할 | 궤적 수 | 프레임 수 |
| --- | ---: | ---: |
| Train | 24 | 1,152 |
| Validation | 16 | 768 |
| Test | 24 | 1,152 |
| 전체 | 64 | 3,072 |

| 패킷 또는 타깃 | 프레임당 크기와 의미 |
| --- | --- |
| `camera_tokens` | 4×21 float32. 카메라 위치/자세, 해상도 정규화 내부 파라미터, 코너·선 수, 단안 유효 및 전체 적합 잔차 |
| `hypothesis_tokens` | 4×15 float32. 세계 병진, 행렬 회전, 가설 유효, 적합 RMS, 로그 조건수 |
| `cross_view_residual_px`, `subset_membership_mask`, `residual_valid_mask`, `view_support` | 각 4×4. 행은 가설, 열은 카메라. 0 채움 결측은 `residual_valid_mask`와 함께 처리 |
| `camera_has_matches`, `hypothesis_valid` | 각 길이 4. 가설이 모두 무효일 수 있음 |
| `target_packet_hypothesis_error`, `target_packet_hypothesis_known` | 각각 4×2와 길이 4. 위치 mm·회전 °, 레이블 사용 가능 여부 |
| `target_reliable_mask`, `target_valid` | 결함을 주입하지 않은 카메라 및 적어도 3대가 정상이라는 합성 정책 대리값 |

표준 가설 행은 `T123,T124,T134,T234`이며 제외 카메라 인덱스는 0-based `[3,2,1,0]`이다. 레거시 `residual_matrix` 등의 제외 카메라 원순서는 반대다. 여덟 이름에 한해 `{name: archive['packet_' + name]}`으로 읽는다. 12,288개의 잠재적인 가설 레이블 중 **10,131개**가 알려져 있고, **330프레임**은 가설 레이블이 모두 무효다. 훈련·평가에서 이 프레임을 버리면 후보 선택의 coverage와 실제 추적 실패를 함께 볼 수 없다. 순위 손실을 가리는 것과 전체 평가 분모를 정하는 것을 분리한다. [입력 allow-list](../../../pose/check_dataset.py) · [열과 행 정의](../../../pose/NEURAL_CONTRACTS.md)

## 종속성과 누출 오해를 피하는 방법

- 위 NPZ는 고정된 `reliable` 추적의 이전 **추정** 자세에서 생성된 패킷이다. 과거 추적기가 시뮬레이션의 GT에 매 프레임 접근하지는 않지만, 시작 자세는 프레임 0의 실제 자세에 잡음을 더한 초기화다. 후보의 오프라인 선택 성능을 같은 모델의 새로운 자율 궤적 결과로 해석할 수 없다. 자율 평가에서는 각 모델이 자신의 마지막 유효 측정과 예측으로 다음 후보를 생성한다. [생성 분기](../../../pose/synthetic_benchmark.py) · [기존 시험 요약](../../../pose/README.md)
- Split, `frame_key`, `trial_key`, 파일 순서와 시뮬레이터 이름은 로더의 선택·결합용이다. 정상/결함·Oracle/중간 GT 정보, `target_*`, 참고 baseline의 점수는 추론 입력으로 제외한다. 스케일러와 threshold는 train/validation에서만 결정하고, 모든 순열에 동일한 값이 들어오는지 테스트한다. 카메라별 intrinsics 및 extrinsics는 추론 가능한 물리 특징이더라도 챌린지와 카메라 조합을 별도로 바꿔 memorization을 점검한다. [계약](../../../pose/NEURAL_CONTRACTS.md)
- `target_reliable_mask`의 양성과 음성은 센서 결함 주입 여부에 가깝다. 결함 시점에서도 일부 정확한 관측이 존재한다. 3대 이상 조건은 합성 정책을 평가하기 위한 정의이며, 최종 기하에서는 1–2대가 충분한 독립 정보량을 제공할 수도 있다. 신뢰도 점수, 위치/회전 오차 기반 오수용, 기하 관측 가능성 및 시점별 대응 정확도를 분리한다. [복구의 측정 기준](../../../evaluation/pose_benchmark/HYPOTHESES_AND_RECOVERY.md)
- Git 인계에는 합성 NPZ와 집계 CSV가 있으며 `git ls-files`에서 원본 녹화 영상, SAM 가중치, 현장 리그·독립 자세 정답이 학습 자료로 추적되지 않았다. 조사 PC에 가중치와 무관한 로컬 산출물이 있을 수 있지만, 학습에 쓸 수 있는 동기 검증 영상과 독립 현장 기준 자세는 이 감사에서 확인하지 않았다. 외부 카메라·장면 도메인에 대한 성능 주장은 별도 시험 자료를 필요로 한다. [Git 제외 규칙](../../../.gitignore) · [실물 시험](../../../docs/guide/15_PHYSICAL_VALIDATION.md)

## 첫 실험을 위한 비교 기준

M04/M05는 고정 패킷의 MLP와 가설별 후보 선택을 먼저 검사할 수 있다. 모델과 기존 All/Best/Reliable 지표를 같은 종류의 후보와 동일 `frame_key`로 맞춘다. 그다음 각자의 이전 추정으로 후보를 다시 생성하는 자율 시험을 실행하고 강한 Phase IV Cross+Verify를 대조군으로 추가한다. 논문식 독립 시험 점수는 최초 확정 후 한 번 보고하고, GT 사용이 다른 Oracle과 현행 방법을 같은 기준으로 점수화하지 않는다. 실물 영상이 확보되면 수동/독립 기준의 작은 holdout과, 광량·실시간 히스테리시스·카메라 재분리 실패를 포함한 현장 평가를 진행한다. [방법 우선순위](METHODOLOGY_SURVEY.md) · [Phase IV 독립 평가 산출물](../../../evaluation/pose_benchmark/results/primary/summary.csv)

# D0 공통 합성 데이터 계약

D0는 학습 경로와 기하 제어군을 같은 관측으로 비교하기 위한 소규모 Layout A 데이터다. Z-up 오른손계에서 카메라 중심은 정면 `(0,−0.65,0)`, 측면 `(+0.65,0,0)`, 상단 `(0,0,+0.65)`, 대각 `(+0.65,−0.65,+0.65)` m다. `K`는 각 카메라에서 동일한 960×720 핀홀 영상 기준 `fx=fy=900`, `cx=480`, `cy=360`이며 직육면체 치수 190×120×85 mm를 알고 있다고 가정한다. 실제 이미지나 센서 노출을 모사한 자료는 아니다.

8개 조건은 정상, 잘못된 대응 후보, 불균등한 검출 품질, 영상 평면 가림, 카메라 유실, 외부 보정 오류, 시간 불일치, 복합 결함이다. 결함 시각과 카메라가 궤적별로 결정되며 시드·모션·검출 잡음은 분할마다 독립이다. 프레임을 섞어 분할하지 않는다. 기본 프레임 수는 궤적당 12개다.

| 분할 | 시작 시드 | 모션 분포 | 궤적 | 프레임 |
| --- | ---: | --- | ---: | ---: |
| 학습 `train` | 72200 | harmonic sweep | 24 | 288 |
| 개발 `dev` | 74100 | phase-shifted arc | 16 | 192 |
| 독립 `lockbox` | 76400 | orbit with reversal | 16 | 192 |

## 파일 및 API

각 분할마다 다음 파일이 생성된다. 숫자 배열은 NumPy NPZ로 저장하고 pickle을 사용하지 않는다. 학습기가 주는 입력만 `inputs.npz`에서 읽으며 `frame_key`는 파일을 결합할 때만 사용한다. 행 인덱스, 시드, 결함 이름을 학습 특징으로 사용하지 않는다.

| 파일 | 내용 |
| --- | --- |
| `inputs.npz` | [기존 추론 계약](../pose/NEURAL_CONTRACTS.md)의 카메라·가설 토큰과 4×4 잔차·마스크에 `prior_rotation`, `prior_translation`, `box_size_m`을 추가했다. 가설 순서는 T123·T124·T134·T234다. |
| `observations.npz` | `points[F,4,18,2]`, `segments[F,4,28,2,2]`, `point_mask`, `segment_mask`, 카메라 중심·`K`·`Rcw`, 영상 크기 및 합성 `time_s`다. 미측정 배열은 0으로 채우고 해당 마스크에서 제외한다. `baseline_best_hypothesis`와 `candidate_cost`는 정답을 사용하지 않는 기존 내부 RMS 기반 진단이다. 후보 유효 마스크가 꺼진 비용 0은 의미가 없다. |
| `index.npz` | `frame_key`, `trial_key`, 분할, 궤적 내 프레임 번호다. |
| `targets.npz` | 별도 정답 자세, 코너·에지 물리 ID, 명목상 기하 가시성, 가림으로 잃은 에지, 결함·신뢰 마스크, 가설별 위치(mm)·회전(°) 오차와 최고 후보 라벨이다. 물리 ID `−1`은 클러터, `−2`는 패딩이다. |
| `target_provenance.json` | 채점과 데이터 감사에서만 읽는 궤적별 결함 종류·카메라·활성 창·강도다. |
| `manifest.json` | 스키마·카메라·시드·모션·배열 파일 SHA-256이다. |

`learning.datasets.iter_trials(split)`은 메모리에서 재현한 `Trial(id, seed, regime, rig, box, initial, frames)`을 제공한다. 프레임의 `detections`는 ​​정답이 제거된 입력이다. `truth`, `labels`, `fault_mask`, `corner_visible`, `edge_visible`, `occlusion_lost`는 평가용이다. 입력 패킷 생성은 기존 추정 대응 `analyze_frame`으로 시작하고 `choose_result(..., "reliable")`의 유효 출력만 자기 다음 prior로 사용한다. 정답은 해당 패킷 생성에 들어가지 않는다. 최초 prior는 합성 첫 자세에 잡음을 준 외부 초기화다.

`load_split(root, "dev")`는 기본값에서 정답 파일을 읽지 않는다. 학습 손실이나 별도 채점기에서만 `with_targets=True`를 지정한다. 프레임 키의 일대일 일치, 배열 길이와 입력 필드 허용 목록을 확인한다. `target_reliable_mask`는 합성 결함을 주입했는지에 대한 라벨이다. 에지의 유효성을 직접 측정한 신뢰도와 다를 수 있다. `target_valid`는 미주입 카메라가 3대 이상인 정책 라벨이고, 실제로 허용 오차 내 후보가 존재하는지와 구분한다. `target_best_hypothesis`는 유효 후보 중 `(위치 mm/20)^2 + (회전 °/8)^2`가 가장 작은 행이며, 유효 후보가 없으면 `−1`이다.

```powershell
python -B -m evaluation.resource_limits --threads 1 --memory-gib 3 learning.datasets --output outputs/d0
python -B -m evaluation.resource_limits --threads 1 --memory-gib 3 learning.test_datasets
```

프로젝트 루트에서 NumPy CPU 환경으로 실행한다. 기본 `outputs/d0`의 배열은 재생성 가능하며 Git에 추가하지 않는다. [생성본의 검증 요약](results/d0_manifest.json) · [독립 기하 및 Oracle 제어군](D0_BASELINES.md)

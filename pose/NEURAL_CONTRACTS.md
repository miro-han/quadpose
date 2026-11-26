# 신뢰도 학습용 계약

`contracts.inference_packet(evidence)`는 추정 대응으로 만든 Evidence를 입력받는다. 관측 세트는 오른손 Z-up 세계, 미터, undistorted-K 픽셀이다. 실물 그룹 실행기가 쓰는 이름 있는 배열은 아래와 같다.

| 입력 | 크기 | 해석 |
|---|---|---|
| camera_tokens | 4 × C | 카메라 중심·회전, 정규화 내부 파라미터, 코너·선 개수, 단안 유효 여부와 전체 가설 잔차. 정확한 열은 CAMERA_FIELDS |
| hypothesis_tokens | 4 × H | 공통 세계에서의 병진·회전 행렬, 유효 여부, 지지 잔차와 조건수. 행 순서 T123,T124,T134,T234. 정확한 열은 HYPOTHESIS_FIELDS |
| cross_view_residual_px | 4 × 4 | 각 가설을 각 카메라의 대응에 투영한 잔차. 사용 불가 셀의 0은 반드시 residual_valid_mask로 가린다 |
| subset_membership_mask | 4 × 4 | 참은 해당 카메라가 가설의 지정 부분집합에 포함된다는 뜻. 거짓인 열이 제외 시점이다 |
| residual_valid_mask / view_support | 각각 4 × 4 | 수치 측정 가능 여부와 실제 지지 여부. 미지원과 미측정을 구분한다 |
| camera_has_matches / hypothesis_valid | 각각 4 | 할당이 한 개라도 있는 카메라와 관측 가능한 유효 자세 가설 |

GT는 inference_packet 인자가 아니다. 미래의 Deep Sets에서는 카메라 토큰을 공유 인코더에 넣고 유효 토큰을 집계한다. cross-attention에서는 가설 토큰을 query, 카메라 토큰을 key/value로 사용하며 membership과 residual은 관계별 입력이 된다. 순서를 바꿀 때 모든 관련 축과 마스크도 함께 순열한다.

### 실험 파일과 라벨

배포된 tokens.npz 안에서 `packet_` 접두사가 붙은 필드는 위 계약을 그대로 사용한다. `target_packet_hypothesis_error`와 `target_packet_hypothesis_known`도 같은 가설 순서다. 학습용 로더는 지정한 입력 필드만 선택하고 GT 로더는 손실 계산 및 평가에서만 불러온다.

합성 벤치마크의 tokens.npz는 카메라별 진단 벡터, LOO 자세·잔차·지지 여부와 연결 키를 저장한다. **파일의 가설 행 순서는 제외한 카메라 1,2,3,4 순서**다. 위 실물 계약 순서로 변환할 때는 행을 [3,2,1,0]으로 재배열하고 열은 유지한다. 합성 원시 잔차의 무한대는 유효성 마스크로 변환한다. 순서에 종속되는 원시 특징 열을 그대로 집합 모델에 넣으면 순열 특성을 잃으므로 이름 있는 관계 행렬을 사용한다.

targets.npz는 동일 frame_key를 가진 평가 전용 파일이다. 카메라별 결함 주입 여부에서 유도한 reliability와 known mask, 실제 T_world_object와 가설별 위치(mm)·회전(°) 오차 및 유효 라벨 마스크가 들어간다. 결함 주입은 그 시점이 부정확할 가능성의 대리 라벨이다. 실측 정답을 확보하지 못한 프레임의 라벨은 unknown으로 둔다. trial_key, split, frame_key 등 결합 메타데이터와 시뮬레이터의 결함 이름은 신경망 특징에서 제외한다. 학습·검증·시험은 궤적/세션 단위로 분리한다. 이 단계의 산출물은 스키마와 합성 자료이며 모델 학습 실행은 없다.

# D0 연구 모델 목록

모든 행은 연구 후보이며 Studio에서 선택하는 추정기는 기존 기하 경로를 사용한다. 데이터 버전 D0는 [생성 명세](DATASETS.md)와 [SHA 목록](results/d0_manifest.json)으로 식별한다.

| 모델 ID | 구조 및 입력 버전 | 시드·학습 설정 | 구현 커밋 | 가중치·해시와 평가 |
|---|---|---|---|---|
| learned_logistic / learned_mlp | 물리 에지 후보 12열, logistic 또는 tanh 16 | [NumPy 모델과 초기화](track_a_models.py), Adam 450회 | `33f9a86` | [소형 NPZ](../experiments/neural/track_a_results/models.npz) · [배열 해시·파라미터](../experiments/neural/track_a_results/training.json) · [프로토콜 및 결과](../experiments/neural/TRACK_A.md) |
| rank_linear / rank_mlp | 14열 유효 후보의 pairwise 순위, 선형 또는 tanh 16 | [NumPy 모델과 초기화](track_a_models.py), Adam 450회 | `33f9a86` | [가설 검증과 진단](../experiments/neural/TRACK_A.md) · [배열 해시와 벽시계](../experiments/neural/track_a_results/training.json) |
| learned_huber | 위에서 선택한 에지 분류기 재사용 | 별도 모델 훈련 없음 | `33f9a86` | [가중치 및 결과](../experiments/neural/TRACK_A.md) |
| supervised / distilled / self_supervised | quadpose-heldout-translation-v0, tanh 은닉 24, 병진 보정 3축 | 시드 907, Adam 0.006, full-batch 100회 | `b0f1afb`, D0 생성기 `2b00497` | [로컬 체크포인트 해시·손실·평가](results/residual_v1.json) |
| behavior_cloning / reinforce | quadpose-policy-v0, tanh 은닉 24, 다섯 개 정책 행동 | 시드 83·41, Adam 0.01로 BC 70회와 순차 갱신 3회 | `bf6d46c` | [시드 83](results/policy_lockbox_83.json) · [시드 41](results/policy_lockbox_41.json) |

재실행 명령과 로컬 체크포인트 경로는 [연구 안내](../docs/research/neural/RESEARCH_OVERVIEW.md)를 따른다. 현재 모델들의 점수에는 검증된 확률 보정이 없다. Brier와 risk–coverage를 내려면 새 독립 평가 및 사전 정의된 사건 단위가 필요하다.

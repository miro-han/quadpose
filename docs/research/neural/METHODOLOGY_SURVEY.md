# Stage 0 · 신경망 방법 조사

Quadpose의 직접 목표는 보정된 네 영상의 특징을 알려진 직육면체에 대응시키고 공통 세계 좌표계에서 자세와 상태를 반환하는 것이다. Stage 0에서는 M01–M12의 투입 데이터와 검증 방법을 비교했다. 아래의 개선 가능성과 일정은 실측치가 아닌 이 코드베이스를 기준으로 한 연구 판단이다. 공개 논문의 데이터, 하드웨어, 정확도는 Quadpose 실험 수치에 포함하지 않았다. 현행 스키마와 실험 기준은 별도 문서에 기록했다. [코드 감사](CODEBASE_AUDIT.md) · [자원 기록](RESOURCE_REPORT.md) · [문헌 목록](REFERENCES.md)

## 방법별 작동 원리

### M01 · 직접 자세 회귀

영상이나 다중 시점 토큰에서 병진과 회전을 바로 예측한다. PoseCNN은 영상 기반 자세 학습의 선례다. 현재 자료에서는 카메라 토큰의 Deep Sets 풀링에 작은 병진·회전 헤드를 붙여 회귀 기준선을 만들 수 있다. 픽셀을 입력하는 모델은 별도 영상 자료가 필요하다. 회귀값만으로는 어느 물리 에지가 선택됐는지 알기 어려우므로 후단에서 기하 지지와 관측 가능성을 따로 검증한다. [PoseCNN](https://arxiv.org/abs/1711.00199) · [Deep Sets](https://papers.nips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)

### M02 · 학습 기반 코너·에지 검출

강체 모델의 코너 위치나 선분을 픽셀 공간에서 검출한다. PVNet은 픽셀별 투표 방향으로 가려진 키포인트의 불확실성까지 추정하고 PnP에 사용했다. SOLD²는 선분 검출과 서술자를 자기지도 방식으로 학습했다. Quadpose에서는 코너 heatmap·선분 검출의 재현율과 위치 오차를 먼저 비교하고, 물리 에지 ID와 자세 지표까지 추적한다. 영상 특징의 반복 검출성과 물체의 특정 모델 ID는 다른 라벨이다. [PVNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html) · [SOLD²](https://openaccess.thecvf.com/content/CVPR2021/html/Pautrat_SOLD2_Self-Supervised_Occlusion-Aware_Line_Description_and_Detection_CVPR_2021_paper.html)

### M03 · 물리 에지 대응 학습

관측 선분과 12개 모델 에지 후보를 그래프로 구성한다. 선 방향, 유한 중첩, 수직 거리, 인접 코너와 시점 정보를 특징으로 사용하고, 모델 에지 점수와 미할당 점수를 학습한다. SuperGlue는 점 특징의 두 집합을 문맥과 부분 배정으로 연결하는 설계 참고다. 같은 물리 에지가 여러 조각으로 관측될 수 있어, 점 대응처럼 모든 관측과 모델 에지를 일대일로 강제하는 규칙은 부적절하다. [SuperGlue](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) · [SOLD²](https://openaccess.thecvf.com/content/CVPR2021/html/Pautrat_SOLD2_Self-Supervised_Occlusion-Aware_Line_Description_and_Detection_CVPR_2021_paper.html)

### M04 · 관측 가중치 학습

현재 재투영 적합에 $\min_T\sum_i w_i\rho(r_i(T))$를 적용한다. 우선 카메라 단위 소형 MLP나 Deep Sets로 센서 신뢰도에 대한 감독 손실을 시험한 뒤 NumPy solver에 가중치를 주입한다. 표본별 `healthy`는 시뮬레이션 결함 주입의 대리값이다. 기하적으로 맞는 관측까지 버릴 수 있으므로 에지 잔차·최종 자세와 가중치가 적용된 랭크를 같이 검사한다. 유효 마스크 이후 양의 가중치와 정규화를 적용하고, 관측 부재는 낮은 가중치의 관측으로 보충하지 않는다. EPro-PnP는 가중 대응과 자세 분포를 같이 학습하는 확장 참고다. [EPro-PnP](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html)

### M05 · 가설 순위 학습

이미 계산된 유효한 세 시점 가설 사이에서 선택한다. 소형 MLP 순위 기준선과, 카메라를 key/value 및 가설을 query로 두고 관계 잔차·멤버십을 주입하는 cross-attention을 비교한다. `target_packet_hypothesis_error`의 mm·° 오차를 공통 판정 스케일로 나눠 점수 라벨로 사용하거나 pairwise 손실을 쓸 수 있다. 유효 후보가 없으면 상실 처리하고 후보가 비슷할 때는 순위 정확도뿐 아니라 최적 후보 대비 regret을 본다. DSAC은 미분 가능한 확률적 선택의 참고이며, 고정 후보 M05는 그 알고리즘을 재현하지 않는다. [DSAC](https://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html) · [Set Transformer](https://proceedings.mlr.press/v97/lee19d)

### M06 · 잔차 자세 학습

유효한 기하 자세 $T_g$에서 짧은 증분 $T'=\exp(\widehat{\Delta\xi})T_g$만 예측한다. 학습 목표는 채택된 GT 방향·좌표 규약에서 $\log(T_{gt}T_g^{-1})$이며 예측 크기와 유효 온셋 범위를 통제한다. 초기 대응이 잘못된 상태에서 큰 증분을 회귀시키기 전에 잔차 개선과 독립 시점 지지를 확인한다. DeepIM은 렌더링과 영상의 차이로 반복 상대 자세를 추정하지만, 토큰 기반 Quadpose 실험에는 렌더링 단계를 추가하지 않아도 된다. [DeepIM](https://openaccess.thecvf.com/content_ECCV_2018/html/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.html)

### M07 · 미분 가능한 기하 최적화

가중 재투영 문제를 PyTorch로 옮겨 제한된 IRLS/Gauss–Newton 반복을 펼치거나 정상점에서 암시적 미분한다. BPnP와 Theseus가 두 방향의 참고다. NumPy solver, 가시성, Greedy/Hungarian 배정, 랭크 거부 및 재초기화는 이 저장소에서 자동미분 그래프로 연결되지 않는다. 고정 대응부터 시작해 동일 입력의 수치 비교 및 유한차분 gradient 검증을 하고, 그다음 손실과 시간을 비교한다. [BPnP](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_End-to-End_Learnable_Geometric_Vision_by_Backpropagating_PnP_Optimization_CVPR_2020_paper.pdf) · [Theseus](https://proceedings.neurips.cc/paper_files/paper/2022/hash/185969291540b3cd86e70c51e8af5d08-Abstract-Conference.html)

### M08 · 자기지도 기하 학습

보정된 다중 영상에서 홀드아웃 재투영, 시간·시점 교환 일관성, 알려진 상자 크기나 실루엣을 감독 신호로 쓴다. Self6D는 실물 레이블 대신 렌더링 기반 기하 제약을 사용했으며, 대상 도메인 훈련에서 깊이 정보를 사용한다. Quadpose의 RGB 기반 설계는 그 데이터 가정을 그대로 사용하기 어렵다. 엣지 일관성만으로 손 경계와 상자 에지를 구분하기 어려우므로, 서로 다른 시점에서 맞는다는 조건과 별도의 레이블 평가가 필요하다. [Self6D](https://arxiv.org/abs/2004.06468)

### M09 · 교사-학생

우수한 시점·수동 라벨·복구가 확인된 기하 추정만 교사 후보로 삼고, 다양하게 누락시킨 입력에서 학생의 신뢰도 및 가설 순위를 학습한다. Mean Teacher의 가중치 EMA 일관성은 한 방식이며, 동결된 기하 교사 증류도 따로 비교할 수 있다. 정답 대응 Oracle은 개발 시 지도 교사 또는 상한 진단으로 사용하고 추론 경로와 실제 시험 점수에서 분리한다. 교사 오류가 학생에 고착되는 비율을 별도 집계한다. [Mean Teacher](https://papers.nips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)

### M10 · 학습 기반 시간 필터

마지막 측정 이력, $\Delta t$, 가려진 시점, 기하 혁신(innovation)으로 속도나 필터 이득을 예측한다. 단순 CV/칼만형 기준선과 혁신에 따른 작은 이득 헤드를 비교하고, 긴 순환 모델은 측정/예측/상실 구분을 유지한 채 평가한다. KalmanNet은 부분적으로 알려진 동역학에서 필터 이득을 학습하는 참고다. 미래 프레임을 본 평활 결과와 인과적 추적 점수를 분리한다. [KalmanNet](https://arxiv.org/abs/2107.10043)

### M11 · 강화학습 및 비용 인식 정책

환경의 상태는 사용 가능한 카메라·가설 토큰, 교차 잔차, $\Delta t$, 최근 거부와 게이트 상태다. 행동 후보는 측정 수용·보류, 가설 선택, 가중치·탐색 범위·정제 횟수 조정, 재초기화 요청, 또는 제한된 $SE(3)$ 증분이다. 보상 예시는 시뮬레이션 자세 오차 감소에 계산 비용·오수용·추적 상실 비용을 뺀 값이다. GT 기반 보상은 훈련 환경에만 남긴다. ReAgent는 점군 등록, SporeAgent는 장면 물리 정보를 사용한 물체 자세 정제의 선례다. 현재 RGB 관측만으로 동일한 상태와 보상을 얻을 수 있다고 가정하지 않는다. [ReAgent](https://openaccess.thecvf.com/content/CVPR2021/html/Bauer_ReAgent_Point_Cloud_Registration_Using_Imitation_and_Reinforcement_Learning_CVPR_2021_paper.html) · [SporeAgent](https://openaccess.thecvf.com/content/WACV2022/html/Bauer_SporeAgent_Reinforced_Scene-Level_Plausibility_for_Object_Pose_Refinement_WACV_2022_paper.html)

| 비교 방법 | 적합한 질문 | 필요한 데이터와 통제 |
| --- | --- | --- |
| 휴리스틱 | 기존 `RecoveryTracker`의 비용·성공 기준 | 같은 초기 자세, 같은 예산과 검증 규칙 |
| Contextual bandit | 한 프레임에서 후보 실행/선택, 미래 효과를 단순화 | 가능한 행동을 같은 상태에서 재실행하거나 행동 확률이 기록된 로그를 사용 |
| Behavior cloning | 잘 검증된 교사 행동을 근사 | 교사 행동·오류 상한, 학습된 정책 재실행의 분포 변화 집계 |
| DQN | 미래 복구 비용이 중요한 이산 행동 선택 | 보상이 있는 재생 전이와 탐험 범위, 기록 밖 행동의 가치는 별도 검증 |
| PPO | 시뮬레이터에서 직접 순차 정책 갱신 | 현재 정책의 새 rollout, 고정 녹화 영상만으로 on-policy 검증을 대체하기 어려움 |
| SAC | 연속 탐색 폭이나 $SE(3)$ 증분 선택 | 행동 범위, 시뮬레이터 재생 버퍼와 early stop, 실패 시 기하 fallback |

녹화 재생을 시험할 때 각 프레임을 모든 허용 계산 행동으로 다시 실행할 수 있으면 공통 영상에서 정책을 직접 비교할 수 있다. 실행 결과가 남지 않은 행동은 단일 정책 로그에서 그대로 복원되지 않는다. 실제 환경에서 GT가 없으면 재투영 잔차만을 보상으로 삼아 잘못된 체계를 강화할 수 있다. 복구율, 오수용, 후보 비용을 분리하고, 축소 정책으로 이 효과를 확인한 뒤 PPO나 SAC의 추가 비용을 판단한다. [Bandit replay](https://arxiv.org/abs/1003.5956) · [DAgger](https://proceedings.mlr.press/v15/ross11a) · [DQN](https://www.nature.com/articles/nature14236) · [PPO](https://arxiv.org/abs/1707.06347) · [SAC](https://proceedings.mlr.press/v80/haarnoja18b)

### M12 · 온라인 적응

장치·조명·표면 조건이 달라질 때 제한된 정규화 통계나 소형 헤드 일부만 재적응한다. Tent는 시험 샘플의 엔트로피를 줄여 적응하는 사례다. 배포 스트림에서 결과가 참이라고 가정한 자기 학습은 부정확한 초기값을 강화할 수 있다. 현재는 동결 모델을 기준선으로 두고, 영상 세션 단위 재생에서 별도 적응 구간·상태 리셋·기하 지지 조건 및 복귀 절차를 둔 방식만 비교한다. [Tent](https://arxiv.org/abs/2006.10726) · [공식 구현](https://github.com/DequanWang/TENT)

## 비교 기준

데이터·비용·연구 적용을 12개 축에 걸쳐 별도로 비교했다. [12축 비교표](METHOD_COMPARISON.md)

## 회전·대칭 규약

| 표현 | 사용할 곳 | 고려할 사항 |
| --- | --- | --- |
| Euler 각 | 사람이 결과를 해석할 때 | 순서·축 고정, 래핑과 gimbal lock에 유의 |
| Axis-angle/회전 벡터 | M06 국소 증분과 기존 최적화 | $\pi$ 근처 로그 분기, 전역 회귀에 불연속 |
| 단위 quaternion | 저장 또는 정규화된 전역 회귀 비교 | $q$와 $-q$가 같은 회전이며 손실에서 부호 동치 반영 |
| 6D 회전 | M01 연속 전역 회귀 출력 | 두 열을 Gram–Schmidt 및 외적으로 $SO(3)$에 복원하고 거의 평행한 출력 검사 |
| 회전 행렬 | 기하 경계·출력·가설의 현재 계약 | $R^\mathsf{T}R=I,\det R=+1$ 검사, 필요 시 적절한 사영 |
| Lie algebra $\mathfrak{se}(3)$ | M06/M07/M11의 제한된 한 단계 행동 | 왼쪽·오른쪽 합성과 병진 규약을 고정하고 기존 `pose.fusion.advance`와 수치 일치 확인 |

병진은 기록된 세계 좌표계에서 미터로 저장하고, 평가에서는 mm로 환산한다. 회전 오차는 $\theta=\arccos(\operatorname{clip}((\operatorname{tr}(R^\mathsf{T}R_{gt})-1)/2,-1,1))$로 구한다. 대칭이 영상에서 구별되지 않는 상자는 유효한 형상 대칭군에서의 최소 오차를 함께 보고하고, 식별 가능한 실제 물리 ID 평가는 별도로 유지한다. 세 변의 길이가 서로 다른 무표식 직육면체도 직각 고유축 기준 180° 회전 대칭이 있다. 이 문제에서 전역 Euler 회귀보다 6D, 작은 보정에는 제한된 Lie 증분이 적합하다는 판단에는 위 가정이 따른다. [Zhou 등](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html) · [BOP](https://arxiv.org/abs/1808.08319)

## Stage 0 결론

현재 스키마로 비교를 시작할 후보는 **M05 가설 순위**와 **M04 신뢰도·가중치**다. 독립 시험에서 All/Best/Cross+Verify, 단순 MLP와 집합 모델을 동일 입력으로 비교하고, 모델을 각자 자율 이력으로 재생해야 한다. 검증에서는 순위·관측 오류뿐 아니라 valid coverage, 잘못 채택한 자세, 가림 중 에지 회수, 복구 시간과 추가 CPU 비용을 각각 기록한다. M03은 물리 대응 라벨과 원본 관측의 개별 스키마 확보 뒤에 적합하다. 영상 기반 고비용 방법과 순차 RL은 독립 실물 데이터 및 측정 결과에 의존한다.

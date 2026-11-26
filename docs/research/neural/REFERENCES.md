# Stage 0 · 조사한 1차 자료

논문 본문·학회 공개본과 저자가 공개한 구현을 우선 확인했다. M번호는 Quadpose의 조사 분류다. 아래 논문의 수치와 모델 동작을 본 저장소의 검증 결과로 사용하지 않는다. [분류와 적용 범위](METHODOLOGY_SURVEY.md)

| 범위 | 원 자료 | 참고한 개념 |
| --- | --- | --- |
| M01 | Xiang 외, *PoseCNN*, RSS 2018. [논문](https://arxiv.org/abs/1711.00199) | 이미지 기반 자세 회귀 사례 |
| M01/M04 | Zaheer 외, *Deep Sets*, NeurIPS 2017. [논문](https://papers.nips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) | 공유 인코더와 집합 함수 |
| M02 | Peng 외, *PVNet: Pixel-Wise Voting Network for 6DoF Pose Estimation*, CVPR 2019. [논문](https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html) | 불확실성을 포함한 코너 위치 투표와 PnP |
| M02/M03 | Pautrat 외, *SOLD²*, CVPR 2021. [논문](https://openaccess.thecvf.com/content/CVPR2021/html/Pautrat_SOLD2_Self-Supervised_Occlusion-Aware_Line_Description_and_Detection_CVPR_2021_paper.html) | 자기지도 선분 검출·서술 |
| M03 | Sarlin 외, *SuperGlue*, CVPR 2020. [논문](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) | 문맥을 이용한 부분 특징 배정 |
| M04/M07 | Chen 외, *EPro-PnP*, CVPR 2022. [논문](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html) | 대응 가중치와 확률적 자세 |
| M05 | Brachmann 외, *DSAC*, CVPR 2017. [논문](https://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html) | 후보 선택에 대한 확률적 미분 |
| M05 | Lee 외, *Set Transformer*, ICML 2019. [논문](https://proceedings.mlr.press/v97/lee19d) | 집합의 attention 기반 인코딩과 집계 |
| M06 | Li 외, *DeepIM*, ECCV 2018. [논문](https://openaccess.thecvf.com/content_ECCV_2018/html/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.html) | 초기 자세에 대한 반복 상대 변환 |
| M07 | Chen 외, *End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization*, CVPR 2020. [논문](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_End-to-End_Learnable_Geometric_Vision_by_Backpropagating_PnP_Optimization_CVPR_2020_paper.pdf) · Pineda 외, *Theseus*, NeurIPS 2022. [논문](https://proceedings.neurips.cc/paper_files/paper/2022/hash/185969291540b3cd86e70c51e8af5d08-Abstract-Conference.html) | 미분 가능한 PnP와 비선형 최적화 |
| M08 | Wang 외, *Self6D*, ECCV 2020. [논문](https://arxiv.org/abs/2004.06468) | 깊이 및 렌더링 기반 자기지도 |
| M09 | Tarvainen 및 Valpola, *Mean teachers are better role models*, NeurIPS 2017. [논문](https://papers.nips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) | 가중치 EMA 교사와 일관성 |
| M10 | Revach 외, *KalmanNet*, IEEE TSP 2022. [저자 공개 논문](https://arxiv.org/abs/2107.10043) | 부분 동역학 지식에 따른 학습 필터 |
| M11 | Bauer 외, *ReAgent*, CVPR 2021. [논문](https://openaccess.thecvf.com/content/CVPR2021/html/Bauer_ReAgent_Point_Cloud_Registration_Using_Imitation_and_Reinforcement_Learning_CVPR_2021_paper.html) · Bauer 외, *SporeAgent*, WACV 2022. [논문](https://openaccess.thecvf.com/content/WACV2022/html/Bauer_SporeAgent_Reinforced_Scene-Level_Plausibility_for_Object_Pose_Refinement_WACV_2022_paper.html) | 순차 강체 등록 및 자세 정제 사례 |
| M11 | Li 외, *Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms*. [논문](https://arxiv.org/abs/1003.5956) · Ross 외, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011. [논문](https://proceedings.mlr.press/v15/ross11a) | 로그 평가 전제와 모방 정책의 분포 변화 |
| M11 | Mnih 외, *Human-level control through deep reinforcement learning*, Nature 2015. [논문](https://www.nature.com/articles/nature14236) · Schulman 외, *Proximal Policy Optimization Algorithms*, 2017. [논문](https://arxiv.org/abs/1707.06347) · Haarnoja 외, *Soft Actor-Critic*, ICML 2018. [논문](https://proceedings.mlr.press/v80/haarnoja18b) | 이산 가치 학습, on-policy 정책, 연속 off-policy 정책 |
| M12 | Wang 외, *Tent*, ICLR 2021. [논문](https://arxiv.org/abs/2006.10726) · [저자 구현](https://github.com/DequanWang/TENT) | 시험 분포에서 엔트로피 최소화 |
| 회전 | Zhou 외, *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019. [논문](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html) | 전역 회전 표현의 연속성 |
| 대칭 평가 | Hodaň 외, *BOP: Benchmark for 6D Object Pose Estimation*, ECCV 2018. [논문](https://arxiv.org/abs/1808.08319) · [공식 챌린지](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/) | 관측·형상 대칭을 고려한 평가 |

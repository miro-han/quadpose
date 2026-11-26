# Phase IV 기하학 참고 자료

| 출처 | 팀 코드와의 관계 |
|---|---|
| Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, Cambridge, 2004. [저자 제공 자료](https://www.robots.ox.ac.uk/~vgg/hzbook/) | 영상 간 일관성, 투영, 보정 및 기하학적 퇴화를 설명하는 근거다. |
| [OpenCV의 PnP 정의와 구현](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) | 3D–2D 코너 대응이 있을 때 단일 시점에서 물체 → 카메라 자세를 구한다. [복구 코드](../../pose/recovery.py)에서 다시 세계 좌표로 변환한 뒤 재검사한다. |
| [선형합 할당 공식 문서 및 Crouse 2016 참고문헌](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | 비용행렬에서 일대일 총비용을 최소화하는 문제 정의. [프로젝트의 구현](../../pose/refinement.py)과 SciPy 문서의 구현은 별개다. |
| Liu et al., *3D Line Mapping Revisited*, CVPR 2023. [저자 논문](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_3D_Line_Mapping_Revisited_CVPR_2023_paper.html) | 2D 선의 다중 시점 지지와 3D 선 삼각측량의 불량 조건을 이해하는 배경이다. |
| [Phase II 추정기](../../pose/README.md) 및 [Phase III 선 ID 검증](../edge_identity/README.md) | 이번 비교의 직접적인 코드·데이터 계보다. 교차 검증 점수, 임계값과 단계 조합의 채택 근거는 팀의 [비교 실험](ABLATION_STUDY.md)에서 확인한다. |

프로젝트 전체의 SAM 2, 광류, 보정, 모델 기반 추적 선행연구와 실제 사용 범위는 [원본 참고문헌](../../docs/REFERENCES.md)과 [가이드 참고문헌](../../docs/guide/REFERENCES.md)에 있다.

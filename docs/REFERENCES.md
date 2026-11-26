# 문헌과 구현 출처

2026-11-25 확인. 아래 번호는 연구 메모와 비교표에 공통으로 사용한다.

1. Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2판, 2004. 에피폴라 기하, 점과 직선의 투영, 삼각측량의 근거. [저자 자료](https://www.robots.ox.ac.uk/~vgg/hzbook/)
2. OpenCV, *Camera Calibration and 3D Reconstruction*. 구현 후보는 undistortPoints, triangulatePoints, projectPoints, solvePnPGeneric이다. 좌표 정규화 여부에 따라 투영행렬의 K 포함 여부를 맞춘다. [공식 문서](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
3. Liu et al., *3D Line Mapping Revisited*, CVPR 2023. 선 삼각측량 퇴화와 다중 시점 선 추적의 근거. 정적 장면 매핑 연구이므로 움직이는 상자에는 동시점 제약 또는 강체 움직임 모델이 필요하다. [논문](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_3D_Line_Mapping_Revisited_CVPR_2023_paper.html) · [LIMAP 공식 구현](https://github.com/cvg/limap)
4. Wang et al., *Deep Active Contour for Real-time 6-DoF Object Tracking*, ICCV 2023. 주어진 초기 자세에서 모델 윤곽을 영상에 투영하는 단일 RGB 추적 연구. [저자 페이지와 구현](https://zju3dv.github.io/deep_ac/)
5. DLR, *M3T: A Multi-region Multi-modality Model-based Tracker*. 공통 강체 자세에 여러 관측을 결합하는 설계 참고. 본 프로젝트에서 사용할 센서 입력은 RGB로 한정한다. [공식 구현 및 논문 안내](https://github.com/DLR-RM/3DObjectTracking/blob/master/M3T/readme.md)
6. OpenCV, *Detection of ArUco Markers*. 사전 ID, 코너 순서 및 평면 마커 검출 구현. [공식 문서](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
7. Lucas & Kanade, *An Iterative Image Registration Technique with an Application to Stereo Vision*, IJCAI 1981. 국소 영상 정합의 근거와 피라미드 LK 구현. [논문](https://publications.ri.cmu.edu/an-iterative-image-registration-technique-with-an-application-to-stereo-vision-ijcai) · [OpenCV API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
8. Hodaň et al., *EPOS: Estimating 6D Pose of Objects With Symmetries*, CVPR 2020. 물체 대칭과 다중 대응을 고려하는 연구. [논문](https://openaccess.thecvf.com/content_CVPR_2020/html/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.html) · [BOP 대칭 평가 정의](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/)
9. Zaheer et al., *Deep Sets*, NeurIPS 2017. 순서에 무관한 관측 집합 집계의 후속 후보. 카메라 신뢰도에 대한 성능 근거는 후속 실험에서 확보해야 한다. [논문](https://arxiv.org/abs/1703.06114)
10. Meta, SAM 2.1 공식 영상 predictor 및 체크포인트. 기존 분할 앞단의 출처. [공식 저장소](https://github.com/facebookresearch/sam2)
11. Eggert, Lorusso & Fisher, *Estimating 3-D rigid body transformations: a comparison of four major algorithms*, 1997. 대응이 주어진 3D 점 집합의 SVD 정합과 퇴화 조건. [논문](https://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf)
12. Vaswani et al., *Attention Is All You Need*, 2017. 질의와 관측 집합 간 attention의 수학적 배경. 카메라 신뢰도 학습은 이 프로젝트에서 별도 검증할 제안이다. [논문](https://arxiv.org/abs/1706.03762)

문헌의 처리 속도와 정확도는 팀 장비에서 측정한 수치로 전용하지 않는다. 본 연구의 비교와 제안은 위 기법을 알려진 직육면체와 네 대의 고정 RGB 카메라에 적용한 설계 판단이다.

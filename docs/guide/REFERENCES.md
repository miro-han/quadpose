# 출처와 구현 범위

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2판, 2004. 투영·에피폴라·삼각측량의 배경이다. [저자 페이지](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- OpenCV, *Camera Calibration and 3D Reconstruction*. 현장 도구의 내부/외부 보정·PnP·좌표 변환의 구현 근거다. [공식 문서](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Liu et al., *3D Line Mapping Revisited*, CVPR 2023. 선 삼각측량의 퇴화와 다중 시점 지지의 배경이다. [논문](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_3D_Line_Mapping_Revisited_CVPR_2023_paper.html)
- Meta, *SAM 2*. 공식 predictor와 Tiny 가중치를 사용했다. 팀의 구현은 프롬프트 UI·결과 저장·관측 처리·기하 융합에 있다. [공식 소스](https://github.com/facebookresearch/sam2) · [동봉한 정확한 버전](../../models/sam2_source.json)
- DeepAC와 M3T는 모델 재투영 및 여러 관측을 통한 자세 개선의 선행 연구다. 이 저장소에서 비교 모델의 논문 성능을 재현한 기록은 없다. [DeepAC](https://zju3dv.github.io/deep_ac/) · [M3T](https://github.com/DLR-RM/3DObjectTracking/blob/master/M3T/readme.md)
- 구현 재현과 제출 시에는 팀의 자체 데이터·코드, 공개 코드의 역할, AI가 보조한 개발·분석 범위를 실제 기여와 대조해 기술한다. 연구용 합성 수치의 원자료와 기존 참고문헌은 [전체 참고문헌](../REFERENCES.md), [Phase IV 출처](../../evaluation/pose_benchmark/REFERENCES.md), [원본 과제 지침](<../source/TP_CV_Term Project.pdf>)을 따른다.

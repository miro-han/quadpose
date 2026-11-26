# 영상에서 기하 관측을 얻기

촬영 UI는 카메라별 AVI와 CSV를 남긴다. 수신 속도가 달라질 수 있으므로 인코더 FPS와 측정 시각을 분리한다. 분할 UI에서는 시작 프레임에 박스를 그리고 필요하면 양·음 클릭을 추가한다. 공식 SAM 2.1 Tiny가 같은 영상 안에서 대상 마스크를 전파한다. 상자의 윗면과 앞면이 같은 마스크에 들어가므로 두 면 사이의 물리 모서리는 마스크만으로 분리되지 않는다. [녹화 코드](../../capture/camera_core.py) · [SAM 연결과 Canny](../../perception/segmentation/sam_session.py)

어댑터는 원본 영상 좌표에서 마스크 주변을 넓히고 LSD 선분 위 표본이 그 영역에 얼마나 들어가는지, Canny 에지가 선을 얼마나 지지하는지 검사한다. 끝이 조금 끊긴 선도 받아들이되 방향이 거의 평행하거나 유한 길이를 지나치게 연장해야 만나는 선쌍의 교점은 거부한다. 한 선이 여러 조각으로 끊기고 여러 교점이 같은 픽셀에서 파생될 수 있다. 이 관측들을 모두 독립 증거로 계산하면 한 면의 정보량을 과대평가한다. [선분 기준 실험](../../perception/edges/check_segments.py) · [실제 어댑터](../../perception/observations/adapter.py)

![실제 영상 11번 프레임에서 마스크와 에지로 선별한 후보](../../perception/observations/sample11/overlay.jpg)

**실제 영상 사례.** 11번 프레임에서 LSD 선분 97개 중 37개와 교점 7개가 남았다. 아래 그림은 여섯 수동 클릭과 가정한 초점거리·치수로 만든 대응 예시다. 5개 점, 8개 선에 ID가 부여됐지만 한 코너 잔차는 약 36.9 px였다. 카메라의 실측 보정과 독립 정답이 없어 이 이미지로 3D 정확도를 판단할 수 없다. [원본 관측](../../perception/observations/sample11/observations.json) · [진단에 사용한 가정](../../perception/observations/illustrative11/prior.json)

![실제 이미지 위에 가정한 모델 투영과 대응 잔차를 그린 진단 예시](../../perception/observations/illustrative11/frame/overlay.jpg)

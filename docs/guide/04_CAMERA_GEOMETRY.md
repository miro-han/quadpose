# 픽셀을 같은 좌표계에 놓는 방법

두 영상에 찍힌 좌표 (120,80)은 서로 다른 광선이다. 픽셀만으로 거리도, 두 관측이 같은 점인지도 정해지지 않는다. 첫 단계는 각 카메라가 물체의 한 점을 어떤 픽셀에 mapping하는지 알아내는 것.

### 좌표와 보정

물체의 꼭짓점 $X_O$를 homogeneous coordinate으로 바꿔 $\tilde X_O=(X_O^T,1)^T$로 나타낸다. ${}^WT_O$는 object $\to$ global, $E_i={}^iT_W$는 global $\to$ camera 변환이다. 카메라 좌표 $\mathbf x_c=(x,y,z)$의 $z>0$일 때 projection 계산해보면

$$
s\tilde u_i=K_i[I\mid0]E_i{}^WT_O\tilde X_O,\qquad
u=f_x x/z+c_x,\quad v=f_y y/z+c_y .
$$

따라서 global 좌표계에서 카메라 중심은 $C_i=-R_i^Tt_i$이다. 카메라 좌표는 오른쪽·아래쪽·전방, 세계는 오른쪽·정면 카메라에서 안쪽·위쪽이다. X를 mm로 쓰고 t를 m로 쓰면 원근 분모까지 잘못된다. 이 프로젝트는 기하 내부 길이를 m로 통일하고 결과 오차를 mm로 변환한다.

왜곡은 정규화된 광선 좌표에서 적용된다. 원본 픽셀로 적합할 때는 왜곡을 포함해 투영하고, 어댑터의 undistorted-K 픽셀을 사용할 때에는 같은 K로 투영한다. 왜곡을 두 번 제거하면 화면 가장자리에서 일관적인 residual이 생긴다. Intrinsic calibration은 알려진 체커보드 칸 좌표와 다양한 보드 자세의 픽셀로, 카메라의 왜곡을 고려해서 각 자세를 reprojection했을 때 오차 최소화 계산을 돌려서 구한다. 실제 보드 칸 크기는 외부 이동량의 길이 단위를 정한다. 고정 리그의 외부 보정(extrinsic calibration)은 세계 기준 보드와 카메라 사이의 변환을 구하고, 직접 기준이 보이지 않는 카메라는 겹침 보드의 상대 변환을 연결한다. 별도 보드 자세에서 교차 투영하고 그래프에 연결되지 않으면 거부한다. [체커보드 도구](../../rig/calibrate.py) · [합성 검증과 현장 지침](../../rig/README.md)

### Epipolar Geometry

이미지에서 렌즈 왜곡을 제거한 픽셀을 $\tilde{u} = (u,v,1)^\mathsf{T}$라고 했을 때, 정규화된 광선 좌표를 $$x = K^{-1}\tilde{u} = \left(\frac{u-c_x}{f_x}, \frac{v-c_y}{f_y}, 1\right)^\mathsf{T}$$로 계산할 수 있다. 이미지에는 깊이 정보가 없기 때문에 1로 둔다. 즉 $(a,b)$에 있는 픽셀에서, 이미지에서 보이는 실제 물체는 정규화된 좌표 $(a,b)$를 기준으로 $\lambda(a,b,1)^\mathsf{T}, \; \lambda > 0$ 위에 있다. $\lambda$는 이미지에서 바로 알 수 없는 실제 깊이.

이때 카메라 1에서 관측한 점의 광선 방향을 $x_1$이라 하면 카메라 1에서 본 좌표를 카메라 2에서 본 좌표로 변환하는 $$X_2 = RX_1+t$$를 계산할 수 있다. 이때 그 광선에 있을 수 있는 실제 3D 점은 카메라 2를 기준으로 $X_2(\lambda) = t + \lambda R x_1$으로 표현된다. $t$는 카메라 2에서 본 카메라 1의 중심 위치, $Rx_1$은 카메라 2 좌표로 표현한 광선의 방향이다. 이때 두 벡터 $t$와 $R x_1$가 만드는 평면을 Epipolar plane이라고 부르는데, 이 plane의 법선벡터는 두 벡터의 외적으로 구할 수 있다. 실제로 검출할 때와 보정할 때는 오차가 있으니 픽셀 영역에서는 선으로부터 몇 px 이내에서 후보를 남긴다. 여기서 undistorted K-coordinate를 썼을 때 Epipolar line은 $l_2 = F \tilde{u}_1$, $F = K_2^{-\mathsf{T}}[t]_\times R K_1^{-1}$로 계산할 수 있다. 후보가 여러 개라면 방향과 모델을 고려해서 가능성 높은 후보를 탐색한다. 

결과적으로 $\tilde{x}_2$는 epipolar plane의 법선벡터와 수직이어야 하므로

$$
\tilde u_2^TF\tilde u_1=0,\qquad F=K_2^{-T}[t]_\times R K_1^{-1}.
$$

이는 두 번째 카메라의 후보를 에피폴라 선 부근으로 좁히는 조건이다. 같은 선 위의 다른 물체나 반복 코너도 통과할 수 있다. 동일 물리 점이 확인되면 각 카메라에 대해 $\tilde u_i\times(P_i\tilde X)=0$의 독립인 두 행을 쌓는다. SVD로 분해해서서 가장 작은 특이값의 우측 벡터가 homogeneous point의 해다. 마지막 성분으로 나눈 후 양의 깊이와 재투영을 검사한다. 광선이 거의 평행하거나 서로 다른 시각의 움직이는 점을 맞추면 깊이가 불안정하다. 삼각측량은 비교 이론이고, 기본 추정 경로는 알려진 상자 모델을 각 영상에 투영한다. [다중 시점 문헌](REFERENCES.md) · [프로젝트 후보 비교](../ALGORITHM_COMPARISON.md)

### 영상선의 inverse projection

픽셀 두 개의 외적 $\ell=\tilde u_a\times\tilde u_b$는 homogeneous 영상선이다. 이 선에 투영되는 모든 세계점은 $\ell^TP_i\tilde X=0$을 만족하므로 $P_i^T\ell$이 global plane이다. 두 평면의 교선은 3D 무한 직선 후보다. 잘못 고른 두 영상선도 일반적으로 교선을 만들며 선분의 실제 양 끝은 그 교선만으로 복원되지 않는다. 이 때문에 알려진 변 길이, 인접 면, 실제 유한 겹침이 중요한 조건이 된다. [선 매핑 연구](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_3D_Line_Mapping_Revisited_CVPR_2023_paper.html)

### 시간도 보정해야 함

수신시각 CSV가 있어서, 프레임을 읽은 시점이 기록된다. 실제 노출과의 차이는 점멸의 대응 시각으로 오프셋·지터를 측정한다. 만약 영상속도가 200 px/s라면 20 ms의 시차만으로 약 4 px의 오차가 생길 수도 있다. 이것만으로도 몇 픽셀의 선 게이트를 넘길 수 있다. [시간 정렬 코드](../../rig/synchronize.py)

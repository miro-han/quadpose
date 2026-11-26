# Phase II 카메라 보정과 시간 정렬

프로젝트 루트에서 기존 Python 환경으로 실행한다. 이미지·AVI/CSV와 JSON 설정은 입력 폴더에 보존하고 매 실행에 새 출력 파일명을 사용한다. 설정 안 상대 경로는 해당 JSON의 위치 기준이다. 예제는 경로와 좌표를 채우는 형식을 보이며 실제 보정 결과를 포함하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m rig.extract_frame --session-dir <calibration-recording> --slot 1 --frame 42 --output outputs\calibration\front_0042.png
.\.venv\Scripts\python.exe -m rig.calibrate --config outputs\calibration\rig_input.json --output outputs\calibration\rig.json
.\.venv\Scripts\python.exe -m rig.synchronize flash --config outputs\sync\flash_input.json --output outputs\sync\flash.json
.\.venv\Scripts\python.exe -m rig.synchronize align --config outputs\sync\align_input.json --output outputs\sync\alignment.json
.\.venv\Scripts\python.exe -B -m unittest rig.test_rig -v
```

## 실물 촬영 절차

1. 프레임을 고정하고 장치·슬롯·해상도·FPS·초점을 기록한다. 기준 좌표는 오른손계·미터 단위다. 정면 카메라는 원점의 −Y 쪽, 측면은 +X 쪽, 상단은 +Z 쪽이다. 정면에서 보아 +X는 오른쪽, +Y는 안쪽, +Z는 위다. 초점·줌·해상도·장치 위치를 바꾼 경우 해당 설정으로 다시 보정한다.
2. 내부 파라미터용 이미지를 카메라별 15–25장 목표로 확보한다. 보드 전체가 나오도록 두고 중앙·가장자리, 다른 거리와 서로 다른 기울기를 사용한다. 보고서는 카메라별 검출 실패, 프레임별 재투영 RMS, 보드 중심 분포와 각도 범위를 남긴다. 기본 기준은 유효 8장, 화면 폭·높이 각각 중심 범위 15% 이상, 보드 내 기울기 차이 15° 이상, 전체 RMS 0.7 px 이하 및 영상별 RMS 1.5 px 이하다.
3. 내부 교점이 가로와 세로에서 다른 개수인 체커보드를 쓴다. 인쇄한 뒤 실제 격자 한 칸을 측정한다. 흰 여백에 O, +X, +Y의 방향을 표시하고 그림 속 교점을 선택한다. `anchors_px`는 **인쇄된 보드에서 같은 물리점을 가리키는** 네 내부 교점의 원본 이미지 `[u, v]` 좌표다. 순서는 O, +X 방향 끝, +Y 방향 끝, +X·+Y 방향 끝이다. 보드 로컬 +Z는 인쇄면 바깥쪽이고 `T_world_board`는 이 보드 좌표에서 세계 좌표로 가는 실측 강체 변환이다.
4. 적어도 한 카메라에서 세계 기준 지그에 보드를 고정해 `solve` 이미지를 얻는다. 평면 자세 후보를 구분할 수 있도록 위치·기울기가 다른 알려진 지그 자세를 사용하는 편이 좋다. 독립 검사에는 다른 알려진 보드 자세의 `check` 이미지와 실측 `T_world_board`를 넣는다. 대상 보드가 가장자리에 닿거나, 어느 이미지에서 보드가 작거나, 두 가능한 자세가 비슷한 오차면 해당 설정을 보류한다.
5. 모든 카메라를 직접 기준 지그에 연결할 수 있으면 모든 카메라에 `anchors`를 제공한다. 직접 닿기 어려운 카메라에는 두 시야에 동시에 보이는 보드의 정지 자세를 한 쌍씩 넣는다. 한 쌍은 **동일한 물리 보드 자세**여야 한다. 보드가 고정된 구간에서 이미지 둘을 추출하고 각각 실제 교점을 표시한다. 간선마다 서로 다른 각도의 `solve` 자세 2개 이상과 별도 `check` 자세 1개 이상을 확보한다. 양·음 대각 카메라가 인쇄면을 동시에 보기 힘들면 중간 카메라를 거쳐 간선을 이어 붙인다. 가능한 경우 중복 간선을 더해 순환 일관성을 검사한다.
6. 움직임 녹화에서는 한 호스트, 한 Recorder 세션을 사용한다. 모든 카메라에 동일한 발광 이벤트가 보이도록 확산광 또는 하나의 신호로 구동한 LED를 놓고 카메라별 고정 ROI를 기록한다. ROI에는 움직이는 상자·손과 밝기가 변하는 배경이 들어오지 않도록 한다. 노출·자동 게인을 가능한 한 고정한다. 발광기 로그의 서로 다른 점등 간격을 입력하고, 최소 다섯 번 점등하며 녹화의 첫·끝에 소등 상태를 포함한다. 유효 구간은 첫 점등부터 마지막 점등까지라서 평가 동작은 두 이벤트 사이에 둔다. 점등은 느린 카메라에서도 연속 2프레임 이상 검출될 만큼 길게 유지한다.
7. `rig.json`의 장치 경로와 해상도를 `alignment.json.streams`와 대조한다. `flash`와 `align` 성공 여부, 지터·드리프트, `valid_group_fraction`, 제외 사유를 확인한다. 최대 시간 잔차는 예상 속도와 오차 예산으로 정한다. 예를 들어 0.25 m/s에서 위치 차이를 3 mm 이내로 제한하려면 12 ms가 한 기준이다. 허용창에 후보가 여럿 나오면 창을 늘려 억지로 맞추지 않고 카메라 간 타이밍과 촬영 주기를 점검한다.

## 설정 형식

### `rig_input.json`

```jsonc
{
  "rig_id": "team7-rig-v1",
  "board": {"cols": 8, "rows": 6, "square_m": 0.028},
  "cameras": {
    "front": {
      "slot": 1,
      "device_path": "copy session.json cameras.1.device.path here",
      "intrinsic_images": ["cal/front_intrinsics_01.png", "…"],
      "anchors": [
        {
          "role": "solve",
          "image": "cal/front_world_01.png",
          "T_world_board": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
          "anchors_px": [[100,110],[250,108],[98,230],[251,235]]
        }
        // Add a held-out anchor with role "check" and its own real measured matrix.
      ]
    },
    "side": {
      "slot": 2,
      "device_path": "copy session.json cameras.2.device.path here",
      "intrinsic_images": ["cal/side_intrinsics_01.png", "…"],
      "anchors": []
    }
  },
  "overlaps": [{
    "source": "front",
    "target": "side",
    "pairs": [{
      "role": "solve",
      "source_image": "cal/front_shared_01.png",
      "target_image": "cal/side_shared_01.png",
      "source_anchors_px": [[100,110],[250,108],[98,230],[251,235]],
      "target_anchors_px": [[140,115],[312,104],[154,230],[318,245]]
    }]
  }]
}
```

예제 주석과 `…`를 지우고 각 카메라별 실제 이미지와 겹침 자료를 채워 JSON으로 저장한다. `quality`로 임계값을 변경할 수 있으며 최종값이 `quality_limits`에 남는다. 직접 앵커의 기본 기준은 맞춤 RMS 1.3 px, 독립 검사 1.8 px 이하다. 겹침 간선은 고정된 각 카메라 내부 파라미터로 `stereoCalibrate`를 실행한다. 기본 맞춤 RMS 1.4 px, 독립 양방향 재투영 2.5 px 이하다. 독립 경로의 차이는 기본 1.5° 및 25 mm 이하여야 한다. 직접 앵커 전용일 경우 `overlaps`를 생략하고 각 카메라에 세계 앵커를 제공한다.

`rig.json`의 `T_camera_world`는 세계 점을 OpenCV 카메라 좌표(+x 오른쪽, +y 아래, +z 전방)로 보낸다. `cameras`의 `K`, `distortion`은 원본 해상도 픽셀 기준이다. Brown 순서는 `[k1, k2, p1, p2, k3]`다. `measured_overlap_edges`는 실제로 맞춘 간선과 독립 오차를 기록하고, `camera_graph`는 보정 뒤 모든 방향의 상대 변환을 완성한다. `T_camera_target_camera_source = T_target_world · inv(T_source_world)`다. `pose_via`에 연결 경로를 기록한다. 겹침 간선으로 세계 자세를 전파한 뒤 별도 간선과 대조한다. 간선의 독립 검사가 있어도 그래프가 트리라면 다른 경로를 통한 순환 검사는 할 수 없다.

### `flash_input.json`

```json
{
  "session_dir": "recordings/20261126_sample",
  "camera_slots": {"front": 1, "side": 2},
  "reference_camera": "front",
  "pulse_intervals_s": [0.72, 1.18, 0.89, 1.49],
  "roi_by_camera": {"front": [0.1, 0.12, 0.21, 0.27], "side": [0.76, 0.11, 0.90, 0.26]},
  "max_schedule_error_ms": 80,
  "max_offset_ms": 150,
  "max_jitter_ms": 20,
  "max_drift_ms": 15,
  "min_flash_delta": 20,
  "min_pulse_frames": 2,
  "max_pulse_frames": 10
}
```

ROI는 `[left, top, right, bottom]` 비율이다. ROI 평균 명도와 이중 문턱으로 점등 첫 프레임을 찾는다. 예상 개수와 비주기 점등 간격이 모든 카메라에서 맞아야 한다. `offset_ns = median(host_i − host_ref)`이고 정렬 시 해당 값을 뺀다. `jitter_p95_ms`와 `jitter_max_ms`는 짝별 오프셋의 중앙값 주변 절대 잔차다. 시각적 검출과 호스트 수신 시각의 결합값이며 센서 노출에 대한 전기적 동기 신호가 아니다. AVI, CSV 및 세션 메타데이터를 해시로 묶어 다른 세션이나 수정한 파일에 재사용하지 않는다.

### `align_input.json`

```json
{
  "session_dir": "recordings/20261126_sample",
  "camera_slots": {"front": 1, "side": 2},
  "reference_camera": "front",
  "flash_report": "flash.json",
  "max_residual_ms": 12,
  "min_views": 2,
  "min_valid_group_fraction": 0.70
}
```

기준 프레임마다 후보가 유일해야 하고 다른 기준 프레임과 같은 영상을 공유하면 두 그룹에서 그 대응을 거부한다. 일부 카메라가 빠져도 `min_views`를 충족하고 모호성이 없으면 그룹을 사용할 수 있다. 기준 프레임 중 플래시 괄호 안의 유효 그룹 비율이 기준보다 낮으면 `status: "rejected"` 결과를 저장하고 종료 코드 2를 반환한다. 처리 입력에는 `status: "passed"`에서 `valid: true`인 그룹을 쓴다. `streams`는 장치·AVI·해상도를, `groups[].frames`는 각 카메라의 `source_frame`, `host_monotonic_ns`, `aligned_time_ns`, `residual_ns`를 제공한다. `missing`은 탈락 캠과 이유를 기록한다.

플래시 자료가 없으면 `allow_unverified_host_only: true`를 설정해 호스트 수신시각 매칭을 탐색할 수 있다. 이 결과는 `unverified_timing: true`이며 움직이는 물체의 실물 정렬 평가에는 플래시 검증 경로를 사용한다. 다른 PC에서 따로 녹화한 영상이나 AVI 재생 FPS와 프레임 번호만으로 같은 시각을 구성할 수 없다.

## 검증 범위

합성 테스트 9개를 통과했다. 렌더링한 체커보드 PNG에서 정면 기준 카메라의 정답 대비 자세 오차는 0.621°·9.19 mm, 겹침을 통해 연결한 측면 카메라는 0.750°·10.53 mm다. 측면과 정면의 겹침 맞춤 RMS는 0.288 px, 독립 양방향 재투영 RMS는 1.000 px 및 0.324 px였다. 합성 AVI/CSV에서는 주입한 7 ms 오프셋을 복원했고 플래시 곡선 내 기준 그룹 78개의 잔차가 0 ns였다. 이 0 ns는 합성 시각의 결과다.

잘못 표시한 보드, 부족하거나 편중된 내부 보정 자료, 연결 끊김, 모순된 그래프, 누락 점등, 지터, 시간 후보의 중복 및 원본 세션 변경을 거부했다. 동봉된 물리 카메라 AVI에서 정확한 프레임을 PNG로 추출했다. 현 PC의 DirectShow에는 OBS Virtual Camera만 있고 Media Foundation에는 장치가 검색되지 않았다. 실물 카메라 보정·공통 플래시 촬영은 현장 절차로 남아 있다.

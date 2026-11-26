# Eight diagonal camera directions and blind pose tracking

## 실행 및 구성

`Team7_Plate_Tracking`에서 다음 명령을 실행한다. NumPy를 사용하며 수치 라이브러리의 스레드를 1개로 제한하고 실험 프로세스에 메모리 커밋 3 GiB 상한을 적용한다. GPU나 OpenCV를 불러오지 않는다.

```powershell
python -B -m evaluation.resource_limits --threads 1 --memory-gib 3 evaluation.layout_pose.benchmark
python -B -m unittest evaluation.layout_pose.test_benchmark -v
```

월드 좌표는 Z-up이다. 정면과 측면, 상단 카메라는 각각 `(0, -0.65, 0)`, `(+0.65, 0, 0)`, `(0, 0, +0.65)` m에 고정한다. 각 대각선은 `x±y±z±` 방향을 따라 모든 카메라를 원점 쪽으로 향하게 한다. 대각선의 중심 거리만 0.650 m로 맞춘 경우와, 실제 1.30 m 정육면체 꼭짓점에 둔 경우의 1.126 m를 구분한다. 초점거리는 900 px이며 각 카메라는 960 × 720 px의 이상적인 핀홀 모델을 사용한다.

190 × 120 × 85 mm 직육면체에 서로 다른 시드의 28프레임 궤적 3개를 적용한다. 반지름 85 mm 구로 손을 모사하고 6개 축 및 8개 대각선 접근 방향을 비교한다. 각 에지의 카메라 광선을 17개 지점에서 검사해 남아 있는 가장 긴 구간만 측정 후보로 사용한다. 기준, 클러터, 손 가림, 네 개의 개별 카메라 누락 조건에서 코너, 에지, 결합 추정기를 각각 실행한다. 추정기는 이전 유효 자세, 무라벨 영상 좌표, 카메라 보정, 물체 치수를 입력으로 받는다. 실제 물리 에지 ID는 성공률 평가에만 사용한다.

`results/by_condition.csv`는 거리·방향·조건·추정기별 자세 RMSE, 95백분위, 실패율 및 물리 에지 대응 결과를 기록한다. `results/by_designated_camera.csv`는 지정 시점에서 사라진 에지를 다른 시점이 기하학적으로 보거나 실제 검출하고 정확히 대응시킨 비율을 구분한다. 통제된 합성 검출 실험이며 바닥, 지지대, 실제 카메라 오차와 실영상 검출 성능은 결과에 포함되지 않는다.

## Experimental design

The two distance modes share all eight directions, three axis cameras, image intrinsics, cuboid dimensions, trial trajectories, hand paths, and per-feature random draws. `equal_radius` places the diagonal at `0.65 / sqrt(3) * (sx, sy, sz)` metres. `cube_vertex` uses `0.65 * (sx, sy, sz)` metres. The cube-vertex direction `x+y-z+` lies in the same octant as the three axis views. `x-y+z-` is its opposite. Moving the diagonal between the two distances also changes its projection scale and perspective.

The three trials use separate trajectory and initial-pose draws. Each starts from a perturbed first-frame truth to simulate a coarse external initialization. Every layout, method, and condition then rolls out independently from its own last valid estimate. Detection dropout, pixel noise, trimming, and endpoint flips are keyed by seed, frame, camera slot, and physical feature ID. A layout with a different visible set therefore keeps the same underlying draws for features that it shares with another layout. The clutter candidates are generated near model projections using a paired candidate ordering and independent noise. Body edges and corners are modeled as genuine detectable features when visible.

`clean` uses nominal noise (0.6 px corner and 0.5 px endpoint standard deviations) and dropout (2.5% corner and 5% edge). All other conditions use 1.25/0.95 px noise, 16%/23% dropout, and up to three clutter samples per feature type and camera. `hand` adds the finite-ray sphere occlusion model. Sphere placement leaves an 8–45 mm gap from the corresponding support plane of the moving cuboid. Each of the fourteen fixed world directions is used for two frames per trial. A hand direction is designated for a camera when the chosen world approach coincides with that camera's axis or diagonal direction. The physical hand position for a given frame is identical across layouts. `missing_*` removes one camera from the noisy/cluttered input for the entire trial. It preserves the other three camera samples exactly.

The estimated methods use the Phase II gated geometric matcher and robust pose solver. The corners contribute pixel reprojection errors. The segments contribute distances to observed supporting lines. No cross-camera physical identities are supplied. Each view matches its own observations against the projected model from the causal prior. Cross-camera edge identities are evaluated after matching. The experiment starts from candidate pixels and uses no image segmentation, real camera stream, or learned fusion.

## Outputs and meaning

| File | Contents |
|---|---|
| `results/manifest.json` | Seeds, dimensions, positions, intrinsics, prior errors and thresholds |
| `results/frame_metrics.csv` | One row per trial, frame, layout, condition and method |
| `results/by_layout.csv` | Pooled summaries for each distance, direction and method |
| `results/by_condition.csv` | Pose and edge metrics split by the seven conditions |
| `results/by_hand_direction.csv` | Pose and edge metrics split by the 14 sphere approach directions |
| `results/by_designated_camera.csv` | Lost-edge rescue results stratified by the impacted camera slot |
| `results/distance_delta.csv` | Cube vertex minus equal radius under matched direction, condition and method |

- Pose RMSE and p95 include held estimates. Failure is an invalid fit or an error above 20 mm or 8°.
- Edge ID precision divides correct physical IDs by all estimated edge assignments. Use the pair-precision column to evaluate claims that two cameras supplied the same model edge ID. A pair is correct when both measurements really belong to that physical ID. An edge in views A, B and C can contribute the three pairs A–B, A–C and B–C. The metric scores physical line identity even when the visible fragments cover disjoint portions of the edge. The geometry sweep separately tracks visibility of identical points along each edge.
- Multiview edge recall divides physical edge IDs correctly assigned in two or more views by the physical edge IDs detected in two or more views. This scores coverage of usable cross-view detections alongside precision.
- Rescue is measured over edges geometrically lost from a designated hand-occluded or missing camera. The geometric, detection, and correct-ID rescue rates ask whether at least one other camera can see, truly detects, or correctly assigns the same physical ID. A partly hidden edge with a sufficiently long remaining fragment is still available in the designated view. The denominator changes with the direction, baseline visibility, and actual occlusion. Use the raw counts when comparing rates.
- Synthetic geometric visibility checks cuboid self-occlusion and usable projected length. A spherical occluder is an optical approximation. This simulation includes no floor, support pole, or physical mounting constraint and does not establish field performance.

## 측정 결과 / Results

28,224개 자세 행을 확인했다. 결합 추정기는 16개 광학 배치와 7개 조건에서 모두 실패율 0%였다. 실제 꼭짓점에서 같은 쪽 `x+y-z+`을 반대쪽 `x-y+z-`으로 바꾸면 손 가림 중 2대 이상에서 볼 수 있는 물리 에지가 평균 9.95개에서 11.01개로 늘었다. 자세 RMSE는 0.778 mm / 0.581°에서 0.798 mm / 0.521°로 바뀌었다. 물리 에지 가시성의 개선과 자세 오차의 변화 방향이 일치하지 않았다.

The table contains the 84 hand-occlusion frames per configuration for the hybrid tracker. All eight layouts in both modes had zero failed hybrid frames.

| Diagonal | 0.650 m RMSE, mm / ° | 0.650 m mean edges seen in ≥2 views | 1.126 m RMSE, mm / ° | 1.126 m mean edges seen in ≥2 views |
|---|---:|---:|---:|---:|
| `x-y-z-` | 0.707 / 0.458 | 10.25 | 0.751 / 0.530 | 10.31 |
| `x-y-z+` | 0.652 / 0.418 | 9.92 | 0.786 / 0.537 | 10.00 |
| `x-y+z-` | 0.696 / 0.430 | 10.93 | 0.798 / 0.521 | 11.01 |
| `x-y+z+` | 0.699 / 0.455 | 10.37 | 0.803 / 0.551 | 10.44 |
| `x+y-z-` | 0.586 / 0.394 | 10.08 | 0.788 / 0.549 | 10.14 |
| `x+y-z+` | 0.798 / 0.538 | 9.90 | 0.778 / 0.581 | 9.95 |
| `x+y+z-` | 1.004 / 0.637 | 10.39 | 0.796 / 0.570 | 10.43 |
| `x+y+z+` | 0.819 / 0.562 | 9.92 | 0.833 / 0.582 | 9.95 |

At the actual cube vertices, the same-side and opposite layouts achieved respectively 68.98% and 68.22% pairwise physical edge-ID precision under hand occlusion. Their recall on physical edges detected by two or more cameras was 83.90% and 81.71%. When the hand targeted one of the three fixed cameras, other cameras correctly assigned 62 of 85 lost edge instances on the same-side layout and 68 of 85 on the opposite layout. Across all designated cameras, both layouts correctly supplied 85 lost edge instances, out of 115 and 104 losses respectively. Those denominators differ because the diagonal is targeted from a different world direction.

At the cube vertices the same-side position's p95 was 1.310 mm / 1.085°, and the opposite position's was 1.746 mm / 1.165°. With the diagonal stream removed, the hybrid estimate was identical across all 16 layouts and had 1.023 mm / 0.626° RMSE. The corner-only tracker failed on 15.48% of those same three-axis frames. With four cameras present and hand occlusion, the equal-radius corner-only `x+y+z-` layout failed on 9.52% of frames, compared with 0% for the edge-only and hybrid trackers there. These results span three paired motion trials. The observed ranking depends on the combination of pose, clutter, occluder direction, and criterion.

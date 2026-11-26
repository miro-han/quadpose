# Synthetic multi-view cuboid tracking

This experiment fits one rigid 6-DoF pose per frame for a 190 × 120 × 85 mm box. It compares corner-only, edge-only, and hybrid projection fitting on the same synthetic detections. Each method runs with a diagonal camera alone and with four calibrated cameras. Both view counts run with estimated and oracle feature associations.

## Run

Run these commands from the project directory with Python 3.10+ and NumPy installed.

```powershell
python experiments/fusion/fusion_poc.py
python experiments/fusion/fusion_poc.py --axis-only
python experiments/fusion/benchmark_fusion.py
python -m unittest discover -s experiments/fusion -v
```

The defaults are 75 frames per trial and seeds 42, 43, and 44. Use `--frames`, `--trials`, `--seed`, and `--output-dir` to change the run. `--axis-only` runs the shared front, side, and top cameras with estimated hybrid fitting. It writes separate files using the same three camera streams as the main benchmark. The stress outages occur at zero-based frames 26–33 and 48–53, so shorter runs can omit them.

## Geometry and observations

- The world axes are +X right, +Y up, and +Z toward the front camera. All cameras aim at the origin of a 1.30 m cube. The front, side, and top camera centers stay at (0, 0, 0.65), (0.65, 0, 0), and (0, 0.65, 0) m. The diagonal camera is at either cube vertex ±(0.65, 0.65, 0.65) m, about 1.126 m from the origin. The cameras use the same ideal pinhole intrinsics, with 850 px focal length, a principal point of (480, 360), and a 960 × 720 image.
- A corner is eligible when it projects inside the image with positive depth and at least one adjoining face is on the camera side of its plane. An edge also needs two eligible image coordinates, at least one camera-facing adjoining face, and a projected span of at least 10 px. Per-feature coverage uses the model IDs only for evaluation. Corner IDs follow lexicographic ((X,Y,Z)) sign order with the negative sign first. Edge IDs follow lexicographic pairs of corner IDs that differ by one sign.
- The simulator applies independent Gaussian noise to both corner pixel coordinates and both endpoints of a line segment. It trims each line endpoint by 4–30% of the full projected length, randomly reverses endpoint order, and shuffles detections. Clutter consists of noisy offsets from visible corners and nearby segments. Each layout uses the same random stream for every corresponding camera and frame, so axis-camera detections are identical between layouts. The diagonal cameras can have different eligible feature sets.

| Scenario | Corner noise σ, px | Edge endpoint noise σ, px | Corner dropout | Edge dropout | Clutter per camera, corners / segments |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.8 | 0.55 | 8% | 12% | 1 / 1 |
| Stress | 1.7 | 1.2 | 45% | 30% | 3 / 3 |

In the stress scenario, every camera loses true corner detections during the first outage, with point clutter still present. During the second outage, the diagonal camera supplies no detections of either type.

## Estimator

Every run starts from the same fixed approximate setup pose. The first frame starts about 11 mm and 4.4° from the simulated pose. This is a local tracking test with a known model, known calibration, synchronized views, and a near-pose initialization. The previous frame's estimate supplies the next prediction. Weak or absent constraints can hold the last pose until enough constraints reappear. A symmetric cuboid can have ambiguities under distant initializations.

The estimated pipeline projects visible model features from the predicted pose and associates them one-to-one with unlabeled detections. Points use pixel distance. Segments use midpoint displacement along and across the model edge, acute orientation difference, and a permitted length range. The oracle pipeline substitutes the simulator's true pairs and discards clutter. Both use the same robust damped SE(3) solver. Corner residuals are horizontal and vertical reprojection errors. Edge residuals are signed distances from both projected model endpoints to the observed supporting line. The detected segment endpoints do not have to be physical corners. The hybrid objective stacks both residual types with the same 3 px Huber threshold.

## Outputs

| File | Content |
|---|---|
| `results/summary.csv` | Translation and geodesic rotation RMSE and 95th percentiles, fit holds, pose failures, and wrong matches for each scenario/layout/view-count/association/method |
| `results/frame_metrics.csv` | Per-frame pose errors, full-rank fit decision, failure flag, and association counts |
| `results/feature_coverage.csv` | Percentage of frames with each corner or edge geometrically eligible or truly detected in at least one or two of the four cameras |
| `results/layout_coverage.csv` | Mean distinct corner or edge count per frame at each visibility threshold |
| `results/config.json` | Seed range, frame counts, rig and target sizes, diagonal centers, thresholds, and stress-outage windows |
| `results/axis_hybrid_*` | The corresponding configuration, per-frame errors, and summary for the axis-only comparator |
| `results/latency.json` | Median and 95th-percentile host wall-clock time for four-view estimated hybrids in the stress scenario |

The invalid-fit rate counts frames where the initial pose Jacobian is rank deficient and the estimate is held. The pose-failure rate counts frames above 20 mm or 8°, including held frames. RMSE and percentiles include every frame. Wrong-match rates count pairs assigned to an incorrect model ID or to clutter. A full-rank fit can still fail the pose threshold when associations are wrong. These metrics describe one synthetic trajectory and two detection models. Real camera calibration, synchronization and feature extraction require separate validation.

## Checked results

In the moving-box run, geometric eligibility in two or more cameras rose from 7.72 to 8.00 corners and from 10.12 to 11.83 edges per frame when the diagonal moved to the opposite vertex. With stress dropout and outages, true detections across two or more cameras rose from 4.12 to 4.28 corners and from 7.27 to 7.68 edges. At the centered, axis-aligned pose, the opposite diagonal increases features visible at least once from 7 to 8 corners and from 9 to 12 edges. Two-view coverage at that exact pose is 7 corners and 9 edges in each rig.

| Estimated hybrid, stress scenario | Translation RMSE, mm | Rotation RMSE, ° | Invalid fits | Pose failures |
|---|---:|---:|---:|---:|
| Same-vertex diagonal only | 20.81 | 1.93 | 8.00% | 16.00% |
| Opposite-vertex diagonal only | 18.89 | 2.12 | 8.00% | 16.44% |
| Three axis cameras | 0.76 | 0.51 | 0% | 0% |
| Three axes plus same-vertex diagonal | 0.66 | 0.45 | 0% | 0% |
| Three axes plus opposite-vertex diagonal | 0.67 | 0.46 | 0% | 0% |

The opposite diagonal improved geometric coverage. The two four-camera hybrid fits gave similar pose errors in this trial. The estimated corner-only tracker exceeded the pose threshold on 28.44% and 24.89% of stress frames, respectively. Both four-camera edge-only runs stayed within the pose thresholds, with 0.74 mm and 0.71 mm translation RMSE.

The timing command warms up for one 75-frame pass per layout and then measures 225 frame updates per layout with detections in memory. On an AMD Ryzen 7 6800H under Windows 11, the same-vertex diagonal measured 16.65 ms median and 23.99 ms at the 95th percentile. The opposite diagonal measured 19.77 ms and 27.51 ms. The timed interval covers model projection, geometric association, residual construction, and the nonlinear solve. Camera capture and synchronization, image decoding, segmentation, feature extraction, simulator work, evaluation, and file I/O are outside the timed interval. End-to-end frame rate depends on the full capture and processing pipeline.

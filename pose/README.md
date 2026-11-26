# Phase II synthetic fusion benchmark

The benchmark compares four-view projection fitting with three-of-four fusion on synthetic cuboid sequences. It also produces disjoint development and held-out data for evaluating a reliability model. The camera inputs are undistorted 2D corner candidates and line segments.

## Reproduce

From `Team7_Plate_Tracking`, using Python 3.10+ and NumPy:

```powershell
python -B -m pose.synthetic_benchmark
python -B -m pose.synthetic_benchmark --autonomous
python -B -m unittest pose.test_fusion -v
```

The paired run defaults to 48 frames per trial and 3 train, 2 validation, and 3 test trials per regime. This yields 64 trials and 3,072 frames. The autonomous run defaults to the 24 held-out test trials. `--split` selects a split and may be repeated. `--frames`, `--trials-per-regime`, and `--output-dir` set the run length, repetitions, and destination. Use a separate output directory for smoke tests.

## Input geometry and independent trials

The world is Z-up. Each rig has three near-axis cameras and one diagonal camera. The sampled rig center-to-face distance is 0.59–0.82 m, camera centers receive independent 25 mm per-axis jitter, and each camera has its own pinhole intrinsics and orientation. Box dimensions vary independently over 140–250 mm, 100–180 mm, and 55–140 mm. The front, side, top, and diagonal camera slots stay fixed. The diagonal can occupy either octant. The motion families differ by split, with a harmonic sweep for training, phase-shifted arc for validation, and orbit with reversal for testing. Each split also uses a distinct seed root. Geometry, motion, initial pose, observation noise, and fault scheduling have separate random streams within each trial.

The starting pose comes from perturbing frame-zero ground truth to simulate external initialization. This dependency is confined to simulation. Per-axis standard deviations are 8 mm and 0.045 rad multiplied by 0.65, 1.20, or 2.60 depending on trial repetition. The manifest records the realized onset errors. The solver has local gates and a hold-last policy. Its recovery tests rely on renewed measurements returning within the gates or on a valid subset recovering from the held pose.

## Fault schedule

| Regime | Injected change during the fault window |
|---|---|
| Clean | Normal pinhole projection, modest noise, trimmed edges, dropout, and nearby clutter |
| Wrong association | One camera loses most true detections and receives a coherent set of displaced decoy corners and segments |
| Unequal quality | One camera has greater corner and endpoint noise and additional independent dropout |
| External occlusion | A moving image-space rectangle hides corners and clips segments. The longest uncovered segment fragment may remain |
| Missing camera | The selected camera slot supplies empty detections |
| Extrinsic miscalibration | One camera renders measurements at a shifted and tilted pose while fusion receives the nominal calibration |
| Time skew | One camera observes the trajectory about five to eight frame periods earlier |
| Simultaneous faults | Two distinct cameras have overlapping fault windows and different fault mechanisms |

Most faults are active between approximately 24% and 72% of a trial. Timing, affected camera identity, and severity are recorded per trial. Physical correspondence labels identify corners and cuboid edges. A decoy or unrelated clutter feature has label −1. A degraded camera can retain correct physical correspondences and can remain usable on some frames, especially under partial occlusion or slow motion during time skew. Policy metrics refer to the injected condition and pose metrics independently check the resulting error.

## Four-view evidence and methods

The estimated path projects the visible cuboid from the predicted pose and uses geometric gating to match corners by pixel distance and segments by midpoint, angle, and length. It freezes those feature assignments for all four candidate fits. Each row of the 4 × 4 diagnostic matrix corresponds to the camera omitted from fitting. Each column scores that camera's prior-frozen assignments at the resulting pose, including the held-out camera when it has sufficient matches. The residual is a median of feature-level pixel errors. A point contributes an image-plane reprojection error. An edge contributes distances to its observed supporting line. Unavailable evaluations are infinite with support false. A camera with fewer than three assigned features cannot participate in a valid triple. The off-diagonal membership is fixed by construction and differs from the data-dependent support. Hypotheses share cameras and a pose prior, so the matrix entries are correlated.

| Method | Pose returned |
|---|---|
| `all` | Single joint robust fit using all available observations |
| `average` | Simple mean of translations and projected mean of rotations over every valid three-view hypothesis |
| `best` | Valid three-view hypothesis with the smallest in-fit RMS pixel residual |
| `reliable` | Best consensus among hypotheses supported by all three participating cameras, followed by joint refinement on the supported views and a new support check |
| `oracle_reliable` | The same reliability rule using true physical correspondence IDs supplied by the simulator |

The oracle receives only the true correspondence labels. It must infer view support from the data and still uses nominal calibration. The estimated path takes only the current pixel detections, model, calibration, and causal prior. The paired experiment uses the same prior from the reliability track to isolate the per-frame fusion decision. The `autonomous_*.csv` tables give each method its own causal pose history and test recovery and drift over time. At an invalid frame the tracking history remains at its last valid estimate.

## Output and metrics

| Output | Content |
|---|---|
| `manifest.json` and `autonomous_manifest.json` | Trial parameters, seeds, calibration, box dimensions, initial errors, and fault windows |
| `frame_metrics.csv` and `autonomous_frame_metrics.csv` | Per-method validity, errors, support and fault masks, physical-ID match counts |
| `summary.csv` and `autonomous_summary.csv` | All-frame and fault-active statistics by split, regime, and method |
| `reacquisition.csv` and `autonomous_reacquisition.csv` | First post-fault onset of three consecutive valid, accurate frames, with censored cases flagged |
| `failure_examples.csv` and `autonomous_failure_examples.csv` | One severe held-out example per affected regime and method, including accepted wrong poses, policy violations, or invalid holds |
| `tokens.npz` | Estimated camera features and pose hypotheses, residual and support matrices, baseline poses, and join indices. `packet_` arrays follow the neural inference contract in T123, T124, T134, T234 order |
| `targets.npz` | Ground-truth poses, camera fault and reliability labels, policy validity, per-hypothesis true errors, and known masks, keyed by `frame_key`. `target_packet_hypothesis_*` follows canonical packet order |

Split, trial, and frame identifiers are for data selection and joining. Model inputs come from the estimated packet fields.

Translation is Euclidean error in millimetres and rotation is the relative SO(3) angle in degrees. RMSE and 95th percentiles include holds. Failure means invalid or above 20 mm or 8°. Wrong-pose given valid divides inaccurate emitted estimates by emitted valid estimates. The separate policy violation rate counts valid outputs with fewer than three uninjected cameras. The aggregate false-valid frame rate combines these two conditions. Physical-ID precision counts the assigned corner or edge IDs from the emitted result's selected views, divided by their corner or edge total. For the average baseline, shared assignments are counted once per camera across its hypotheses. Invalid outputs contribute no used correspondences. This evaluates precision over assignments made, while assignment coverage and recall require separate denominators.

## Held-out autonomous results

The table pools 483 active-fault frames from 21 faulted test trials. Rotation uses the geodesic error in degrees. The policy column measures the share of emitted estimates made with fewer than three uninjected views.

| Method | RMSE, mm / ° | p95, mm / ° | Invalid | Wrong pose given valid | Policy violation given valid | Corner / edge ID precision |
|---|---:|---:|---:|---:|---:|---:|
| All | 3.85 / 2.21 | 4.63 / 1.66 | 0% | 1.24% | 12.42% | 88.70% / 89.12% |
| Average triple | 4.11 / 2.18 | 6.59 / 2.20 | 0% | 1.66% | 12.42% | 88.64% / 89.04% |
| Best triple | 2.00 / 0.77 | 2.60 / 1.07 | 0% | 0% | 12.42% | 93.32% / 93.23% |
| Reliability and joint refinement | 22.07 / 6.56 | 64.84 / 19.34 | 13.46% | 0% | 9.09% | 94.38% / 94.54% |
| Reliability with oracle correspondences | 4.96 / 0.83 | 1.48 / 0.96 | 4.14% | 0% | 8.64% | 100% / 100% |

The best triple gave the lowest pooled held-out error. The reliability rule tended to exclude clutter and kept every emitted pose inside the error thresholds. Its holds increased overall error. Three reliability trials ended the fault window in a failed state, and none regained a stable three-frame track during the remaining observation window. The initial test errors ranged from 4.95 to 43.50 mm and from 0.66° to 20.97°. For example, `test-time_skew-02` is invalid at frame 0 with 26.42 mm and 12.09° error before the fault starts. `test-unequal_quality-01` is still invalid after the fault at frame 46 with 82.85 mm and 22.90° error. Both cases expose the risk of a local tracker holding a pose outside its association gates.

The reliability rule emitted 46 of the 69 simultaneous-fault frames. On 38 of those 46 frames fewer than three cameras were uninjected. For example, `test-simultaneous-02` frame 32 is accurate to 1.36 mm and 1.65°, yet accepts a three-view support set containing an injected-fault camera. Agreement with model projections can admit affected views when the fault residual is small. Policy validity needs independent sensor-health evidence if it is to carry a stronger guarantee.

The paired comparison answers a different question. Under the common, sometimes stale reliability prior, the best triple has 21.92 mm and 6.44° pooled RMSE and emits 443 of 483 active frames, with 0.90% wrong poses among emitted estimates. The all-view fit has 63.19 mm and 8.55° RMSE over the same frames and emits 475 times, with 7.58% wrong poses among emitted estimates. Its stronger autonomous performance shows the importance of reporting the source of the frame prior alongside fusion metrics.

The project uses simulated detection pixels and known object dimensions. Performance with unknown scale, calibration error beyond the tested range, far-away initialization, occluders spanning several cameras, or different physical parts and extractors requires further validation.

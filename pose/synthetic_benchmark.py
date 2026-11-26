"""Generate held-out fault trials and evaluate diagnostic four-camera box fusion."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from pose.contracts import CAMERA_FIELDS, HYPOTHESIS_FIELDS, OMITTED, inference_packet
from .fusion import (
    TOKEN_FIELDS, AssociationLabels, BoxModel, Camera, Detections, Pose, analyze_frame,
    analyze_oracle_frame, choose_result, make_tokens, project, rotation_vector, visible,
)


FAULTS = ("wrong_association", "unequal_quality", "occlusion", "missing", "extrinsics", "time_skew", "simultaneous")
REGIMES = ("clean", *FAULTS)
SPLITS = ("train", "validation", "test")
SEED_ROOTS = (120000, 310000, 880000)
DEFAULT_TRIALS = (3, 2, 3)
METHODS = ("all", "average", "best", "reliable", "oracle_reliable")
TRANSLATION_LIMIT_MM = 20.0
ROTATION_LIMIT_DEG = 8.0


@dataclass(frozen=True)
class TrialSpec:
    split: str
    regime: str
    repeat: int
    seed: int
    frames: int

    @property
    def trial_id(self):
        return f"{self.split}-{self.regime}-{self.repeat:02d}"


@dataclass(frozen=True)
class Fault:
    kind: str
    camera: int
    start: int
    stop: int
    severity: float
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class Truth:
    pose: Pose
    labels: tuple[AssociationLabels, ...]
    fault_mask: np.ndarray


@dataclass(frozen=True)
class Frame:
    detections: tuple[Detections, ...]
    truth: Truth


@dataclass(frozen=True)
class Trial:
    spec: TrialSpec
    box: BoxModel
    cameras: tuple[Camera, ...]
    initial: Pose
    faults: tuple[Fault, ...]
    motion_family: str
    frames: tuple[Frame, ...]


def trial_specs(split, trials=None, frames=48):
    split_id = SPLITS.index(split)
    trials = DEFAULT_TRIALS[split_id] if trials is None else trials
    return [TrialSpec(split, regime, index, SEED_ROOTS[split_id] + 1000 * index + regime_id, frames)
            for index in range(trials) for regime_id, regime in enumerate(REGIMES)]


def rig_camera(name, center, intrinsic, image_size):
    forward = -center / np.linalg.norm(center)
    up = np.array([0., 1., 0.]) if name == "top" else np.array([0., 0., 1.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return Camera(name, intrinsic, np.stack((right, down, forward)), center, image_size)


def camera_rig(rng, opposite):
    half_side = rng.uniform(0.59, 0.82)
    signs = -1 if opposite else 1
    centers = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1], [signs, -signs, signs]], dtype=float) * half_side
    centers += rng.normal(0, 0.025, centers.shape)
    cameras = []
    for name, center in zip(("front", "side", "top", "diagonal"), centers, strict=True):
        focal = rng.uniform(760, 1110)
        k = np.array([[focal, 0, 480 + rng.normal(0, 12)], [0, focal * rng.uniform(0.95, 1.05),
                       360 + rng.normal(0, 12)], [0, 0, 1]])
        cameras.append(rig_camera(name, center, k, (960, 720)))
    return tuple(cameras)


def motion_function(split, rng, frames):
    family = ("harmonic_sweep", "phase_shifted_arc", "orbit_with_reversal")[SPLITS.index(split)]
    phase = rng.uniform(-np.pi, np.pi, 3)
    translation_scale = rng.uniform([0.025, 0.022, 0.015], [0.064, 0.057, 0.033])
    rotation_scale = rng.uniform([0.06, 0.09, 0.15], [0.18, 0.24, 0.37])
    center = rng.uniform(-0.026, 0.026, 3)
    orientation = rng.uniform(-0.22, 0.22, 3)
    speed = rng.uniform(3.3, 5.8) / max(frames - 1, 1)
    translation_frequency = np.array(((1, 1.31, 0.69), (0.94, 1.2, 1.15), (0.9, 1.01, 0.44)))[SPLITS.index(split)]
    rotation_frequency = np.array(((0.84, 1.11, 1.35), (1.12, 0.93, 0.67), (0.68, 0.93, 1.27)))[SPLITS.index(split)]

    def pose_at(frame):
        progress = frame * speed
        if split == "train":
            basis = np.sin(progress * translation_frequency + phase)
            rotation_basis = np.cos(progress * rotation_frequency - phase)
        elif split == "validation":
            basis = np.sin(progress * translation_frequency + phase) * np.cos(0.23 * progress + phase[::-1])
            rotation_basis = np.sin(progress * rotation_frequency + phase[::-1])
        else:
            reversal = progress + 0.63 * np.sin(2.1 * progress + phase[0])
            basis = np.sin(reversal * translation_frequency + phase)
            rotation_basis = np.cos(reversal * rotation_frequency + phase[::-1])
        return Pose(rotation_vector(orientation + rotation_scale * rotation_basis), center + translation_scale * basis)

    return family, pose_at


def fault_schedule(spec, rng):
    if spec.regime == "clean":
        return ()
    kinds = (spec.regime,)
    if spec.regime == "simultaneous":
        combinations = (("wrong_association", "missing"), ("extrinsics", "time_skew"),
                        ("occlusion", "unequal_quality"), ("missing", "extrinsics"))
        split_offset = 2 * SPLITS.index(spec.split)
        kinds = combinations[(spec.repeat + split_offset) % len(combinations)]
    slots = rng.choice(4, len(kinds), replace=False)
    return tuple(Fault(kind, int(slot), int(spec.frames * 0.24) + index * 2, int(spec.frames * 0.72) - index,
                       float(rng.uniform(0.8, 1.25)), tuple(rng.normal(size=3)))
                 for index, (kind, slot) in enumerate(zip(kinds, slots, strict=True)))


def longest_uncovered_segment(segment, low, high):
    direction = segment[1] - segment[0]
    enter, leave = 0., 1.
    for coordinate, movement, minimum, maximum in zip(segment[0], direction, low, high, strict=True):
        if abs(movement) < 1e-8:
            if not minimum <= coordinate <= maximum:
                return segment
            continue
        bounds = sorted(((minimum - coordinate) / movement, (maximum - coordinate) / movement))
        enter, leave = max(enter, bounds[0]), min(leave, bounds[1])
    if enter >= leave:
        return segment
    if enter >= 1 - leave:
        return np.stack((segment[0], segment[0] + enter * direction))
    return np.stack((segment[0] + leave * direction, segment[1]))


def shuffle(pixels, labels, rng):
    order = rng.permutation(len(labels))
    return pixels[order], labels[order]


def simulate_view(box, nominal, pose_at, frame, fault, noise_sigma, rng):
    kind = None if fault is None else fault.kind
    if kind == "missing":
        return Detections(np.empty((0, 2)), np.empty((0, 2, 2))), AssociationLabels(np.empty(0, int), np.empty(0, int))
    camera, sample_frame = nominal, frame
    if kind == "extrinsics":
        unit = np.asarray(fault.offset) / np.linalg.norm(fault.offset)
        camera = replace(nominal, center=nominal.center + unit * 0.025 * fault.severity,
                         rotation=rotation_vector(unit[::-1] * 0.035 * fault.severity) @ nominal.rotation)
    if kind == "time_skew":
        sample_frame -= int(round(6 * fault.severity))
    pose = pose_at(sample_frame)
    point_ids, points, edge_ids, edges = visible(box, pose, camera)
    trims = rng.uniform(0.03, 0.28, (len(edges), 2, 1))
    edges = edges + trims * (edges[:, ::-1] - edges)
    keep_points = rng.random(len(points)) > rng.uniform(0.04, 0.18)
    keep_edges = rng.random(len(edges)) > rng.uniform(0.05, 0.21)
    px_noise, edge_noise = noise_sigma, noise_sigma * 0.72
    if kind == "unequal_quality":
        px_noise *= 5.6 * fault.severity
        edge_noise *= 5.1 * fault.severity
        keep_points &= rng.random(len(points)) > 0.45
        keep_edges &= rng.random(len(edges)) > 0.40
    if kind == "occlusion":
        midpoint = project(np.zeros((1, 3)), pose, camera)[0][0]
        half_size = np.array([69, 52]) * fault.severity
        shift = np.asarray(fault.offset[:2]) * 24 + np.array([18 * np.sin(frame * 0.22), 15 * np.cos(frame * 0.16)])
        low, high = midpoint + shift - half_size, midpoint + shift + half_size
        keep_points &= ~np.all((points >= low) & (points <= high), axis=1)
        if len(edges):
            edges = np.stack([longest_uncovered_segment(edge, low, high) for edge in edges])
            keep_edges &= np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1) >= 9
    true_points, true_edges = points[keep_points], edges[keep_edges]
    corner_labels, edge_labels = point_ids[keep_points], edge_ids[keep_edges]
    true_points = true_points + rng.normal(0, px_noise, true_points.shape)
    true_edges = true_edges + rng.normal(0, edge_noise, true_edges.shape)
    corner_clutter = points[rng.integers(len(points), size=2)] + rng.normal(0, 20, (2, 2)) if len(points) else np.empty((0, 2))
    edge_clutter = edges[rng.integers(len(edges), size=2)] + rng.normal(0, 16, (2, 2, 2)) if len(edges) else np.empty((0, 2, 2))
    if kind == "wrong_association":
        offset = np.asarray(fault.offset[:2])
        offset *= 18 * fault.severity / max(np.linalg.norm(offset), 1e-8)
        corner_clutter = np.concatenate((corner_clutter, points + offset))
        edge_clutter = np.concatenate((edge_clutter, edges + offset))
        true_points, corner_labels = true_points[:1], corner_labels[:1]
        true_edges, edge_labels = true_edges[:1], edge_labels[:1]
    points, corner_labels = shuffle(np.concatenate((true_points, corner_clutter)),
                                    np.concatenate((corner_labels, np.full(len(corner_clutter), -1))), rng)
    edges = np.concatenate((true_edges, edge_clutter))
    reversed_rows = rng.random(len(edges)) > 0.5
    edges[reversed_rows] = edges[reversed_rows, ::-1]
    edges, edge_labels = shuffle(edges, np.concatenate((edge_labels, np.full(len(edge_clutter), -1))), rng)
    return Detections(points, edges), AssociationLabels(corner_labels, edge_labels)


def generate_trial(spec):
    geometry_seed, motion_seed, prior_seed, noise_seed, fault_seed = np.random.SeedSequence(spec.seed).spawn(5)
    geometry_rng = np.random.default_rng(geometry_seed)
    cameras = camera_rig(geometry_rng, bool(geometry_rng.integers(0, 2)))
    box = BoxModel(tuple(geometry_rng.uniform([0.14, 0.10, 0.055], [0.25, 0.18, 0.14])))
    motion_family, pose_at = motion_function(spec.split, np.random.default_rng(motion_seed), spec.frames)
    faults = fault_schedule(spec, np.random.default_rng(fault_seed))
    prior_rng = np.random.default_rng(prior_seed)
    first = pose_at(0)
    prior_scale = (0.65, 1.2, 2.6)[spec.repeat % 3]
    initial = Pose(rotation_vector(prior_rng.normal(0, 0.045 * prior_scale, 3)) @ first.rotation,
                   first.translation + prior_rng.normal(0, 0.008 * prior_scale, 3))
    frames = []
    for frame in range(spec.frames):
        active = {fault.camera: fault for fault in faults if fault.start <= frame < fault.stop}
        samples = []
        for camera_index, camera in enumerate(cameras):
            frame_rng = np.random.default_rng(np.random.SeedSequence([*noise_seed.generate_state(2), frame, camera_index]))
            sigma = frame_rng.uniform(0.65, 1.45)
            samples.append(simulate_view(box, camera, pose_at, frame, active.get(camera_index), sigma, frame_rng))
        detections, labels = zip(*samples, strict=True)
        frames.append(Frame(detections, Truth(pose_at(frame), labels, np.isin(np.arange(4), list(active)))))
    return Trial(spec, box, cameras, initial, faults, motion_family, tuple(frames))


def pose_error(pose, truth):
    translation = 1000 * np.linalg.norm(pose.translation - truth.translation)
    cosine = (np.trace(pose.rotation @ truth.rotation.T) - 1) / 2
    return float(translation), float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def correspondence_counts(evidence, prediction, labels):
    corners = edges = correct_corners = correct_edges = 0
    if prediction.valid:
        for support, match, truth in zip(prediction.support, evidence.matches, labels, strict=True):
            if not support:
                continue
            point_model, point_obs = match.point_pairs.T
            edge_model, edge_obs = match.segment_pairs.T
            corners += len(point_model)
            edges += len(edge_model)
            correct_corners += np.count_nonzero(point_model == truth.corner_ids[point_obs])
            correct_edges += np.count_nonzero(edge_model == truth.edge_ids[edge_obs])
    return dict(corner_matches=corners, correct_corner_matches=correct_corners,
                edge_matches=edges, correct_edge_matches=correct_edges)


def evaluate_trial(trial, export_tokens=True, autonomous=False):
    rows, token_rows = [], []
    priors = {method: trial.initial for method in METHODS}
    last_fault_stop = max((fault.stop for fault in trial.faults), default=-1)
    for frame_index, frame in enumerate(trial.frames):
        caches = {}
        estimates, evidence_by_method = {}, {}
        for method in METHODS:
            is_oracle = method.startswith("oracle")
            prior = priors[method if autonomous else "reliable"]
            cache_key = id(prior), is_oracle
            if cache_key not in caches:
                if is_oracle:
                    caches[cache_key] = analyze_oracle_frame(prior, trial.cameras, frame.detections, trial.box, frame.truth.labels)
                else:
                    caches[cache_key] = analyze_frame(prior, trial.cameras, frame.detections, trial.box)
            evidence_by_method[method] = caches[cache_key]
            estimates[method] = choose_result(caches[cache_key], "reliable" if is_oracle else method)
        true_pose = frame.truth.pose
        healthy = ~frame.truth.fault_mask
        for method, prediction in estimates.items():
            translation, rotation = pose_error(prediction.pose, true_pose)
            inaccurate = translation > TRANSLATION_LIMIT_MM or rotation > ROTATION_LIMIT_DEG
            counts = correspondence_counts(evidence_by_method[method], prediction, frame.truth.labels)
            rows.append(dict(split=trial.spec.split, trial=trial.spec.trial_id, regime=trial.spec.regime,
                             frame=frame_index, in_fault=not np.all(healthy), recovery=last_fault_stop >= 0 and frame_index >= last_fault_stop,
                             method=method, valid=prediction.valid, pose_failed=inaccurate, failed=inaccurate or not prediction.valid,
                             false_valid=prediction.valid and (inaccurate or healthy.sum() < 3),
                             inaccurate_valid=prediction.valid and inaccurate, fewer_than_3_healthy=healthy.sum() < 3,
                             policy_violation_valid=prediction.valid and healthy.sum() < 3,
                             translation_mm=translation, rotation_deg=rotation, support_mask=mask_string(prediction.support),
                             fault_mask=mask_string(frame.truth.fault_mask), faulty_views_accepted=int(np.count_nonzero(prediction.support & ~healthy)),
                             valid_support_count=int(prediction.support.sum()), **counts))
        reliable = estimates["reliable"]
        if export_tokens:
            evidence, prior = evidence_by_method["reliable"], priors["reliable"]
            hypothesis_errors = np.asarray([pose_error(fit.pose, true_pose) for fit in evidence.hypotheses])
            hypothesis_known = np.asarray([fit.valid for fit in evidence.hypotheses])
            packet = {f"packet_{key}": value for key, value in inference_packet(evidence).items()}
            token_rows.append(dict(
                tokens=make_tokens(evidence), residual_matrix=evidence.residuals,
                support_matrix=evidence.support, local_rotation=[fit.pose.rotation for fit in evidence.local],
                local_translation=[fit.pose.translation for fit in evidence.local], local_valid=[fit.valid for fit in evidence.local],
                hypothesis_rotation=[fit.pose.rotation for fit in evidence.hypotheses],
                hypothesis_translation=[fit.pose.translation for fit in evidence.hypotheses],
                hypothesis_valid=[fit.valid for fit in evidence.hypotheses],
                prior_rotation=prior.rotation, prior_translation=prior.translation,
                reliable_rotation=reliable.pose.rotation, reliable_translation=reliable.pose.translation,
                reliable_support=reliable.support, reliable_valid=reliable.valid,
                target_rotation=true_pose.rotation, target_translation=true_pose.translation,
                target_fault_mask=frame.truth.fault_mask, target_reliable_mask=healthy,
                target_reliability_known=np.ones(4, dtype=bool), target_valid=healthy.sum() >= 3,
                target_valid_known=True, target_hypothesis_error=hypothesis_errors,
                target_hypothesis_known=hypothesis_known, target_packet_hypothesis_error=hypothesis_errors[OMITTED],
                target_packet_hypothesis_known=hypothesis_known[OMITTED],
                split=trial.spec.split, trial_key=trial.spec.seed, frame=frame_index,
                frame_key=f"{trial.spec.seed}:{frame_index}", regime=trial.spec.regime, trial=trial.spec.trial_id, **packet,
            ))
        # Paired mode uses the reliable estimated track prior. Autonomous mode rolls out each algorithm independently.
        for method, prediction in estimates.items():
            if prediction.valid:
                priors[method] = prediction.pose
    return rows, token_rows


def mask_string(mask):
    return "".join("1" if bit else "0" for bit in mask)


def summarize(rows):
    groups = {}
    for row in rows:
        stages = ["all_frames", "active_fault"] if row["in_fault"] else ["all_frames"]
        for stage in stages:
            key = row["split"], row["regime"], stage, row["method"]
            groups.setdefault(key, []).append(row)
    aggregate = []
    for (split, regime, stage, method), group in groups.items():
        translations = np.array([row["translation_mm"] for row in group])
        rotations = np.array([row["rotation_deg"] for row in group])
        rate = lambda field: 100 * np.mean([row[field] for row in group])
        totals = {key: sum(row[key] for row in group) for key in (
            "corner_matches", "correct_corner_matches", "edge_matches", "correct_edge_matches",
        )}
        valid_count = sum(row["valid"] for row in group)
        conditional = lambda field: 100 * sum(row[field] for row in group) / valid_count if valid_count else None
        precision = lambda feature: (100 * totals[f"correct_{feature}_matches"] / totals[f"{feature}_matches"]
                                     if totals[f"{feature}_matches"] else None)
        aggregate.append(dict(split=split, regime=regime, stage=stage, method=method, frames=len(group),
                              translation_rmse_mm=np.sqrt(np.mean(translations**2)), rotation_rmse_deg=np.sqrt(np.mean(rotations**2)),
                              translation_p95_mm=np.percentile(translations, 95), rotation_p95_deg=np.percentile(rotations, 95),
                              failure_rate_pct=rate("failed"), pose_failure_rate_pct=rate("pose_failed"),
                              invalid_rate_pct=100 - rate("valid"), false_valid_rate_pct=rate("false_valid"),
                              inaccurate_valid_rate_pct=rate("inaccurate_valid"),
                              wrong_pose_given_valid_pct=conditional("inaccurate_valid"),
                              fewer_than_3_healthy_given_valid_pct=conditional("policy_violation_valid"),
                              policy_violation_valid_rate_pct=rate("policy_violation_valid"),
                              faulty_views_accepted_mean=np.mean([row["faulty_views_accepted"] for row in group]),
                              corner_precision_pct=precision("corner"), edge_precision_pct=precision("edge"), **totals))
    return aggregate


def reacquisition(rows):
    group = {}
    for row in rows:
        if row["regime"] != "clean":
            group.setdefault((row["split"], row["regime"], row["trial"], row["method"]), []).append(row)
    result = []
    for (split, regime, trial, method), sequence in group.items():
        tail = [row for row in sequence if row["recovery"]]
        active = [row for row in sequence if row["in_fault"]]
        if not tail or not active:
            continue
        had_preceding_failure = active[-1]["failed"]
        stable_recovery = next((index for index in range(len(tail) - 2)
                                if all(not row["failed"] for row in tail[index:index + 3])), None)
        result.append(dict(split=split, regime=regime, trial=trial, method=method, failed_at_fault_end=had_preceding_failure,
                           frames_to_stable_reacquisition=stable_recovery, censored=stable_recovery is None,
                           available_post_fault_frames=len(tail)))
    return result


def failure_examples(rows):
    worst = {}
    for row in rows:
        if row["split"] != "test" or not (row["failed"] or row["false_valid"]):
            continue
        key = row["regime"], row["method"]
        quality = (row["inaccurate_valid"], row["false_valid"],
                   row["translation_mm"] / TRANSLATION_LIMIT_MM + row["rotation_deg"] / ROTATION_LIMIT_DEG)
        if key not in worst or quality > worst[key][0]:
            category = ("wrong_pose_emitted" if row["inaccurate_valid"] else
                        "fewer_than_three_healthy" if row["policy_violation_valid"] else "invalid_or_held")
            worst[key] = quality, dict(row, failure_category=category)
    return [entry for _, entry in sorted(worst.values(), key=lambda item: item[0], reverse=True)]


def trial_manifest(trial):
    return dict(**asdict(trial.spec), trial_id=trial.spec.trial_id, box_m=trial.box.size_m, motion=trial.motion_family,
                faults=[asdict(fault) for fault in trial.faults],
                initial_rotation=trial.initial.rotation.tolist(), initial_translation=trial.initial.translation.tolist(),
                initial_error_mm_deg=pose_error(trial.initial, trial.frames[0].truth.pose),
                cameras=[dict(name=camera.name, center_m=camera.center.tolist(),
                              world_to_camera=camera.rotation.tolist(), intrinsic=camera.intrinsic.tolist())
                         for camera in trial.cameras])


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--trials-per-regime", type=int, help="Override the train/validation/test trial counts")
    parser.add_argument("--split", choices=SPLITS, action="append", help="Repeat to select multiple splits")
    parser.add_argument("--autonomous", action="store_true", help="Use independent tracker rollouts, with test as the default split")
    args = parser.parse_args()
    if args.frames < 16 or (args.trials_per_regime is not None and args.trials_per_regime < 1):
        parser.error("use at least 16 frames and one trial per regime")
    measurements, token_rows, trials = [], [], []
    splits = args.split or (("test",) if args.autonomous else SPLITS)
    for split in splits:
        for spec in trial_specs(split, args.trials_per_regime, args.frames):
            trial = generate_trial(spec)
            scores, tokens = evaluate_trial(trial, export_tokens=not args.autonomous, autonomous=args.autonomous)
            measurements.extend(scores)
            token_rows.extend(tokens)
            trials.append(trial_manifest(trial))
        print(f"Finished {split}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "autonomous_" if args.autonomous else ""
    write_csv(args.output_dir / f"{prefix}frame_metrics.csv", measurements)
    write_csv(args.output_dir / f"{prefix}summary.csv", summarize(measurements))
    write_csv(args.output_dir / f"{prefix}reacquisition.csv", reacquisition(measurements))
    failures = failure_examples(measurements)
    if failures:
        write_csv(args.output_dir / f"{prefix}failure_examples.csv", failures)
    (args.output_dir / f"{prefix}manifest.json").write_text(json.dumps(dict(
        translation_limit_mm=TRANSLATION_LIMIT_MM, rotation_limit_deg=ROTATION_LIMIT_DEG, trials=trials,
    ), indent=2) + "\n", encoding="utf-8")
    if args.autonomous:
        print(f"Wrote independent estimates from {len(trials)} trials to {args.output_dir}")
        return
    token_arrays = {key: np.asarray([row[key] for row in token_rows]) for key in token_rows[0]}
    targets = {key: value for key, value in token_arrays.items() if key.startswith("target_") or key in ("regime", "trial")}
    indices = {key: token_arrays[key] for key in ("frame_key", "trial_key", "frame", "split")}
    evidence = {key: value for key, value in token_arrays.items() if key not in targets and key not in indices}
    np.savez_compressed(args.output_dir / "tokens.npz", token_fields=np.asarray(TOKEN_FIELDS),
                        packet_camera_fields=np.asarray(CAMERA_FIELDS), packet_hypothesis_fields=np.asarray(HYPOTHESIS_FIELDS),
                        packet_omitted_camera=OMITTED, **indices, **evidence)
    np.savez_compressed(args.output_dir / "targets.npz", **indices, **targets)
    print(f"Wrote {len(token_rows)} frames from {len(trials)} trials to {args.output_dir}")


if __name__ == "__main__":
    main()

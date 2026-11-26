"""Run paired and causal Track A experiments on the frozen D0 development split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from pose.fusion import AssociationLabels, Detections, analyze_frame, choose_result
from pose.hypotheses import evaluate_hypotheses
from pose.synthetic_benchmark import pose_error

from .datasets import iter_trials
from .track_a_geometry import (
    EDGE_FIELDS, RANK_FIELDS, broad_pairs, candidate_targets, edge_weights, fitted_pose, geometry_pairs, learned_pairs,
    make_edge_candidates, rank_features, select_rank,
)
from .track_a_models import fit


ASSOCIATION = ("gated_geometry", "broad_geometry", "learned_logistic", "learned_mlp", "oracle_edges")
RANK = ("rank_min_rms", "rank_cross", "rank_linear", "rank_mlp", "rank_oracle")
WEIGHT = ("huber", "learned_huber")
THRESHOLDS = (0.35, 0.50, 0.65, 0.75, 0.85, 0.90)
POSE_LIMIT = (20, 8)


@dataclass(frozen=True)
class OpenFrame:
    trial: object
    number: int
    frame: object
    prior: object
    candidates: tuple
    matrix: object
    features: np.ndarray
    valid: np.ndarray
    pose_cost: np.ndarray


def pose_costs(matrix, truth, valid):
    error = np.array([pose_error(row.fit.pose, truth) for row in matrix.triples])
    return np.where(valid, np.sum(error / POSE_LIMIT, axis=1), np.inf)


def open_frames(split, repeats):
    """Use the same previous accepted Phase II reliable estimate as the D0 packet export."""
    for trial in iter_trials(split, repeats=repeats):
        prior = trial.initial
        for number, frame in enumerate(trial.frames):
            matrix = evaluate_hypotheses(prior, trial.rig, frame.detections, trial.box)
            candidates = tuple(make_edge_candidates(prior, camera, obs, trial.box)
                               for camera, obs in zip(trial.rig, frame.detections, strict=True))
            features, valid = rank_features(matrix)
            yield OpenFrame(trial, number, frame, prior, candidates, matrix, features, valid, pose_costs(matrix, frame.truth, valid))
            estimated = choose_result(analyze_frame(prior, trial.rig, frame.detections, trial.box), "reliable")
            if estimated.valid:
                prior = estimated.pose


def training_arrays(frames):
    edge_features, edge_labels = [], []
    rank_left, rank_right, rank_labels = [], [], []
    for item in frames:
        for candidates, labels in zip(item.candidates, item.frame.labels, strict=True):
            edge_features.append(candidates.features)
            edge_labels.append(candidate_targets(candidates, labels.edge_ids))
        for first, second in combinations(np.flatnonzero(item.valid), 2):
            if abs(item.pose_cost[first] - item.pose_cost[second]) < 0.015:
                continue
            rank_left.append(item.features[first])
            rank_right.append(item.features[second])
            rank_labels.append(item.pose_cost[first] < item.pose_cost[second])
    return ((np.concatenate(edge_features), np.concatenate(edge_labels)),
            (np.asarray(rank_left), np.asarray(rank_right), np.asarray(rank_labels)))


def oracle_index(cost):
    return int(np.argmin(cost)) if np.any(np.isfinite(cost)) else None


def identity_metrics(pairs, frame):
    claims = np.zeros((4, 12), dtype=bool)
    correct = np.zeros_like(claims)
    matched, exact, truths = 0, 0, 0
    for view, ((_, edges), labels) in enumerate(zip(pairs, frame.labels, strict=True)):
        true = labels.edge_ids[edges[:, 1]]
        equal = edges[:, 0] == true
        matched += len(edges)
        exact += np.count_nonzero(equal)
        truths += np.count_nonzero(labels.edge_ids >= 0)
        claims[view, edges[:, 0]] = True
        correct[view, edges[equal, 0]] = True
    joint_claimed = sum(np.count_nonzero(claims[a] & claims[b]) for a, b in combinations(range(4), 2))
    joint_correct = sum(np.count_nonzero(correct[a] & correct[b]) for a, b in combinations(range(4), 2))
    rescued = sum(np.count_nonzero(frame.occlusion_lost[view] & np.any(correct[np.arange(4) != view], axis=0))
                  for view in range(4))
    return dict(detected_true=int(truths), claimed=int(matched), correct=int(exact), cross_pairs=joint_claimed,
                correct_cross_pairs=joint_correct, occluded=int(np.count_nonzero(frame.occlusion_lost)), rescued=rescued)


def zero_matches(pairs):
    return tuple((points[:0], edges[:0]) for points, edges in pairs)


def fragment_frame(frame):
    """Lockbox-only stress transform of every long segment, including clutter."""
    detected, labelled = [], []
    for view, labels in zip(frame.detections, frame.labels, strict=True):
        segments, ids = [], []
        for edge, ident in zip(view.segments, labels.edge_ids, strict=True):
            span = edge[1] - edge[0]
            if np.linalg.norm(span) >= 34:
                segments.extend(((edge[0], edge[0] + .42 * span), (edge[0] + .56 * span, edge[1])))
                ids.extend((ident, ident))
            else:
                segments.append(edge)
                ids.append(ident)
        detected.append(Detections(view.points, np.asarray(segments, dtype=float).reshape(-1, 2, 2)))
        labelled.append(AssociationLabels(labels.corner_ids, np.asarray(ids, dtype=int)))
    return replace(frame, detections=tuple(detected), labels=tuple(labelled))


def pairs_for(method, trial, frame, prior, candidates, model=None, threshold=None):
    if method == "gated_geometry" or method in WEIGHT:
        return geometry_pairs(prior, trial.rig, frame.detections, trial.box)
    if method == "broad_geometry":
        return tuple(broad_pairs(item) for item in candidates)
    if method == "oracle_edges":
        return tuple((item.point_pairs, np.column_stack((labels.edge_ids[labels.edge_ids >= 0], np.flatnonzero(labels.edge_ids >= 0))))
                     for item, labels in zip(candidates, frame.labels, strict=True))
    return tuple(learned_pairs(item, model, threshold) for item in candidates)


def choose_threshold(dev, model):
    aggregate = {}
    for threshold in THRESHOLDS:
        correct = claimed = detected = 0
        for item in dev:
            pairs = tuple(learned_pairs(candidate, model, threshold) for candidate in item.candidates)
            counts = identity_metrics(pairs, item.frame)
            correct += counts["correct"]
            claimed += counts["claimed"]
            detected += counts["detected_true"]
        aggregate[threshold] = 2 * correct / max(claimed + detected, 1)
    chosen = max(aggregate, key=lambda value: (aggregate[value], value))
    return chosen, aggregate


def validate_rank(dev, method, model):
    regret = []
    for item in dev:
        selected = select_rank(item.matrix, item.features, item.valid, method, model)
        if selected is not None:
            regret.append(float(item.pose_cost[selected] - np.min(item.pose_cost)))
    return float(np.mean(regret)) if regret else float("inf")


def frame_record(trial, frame, number, mode, track, method, fit_pose, valid, elapsed, *, pairs=None,
                 selected=None, oracle=None, regret=None):
    mm, degrees = pose_error(fit_pose, frame.truth)
    wrong = mm > POSE_LIMIT[0] or degrees > POSE_LIMIT[1]
    claims = identity_metrics(pairs if valid else zero_matches(pairs), frame) if pairs is not None else {}
    return dict(mode=mode, track=track, method=method, trial=trial.id, regime=trial.regime, frame=number,
                active_fault=bool(np.any(frame.fault_mask)), valid=bool(valid), translation_mm=mm, rotation_deg=degrees,
                failed=not valid or wrong, inaccurate_valid=bool(valid and wrong), elapsed_ms=elapsed,
                selected=selected, oracle=oracle, regret=regret, **claims)


def run_frame(trial, frame, number, prior, mode, track, method, model=None, threshold=None, shared=None):
    tick = perf_counter_ns()
    if track == "rank":
        if shared is None:
            matrix = evaluate_hypotheses(prior, trial.rig, frame.detections, trial.box)
            features, valid = rank_features(matrix)
        else:
            matrix, features, valid = shared.matrix, shared.features, shared.valid
        selected = None if method == "rank_oracle" else select_rank(matrix, features, valid, method, model)
        elapsed = (perf_counter_ns() - tick) / 1e6
        cost = pose_costs(matrix, frame.truth, valid) if shared is None else shared.pose_cost
        oracle = oracle_index(cost)
        if method == "rank_oracle":
            selected = oracle
        fit_pose = prior if selected is None else matrix.triples[selected].fit.pose
        record = frame_record(trial, frame, number, mode, track, method, fit_pose, selected is not None, elapsed,
                              selected=selected, oracle=oracle, regret=cost[selected] - cost[oracle] if selected is not None else None)
    else:
        candidates = None
        if method not in ("gated_geometry", "huber"):
            candidates = (tuple(make_edge_candidates(prior, camera, obs, trial.box)
                                for camera, obs in zip(trial.rig, frame.detections, strict=True))
                          if shared is None else shared.candidates)
        pairs = pairs_for(method, trial, frame, prior, candidates, model, threshold)
        weights = None
        if method == "learned_huber":
            weights = tuple(edge_weights(item, model, pair[1]) for item, pair in zip(candidates, pairs, strict=True))
        fitted = fitted_pose(prior, trial.rig, frame.detections, trial.box, pairs, weights)
        fit_pose = fitted.pose
        elapsed = (perf_counter_ns() - tick) / 1e6
        record = frame_record(trial, frame, number, mode, track, method, fit_pose, fitted.valid, elapsed, pairs=pairs)
    return record, fit_pose


def summarize(rows):
    groups = {}
    for row in rows:
        key = row["mode"], row["track"], row["method"]
        groups.setdefault((*key, "all"), []).append(row)
        groups.setdefault((*key, row["regime"]), []).append(row)
    out = []
    for (mode, track, method, regime), group in groups.items():
        translation = np.array([row["translation_mm"] for row in group])
        rotation = np.array([row["rotation_deg"] for row in group])
        elapsed = [row["elapsed_ms"] for row in group]
        valid = [row for row in group if row["valid"]]
        result = dict(mode=mode, track=track, method=method, regime=regime, frames=len(group), valid=len(valid),
                      translation_rmse_mm=float(np.sqrt(np.mean(translation**2))), rotation_rmse_deg=float(np.sqrt(np.mean(rotation**2))),
                      translation_p95_mm=float(np.percentile(translation, 95)), rotation_p95_deg=float(np.percentile(rotation, 95)),
                      failure_rate=float(np.mean([row["failed"] for row in group])),
                      inaccurate_given_valid=float(np.mean([row["inaccurate_valid"] for row in valid])) if valid else None,
                      p50_ms=float(np.median(elapsed)), p95_ms=float(np.percentile(elapsed, 95)))
        if track == "rank":
            result.update(rank_oracle_candidate_frames=sum(row["oracle"] is not None for row in group),
                          top1_among_selected=float(np.mean([row["selected"] == row["oracle"] for row in valid])) if valid else None,
                          mean_regret_among_selected=float(np.mean([row["regret"] for row in valid])) if valid else None)
        else:
            totals = {name: sum(row[name] for row in group)
                      for name in ("claimed", "correct", "detected_true", "cross_pairs", "correct_cross_pairs", "occluded", "rescued")}
            count = lambda numerator, denominator: totals[numerator] / totals[denominator] if totals[denominator] else None
            result.update(**totals, edge_precision=count("correct", "claimed"), edge_recall=count("correct", "detected_true"),
                          edge_F1=(2 * totals["correct"] / (totals["claimed"] + totals["detected_true"])
                                   if totals["claimed"] + totals["detected_true"] else None),
                          physical_pair_accuracy=count("correct_cross_pairs", "cross_pairs"), occluded_edge_recovery=count("rescued", "occluded"))
        out.append(result)
    return out


def write_rows(path, rows):
    columns = dict.fromkeys(field for row in rows for field in row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def train_models(edge_arrays, rank_arrays):
    models, timings = {}, {}
    configurations = (("learned_logistic", *edge_arrays, None, 0), ("learned_mlp", *edge_arrays, None, 16),
                      ("rank_linear", rank_arrays[0], rank_arrays[2], rank_arrays[1], 0),
                      ("rank_mlp", rank_arrays[0], rank_arrays[2], rank_arrays[1], 16))
    for name, features, labels, paired, hidden in configurations:
        start = perf_counter_ns()
        models[name] = fit(features, labels, hidden_width=hidden, paired=paired)
        timings[name] = (perf_counter_ns() - start) / 1e6
    return models, timings


def save_checkpoints(output, models, timings, preparation_ms):
    arrays, info = {}, {}
    for name, model in models.items():
        digest = hashlib.sha256()
        weights = dict(center=model.center, scale=model.scale)
        weights.update({f"w{i}": values for i, values in enumerate(model.weights)})
        for key, values in weights.items():
            digest.update(f"{key} {values.dtype.str} {values.shape}".encode())
            digest.update(np.ascontiguousarray(values).tobytes())
            arrays[f"{name}_{key}"] = values
        info[name] = {"sha256_arrays": digest.hexdigest(), "optimization_ms": timings[name],
                      "parameter_count": sum(value.size for value in model.weights)}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "models.npz", **arrays)
    (output / "training.json").write_text(json.dumps({"train_feature_preparation_ms": preparation_ms, "models": info},
                                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/neural/track_a_results"))
    parser.add_argument("--repeats", type=int, help="Use 1 repeat per regime as a smoke check")
    parser.add_argument("--prepare-only", action="store_true", help="Export the train checkpoints without reading lockbox")
    args = parser.parse_args()
    start = perf_counter_ns()
    edge_arrays, rank_arrays = training_arrays(open_frames("train", args.repeats))
    preparation_ms = (perf_counter_ns() - start) / 1e6
    models, timings = train_models(edge_arrays, rank_arrays)
    save_checkpoints(args.output, models, timings, preparation_ms)
    if args.prepare_only:
        print(f"Exported {len(models)} locked train checkpoints", flush=True)
        return
    dev = list(open_frames("dev", args.repeats))
    threshold_search = {name: choose_threshold(dev, models[name]) for name in ("learned_logistic", "learned_mlp")}
    threshold = {name: score[0] for name, score in threshold_search.items()}
    rank_validation = {name: validate_rank(dev, name, models[name]) for name in ("rank_linear", "rank_mlp")}
    selected_assoc = max(threshold, key=lambda name: threshold_search[name][1][threshold[name]])
    selected_rank = min(rank_validation, key=rank_validation.get)
    models["learned_huber"] = models[selected_assoc]
    del dev
    print(f"Development choice {selected_assoc}, threshold {threshold[selected_assoc]} and {selected_rank}", flush=True)
    rows = []
    groups = {"association": ASSOCIATION, "rank": RANK, "weight": WEIGHT}
    lockbox = open_frames("lockbox", args.repeats)
    for item in lockbox:
        for track, methods in groups.items():
            for method in methods:
                row, _ = run_frame(item.trial, item.frame, item.number, item.prior, "open", track, method,
                                   models.get(method), threshold.get(method), shared=item)
                rows.append(row)
        if item.number % 4 == 3:
            fragment = fragment_frame(item.frame)
            for method in ("gated_geometry", "broad_geometry", selected_assoc, "oracle_edges"):
                row, _ = run_frame(item.trial, fragment, item.number, item.prior, "open_fragment", "association",
                                   method, models.get(method), threshold.get(method))
                rows.append(row)
    print("Completed open-loop lockbox", flush=True)
    closed_groups = {"association": ("gated_geometry", "broad_geometry", selected_assoc),
                     "rank": ("rank_min_rms", "rank_cross", selected_rank, "rank_oracle"), "weight": WEIGHT}
    for trial in iter_trials("lockbox", repeats=args.repeats):
        for track, methods in closed_groups.items():
            for method in methods:
                prior = trial.initial
                for number, frame in enumerate(trial.frames):
                    row, current = run_frame(trial, frame, number, prior, "closed", track, method,
                                             models.get(method), threshold.get(method))
                    if row["valid"]:
                        prior = current
                    rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "frames.csv", rows)
    write_rows(args.output / "summary.csv", summarize(rows))
    report = dict(dataset="D0", repeats_override=args.repeats, edge_training_pairs=len(edge_arrays[0]),
                  edge_training_positive_pairs=int(edge_arrays[1].sum()), ranking_training_pairs=len(rank_arrays[0]),
                  selected_association=selected_assoc, selected_ranker=selected_rank, dev_edge_threshold_search=threshold_search,
                  dev_rank_mean_regret=rank_validation, edge_features=EDGE_FIELDS, rank_features=RANK_FIELDS,
                  pose_failure_mm_deg=POSE_LIMIT, lockbox_rows=len(rows))
    (args.output / "protocol.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Completed closed-loop lockbox and wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()

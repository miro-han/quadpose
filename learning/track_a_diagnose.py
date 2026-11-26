"""Replay a failed Track A trial with frozen weights and write evaluator-only evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from pose.fusion import EDGES, build_problem, huber, project, visible
from pose.refinement import Config, edge_gate, final_observability, view_problems
from pose.synthetic_benchmark import pose_error

from .datasets import iter_trials
from .track_a import run_frame, summarize, write_rows
from .track_a_geometry import fitted_pose, learned_pairs, make_edge_candidates
from .track_a_models import Model


def load_model(checkpoint, name):
    with np.load(checkpoint, allow_pickle=False) as arrays:
        values = tuple(arrays[f"{name}_w{i}"] for i in range(4) if f"{name}_w{i}" in arrays)
        return Model(arrays[f"{name}_center"], arrays[f"{name}_scale"], values)


def finite(value):
    return float(value) if np.isfinite(value) else None


def replay(model, threshold, trial):
    prior = trial.initial
    rows = []
    for number, frame in enumerate(trial.frames):
        candidates = tuple(make_edge_candidates(prior, camera, observed, trial.box)
                           for camera, observed in zip(trial.rig, frame.detections, strict=True))
        pairs = tuple(learned_pairs(item, model, threshold) for item in candidates)
        fit = fitted_pose(prior, trial.rig, frame.detections, trial.box, pairs)
        matches = view_problems(trial.rig, frame.detections, trial.box, pairs)
        problem = build_problem(trial.box, matches)
        geometry, supports = final_observability(fit.pose, trial.box, matches, pairs, Config(), problem)
        input_residual, jacobian = problem.evaluate(prior, jacobian=True)
        output_residual = problem.evaluate(fit.pose)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        cameras = []
        for camera, raw, (points, edges), labels, fault, candidate in zip(
            trial.rig, frame.detections, pairs, frame.labels, frame.fault_mask, candidates, strict=True,
        ):
            shown_corners, _, shown_edges, _ = visible(trial.box, prior, camera)
            image, depth = project(trial.box.vertices, fit.pose, camera)
            front = depth > 0
            inside = front & np.all((image >= 0) & (image < camera.image_size), axis=1)
            confidence = {(int(candidate.edge_ids[row]), int(col)): float(p) for row, col, p in
                          zip(candidate.rows, candidate.columns, model.probability(candidate.features), strict=True)}
            after_edges = visible(trial.box, fit.pose, camera)[2]
            edge_trace = []
            for model_id, index in edges:
                projected = image[EDGES[model_id]]
                observed = raw.segments[index]
                final_gate = edge_gate(projected[None], observed[None], np.array([model_id]), Config(), verification=True)
                edge_trace.append(dict(claimed_id=int(model_id), observation=int(index), physical_id=int(labels.edge_ids[index]),
                    score=confidence[int(model_id), int(index)], output_model_edge_visible=bool(model_id in after_edges),
                    output_projected_length_px=float(np.linalg.norm(projected[1] - projected[0])),
                    observed_length_px=float(np.linalg.norm(observed[1] - observed[0])),
                    postfit_geometric_gate=bool(np.isfinite(final_gate.cost[0, 0]))))
            cameras.append(dict(
                name=camera.name, fault_active=bool(fault), input_corner_count=len(raw.points), input_edge_count=len(raw.segments),
                prior_visible_model_corner_count=len(shown_corners), prior_visible_model_edge_count=len(shown_edges),
                output_vertices_in_front=int(np.count_nonzero(front)), output_vertices_inside=int(np.count_nonzero(inside)),
                pose_min_depth_m=float(depth.min()), pose_max_depth_m=float(depth.max()),
                corners=[dict(claimed_id=int(model_id), observation=int(index), physical_id=int(labels.corner_ids[index]))
                         for model_id, index in points],
                edges=edge_trace,
            ))
        rows.append(dict(
            frame=number, accepted=fit.valid, translation_error_mm=pose_error(fit.pose, frame.truth)[0],
            rotation_error_deg=pose_error(fit.pose, frame.truth)[1], prior_translation_m=prior.translation.tolist(),
            candidate_translation_m=fit.pose.translation.tolist(), truth_translation_m=frame.truth.translation.tolist(),
            prior_to_candidate_mm=1000 * float(np.linalg.norm(fit.pose.translation - prior.translation)),
            candidate_residual_equations=len(problem.vertices), initial_singular_values=singular.tolist(),
            reported_initial_condition=finite(fit.condition), prior_huber=float(huber(input_residual)) if len(input_residual) else None,
            final_huber=float(huber(output_residual)) if len(output_residual) else None, final_residual_rms_px=finite(fit.rms),
            rechecked_rank=geometry.rank, rechecked_normalized_min_eigenvalue=geometry.min_eigenvalue,
            rechecked_normalized_condition=finite(geometry.condition), rechecked_view_support=supports.tolist(), cameras=cameras,
        ))
        if fit.valid:
            prior = fit.pose
    return rows


def axis_fixture(model, threshold, model_name):
    from .occlusion_sanity import iter_axis_trials

    rows = []
    for occluded in (False, True):
        for trial in iter_axis_trials(occluded):
            for method in ("gated_geometry", "broad_geometry", model_name):
                prior = trial.initial
                for number, frame in enumerate(trial.frames):
                    row, pose = run_frame(trial, frame, number, prior, "axis_fixture_closed", "association", method,
                                          model, threshold)
                    rows.append(row)
                    if row["valid"]:
                        prior = pose
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/neural/track_a_results"))
    parser.add_argument("--trial", default="D0-lockbox-76405")
    parser.add_argument("--axis-fixture", action="store_true", help="Evaluate the paired axis-ray sphere fixture too")
    args = parser.parse_args()
    protocol = json.loads((args.results / "protocol.json").read_text(encoding="utf-8"))
    name = protocol["selected_association"]
    threshold = protocol["dev_edge_threshold_search"][name][0]
    model = load_model(args.results / "models.npz", name)
    trial = next(trial for trial in iter_trials("lockbox") if trial.id == args.trial)
    rows = replay(model, threshold, trial)
    (args.results / "outlier_trace.json").write_text(json.dumps(dict(trial=trial.id, regime=trial.regime, model=name,
        frozen_dev_threshold=threshold, frames=rows), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.axis_fixture:
        axes = axis_fixture(model, threshold, name)
        write_rows(args.results / "axis_fixture_frames.csv", axes)
        write_rows(args.results / "axis_fixture_summary.csv", summarize(axes))
        print(f"Wrote {len(axes)} fixed-threshold axis fixture rows")


if __name__ == "__main__":
    main()

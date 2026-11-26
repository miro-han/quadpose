"""Observation-only state and action-conditioned Quadpose replay."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from pose.fusion import associate_view, build_problem, solve_pose, view_match
from pose.hypotheses import evaluate_hypotheses, pose_distance, select_pose
from pose.recovery import RecoveryConfig, partial_pose, predict_pose

ACTIONS = ("all", "cross", "cross_verify", "partial", "predict")


@dataclass(frozen=True)
class State:
    history: tuple
    misses: int = 0


def features(state, data):
    counts = np.array([(len(item.points), len(item.segments)) for item in data])
    speed = pose_distance(state.history[-2][1], state.history[-1][1]) if len(state.history) == 2 else (0., 0.)
    return np.r_[np.log1p(counts.ravel()), min(state.misses, 8), speed[0] * 100, speed[1] / 20]


def action_mask(data):
    present = sum(len(item.points) + len(item.segments) >= 3 for item in data)
    return np.array([present >= 1, present >= 3, present >= 3, 1 <= present <= 2, True])


def transition(state, action, stamp, cameras, data, box):
    """Run only the chosen strategy; no candidate fits enter the policy state."""
    started = perf_counter()
    prior = state.history[-1][1]
    if state.misses < 3 and len(state.history) > 1:
        prior = predict_pose(list(state.history), stamp)
    if action == "all":
        matches = [view_match(camera, detection, *associate_view(prior, camera, detection, box))
                   for camera, detection in zip(cameras, data, strict=True)]
        choice = solve_pose(prior, build_problem(box, matches))
    elif action == "partial":
        choice = partial_pose(prior, cameras, data, box, 1. + .55 * min(state.misses, 3), RecoveryConfig())
    elif action == "predict":
        choice = None
    else:
        matrix = evaluate_hypotheses(prior, cameras, data, box)
        choice = select_pose(matrix, action)
    measured = choice is not None and choice.valid
    if measured:
        next_state = State((state.history + ((stamp, choice.pose),))[-2:])
        return next_state, choice.pose, "measured", (perf_counter() - started) * 1e3
    status = "predicted" if state.misses < 3 else "lost"
    return State(state.history, state.misses + 1), prior, status, (perf_counter() - started) * 1e3


def utility(pose, status, truth, action):
    """Targets used in simulation only. Fixed proxy costs avoid learning host-load artifacts."""
    mm, deg = pose_distance(pose, truth)
    mm *= 1000.
    accurate = mm <= 20 and deg <= 8
    cost = {"all": .03, "cross": .10, "cross_verify": .18, "partial": .06, "predict": .0}[action]
    reward = (1. if status == "measured" and accurate else
              -2. if status == "measured" else -.9 if status == "lost" else -.45)
    return reward - min(2., mm/40 + deg/25) - cost, mm, deg


def heuristic(mask):
    return 1 if mask[1] else 3 if mask[3] else 0 if mask[0] else 4

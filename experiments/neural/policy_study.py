"""D0 behavior cloning vs sequential policy gradients on action-conditioned replay."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

from learning.datasets import iter_trials
from learning.policy import ACTIONS, State, action_mask, features, heuristic, transition, utility
import numpy as np
import torch
from torch import nn


def run(trials, model=None, *, collect=False, explore=False, all_camera=False):
    records, train_rows, episode_losses = [], [], []
    for trial in trials:
        state = State(((-1/30, trial.initial),))
        logs, rewards = [], []
        for i, frame in enumerate(trial.frames):
            x = features(state, frame.detections)
            mask = action_mask(frame.detections)
            stamp = i / 30
            args = (stamp, trial.rig, frame.detections, trial.box)
            decision_start = perf_counter()
            if model is None:
                a = (0 if mask[0] else 4) if all_camera else heuristic(mask)
            else:
                logits = model(torch.as_tensor(x, dtype=torch.float32)).masked_fill(~torch.as_tensor(mask), -1e8)
                distribution = torch.distributions.Categorical(logits=logits)
                sample = distribution.sample() if explore else logits.argmax()
                a = sample.item()
                if explore:
                    logs.append(distribution.log_prob(sample))
            decision_ms = (perf_counter() - decision_start) * 1000
            if collect:
                # One-step teacher branches from the identical realized state.
                alternatives = [transition(state, ACTIONS[j], *args) if m else None for j,m in enumerate(mask)]
                targets = [utility(v[1], v[2], frame.truth, ACTIONS[j])[0] if v is not None else -1e9
                           for j,v in enumerate(alternatives)]
                train_rows.append((x, mask, int(np.argmax(targets))))
                update = alternatives[a]
            else:
                update = transition(state, ACTIONS[a], *args)
            state, pose, status, latency = update
            reward, mm, deg = utility(pose, status, frame.truth, ACTIONS[a])
            rewards.append(reward)
            records.append(dict(trial=trial.id, frame=i, action=ACTIONS[a], status=status, reward=reward,
                                mm=mm, deg=deg, latency_ms=latency, decision_ms=decision_ms))
        if explore:
            discounted = []
            future = 0.
            for reward in reversed(rewards):
                future = reward + .9 * future
                discounted.append(future)
            returns = torch.tensor(discounted[::-1])
            advantage = (returns-returns.mean())/(returns.std(unbiased=False)+1e-6)
            episode_losses.append(-(torch.stack(logs)*advantage).mean())
    return records, train_rows, episode_losses


def summary(rows):
    measured = [x for x in rows if x["status"] == "measured"]
    correct = sum(x["mm"]<=20 and x["deg"]<=8 for x in measured)
    return dict(frames=len(rows), measured=len(measured), success_pct=100*correct/len(rows),
                false_valid_pct=100*(len(measured)-correct)/max(len(measured),1),
                return_per_frame=float(np.mean([x["reward"] for x in rows])),
                translation_rmse_mm=float(np.sqrt(np.mean([x["mm"]**2 for x in rows]))),
                rotation_rmse_deg=float(np.sqrt(np.mean([x["deg"]**2 for x in rows]))),
                backend_p95_ms=float(np.quantile([x["latency_ms"] for x in rows],.95)),
                decision_p95_ms=float(np.quantile([x["decision_ms"] for x in rows],.95)),
                actions=dict(Counter(row["action"] for row in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12, help="maximum trials per split")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--eval-split", choices=("dev", "lockbox"), default="dev")
    parser.add_argument("--resume-rl", type=Path, help="Warm-start RL from a previous policy study checkpoint")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    from itertools import islice
    train = list(islice(iter_trials("train"), args.limit))
    evaluation = list(islice(iter_trials(args.eval_split), args.limit))
    started = perf_counter()
    _, samples, _ = run(train, collect=True)
    generation_seconds = perf_counter() - started
    x, masks, labels = zip(*samples)
    x = torch.tensor(np.stack(x), dtype=torch.float32)
    masks = torch.tensor(np.stack(masks))
    labels = torch.tensor(labels)
    model = nn.Sequential(nn.Linear(x.shape[1], 24), nn.Tanh(), nn.Linear(24, len(ACTIONS)))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses = []
    started = perf_counter()
    for _ in range(70):
        loss = nn.functional.cross_entropy(model(x).masked_fill(~masks, -1e8), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    bc_seconds = perf_counter() - started
    rl = copy.deepcopy(model)
    if args.resume_rl is not None:
        rl.load_state_dict(torch.load(args.resume_rl, map_location="cpu", weights_only=True))
    optimizer = torch.optim.Adam(rl.parameters(), lr=.002)
    started = perf_counter()
    rl_losses = []
    train_rewards = []
    for _ in range(3):
        rows, _, terms = run(train, rl, explore=True)
        loss = torch.stack(terms).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rl_losses.append(loss.item())
        train_rewards.append(summary(rows)["return_per_frame"])
    rl_seconds = perf_counter() - started
    args.output.mkdir(parents=True)
    reports = {}
    for name,policy in (("heuristic",None),("all_camera_velocity",None),("behavior_cloning",model),("reinforce",rl)):
        with torch.no_grad():
            rows, _, _ = run(evaluation, policy, all_camera=name=="all_camera_velocity")
        with (args.output / (name+".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        reports[name] = summary(rows)
    details = dict(schema="quadpose-policy-v0", seed=args.seed, train=[x.id for x in train],
                   evaluation_split=args.eval_split, evaluation=[x.id for x in evaluation],
                   actions=ACTIONS, losses=losses, rl_losses=rl_losses,
                   rl_train_reward=train_rewards, generation_seconds=generation_seconds,
                   bc_seconds=bc_seconds, rl_seconds=rl_seconds,
                   parent_checkpoint_sha256=(hashlib.sha256(args.resume_rl.read_bytes()).hexdigest()
                                             if args.resume_rl is not None else None),
                   parameters=sum(p.numel() for p in model.parameters()), results=reports)
    for name, network in (("behavior_cloning",model),("reinforce",rl)):
        checkpoint = args.output / (name+".pt")
        torch.save(network.state_dict(), checkpoint)
        details.setdefault("checkpoint_sha256", {})[name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (args.output / "summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

"""Paired trajectory bootstrap for closed-loop edge association comparisons."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ("broad_geometry", "learned_mlp")


def per_trajectory(rows, method):
    trials = sorted({row["trial"] for row in rows})
    records = []
    for trial in trials:
        group = [r for r in rows if r["trial"] == trial and r["method"] == method]
        if not group:
            raise ValueError(f"Missing {method} on {trial}")
        records.append([len(group), sum(r["failed"] == "True" for r in group),
                        sum(float(r["translation_mm"])**2 for r in group),
                        sum(int(r["correct"]) for r in group),
                        sum(int(r["claimed"])+int(r["detected_true"]) for r in group)])
    return np.array(records)


def metrics(data):
    frames, failure, squared, correct, possible = np.moveaxis(data, -1, 0)
    return np.stack((100*(1-failure/frames), np.sqrt(squared/frames), 200*correct/possible), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/neural/track_a_results/frames.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["mode"]=="closed" and r["track"]=="association"
                and r["method"] in METHODS]
    base, learned = [per_trajectory(rows, method) for method in METHODS]
    seed, draws = 8841, 5000
    indices = np.random.default_rng(seed).integers(len(base), size=(draws,len(base)))
    delta = metrics(learned[indices].sum(axis=1)) - metrics(base[indices].sum(axis=1))
    observed = metrics(learned.sum(axis=0)) - metrics(base.sum(axis=0))
    intervals = np.quantile(delta,[.025,.975],axis=0)
    result = dict(trials=len(base), draws=draws, seed=seed, baseline=METHODS[0], learned=METHODS[1],
                  differences={key: dict(observed=float(observed[i]), p2_5=float(intervals[0,i]),
                                         p97_5=float(intervals[1,i]))
                               for i,key in enumerate(("success_pp", "translation_rmse_mm", "edge_f1_pp"))})
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2)


if __name__ == "__main__":
    main()

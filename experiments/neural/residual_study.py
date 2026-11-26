"""Compare bounded translation correction from GT, teacher, and held-out reprojection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from learning.datasets import iter_trials
from learning.policy import State, features
from pose.hypotheses import evaluate_hypotheses, pose_distance
from pose.fusion import Detections, Pose
from pose.recovery import predict_pose

MODES = ("supervised", "distilled", "self_supervised")


def extract(split):
    samples = []
    for trial in iter_trials(split):
        history = ((-1/30, trial.initial),)
        for index, frame in enumerate(trial.frames):
            prior = history[-1][1]
            matrix = evaluate_hypotheses(prior, trial.rig, frame.detections, trial.box)
            # Fixed held-out slot index, assigned from the prior before fitting any triple.
            triple = matrix.triples[index % 4]
            fit_inputs = list(frame.detections)
            fit_inputs[triple.omitted] = Detections(np.empty((0, 2)), np.empty((0, 2, 2)))
            feature_history = history
            if matrix.all.fit.valid:
                history = (history + ((index/30, matrix.all.fit.pose),))[-2:]
            if not triple.fit.valid:
                continue
            base = triple.fit.pose
            held = matrix.problems[triple.omitted]
            residual, jacobian = held.evaluate(base, jacobian=True)
            if not len(residual):
                continue
            teacher = matrix.all
            # Sample gate uses inference-time evidence only.
            confident = teacher.fit.valid and np.count_nonzero(teacher.supported)>=3 and teacher.fit.rms<=4
            x = np.r_[features(State(feature_history), fit_inputs), triple.omitted == np.arange(4),
                      np.log1p(triple.fit.rms), base.translation*10]
            samples.append((trial.id,index,x,base,frame.truth,teacher.fit.pose,confident,held))
    return samples


class Correction(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.register_buffer("mean", x.mean(dim=0))
        self.register_buffer("scale", x.std(dim=0,unbiased=False).clamp_min(.08))
        self.layers = nn.Sequential(nn.Linear(x.shape[-1], 24), nn.Tanh(), nn.Linear(24, 3))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return .025 * torch.tanh(self.layers((x-self.mean)/self.scale))


def tensor(x):
    return torch.as_tensor(np.array(x), dtype=torch.float32)


def training(samples, mode, device):
    if mode == "distilled":
        samples = [item for item in samples if item[6]]
    x = tensor([s[2] for s in samples]).to(device)
    t = [s[3].translation for s in samples]
    if mode != "self_supervised":
        target = tensor([(s[5] if mode=="distilled" else s[4]).translation-old
                         for s,old in zip(samples,t,strict=True)]).to(device)
    else:
        max_rows = max(len(s[7].vertices) for s in samples)
        res = np.zeros((len(samples),max_rows),np.float32)
        derivative = np.zeros((*res.shape, 3),np.float32)
        valid = np.zeros(res.shape,bool)
        for i,s in enumerate(samples):
            r,j = s[7].evaluate(s[3],jacobian=True)
            res[i,:len(r)], derivative[i,:len(r)], valid[i,:len(r)] = r,j[:,3:],True
        residual, jac = tensor(res).to(device),tensor(derivative).to(device)
        valid = torch.as_tensor(valid,device=device)
    model = Correction(x).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.006)
    if device=="cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    losses = []
    for _ in range(100):
        delta = model(x)
        if mode=="self_supervised":
            # Differentiation of a first-order reprojection model at frozen pose/association.
            projected = residual + torch.einsum("nki,ni->nk",jac,delta)
            loss = nn.functional.huber_loss(projected[valid],torch.zeros_like(projected[valid]),delta=3.)
            loss = loss + .15*(delta*100).square().mean()
        else:
            loss = nn.functional.huber_loss(delta*100,target*100,delta=.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    if device=="cuda":
        torch.cuda.synchronize()
    return model, dict(seconds=perf_counter()-started, rows=len(x), loss=[v.item() for v in losses])


def report(samples, model, device):
    prior_errors, errors, residuals = [], [], []
    with torch.no_grad():
        for trial,index,x,base,truth,_,_,held in samples:
            started = perf_counter()
            delta = (model(tensor(x).to(device)).cpu().numpy() if model is not None else np.zeros(3))
            duration = (perf_counter()-started)*1e3
            new_pose = Pose(base.rotation,base.translation+delta)
            old, _ = pose_distance(base,truth)
            position,rotation = pose_distance(new_pose,truth)
            prior_errors.append(old)
            errors.append(position)
            residuals.append(dict(trial=trial,frame=index,translation_mm=position*1000,rotation_deg=rotation,
                                  worsened=position>old+1e-7,heldout_median_px=float(np.median(np.abs(held.evaluate(new_pose)))),
                                  correction_ms=duration))
    return residuals, dict(conditional_frames=len(errors),translation_rmse_mm=float(1000*np.sqrt(np.mean(np.square(errors)))),
                           translation_p95_mm=float(1000*np.quantile(errors,.95)),
                           worsened_pct=100*sum(r["worsened"] for r in residuals)/len(residuals),
                           heldout_px_p50=float(np.median([r["heldout_median_px"] for r in residuals])),
                           correction_p95_ms=float(np.quantile([r["correction_ms"] for r in residuals],.95)))


def temporal_rollout():
    """Evaluate causal extrapolation and fixed/learned gain on separate dev trajectories."""
    from pose.hypotheses import select_pose
    outcomes = {name: [] for name in ("hold","velocity","fixed_gain","adaptive_fit")}
    rng = []
    for trial in iter_trials("train"):
        h = ((-1/30,trial.initial),)
        for i,frame in enumerate(trial.frames):
            p = h[-1][1]
            candidate = select_pose(evaluate_hypotheses(p,trial.rig,frame.detections,trial.box),"all")
            if candidate.valid:
                diff = candidate.pose.translation - p.translation
                den = diff @ diff
                if den > 1e-9:
                    z = np.clip((frame.truth.translation-p.translation)@diff/den,0,1)
                    rng.append((candidate.diagnostics["residual_rms_px"],z))
                h = (h+((i/30,candidate.pose),))[-2:]
    gain = np.linalg.lstsq(np.array([[1,np.log1p(r)] for r,_ in rng]), np.array([z for _,z in rng]),rcond=None)[0]
    for trial in iter_trials("dev"):
        for name, rows in outcomes.items():
            h = ((-1/30,trial.initial),)
            for i,frame in enumerate(trial.frames):
                stamp = i/30
                p = predict_pose(list(h),stamp) if name=="velocity" and len(h)==2 else h[-1][1]
                candidate = select_pose(evaluate_hypotheses(p,trial.rig,frame.detections,trial.box),"all")
                if candidate.valid:
                    pose = candidate.pose
                    if name in ("fixed_gain","adaptive_fit"):
                        a = .7 if name=="fixed_gain" else np.clip(gain@[1,np.log1p(candidate.diagnostics["residual_rms_px"])],.1,1.)
                        pose = Pose(pose.rotation,(1-a)*p.translation+a*pose.translation)
                    h=(h+((stamp,pose),))[-2:]
                else:
                    pose = p
                mm,deg = pose_distance(pose,frame.truth)
                rows.append((mm*1000,deg,candidate.valid))
    result = {}
    for name,rows in outcomes.items():
        a = np.array(rows)
        result[name] = dict(frames=len(a), translation_rmse_mm=float(np.sqrt(np.mean(a[:,0]**2))),
                            rotation_rmse_deg=float(np.sqrt(np.mean(a[:,1]**2))),
                            success_pct=float(100*np.mean((a[:,0]<=20)&(a[:,1]<=8)&(a[:,2]>0))))
    return dict(gain=gain.tolist(), results=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cuda",action="store_true",help="compare supervised GPU training with CPU")
    parser.add_argument("--temporal",action="store_true")
    parser.add_argument("--seed",type=int,default=907)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    if args.cuda and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    start=perf_counter()
    train,dev=extract("train"),extract("dev")
    generation_seconds=perf_counter()-start
    args.output.mkdir(parents=True)
    results={}
    digest={}
    fields=[]
    options=[("geometry",None,"cpu")]+[(mode,mode,"cpu") for mode in MODES]
    if args.cuda:
        options.append(("supervised_cuda","supervised","cuda"))
        torch.empty(1,device="cuda").add_(1)
        torch.cuda.synchronize()
    for label,mode,device in options:
        torch.manual_seed(args.seed)
        if device=="cuda":
            torch.cuda.reset_peak_memory_stats()
        model,stats=training(train,mode,device) if mode else (None,{})
        rows,metric=report(dev,model,device)
        results[label]=metric|stats
        if device=="cuda":
            results[label]["torch_peak_allocated_mib"]=torch.cuda.max_memory_allocated()/2**20
        if model is not None:
            path=args.output/(label+".pt")
            torch.save(model.state_dict(),path)
            digest[label]=hashlib.sha256(path.read_bytes()).hexdigest()
        fields.extend(dict(model=label,**row) for row in rows)
    with (args.output/"frames.csv").open("w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(fields[0]))
        writer.writeheader()
        writer.writerows(fields)
    result=dict(seed=args.seed, generation_seconds=generation_seconds, train_rows=len(train),
                dev_rows=len(dev), results=results, checkpoint_sha256=digest,
                input_schema="quadpose-heldout-translation-v0")
    if args.temporal:
        result["temporal"]=temporal_rollout()
    (args.output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()


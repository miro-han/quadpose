"""Train two small camera reliability classifiers on NumPy CPU and explicit AMD OpenCL."""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import time

from evaluation.resource_limits import limit_process


if __name__ == "__main__":
    limit_process(threads=1, memory_gib=3)

import numpy as np

from experiments.amd.opencl.measure import process_video_memory
from .model import AMDBatch, Batch, Model, cpu_forward_backward
from .native import AMD


ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "validation", "test")
SAMPLES_PER_TRIAL = (16, 12, 12)
RATE = np.float32(0.18)
WARMUP_STEPS = 4


def selected_indices(data, split, count):
    indices = []
    for trial in np.unique(data["trial_key"][data["split"] == split]):
        trial_rows = np.flatnonzero((data["split"] == split) & (data["trial_key"] == trial))
        ordered = trial_rows[np.argsort(data["frame"][trial_rows])]
        indices.extend(ordered[np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)])
    return np.asarray(indices, dtype=int)


def standardize_with_bias(array, train):
    train_data = array[train]
    center = train_data.mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(train_data.std(axis=(0, 1), keepdims=True), 0.035)
    standardized = np.clip((array - center) / scale, -5, 5)
    return np.concatenate((standardized, np.ones((*array.shape[:2], 1), dtype=np.float32)), axis=2)


def load_batches(root):
    with np.load(root / "tokens.npz") as data, np.load(root / "targets.npz") as targets:
        evidence_keys, target_keys = data["frame_key"], targets["frame_key"]
        if len(np.unique(target_keys)) != len(target_keys) or len(evidence_keys) != len(target_keys):
            raise ValueError("Expected unique, paired Phase II frame identifiers")
        target_by_key = dict(zip(target_keys, range(len(target_keys)), strict=True))
        target_rows = np.array([target_by_key[key] for key in evidence_keys])
        if not np.array_equal(data["split"], targets["split"][target_rows]):
            raise ValueError("Evidence and target splits disagree")
        indices = {split: selected_indices(data, split, count) for split, count in zip(SPLITS, SAMPLES_PER_TRIAL, strict=True)}
        cameras = standardize_with_bias(data["packet_camera_tokens"], indices["train"])
        hypotheses = standardize_with_bias(data["packet_hypothesis_tokens"], indices["train"])
        valid = data["packet_hypothesis_valid"]
        residual = np.minimum(data["packet_cross_view_residual_px"] / 10, 3)
        residual_valid = data["packet_residual_valid_mask"]
        membership = data["packet_subset_membership_mask"]
        relation = np.where(residual_valid, -residual + 0.20 * ~membership, -1.5).transpose(0, 2, 1)
        labels = targets["target_reliable_mask"][target_rows]
        if not np.all(targets["target_reliability_known"][target_rows]):
            raise ValueError("Unexpected unavailable reliability labels")
        batches = {split: Batch(*(np.ascontiguousarray(array[rows], dtype=np.float32) for array in
                                  (cameras, hypotheses, valid, relation, labels))) for split, rows in indices.items()}
        trial_counts = {split: len(np.unique(data["trial_key"][rows])) for split, rows in indices.items()}
    return batches, trial_counts


def metrics(result, truth):
    loss, probabilities, _ = result
    truth = truth > 0
    prediction = probabilities >= 0.5
    sensitivity = float(np.mean(prediction[truth]))
    specificity = float(np.mean(~prediction[~truth]))
    return dict(weighted_loss=loss, accuracy=float(np.mean(prediction == truth)),
                balanced_accuracy=(sensitivity + specificity) / 2, healthy_recall=sensitivity,
                faulty_recall=specificity)


def verify_gradient(model, batch, theta, weights, amd_batch):
    reference = cpu_forward_backward(model, batch, theta, weights)
    device = amd_batch.evaluate(with_gradient=True)
    gradient_max = float(np.max(np.abs(reference[2] - device[2])))
    probability_max = float(np.max(np.abs(reference[1] - device[1])))
    loss_difference = abs(reference[0] - device[0])
    if max(gradient_max, probability_max, loss_difference) > 2e-5:
        raise RuntimeError("AMD and reference gradients or predictions differ")
    indices = np.unique(np.r_[np.linspace(0, len(theta) - 1, 12, dtype=int), model.bias, len(theta) - 1])
    derivative_errors = []
    epsilon = np.float32(0.002)
    for index in indices:
        offset = np.zeros_like(theta)
        offset[index] = epsilon
        high = cpu_forward_backward(model, batch, theta + offset, weights)[0]
        low = cpu_forward_backward(model, batch, theta - offset, weights)[0]
        derivative_errors.append(abs((high - low) / (2 * epsilon) - reference[2][index]))
    maximum = float(max(derivative_errors))
    if maximum > 5e-4:
        raise RuntimeError("Analytic gradients disagree with finite differences")
    amd_batch.step(RATE)
    cpu_updated = theta - RATE * reference[2]
    step_difference = float(np.max(np.abs(amd_batch.read_parameters() - cpu_updated)))
    if step_difference > 2e-5:
        raise RuntimeError("AMD optimizer update differs from CPU reference")
    return dict(max_cpu_amd_gradient_delta=gradient_max, max_cpu_amd_probability_delta=probability_max,
                absolute_loss_delta=loss_difference, finite_difference_coordinates=len(indices),
                max_cpu_finite_difference_gradient_delta=maximum, max_first_sgd_update_delta=step_difference,
                cpu_initial_gradient_l2=float(np.linalg.norm(reference[2])), amd_initial_gradient_l2=float(np.linalg.norm(device[2])))


def timed_steps(model, batch, theta, weights, gpu, steps):
    cpu = theta.copy()
    for _ in range(WARMUP_STEPS):
        cpu -= RATE * cpu_forward_backward(model, batch, cpu, weights)[2]
        gpu.step(RATE)
    cpu = theta.copy()
    gpu.reset(theta)
    elapsed = {"cpu": [], "amd": []}
    for _ in range(steps):
        started = time.perf_counter_ns()
        cpu -= RATE * cpu_forward_backward(model, batch, cpu, weights)[2]
        elapsed["cpu"].append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        gpu.step(RATE)
        elapsed["amd"].append((time.perf_counter_ns() - started) / 1e6)
    timing = {mode: dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                        total_ms=float(sum(values)), step_ms=values) for mode, values in elapsed.items()}
    return cpu, gpu.read_parameters(), timing


def train_pair(amd, batches, weights, attention, steps):
    training = batches["train"]
    model = Model(training.cameras.shape[2], training.hypotheses.shape[2], attention)
    initial = model.initial(213 + int(attention))
    compile_start = time.perf_counter_ns()
    program = AMDBatch.program(amd, model)
    compilation_ms = (time.perf_counter_ns() - compile_start) / 1e6
    gpu = {key: AMDBatch(amd, model, batch, initial, weights, program) for key, batch in batches.items()}
    verified = verify_gradient(model, training, initial, weights, gpu["train"])
    gpu["train"].reset(initial)
    initial_results = {key: metrics(cpu_forward_backward(model, batch, initial, weights), batch.labels)
                       for key, batch in batches.items()}
    theta_cpu, theta_amd, timing = timed_steps(model, training, initial, weights, gpu["train"], steps)
    max_parameter_delta = float(np.max(np.abs(theta_cpu - theta_amd)))
    if max_parameter_delta > 1e-4:
        raise RuntimeError("CPU and AMD parameter drift exceeds numeric tolerance")
    evaluation = {}
    regions = dict(camera_embedding=slice(0, model.camera_end), readout=slice(model.readout_start, model.bias + 1))
    if attention:
        regions.update(hypothesis_embedding=slice(model.camera_end, model.readout_start), relation_coefficient=slice(model.bias + 1, None))
    parameter_delta = {key: float(np.linalg.norm(theta_amd[region] - initial[region])) for key, region in regions.items()}
    if not all(delta > 1e-5 for delta in parameter_delta.values()):
        raise RuntimeError("An expected trainable AMD parameter block did not update")
    for key, batch in batches.items():
        gpu[key].reset(theta_amd)
        cpu_results = cpu_forward_backward(model, batch, theta_cpu, weights)
        amd_results = gpu[key].evaluate()
        evaluation[key] = dict(initial=initial_results[key], cpu=metrics(cpu_results, batch.labels),
                               amd=metrics(amd_results, batch.labels),
                               max_probability_delta=float(np.max(np.abs(cpu_results[1] - amd_results[1]))))
    return dict(parameters=model.count, compile_ms=compilation_ms, numeric_verification=verified,
                amd_parameter_block_l2_change=parameter_delta, max_final_cpu_amd_parameter_delta=max_parameter_delta,
                timing=timing, metrics=evaluation)


def loaded_gpu_modules():
    get_module = ct.WinDLL("kernel32", use_last_error=True).GetModuleHandleW
    get_module.argtypes = [ct.c_wchar_p]
    get_module.restype = ct.c_void_p
    names = ("OpenCL.dll", "nvcuda.dll", "nvopencl64.dll", "DirectML.dll", "torch_directml.dll")
    return {name: bool(get_module(name)) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "pose/results")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.steps <= 100:
        parser.error("use a new output path and 5–100 training steps")
    memory = {"before": process_video_memory()}
    if memory["before"]["0x10de"] != [0, 0] or any(loaded_gpu_modules().values()):
        raise RuntimeError("Expected a clean standalone process with no NVIDIA or generic runtime activity")
    batches, trials = load_batches(args.dataset)
    healthy_fraction = float(batches["train"].labels.mean())
    weights = (np.float32(0.5 / healthy_fraction), np.float32(0.5 / (1 - healthy_fraction)))
    with AMD() as amd:
        device = dict(runtime=str(amd.path), name=amd.name, vendor=amd.vendor, driver=amd.driver)
        results = {model: train_pair(amd, batches, weights, index == 1, args.steps)
                   for index, model in enumerate(("deep_sets", "cross_attention"))}
        memory["after_training_before_release"] = process_video_memory()
        modules = loaded_gpu_modules()
        if memory["after_training_before_release"]["0x10de"] != [0, 0] or any(modules.values()):
            raise RuntimeError("Unexpected NVIDIA, generic ICD or DirectML state")
    memory["after_release"] = process_video_memory()
    data = {key: dict(frames=len(batch.labels), trials=trials[key], healthy=int(batch.labels.sum()),
                      faulty=int(batch.labels.size - batch.labels.sum())) for key, batch in batches.items()}
    result = dict(pid=os.getpid(), device=device, threads=1, memory_cap_gib=3, steps=args.steps, warmup_steps=WARMUP_STEPS,
                  learning_rate=float(RATE), class_weights=list(map(float, weights)), dataset=data,
                  process_local_nonlocal_dxgi_bytes=memory, tracked_module_loaded=modules, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Trained Deep Sets and cross-attention on {device['name']} with CPU reference; wrote {args.output}")


if __name__ == "__main__":
    main()

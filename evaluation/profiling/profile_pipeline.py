"""Profile SAM 2.1 device work and the surrounding CPU/I/O pipeline."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
import torch

from lk_mask import LKMaskTracker


ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / 'data/sample_videos/WIN_20261124_20_05_34_Pro.mp4'
CHECKPOINT = ROOT / 'models/sam2.1_hiera_tiny.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'
BOX = (831.5766738660907, 272.9157667386609, 1237.451403887689, 706.7818574514039)
PRECISIONS = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def wall(operation):
    started = perf_counter()
    value = operation()
    return value, (perf_counter() - started) * 1000


def cuda_timing(operation):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    started = perf_counter()
    start.record()
    value = operation()
    end.record()
    end.synchronize()
    return value, {'wall_ms': (perf_counter() - started) * 1000,
                   'cuda_stream_ms': start.elapsed_time(end)}


def decode(video, count):
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not fps > 0:
            raise RuntimeError(f'Cannot read video: {video}')
        frames = []
        for index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Could not decode frame {index}')
            frames.append(frame)
        return frames, fps
    finally:
        capture.release()


def encode(ext, image, params=()):
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f'Failed to encode {ext}')
    return buffer


def prepare_frames(frames, folder, image_size, load_jpeg):
    jpegs, encode_ms = wall(lambda: [encode('.jpg', frame, (cv2.IMWRITE_JPEG_QUALITY, 95))
                                     for frame in frames])

    def save_jpegs():
        for index, jpeg in enumerate(jpegs):
            (folder / f'{index:06d}.jpg').write_bytes(jpeg)

    _, write_ms = wall(save_jpegs)
    loaded, load_ms = wall(lambda: load_jpeg(str(folder), image_size, offload_video_to_cpu=True))
    stages = {'jpeg_encode_ms': encode_ms, 'jpeg_file_write_ms': write_ms,
              'jpeg_read_decode_resize_normalize_ms': load_ms}
    return loaded, stages


class SamMasks:
    """SAM initialization and full causal propagation over preprocessed tensors."""

    def __init__(self, model):
        self.model = model

    def predict(self, images, height, width, box, precision, *, propagate=True, offload=False, tracer=None):
        dtype = PRECISIONS[precision]
        autocast = torch.autocast('cuda', dtype=dtype) if dtype is not None else nullcontext()
        with torch.inference_mode(), autocast:
            # Inject only the video loader. State construction, prompts and propagation are upstream.
            module = 'sam2.sam2_video_predictor.load_video_frames'
            def preloaded(**_):
                return images, height, width

            with patch(module, new=preloaded):
                state, init = cuda_timing(lambda: self.model.init_state(
                    'preloaded', offload_video_to_cpu=offload, offload_state_to_cpu=offload))
            prompt, prompt_time = cuda_timing(lambda: self.model.add_new_points_or_box(
                state, frame_idx=0, obj_id=1, box=np.asarray(box, dtype=np.float32)))
            phases = {'init': init, 'prompt': prompt_time}
            if propagate:
                def sequence():
                    logits = []
                    for index, _, scores in self.model.propagate_in_video(state):
                        if index != len(logits):
                            raise RuntimeError(f'Unexpected frame order at {index}')
                        logits.append(scores[0, 0])
                    if len(logits) != len(images):
                        raise RuntimeError('Model did not return all frames')
                    return logits

                with tracer if tracer is not None else nullcontext():
                    logits, phases['propagate'] = cuda_timing(sequence)
            else:
                logits = [prompt[2][0, 0]]

            def threshold():
                finite = torch.stack([torch.isfinite(scores).all() for scores in logits])
                binary = torch.stack([scores > 0 for scores in logits])
                return finite, binary

            (finite, masks), phases['threshold_and_finite_check'] = cuda_timing(threshold)
            masks_cpu, phases['download_mask'] = cuda_timing(lambda: masks.cpu().numpy())
            invalid = np.flatnonzero(~finite.cpu().numpy()).tolist()
            return masks_cpu, phases, invalid


def fidelity(reference, predicted):
    comparable = []
    for index, (left, right) in enumerate(zip(reference, predicted, strict=True)):
        if right is None:
            continue
        union = np.count_nonzero(left | right)
        if union:
            comparable.append((index, np.count_nonzero(left & right) / union))
    if not comparable:
        return {'frames_compared': 0, 'mean_iou': None, 'min_iou': None, 'worst_frame': None}
    worst_frame, minimum = min(comparable, key=lambda pair: pair[1])
    return {'frames_compared': len(comparable), 'mean_iou': statistics.mean(iou for _, iou in comparable),
            'min_iou': minimum, 'worst_frame': worst_frame}


def profile_modes(segmenter, images_cpu, height, width, box, repeats):
    cases = ('fp32', 'fp32_offload', 'fp16', 'bf16')
    modes = {}
    reference = None
    schedule = []
    for repeat in range(repeats):
        shift = repeat % len(cases)
        order = cases[shift:] + cases[:shift]
        schedule.append(order)
        for name in order:
            if name == 'bf16' and not torch.cuda.is_bf16_supported(including_emulation=False):
                modes[name] = {'status': 'unsupported', 'fallback': None, 'reason': 'No native CUDA BF16'}
                continue
            offload = name == 'fp32_offload'
            precision = 'fp32' if offload else name
            result = modes.setdefault(name, {'status': 'ok', 'fallback': None, 'samples': [],
                                             'fidelity_to_fp32': [], 'empty_masks': [], 'nonfinite_frames': []})
            if result['status'] == 'runtime_error':
                continue
            images = images_cpu
            upload = None
            before_input = torch.cuda.memory_allocated() / 2**20
            try:
                if not offload:
                    images, upload = cuda_timing(lambda: images_cpu.to('cuda:0'))
                if repeat == 0:
                    segmenter.predict(images[:1], height, width, box, precision,
                                      propagate=False, offload=offload)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                masks, stages, invalid = segmenter.predict(images, height, width, box, precision, offload=offload)
                result['samples'].append({'phases': stages, 'upload_all': upload,
                                          'allocated_before_input_mib': before_input,
                                          'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20})
                result['nonfinite_frames'].append(invalid)
                result['empty_masks'].append(np.flatnonzero(~masks.any(axis=(1, 2))).tolist())
                if name == 'fp32' and reference is None:
                    reference = masks
                result['fidelity_to_fp32'].append(fidelity(reference, masks))
            except RuntimeError as exc:
                if name == 'fp32':
                    raise
                result.update(status='runtime_error', reason=str(exc))
            finally:
                del images
    for result in modes.values():
        if result['status'] != 'ok':
            continue
        if any(result['nonfinite_frames']) or any(result['empty_masks']):
            result['status'] = 'invalid_output'
        result['median_propagate'] = {
            clock: statistics.median(sample['phases']['propagate'][clock] for sample in result['samples'])
            for clock in ('wall_ms', 'cuda_stream_ms')}
    if modes['fp32']['status'] != 'ok':
        raise RuntimeError('Baseline produced invalid masks')
    return reference, modes, schedule


def run_tracker(tracker, frames, initial):
    first, initialization_ms = wall(lambda: tracker.initialize(frames[0], initial))
    observations = [first]
    elapsed = []
    for frame in frames[1:]:
        obs, duration = wall(lambda: tracker.update(frame))
        observations.append(obs)
        elapsed.append(duration)
    valid = [duration for obs, duration in zip(observations[1:], elapsed) if obs.mask is not None]
    return observations, {'initialize_cpu_ms': initialization_ms, 'update_total_ms': sum(elapsed),
                          'valid_update_count': len(valid),
                          'valid_update_p50_ms': statistics.median(valid) if valid else None,
                          'update_frame_ms': elapsed}


def interval_union(intervals):
    active_time = 0.
    end = float('-inf')
    for beginning, finish in sorted(intervals):
        active_time += max(0., finish - max(beginning, end))
        end = max(end, finish)
    return active_time


def profile_kernels(segmenter, images, height, width, box, folder):
    from torch.profiler import ProfilerActivity, profile, supported_activities

    if ProfilerActivity.CUDA not in supported_activities():
        return {'status': 'unavailable', 'reason': 'Torch profiler lacks CUDA activity support'}
    try:
        tracer = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        _, timings, _ = segmenter.predict(images, height, width, box, 'fp32', tracer=tracer)
        path = folder / 'trace.json'
        tracer.export_chrome_trace(str(path))
        events = json.loads(path.read_text(encoding='utf-8'))['traceEvents']
    except RuntimeError as exc:
        return {'status': 'unavailable', 'reason': str(exc)}

    categories = {'kernel', 'gpu_memcpy', 'gpu_memset'}
    intervals = {name: [] for name in categories}
    for event in events:
        if event.get('cat') in categories and event.get('ph') == 'X':
            intervals[event['cat']].append((event['ts'], event['ts'] + event['dur']))
    if not intervals['kernel']:
        return {'status': 'unavailable', 'reason': 'Trace contained no CUDA kernel activity'}
    combined = [interval for items in intervals.values() for interval in items]
    gpu_span = max(end for _, end in combined) - min(start for start, _ in combined)
    gpu_active = interval_union(combined)
    return {'status': 'ok', 'kernel_count': len(intervals['kernel']),
            'kernel_active_union_ms': interval_union(intervals['kernel']) / 1000,
            'kernel_duration_sum_ms': sum(end - start for start, end in intervals['kernel']) / 1000,
            'copy_active_union_ms': interval_union(intervals['gpu_memcpy']) / 1000,
            'gpu_active_union_ms': gpu_active / 1000, 'gpu_activity_span_ms': gpu_span / 1000,
            'gpu_activity_gap_ms': (gpu_span - gpu_active) / 1000,
            'profiled_propagate': timings['propagate']}


def profile_output(folder, frames, masks, fps, visualize, *, verify=True):
    pairs, visualization_ms = wall(lambda: [visualize(frame, mask) for frame, mask in zip(frames, masks)])
    pngs, png_encode_ms = wall(lambda: [(encode('.png', mask.astype(np.uint8) * 255), encode('.png', edges))
                                        for mask, (_, edges) in zip(masks, pairs)])

    def save_artifacts():
        for index, (mask, edge) in enumerate(pngs):
            (folder / f'{index:06d}_mask.png').write_bytes(mask)
            (folder / f'{index:06d}_edge.png').write_bytes(edge)
        (folder / 'frames.jsonl').write_text(''.join(json.dumps({'frame': index}) + '\n'
                                                  for index in range(len(masks))), encoding='utf-8')

    def write_preview():
        height, width = frames[0].shape[:2]
        video = folder / 'preview.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        try:
            if not writer.isOpened():
                raise RuntimeError('Cannot create output video')
            for overlay, _ in pairs:
                writer.write(overlay)
        finally:
            writer.release()
        return video

    # Keep verification separate from the measured writer.
    _, file_write_ms = wall(save_artifacts)
    preview, video_encode_write_ms = wall(write_preview)
    if verify:
        decode(preview, len(frames))
    return {'overlay_and_edge_cpu_ms': visualization_ms, 'png_encode_cpu_ms': png_encode_ms,
            'png_json_file_write_ms': file_write_ms, 'mp4_encode_and_write_ms': video_encode_write_ms,
            'output_bytes': sum(path.stat().st_size for path in folder.iterdir())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=VIDEO)
    parser.add_argument('--box', nargs=4, type=float, default=BOX)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--kernel-trace', action='store_true', help='Capture one extra FP32 CUDA activity trace')
    parser.add_argument('--e2e-only', choices=PRECISIONS, help='Run one clip-to-output wall benchmark')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 3 <= args.frames <= 120 or args.repeats < 1:
        parser.error('Use 3–120 frames and at least one repeat')
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    if args.output is None:
        filename = f'e2e_{args.e2e_only}.json' if args.e2e_only else 'profile.json'
        args.output = Path(__file__).parent / 'results' / filename
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT / 'third_party/sam2'))
    sys.path.insert(0, str(ROOT / 'segmentation'))
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.misc import load_video_frames_from_jpg_images
    from sam_session import visualize

    clip_started = perf_counter()
    ((frames, fps), decode_ms) = wall(lambda: decode(args.video, args.frames))
    height, width = frames[0].shape[:2]
    x0, y0, x1, y1 = args.box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        parser.error('Use an ordered box in source-frame coordinates')
    model, model_load = cuda_timing(lambda: build_sam2_video_predictor(
        CONFIG, str(CHECKPOINT), device='cuda:0', apply_postprocessing=False))
    segmenter = SamMasks(model)
    with tempfile.TemporaryDirectory(prefix='profile-', dir=args.output.parent) as temporary:
        scratch = Path(temporary)
        inputs = scratch / 'input'
        inputs.mkdir()
        (images_cpu, h, w), preprocessing = prepare_frames(
            frames, inputs, model.image_size, load_video_frames_from_jpg_images)
        if (h, w) != (height, width):
            raise RuntimeError('Preprocessed sequence has inconsistent video geometry')
        if args.e2e_only:
            images_gpu, upload = cuda_timing(lambda: images_cpu.to('cuda:0'))
            masks, inference, invalid = segmenter.predict(images_gpu, height, width, args.box, args.e2e_only)
            if invalid or not masks.any(axis=(1, 2)).all():
                raise RuntimeError(f'Invalid predictions, frames with NaN/Inf: {invalid}')
            output = scratch / 'output'
            output.mkdir()
            storage = profile_output(output, frames, masks, fps, visualize, verify=False)
            elapsed = (perf_counter() - clip_started) * 1000
            decode(output / 'preview.mp4', len(frames))
            with args.video.open('rb') as stream:
                sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            report = {'video': args.video.name, 'video_sha256': sha256,
                      'width': width, 'height': height, 'box': args.box,
                      'frames': args.frames, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                      'precision': args.e2e_only, 'tf32': False, 'clip_to_output_wall_ms': elapsed,
                      'model_load': model_load, 'decode_ms': decode_ms, 'preprocessing': preprocessing,
                      'upload_all': upload, 'inference': inference, 'output': storage}
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(f'Wrote {args.output}', flush=True)
            return
        reference, modes, schedule = profile_modes(
            segmenter, images_cpu, height, width, args.box, args.repeats)
        offload = modes.pop('fp32_offload')
        kernel_trace = None
        if args.kernel_trace:
            images_gpu = images_cpu.to('cuda:0')
            kernel_trace = profile_kernels(segmenter, images_gpu, height, width, args.box, scratch)
            del images_gpu
        first_gpu, seed_transfer = cuda_timing(lambda: images_cpu[:1].to('cuda:0'))
        seed, seed_model, invalid = segmenter.predict(first_gpu, height, width, args.box, 'fp32', propagate=False)
        if invalid:
            raise RuntimeError('Initial SAM mask contains nonfinite logits')
        observations, lk = run_tracker(LKMaskTracker(), frames, seed[0])
        masks = [observation.mask for observation in observations]
        hybrid = {'upload_first': seed_transfer, 'first_sam': seed_model, 'lk_cpu': lk,
                  'valid_frames': sum(mask is not None for mask in masks),
                  'fidelity_to_sam_fp32': fidelity(reference, masks),
                  'frames': [{'frame': index, 'reason': obs.reason, 'matches': obs.matches, 'inliers': obs.inliers}
                             for index, obs in enumerate(observations)]}
        out = scratch / 'output'
        out.mkdir()
        storage = profile_output(out, frames, reference, fps, visualize)

    with args.video.open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    results = {'video': str(args.video.name), 'video_sha256': sha256, 'width': width, 'height': height,
               'first_frame': 0, 'frames': args.frames, 'video_fps': fps, 'box': args.box,
               'gpu': torch.cuda.get_device_name(0), 'runtime': torch.version.cuda,
               'torch': torch.__version__, 'opencv': cv2.__version__, 'cpu_threads': 2,
               'tf32': False, 'sam_source_commit': (ROOT / 'third_party/sam2_commit.txt').read_text().strip(),
               'model_load': model_load, 'source_decode_ms': decode_ms,
               'preprocessing': preprocessing, 'precision': modes, 'run_order': schedule,
               'fp32_kernel_trace': kernel_trace,
               'fp32_offload': offload,
               'initial_sam_lk': hybrid, 'output': storage}
    args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()

"""Replay the real 30-frame observation chain on CPU and explicit AMD UMat."""

import argparse
import json
import os
from pathlib import Path, PureWindowsPath
from time import perf_counter_ns

from evaluation.resource_limits import limit_process

ROOT = Path(__file__).resolve().parents[4]
VIDEO = ROOT / 'data/sample_videos/WIN_20261124_20_05_34_Pro.mp4'
SAM = ROOT / 'outputs/segmentation/20261124_215925_8f211c'
LSD = ROOT / 'outputs/edges/lsd_baseline'
COUNT = 30


def select_runtime(cpu_only):
    """Match phase3.amd.opencl.measure's AMD-only runtime and device filter."""
    windows = os.environ.get('WINDIR')
    drivers = sorted((Path(windows) / 'System32/DriverStore/FileRepository').glob('amdocl.inf*/amdocl64.dll')) if windows else []
    amd_runtime = drivers[0] if not cpu_only and len(drivers) == 1 else None
    os.environ['OPENCV_OPENCL_RUNTIME'] = str(amd_runtime) if amd_runtime else 'disabled'
    os.environ['OPENCV_OPENCL_DEVICE'] = 'AMD:GPU:gfx1035'
    os.environ['OPENCV_OPENCL_CACHE_ENABLE'] = '0'
    return amd_runtime, len(drivers)


def enable_device(cv, runtime, available_drivers, cpu_only):
    if runtime is None:
        reason = 'CPU-only requested' if cpu_only else f'Expected one AMD runtime, found {available_drivers}'
        return {'available': False, 'reason': reason}
    cv.ocl.setUseOpenCL(True)
    if not cv.ocl.useOpenCL():
        return {'available': False, 'reason': 'OpenCV could not activate the explicit AMD GPU', 'runtime': str(runtime)}
    device = cv.ocl.Device_getDefault()
    if not device.isAMD() or device.name() != 'gfx1035':
        cv.ocl.setUseOpenCL(False)
        return {'available': False, 'reason': 'Explicit device identity did not match AMD gfx1035', 'runtime': str(runtime)}
    return {'available': True, 'device': device.name(), 'vendor': device.vendorName(),
            'driver': device.driverVersion(), 'runtime': str(runtime)}


def json_lines(path, keys, field):
    rows = {}
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            if row[field] in keys:
                rows[row[field]] = row
            if len(rows) == len(keys):
                break
    if rows.keys() != keys:
        raise ValueError(f'Missing requested source frames in {path}')
    return rows


def load_clip(cv, adapter):
    sam = adapter.read_json(SAM / 'session.json')
    lsd = adapter.read_json(LSD / 'summary.json')
    if sam['status'] != 'completed' or sam['written_frames'] != COUNT:
        raise ValueError('Expected the archived completed 30-frame segmentation session')
    if any(PureWindowsPath(meta['source']).name != VIDEO.name for meta in (sam, lsd)):
        raise ValueError('Archive provenance does not match the source clip')
    start = sam['start_frame']
    indices = set(range(start, start + COUNT))
    masks = json_lines(SAM / 'frames.jsonl', indices, 'source_frame')
    segments = json_lines(LSD / 'segments.jsonl', indices, 'frame_index')
    cap = cv.VideoCapture(str(VIDEO))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open the archived clip')
        if start:
            cap.set(cv.CAP_PROP_POS_FRAMES, start)
        for index in sorted(indices):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Could not decode source frame {index}')
            name = f'{masks[index]["local_frame"]:06d}.png'
            mask = adapter.binary_image(SAM / 'masks' / name, frame.shape[:2]).astype('uint8')
            lsd_row = segments[index]
            if lsd_row['coordinate_space'] != 'original_distorted_pixels' or lsd_row['camera_id'] != lsd['camera_id']:
                raise ValueError(f'Unexpected LSD coordinates or camera at source frame {index}')
            frames.append((frame, mask, lsd_row['segments'], name))
    finally:
        cap.release()
    return frames


def selected_edges(cv, frame, mask, kernel, amd):
    if amd:
        frame, mask = cv.UMat(frame), cv.UMat(mask)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv.Canny(blurred, 15, 45, L2gradient=True)
    region = cv.dilate(mask, kernel)
    selected = cv.bitwise_and(edges, edges, mask=region)
    if amd:
        selected = selected.get()
        cv.ocl.finish()
    return selected


def geometry(cv, np, adapter, mask, selected, archived, kernel5, kernel7):
    roi = cv.dilate(mask, kernel7)
    edge_band = cv.dilate((selected != 0).astype(np.uint8), kernel5)
    height, width = mask.shape
    candidates = []
    for line in archived:
        raw = np.asarray(line['endpoints_px'], dtype=float).reshape(2, 2)
        x, y = np.rint(np.linspace(*raw, 21)).astype(int).T
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x[inside], y[inside]
        reason = None
        if np.sum(roi[y, x]) / 21 < .7:
            reason = 'outside_roi'
        elif np.linalg.norm(raw[1] - raw[0]) < 15:
            reason = 'short_segment'
        elif np.sum(edge_band[y, x]) / 21 < .35:
            reason = 'weak_canny_support'
        candidates.append(dict(local_id=line['local_id'], xy=raw.tolist(), reason=reason))
    corners = adapter.corners_from_segments(candidates)
    kept = []
    for corner in corners:
        x, y = np.rint(corner['xy']).astype(int)
        if 0 <= x < width and 0 <= y < height and roi[y, x]:
            kept.append(corner['xy'])
    status = tuple(row['reason'] for row in candidates)
    ids = {row['local_id'] for row in candidates if row['reason'] is None}
    return status, ids, np.asarray(kept).reshape(-1, 2)


def pipeline(cv, np, adapter, frames, amd, keep=False):
    kernel5 = np.ones((5, 5), np.uint8)
    kernel7 = np.ones((7, 7), np.uint8)
    edge_time = geometry_time = 0
    results = []
    for frame, mask, lines, name in frames:
        start = perf_counter_ns()
        edges = selected_edges(cv, frame, mask, kernel5, amd)
        middle = perf_counter_ns()
        observations = geometry(cv, np, adapter, mask, edges, lines, kernel5, kernel7)
        edge_time += middle - start
        geometry_time += perf_counter_ns() - middle
        if keep:
            results.append((edges, observations, name))
    return results, edge_time / 1e6, geometry_time / 1e6


def output_agreement(cv, np, cpu, other):
    edge_equal = total_pixels = intersection = union = exact_archive = 0
    ids_intersection = ids_union = reasons_equal = reasons_total = 0
    corners_cpu = corners_other = exact_corner_frames = 0
    corner_errors = []
    for (edge_a, (reasons_a, ids_a, corners_a), key), (edge_b, (reasons_b, ids_b, corners_b), _) in zip(cpu, other, strict=True):
        reference = cv.imread(str(SAM / 'selected_edges' / key), cv.IMREAD_GRAYSCALE)
        exact_archive += int(np.array_equal(reference, edge_a))
        edge_equal += int(np.count_nonzero(edge_a == edge_b))
        total_pixels += edge_a.size
        intersection += int(np.count_nonzero((edge_a > 0) & (edge_b > 0)))
        union += int(np.count_nonzero((edge_a > 0) | (edge_b > 0)))
        ids_intersection += len(ids_a & ids_b)
        ids_union += len(ids_a | ids_b)
        reasons_equal += sum(a == b for a, b in zip(reasons_a, reasons_b, strict=True))
        reasons_total += len(reasons_a)
        corners_cpu += len(corners_a)
        corners_other += len(corners_b)
        exact_corner_frames += int(np.array_equal(corners_a, corners_b))
        if len(corners_a) and len(corners_b):
            distances = np.linalg.norm(corners_a[:, None] - corners_b, axis=-1)
            left, right = set(), set()
            for i, j in np.column_stack(np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
                if distances[i, j] > 1:
                    break
                if i not in left and j not in right:
                    left.add(i)
                    right.add(j)
                    corner_errors.append(float(distances[i, j]))
    return dict(pixel_agreement=edge_equal / total_pixels, edge_foreground_iou=intersection / union if union else None,
                cpu_exact_archived_edge_frames=exact_archive, segment_candidate_id_iou=ids_intersection / ids_union if ids_union else None,
                segment_reason_agreement=reasons_equal / reasons_total, corner_count_cpu=corners_cpu,
                corner_count_comparison=corners_other, corner_frames_exact=exact_corner_frames,
                matched_corners_within_1px=len(corner_errors),
                corner_match_median_px=float(np.median(corner_errors)) if corner_errors else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--memory-gib', type=float, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3:
        parser.error('Use a new output path and at least three paired repeats')
    limit_process(args.threads, args.memory_gib)
    runtime, count = select_runtime(args.cpu_only)
    import cv2 as cv
    import numpy as np
    from perception.observations import adapter

    cv.setNumThreads(args.threads)
    amd = enable_device(cv, runtime, count, args.cpu_only)
    modes = [False, True] if amd['available'] else [False]
    preloaded = load_clip(cv, adapter)
    samples = {mode: {scope: [] for scope in ('full_clip_with_io', 'preloaded_clip')} for mode in modes}
    outputs = {}
    for mode in modes:
        cv.ocl.setUseOpenCL(mode)
        outputs[mode], *_ = pipeline(cv, np, adapter, preloaded, mode, keep=True)  # warmup and untimed agreement
    for repeat in range(args.repeats):
        for scope in ('full_clip_with_io', 'preloaded_clip'):
            for mode in modes[::(-1 if repeat % 2 else 1)]:
                cv.ocl.setUseOpenCL(mode)
                if mode and not cv.ocl.useOpenCL():
                    raise RuntimeError('Lost the selected AMD GPU during the paired run')
                start = perf_counter_ns()
                frames = load_clip(cv, adapter) if scope == 'full_clip_with_io' else preloaded
                loaded = perf_counter_ns()
                _, edge_ms, geometry_ms = pipeline(cv, np, adapter, frames, mode)
                wall_ms = (perf_counter_ns() - start) / 1e6
                samples[mode][scope].append((wall_ms, (loaded - start) / 1e6, edge_ms, geometry_ms))
                del frames
    def summarize(run):
        values = np.asarray(run)
        return {name: {'p50_ms': float(np.quantile(values[:, column], .5)),
                       'p95_ms': float(np.quantile(values[:, column], .95))}
                for column, name in enumerate(('wall', 'io_and_decode', 'selected_canny', 'cpu_filter_and_intersections'))}

    if amd['available']:
        compared = outputs[True]
    else:
        compared, *_ = pipeline(cv, np, adapter, preloaded, False, keep=True)
    comparison = output_agreement(cv, np, outputs[False], compared)
    report = {'mode': 'paired_cpu_amd_umat' if amd['available'] else 'cpu_only', 'amd': amd, 'opencv': cv.__version__,
              'video': str(VIDEO.relative_to(ROOT)), 'sam_archive': str(SAM.relative_to(ROOT)), 'lsd_archive': str(LSD.relative_to(ROOT)),
              'frames': COUNT, 'repeats': args.repeats, 'warmup': 'one untimed preloaded clip per mode',
              'threads': args.threads, 'memory_cap_gib': args.memory_gib, 'timing_unit': 'milliseconds per 30-frame clip',
              'scope': 'Shared files and CPU geometry; AMD UMat only for selected Canny path, including transfers and synchronization',
              'excluded': ['SAM inference', 'LSD extraction', 'calibration', 'multi-camera fusion'],
              'agreement_comparison': 'cpu_vs_amd' if amd['available'] else 'cpu_self_check_no_amd', 'agreement': comparison,
              'timings': {('amd_umat' if mode else 'cpu'): {scope: summarize(run) for scope, run in entries.items()}
                          for mode, entries in samples.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print(f'{report["mode"]}: {args.output}')


if __name__ == '__main__':
    main()

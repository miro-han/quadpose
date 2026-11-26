"""Measure common flash timing and make one-to-one host-time frame groups."""

import argparse
import bisect
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from rig.common import configure_console, integer, load_json, number, require, write_json

NS_MS = 1_000_000
NS_S = NS_MS * 1000


@dataclass
class Stream:
    slot: int
    video: Path
    stamps: list[int]
    size: tuple[int, int]
    device_path: str


@dataclass
class Recording:
    key: str
    directory: Path
    reference: str
    streams: dict[str, Stream]


def recording(config, directory):
    folder = (directory / config['session_dir']).resolve()
    metadata_path = folder / 'session.json'
    raw_metadata = metadata_path.read_bytes()
    metadata = load_json(metadata_path)
    require(metadata['status'] == 'completed', f'Recorder session must be completed: {folder}')
    require(type(metadata['start_monotonic_ns']) is int and type(metadata['stop_monotonic_ns']) is int and
            metadata['start_monotonic_ns'] < metadata['stop_monotonic_ns'], 'Invalid recorder start or stop clock')
    mapping = config['camera_slots']
    require(isinstance(mapping, dict) and 2 <= len(mapping) <= 4 and len(set(mapping.values())) == len(mapping),
            'Specify two to four unique camera slots')
    require(set(metadata['written_frames']) == {str(slot) for slot in mapping.values()},
            'camera_slots must explicitly map every recorded camera slot')
    reference = config['reference_camera']
    require(reference in mapping, 'reference_camera must be an ID from camera_slots')
    digest = hashlib.sha256(raw_metadata)
    streams = {}
    for name, slot in sorted(mapping.items()):
        integer(slot, f'{name} slot', minimum=1)
        info = metadata['cameras'][str(slot)]
        device_path = info['device']['path']
        require(isinstance(device_path, str) and device_path.strip(), f'{name}: recorder device path is missing')
        size = tuple(info['actual']['size_px'])
        require(len(size) == 2 and all(type(value) is int and value > 0 for value in size),
                f'{name}: invalid video size')
        timestamps = folder / f'camera_{slot}_timestamps.csv'
        digest.update(name.encode('utf-8') + b'\x00' + timestamps.read_bytes())
        with timestamps.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        try:
            indexes = [int(row['video_frame_index']) for row in rows]
            stamps = [int(row['host_monotonic_ns']) for row in rows]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'{name}: invalid recorder frame index or host timestamp') from error
        require(len(stamps) == metadata['written_frames'][str(slot)] and len(stamps) >= 2 and
                indexes == list(range(len(stamps))), f'{name}: missing recorder frame indexes')
        require(metadata['start_monotonic_ns'] <= stamps[0] < stamps[-1] <= metadata['stop_monotonic_ns'] and
                all(left < right for left, right in zip(stamps, stamps[1:])),
                f'{name}: timestamps must strictly increase within the recording gate')
        video = folder / f'camera_{slot}.avi'
        require(video.is_file(), f'{name}: missing recorder AVI')
        with video.open('rb') as handle:
            video_hash = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                video_hash.update(block)
            digest.update(video_hash.digest())
        streams[name] = Stream(slot, video, stamps, size, device_path)
    require(len({stream.device_path for stream in streams.values()}) == len(streams),
            'Recorder camera device paths must be distinct')
    return Recording(digest.hexdigest(), folder, reference, streams)


def decode_brightness(stream, roi):
    bounds = np.asarray(roi, dtype=float)
    require(bounds.shape == (4,) and np.isfinite(bounds).all() and
            0 <= bounds[0] < bounds[2] <= 1 and 0 <= bounds[1] < bounds[3] <= 1,
            'Flash ROI must be normalized [left, top, right, bottom] within the frame')
    width, height = stream.size
    x0, y0, x1, y1 = np.round(bounds * [width, height, width, height]).astype(int)
    require(x1 - x0 >= 3 and y1 - y0 >= 3, 'Flash ROI must contain at least 3x3 pixels')
    cap = cv2.VideoCapture(str(stream.video))
    try:
        require(cap.isOpened(), f'Cannot open flash video {stream.video}')
        values = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape[1::-1] == stream.size, f'Unexpected frame size in {stream.video}')
            gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            values.append(float(np.mean(gray)))
    finally:
        cap.release()
    require(len(values) == len(stream.stamps), f'AVI/CSV frame count mismatch in {stream.video}')
    return np.asarray(values)


def detect_pulses(brightness, delta, min_frames, max_frames):
    lower = brightness[brightness <= np.quantile(brightness, .65)]
    baseline = float(np.median(lower))
    noise_mad = float(np.median(np.abs(lower - baseline)))
    high = baseline + max(delta, 8 * noise_mad)
    low = baseline + (high - baseline) * .5
    require(high <= 255, 'Flash ROI is too bright or unstable to measure upward pulses')
    require(brightness[0] < low and brightness[-1] < low, 'Flash recording must start and end with the light off')
    events = []
    begin = None
    for frame, value in enumerate(brightness):
        if begin is None and value >= high:
            begin = frame
        elif begin is not None and value < low:
            require(min_frames <= frame - begin <= max_frames,
                    f'A flash pulse lasts {frame - begin} frames, outside configured limits')
            events.append(begin)
            begin = None
    return events, {'baseline': baseline, 'dark_mad': noise_mad, 'on_threshold': high, 'off_threshold': low}


def flash(config, session):
    supplied = config['pulse_intervals_s']
    require(isinstance(supplied, list) and len(supplied) >= 4, 'Supply at least five flashes with a known interval sequence')
    intervals = [number(item, 'pulse_interval_s') * NS_S for item in supplied]
    schedule_tol_ns = number(config.get('max_schedule_error_ms', 80), 'max_schedule_error_ms') * NS_MS
    require(min(intervals) > 2 * schedule_tol_ns and max(intervals) - min(intervals) > 2 * schedule_tol_ns,
            'Use a nonperiodic pulse schedule with distinguishable intervals')
    rois = config['roi_by_camera']
    require(set(rois) == set(session.streams), 'Every selected camera must have a flash ROI')
    delta = number(config.get('min_flash_delta', 20), 'min_flash_delta', maximum=255)
    min_frames = integer(config.get('min_pulse_frames', 2), 'min_pulse_frames', minimum=1)
    max_frames = integer(config.get('max_pulse_frames', 10), 'max_pulse_frames', minimum=min_frames)
    max_offset = number(config.get('max_offset_ms', 150), 'max_offset_ms') * NS_MS
    max_jitter = number(config.get('max_jitter_ms', 20), 'max_jitter_ms') * NS_MS
    max_drift = number(config.get('max_drift_ms', 15), 'max_drift_ms') * NS_MS
    edges = {}
    for camera, stream in session.streams.items():
        try:
            frames, signal = detect_pulses(decode_brightness(stream, rois[camera]), delta, min_frames, max_frames)
            require(len(frames) == len(intervals) + 1, f'Detected {len(frames)} flashes, expected {len(intervals) + 1}')
            timestamps = [stream.stamps[frame] for frame in frames]
            errors = [abs(right - left - nominal) for left, right, nominal in zip(timestamps, timestamps[1:], intervals)]
            require(max(errors) <= schedule_tol_ns, f'Flash intervals do not match the schedule, worst error {max(errors) / NS_MS:.2f} ms')
        except (ValueError, cv2.error) as error:
            raise ValueError(f'{camera}: {error}') from error
        edges[camera] = {'source_frames': frames, 'host_monotonic_ns': timestamps,
                         'signal': signal, 'max_schedule_error_ms': max(errors) / NS_MS}

    reference = edges[session.reference]['host_monotonic_ns']
    for camera, result in edges.items():
        offsets = [host - reference_host for host, reference_host in zip(result['host_monotonic_ns'], reference)]
        location = round(median(offsets))
        residuals = [abs(value - location) for value in offsets]
        half = len(offsets) // 2
        drift = abs(median(offsets[:half]) - median(offsets[-half:]))
        require(abs(location) <= max_offset, f'{camera}: host offset {location / NS_MS:.2f} ms exceeds limit')
        require(max(residuals) <= max_jitter and drift <= max_drift,
                f'{camera}: unstable flash timing (jitter {max(residuals) / NS_MS:.2f} ms, drift {drift / NS_MS:.2f} ms)')
        result.update(offset_ns=location, jitter_p95_ms=float(np.quantile(residuals, .95) / NS_MS),
                      jitter_max_ms=max(residuals) / NS_MS, drift_ms=drift / NS_MS,
                      paired_offsets_ns=offsets, device_path=session.streams[camera].device_path)
    return {'schema_version': 1, 'status': 'passed', 'session_key': session.key, 'recording_dir': str(session.directory),
            'reference_camera': session.reference, 'clock': 'recorder host_monotonic_ns after read; flash-constrained',
            'valid_aligned_interval_ns': [reference[0], reference[-1]], 'cameras': edges,
            'pulse_intervals_s': supplied, 'limits_ms': {'schedule': schedule_tol_ns / NS_MS, 'offset': max_offset / NS_MS,
                                                        'jitter': max_jitter / NS_MS, 'drift': max_drift / NS_MS}}


def alignment(config, session, directory):
    tolerance = round(number(config.get('max_residual_ms', 12), 'max_residual_ms') * NS_MS)
    require(tolerance > 0, 'Frame-group tolerance must be at least one nanosecond')
    minimum = integer(config.get('min_views', 2), 'min_views', minimum=2)
    require(minimum <= len(session.streams), 'min_views exceeds the recorded camera count')
    required_fraction = number(config.get('min_valid_group_fraction', .70), 'min_valid_group_fraction', maximum=1)
    supplied = config.get('flash_report')
    require(supplied or config.get('allow_unverified_host_only') is True,
            'Provide a flash report or explicitly opt in to unverified host-only matching')
    report = load_json(directory / supplied) if supplied else None
    names = set(session.streams)
    offsets = dict.fromkeys(names, 0)
    bracket = None
    if report is not None:
        require(report['schema_version'] == 1 and report['status'] == 'passed' and report['session_key'] == session.key
                and report['reference_camera'] == session.reference and set(report['cameras']) == names,
                'Flash report does not match this recording, camera mapping, or reference')
        for name, stream in session.streams.items():
            camera = report['cameras'][name]
            require(camera['device_path'] == stream.device_path and type(camera['offset_ns']) is int,
                    f'{name}: flash report device or clock offset is invalid')
            offsets[name] = camera['offset_ns']
        bracket = report['valid_aligned_interval_ns']
        reference_stamps = session.streams[session.reference].stamps
        require(offsets[session.reference] == 0 and len(bracket) == 2 and
                all(type(value) is int for value in bracket) and reference_stamps[0] <= bracket[0] < bracket[1] <= reference_stamps[-1],
                'Flash bracket is invalid')

    clocks = {name: [stamp - offsets[name] for stamp in stream.stamps] for name, stream in session.streams.items()}
    groups = []
    reference = session.reference
    candidates_by_group = []
    claimed = Counter()
    for t in clocks[reference]:
        inside = bracket is None or bracket[0] <= t <= bracket[1]
        group_candidates = {}
        if inside:
            for name, stamps in clocks.items():
                if name == reference:
                    continue
                lower = bisect.bisect_left(stamps, t - tolerance)
                upper = bisect.bisect_right(stamps, t + tolerance)
                group_candidates[name] = list(range(lower, upper))
                claimed.update((name, frame) for frame in group_candidates[name])
        candidates_by_group.append(group_candidates)

    for source_frame, (t, possibilities) in enumerate(zip(clocks[reference], candidates_by_group)):
        used = {reference: {'source_frame': source_frame, 'host_monotonic_ns': t, 'aligned_time_ns': t, 'residual_ns': 0}}
        missing = {}
        for name, frames in possibilities.items():
            if not frames:
                missing[name] = 'no_frame_within_tolerance'
            elif len(frames) > 1:
                missing[name] = 'multiple_frames_within_tolerance'
            elif claimed[name, frames[0]] > 1:
                missing[name] = 'frame_eligible_for_multiple_reference_groups'
            else:
                frame = frames[0]
                adjusted = clocks[name][frame]
                used[name] = {'source_frame': frame, 'host_monotonic_ns': session.streams[name].stamps[frame],
                              'aligned_time_ns': adjusted, 'residual_ns': adjusted - t}
        reason = None
        if not possibilities:
            reason = 'outside_verified_flash_interval'
            missing.update({name: reason for name in session.streams if name != reference})
        elif any(reason.startswith(('multiple_', 'frame_eligible_')) for reason in missing.values()):
            reason = 'ambiguous_frame_match'
        elif len(used) < minimum:
            reason = 'insufficient_views'
        groups.append({'group_id': source_frame, 'reference_source_frame': source_frame,
                       'aligned_time_ns': t, 'valid': reason is None, 'reason': reason,
                       'frames': used, 'missing': missing})
    valid = sum(group['valid'] for group in groups)
    eligible = sum(bool(possibilities) for possibilities in candidates_by_group)
    valid_fraction = valid / eligible if eligible else 0.
    passed = valid_fraction >= required_fraction
    streams = {name: {'slot': stream.slot, 'video': stream.video.name, 'device_path': stream.device_path,
                      'image_size_px': list(stream.size)} for name, stream in session.streams.items()}
    return {'schema_version': 1, 'status': 'passed' if passed else 'rejected', 'session_key': session.key,
            'recording_dir': str(session.directory), 'reference_camera': reference,
            'mode': 'flash_corrected_host' if report is not None else 'unverified_host_only',
            'max_residual_ns': tolerance, 'min_views': minimum, 'valid_groups': valid, 'eligible_groups': eligible,
            'total_groups': len(groups), 'valid_group_fraction': valid_fraction,
            'min_valid_group_fraction': required_fraction,
            'rejection_reason': None if passed else 'insufficient_valid_group_fraction',
            'offsets_ns': offsets, 'valid_aligned_interval_ns': bracket,
            'unverified_timing': report is None, 'streams': streams, 'groups': groups}


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('flash', 'align'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.config.resolve().parent
    try:
        require(not args.output.exists(), f'Choose a new output path: {args.output}')
        config = load_json(args.config)
        session = recording(config, directory)
        result = flash(config, session) if args.action == 'flash' else alignment(config, session, directory)
        write_json(args.output, result)
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        parser.exit(2, f'{args.action} rejected: {error}\n')
    print(f'{args.action}: {result["status"]} -> {args.output}')
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())

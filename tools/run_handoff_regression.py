"""Replay the supplied edge and recorder checks into a new output directory."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import cv2

ROOT = Path(__file__).resolve().parents[1]
from capture import verify_recorder

SAMPLES = ROOT / 'data' / 'sample_videos'
LONG_VIDEO = SAMPLES / 'WIN_20261124_20_05_34_Pro.mp4'
SHORT_VIDEO = SAMPLES / 'WIN_20261124_20_04_41_Pro.mp4'
RECORDER = ROOT / 'outputs' / 'camera' / 'recorder_checks'
MANIFEST = 'FULL_PACKAGE_MANIFEST.json'


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(source):
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def decoded_count(path):
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        require(cap.isOpened(), f'Cannot open {path}')
        while cap.read()[0]:
            count += 1
        return count
    finally:
        cap.release()


def run_edge(script, video, folder, *options):
    subprocess.run([sys.executable, str(ROOT / 'perception' / 'edges' / script),
                    '--video', str(video), '--output', str(folder), *options], check=True)


def check_canny(output):
    run_edge('check_edges.py', LONG_VIDEO, output, '--low', '30', '--high', '90')
    meta = load(output / 'run_info.json')
    decoded = decoded_count(output / 'edge_preview.mp4')
    require(decoded == meta['frames_processed'] == meta['reported_frames'] == 530, meta)
    require(meta['source_size_px'] == [1920, 1080] and meta['canny_thresholds'] == [30, 90], meta)
    require(meta['coordinate_space'] == 'original_distorted_pixels', meta)
    for pattern in ('frame_*_original.png', 'frame_*_edges.png', 'frame_*_overlay.png'):
        require(len(list(output.glob(pattern))) == 6, f'Expected six Canny images: {pattern}')
    require((output / 'threshold_comparison.jpg').stat().st_size > 0, 'Empty Canny comparison')
    return {'processed_frames': 530, 'decoded_preview_frames': decoded, 'sample_frames': 6}


def check_lsd(output, video, camera_id, roi, expected_frames):
    run_edge('check_segments.py', video, output, '--camera-id', camera_id, '--roi', *roi,
             '--min-length', '25', '--min-support', '0.35')
    summary = load(output / 'summary.json')
    decoded = decoded_count(output / 'segment_preview.mp4')
    counts = []
    with (output / 'segments.jsonl').open(encoding='utf-8') as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            require(row['frame_index'] == index and row['camera_id'] == camera_id and
                    row['coordinate_space'] == 'original_distorted_pixels', f'Invalid LSD frame {index}')
            candidates = 0
            for local_id, segment in enumerate(row['segments']):
                require(segment['local_id'] == local_id and segment['physical_edge_id'] is None and
                        len(segment['endpoints_px']) == 4 and
                        0 <= segment['canny_support'] <= 1 and
                        segment['candidate'] == (not segment['rejection_reasons']),
                        f'Invalid LSD segment {local_id} in frame {index}')
                candidates += segment['candidate']
            counts.append(candidates)
    require(len(counts) == decoded == summary['processed_frames'] == expected_frames, summary)
    require(sum(counts) > 0 and abs(sum(counts) / len(counts) - summary['mean_candidates']) < 1e-9, summary)
    return {'processed_frames': expected_frames, 'decoded_preview_frames': decoded,
            'jsonl_rows': len(counts), 'mean_candidates': summary['mean_candidates']}


def check_synthetic(output):
    verify_recorder.QA = output
    return verify_recorder.synthetic_check()


def check_archived_camera():
    results = {}
    for group, name, count in [('hardware', '20261124_205224_e12951', 49),
                               ('single_hardware', '20261124_205126_1ea7a6', 32)]:
        session = RECORDER / group / name
        report = verify_recorder.verify_session(session, expected=count)
        require(set(report) == ({'1', '2'} if group == 'hardware' else {'1'}), report)
        if group == 'hardware':
            require(report == load(RECORDER / 'hardware_report.json')['cameras'],
                    'Decoded camera report differs from the archived report')
        results[group] = {'session': name, 'cameras': report}
    return results


def check_archive(archive_path):
    prefix = 'Team7_Plate_Tracking/'
    immutable = ('data/sample_videos/', 'outputs/', 'models/', 'third_party/', 'docs/source/')
    with zipfile.ZipFile(archive_path) as archive:
        manifest_bytes = archive.read(prefix + MANIFEST)
        require((ROOT / MANIFEST).read_bytes() == manifest_bytes, 'Extracted manifest differs from ZIP')
        rows = json.loads(manifest_bytes)
        expected = {prefix + row['path'] for row in rows} | {prefix + MANIFEST}
        require(set(archive.namelist()) == expected and len(archive.namelist()) == len(expected),
                'ZIP entries differ from manifest')
        verified_extracted = 0
        for row in rows:
            relative = row['path']
            entry = prefix + relative
            require(archive.getinfo(entry).file_size == row['bytes'], f'ZIP size mismatch: {relative}')
            with archive.open(entry) as source:
                require(sha256(source) == row['sha256'],
                        f'ZIP SHA-256 mismatch: {relative}')
            if relative.startswith(immutable):
                extracted = ROOT / relative
                require(extracted.stat().st_size == row['bytes'], f'Extracted size mismatch: {relative}')
                with extracted.open('rb') as source:
                    require(sha256(source) == row['sha256'],
                            f'Extracted SHA-256 mismatch: {relative}')
                verified_extracted += 1
    with archive_path.open('rb') as source:
        archive_digest = sha256(source)
    return {'zip_manifest_entries': len(rows), 'preserved_extracted_entries': verified_extracted,
            'zip_sha256': archive_digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New directory for all test outputs')
    parser.add_argument('--zip', type=Path, default=ROOT.parent / 'Team7_Complete_Handoff.zip',
                        help='Original full handoff ZIP')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Run without Python optimization because the recorder verifier uses assertions')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = {
        'canny_long': lambda: check_canny(output / 'canny_long'),
        'lsd_long': lambda: check_lsd(output / 'lsd_long', LONG_VIDEO, 'camera_1',
                                     ('0.40', '0.20', '0.85', '0.92'), 530),
        'lsd_short': lambda: check_lsd(output / 'lsd_short', SHORT_VIDEO, 'clip_200441',
                                      ('0.40', '0.45', '0.65', '0.98'), 50),
        'synthetic_recorder': lambda: check_synthetic(output / 'recorder'),
        'archived_hardware_replay': check_archived_camera,
        'archive_integrity': lambda: check_archive(args.zip),
    }
    results = {'python': sys.version.split()[0], 'opencv': cv2.__version__, 'checks': {}}
    for name, check in checks.items():
        try:
            results['checks'][name] = {'status': 'PASS', 'result': check()}
        except Exception as exc:
            results['checks'][name] = {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}
        print(f'{name}: {results["checks"][name]["status"]}', flush=True)
        (output / 'report.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return int(any(item['status'] != 'PASS' for item in results['checks'].values()))


if __name__ == '__main__':
    raise SystemExit(main())

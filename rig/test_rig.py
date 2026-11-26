"""Synthetic checkerboard images, recorded-style AVI/CSV, and fail-closed cases."""

import copy
import csv
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from rig.calibrate import anchor_graph, calibrate_intrinsics, pose_difference, quality_config, reorder_anchor
from rig.common import load_json, write_json
from rig.synchronize import NS_MS, alignment, flash, recording

HERE = Path(__file__).resolve().parent
COLS, ROWS, SQUARE = 8, 6, .028
SIZE = (640, 480)
K = np.array([[570., 0, 320], [0, 555, 240], [0, 0, 1]])
MID_BOARD = np.array([SQUARE * (COLS - 1) / 2, SQUARE * (ROWS - 1) / 2, 0])
WORLD_POSES = {
    'front': ([-.10, -.50, .55], [0, 0, 0]),
    'side': ([.48, -.15, .56], [0, 0, 0]),
}


def orient(euler):
    return cv2.Rodrigues(np.radians(np.asarray(euler, dtype=float)))[0]


def look_at(center, target):
    center = np.asarray(center)
    forward = np.subtract(target, center)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = [right, down, forward]
    pose[:3, 3] = -pose[:3, :3] @ center
    return pose


def board_at(angles, center):
    pose = np.eye(4)
    pose[:3, :3] = orient(angles)
    pose[:3, 3] = center - pose[:3, :3] @ MID_BOARD
    return pose


def checkerboard(path, board_to_camera):
    tile = 64
    image = np.zeros(((ROWS + 1) * tile, (COLS + 1) * tile), dtype=np.uint8)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            image[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = 240 if (r + c) % 2 else 15
    outer = np.array([[-1, -1, 0], [COLS, -1, 0], [COLS, ROWS, 0], [-1, ROWS, 0]], dtype=float) * SQUARE
    extremes = np.array([[0, 0, 0], [COLS - 1, 0, 0], [0, ROWS - 1, 0], [COLS - 1, ROWS - 1, 0]], dtype=float) * SQUARE
    rvec = cv2.Rodrigues(board_to_camera[:3, :3])[0]
    tvec = board_to_camera[:3, 3]

    def project(points):
        return cv2.projectPoints(points, rvec, tvec, K, None)[0].reshape(-1, 2)

    h, w = image.shape
    source = np.array([[-.5, -.5], [w - .5, -.5], [w - .5, h - .5], [-.5, h - .5]], dtype=np.float32)
    warp = cv2.getPerspectiveTransform(source, project(outer).astype(np.float32))
    picture = cv2.warpPerspective(image, warp, SIZE, borderValue=160)
    cv2.imencode('.png', picture)[1].tofile(str(path))
    return project(extremes).tolist()


def calibration_fixture(folder):
    paths = []
    for index, (angles, center) in enumerate([
            ([10, -18, 8], [-.11, -.08, .59]), ([-18, 7, -13], [.12, -.07, .62]),
            ([-8, -17, 2], [-.10, .10, .66]), ([12, 16, 20], [.11, .08, .57]),
            ([14, -22, -18], [-.02, -.10, .55]), ([-24, 17, 9], [.04, .10, .60]),
            ([27, 19, 12], [-.13, .02, .65]), ([17, 26, -3], [.13, -.01, .63]),
            ([-25, 27, 23], [.01, .04, .51]), ([0, -28, -22], [0, 0, .55])]):
        name = f'intrinsic_{index}.png'
        transform = board_at(angles, np.array(center))
        transform[:3, :3] = np.diag([1., -1., -1.]) @ transform[:3, :3]
        transform[:3, 3] = center - transform[:3, :3] @ MID_BOARD
        checkerboard(folder / name, transform)
        paths.append(name)

    known = {name: look_at(*inputs) for name, inputs in WORLD_POSES.items()}
    config = {'rig_id': 'synthetic_rig', 'board': {'cols': COLS, 'rows': ROWS, 'square_m': SQUARE},
              'cameras': {name: {'slot': index, 'device_path': 'synthetic-' + name,
                                 'intrinsic_images': paths, 'anchors': []}
                          for index, name in enumerate(known, 1)}}
    for index, (role, angles, center) in enumerate([
            ('solve', [15, -7, 5], [0., -.02, 0]), ('solve', [-15, 9, 0], [.03, .05, .04]),
            ('check', [8, 5, -12], [-.04, .00, -.015])]):
        board = board_at(angles, np.array(center))
        name = f'anchor_{index}.png'
        points = checkerboard(folder / name, known['front'] @ board)
        config['cameras']['front']['anchors'].append(
            {'image': name, 'role': role, 'T_world_board': board.tolist(), 'anchors_px': points})

    pairs = []
    for index, (role, angles, center) in enumerate([
            ('solve', [18, 15, 5], [.03, -.04, .02]), ('solve', [-16, -15, 3], [-.03, .05, .01]),
            ('solve', [20, -9, 1], [0., .01, .045]), ('check', [1, -5, 15], [.03, .01, -.015])]):
        board = board_at(angles, np.array(center))
        row = {'role': role}
        for direction, view in (('source', 'front'), ('target', 'side')):
            name = f'shared_{view}_{index}.png'
            row[f'{direction}_image'] = name
            row[f'{direction}_anchors_px'] = checkerboard(folder / name, known[view] @ board)
        pairs.append(row)
    config['overlaps'] = [{'source': 'front', 'target': 'side', 'pairs': pairs}]
    return config, known


def flash_fixture(folder, *, missed=False, jitter_ms=0):
    frames = 97
    pulses = [7, 17, 34, 46, 68, 84]
    offsets = {'front': 0, 'side': 7 * NS_MS}
    config = {'session_dir': '.', 'camera_slots': {'front': 1, 'side': 2}, 'reference_camera': 'front',
              'pulse_intervals_s': [(right - left) * .04 for left, right in zip(pulses, pulses[1:])],
              'max_schedule_error_ms': 40, 'max_jitter_ms': 5,
              'roi_by_camera': {'front': [.25, .25, .75, .75], 'side': [.25, .25, .75, .75]}}
    cameras = {}
    start = 65_000_000_000
    for view, slot in config['camera_slots'].items():
        active = pulses[1:] if missed and view == 'side' else pulses
        writer = cv2.VideoWriter(str(folder / f'camera_{slot}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
        if not writer.isOpened():
            raise RuntimeError('Synthetic AVI writer failed')
        cameras[str(slot)] = {'actual': {'size_px': [64, 48]}, 'device': {'path': 'dev-' + view}}
        try:
            with (folder / f'camera_{slot}_timestamps.csv').open('w', newline='', encoding='utf-8') as handle:
                timestamps = csv.writer(handle)
                timestamps.writerow(['video_frame_index', 'host_monotonic_ns'])
                for index in range(frames):
                    image = np.full((48, 64, 3), 23, dtype=np.uint8)
                    if any(0 <= index - event < 3 for event in active):
                        image[12:36, 16:48] = 210
                    writer.write(image)
                    perturbation = jitter_ms * NS_MS if view == 'side' and index == pulses[-2] else 0
                    timestamps.writerow([index, start + (index + 1) * 40 * NS_MS + offsets[view] + perturbation])
        finally:
            writer.release()
    write_json(folder / 'session.json', {'status': 'completed', 'start_monotonic_ns': start,
                                        'stop_monotonic_ns': start + 100 * 40 * NS_MS,
                                        'written_frames': {slot: frames for slot in cameras}, 'cameras': cameras})
    return config, offsets, pulses


class CalibrationTests(unittest.TestCase):
    def test_actual_pngs_world_and_overlap_via_cli(self):
        with TemporaryDirectory() as label:
            folder = Path(label)
            config, known = calibration_fixture(folder)
            write_json(folder / 'calibration.json', config)
            result = subprocess.run([sys.executable, '-B', '-m', 'rig.calibrate',
                                     '--config', str(folder / 'calibration.json'),
                                     '--output', str(folder / 'rig.json')], capture_output=True, text=True, cwd=HERE.parent)
            self.assertEqual(result.returncode, 0, result.stderr)
            rig = load_json(folder / 'rig.json')
            self.assertEqual(rig['cameras']['side']['pose_via'], ['front'])
            self.assertEqual(len(rig['camera_graph']), 2)
            self.assertEqual(len(rig['measured_overlap_edges']), 1)
            for camera, pose in known.items():
                degrees, distance = pose_difference(np.asarray(rig['cameras'][camera]['T_camera_world']), pose)
                self.assertLess(degrees, 1.)
                self.assertLess(distance, .015)
                np.testing.assert_allclose(np.diag(rig['cameras'][camera]['K'])[:2], [570, 555], rtol=.02)

    def test_bad_labels_rejected(self):
        board = np.array([(c * 30, r * 30) for r in range(6) for c in range(8)], dtype=float)
        with self.assertRaisesRegex(ValueError, 'ambiguous board orientation'):
            reorder_anchor(board, (8, 6), [[105., 75.]] * 4, 20., 15.)

    def test_insufficient_and_undiverse_intrinsics_rejected(self):
        board = np.array([(c * .03, r * .03, 0) for r in range(6) for c in range(8)], dtype=float)
        imaged = (board[:, :2] * 1000 + [200, 200]).astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'Insufficient usable'):
            calibrate_intrinsics([imaged] * 3, board, SIZE, quality_config({}))
        with self.assertRaisesRegex(ValueError, 'cover too little'):
            calibrate_intrinsics([imaged] * 8, board, SIZE, quality_config({}))

    def test_graph_rejects_missing_and_cycle(self):
        checks = quality_config({})
        initial = {'a': {'T_camera_world': np.eye(4).tolist(), 'pose_via': []}, 'b': {}, 'c': {}}

        def edge(source, target, shift):
            t = np.eye(4)
            t[0, 3] = shift
            return {'source': source, 'target': target, 'T_camera_target_camera_source': t.tolist()}

        with self.assertRaisesRegex(ValueError, 'disconnected'):
            anchor_graph(copy.deepcopy(initial), [edge('a', 'b', .40)], checks)
        with self.assertRaisesRegex(ValueError, 'Inconsistent overlap'):
            anchor_graph(copy.deepcopy(initial), [edge('a', 'b', .40), edge('b', 'c', .40), edge('a', 'c', .54)], checks)


class TimingTests(unittest.TestCase):
    def test_extraction_and_timing_pipeline_cli(self):
        with TemporaryDirectory() as label:
            folder = Path(label)
            config, _, pulses = flash_fixture(folder)
            write_json(folder / 'flash_input.json', config)
            write_json(folder / 'align_input.json', dict(config, flash_report='flash.json'))
            commands = [
                ['extract_frame.py', '--session-dir', str(folder), '--slot', '2', '--frame',
                 str(pulses[0]), '--output', str(folder / 'pulse.png')],
                ['synchronize.py', 'flash', '--config', str(folder / 'flash_input.json'),
                 '--output', str(folder / 'flash.json')],
                ['synchronize.py', 'align', '--config', str(folder / 'align_input.json'),
                 '--output', str(folder / 'alignment.json')],
            ]
            for script, *arguments in commands:
                result = subprocess.run([sys.executable, '-B', '-m', f'rig.{script[:-3]}', *arguments],
                                        capture_output=True, text=True, cwd=HERE.parent)
                self.assertEqual(result.returncode, 0, result.stderr)
            image = cv2.imread(str(folder / 'pulse.png'), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(image[24, 32], 190)
            self.assertEqual(load_json(folder / 'alignment.json')['valid_groups'], 78)

    def test_flash_offsets_and_ambiguous_alignment(self):
        with TemporaryDirectory() as label:
            folder = Path(label)
            config, offsets, pulses = flash_fixture(folder)
            session = recording(config, folder)
            result = flash(config, session)
            self.assertEqual({cam: row['offset_ns'] for cam, row in result['cameras'].items()}, offsets)
            write_json(folder / 'flash.json', result)
            paired = alignment(dict(config, flash_report='flash.json'), session, folder)
            self.assertEqual(paired['valid_groups'], pulses[-1] - pulses[0] + 1)
            self.assertEqual(paired['groups'][0]['reason'], 'outside_verified_flash_interval')
            self.assertEqual(paired['groups'][30]['frames']['side']['residual_ns'], 0)
            ambiguous = alignment(dict(config, flash_report='flash.json', max_residual_ms=41), session, folder)
            self.assertEqual(ambiguous['status'], 'rejected')
            self.assertEqual(ambiguous['groups'][30]['reason'], 'ambiguous_frame_match')
            self.assertEqual(ambiguous['rejection_reason'], 'insufficient_valid_group_fraction')
            with self.assertRaisesRegex(ValueError, 'flash report'):
                alignment(config, session, folder)

    def test_timing_report_is_bound_to_original_recording(self):
        with TemporaryDirectory() as label:
            folder = Path(label)
            config, _, _ = flash_fixture(folder)
            write_json(folder / 'flash.json', flash(config, recording(config, folder)))
            with (folder / 'camera_2.avi').open('ab') as replacement:
                replacement.write(b'new file bytes')
            session = recording(config, folder)
            with self.assertRaisesRegex(ValueError, 'does not match this recording'):
                alignment(dict(config, flash_report='flash.json'), session, folder)

    def test_unverified_and_incomplete_alignment_rejected_for_delivery(self):
        with TemporaryDirectory() as label:
            folder = Path(label)
            config, _, _ = flash_fixture(folder)
            session = recording(config, folder)
            result = alignment(dict(config, allow_unverified_host_only=True, max_residual_ms=5), session, folder)
            self.assertTrue(result['unverified_timing'])
            self.assertEqual(result['status'], 'rejected')
            self.assertEqual(result['groups'][30]['reason'], 'insufficient_views')

    def test_missed_flash_and_jitter_rejected(self):
        for options, expected in [({'missed': True}, 'Detected 5 flashes'), ({'jitter_ms': 14}, 'unstable flash timing')]:
            with self.subTest(case=options), TemporaryDirectory() as label:
                folder = Path(label)
                config, _, _ = flash_fixture(folder, **options)
                with self.assertRaisesRegex(ValueError, expected):
                    flash(config, recording(config, folder))


if __name__ == '__main__':
    unittest.main(verbosity=2)

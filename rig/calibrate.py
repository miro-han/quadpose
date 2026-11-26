"""Calibrate intrinsics and world-anchored extrinsics from checkerboard images."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rig.common import CONVENTION, configure_console, integer, load_json, number, require, rigid_matrix, write_json


@dataclass
class Anchor:
    world: np.ndarray
    pixels: np.ndarray
    board_to_world: np.ndarray
    source: str


def read_corners(path, pattern):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    require(image is not None, f'Cannot read calibration image {path}')
    ok, corners = cv2.findChessboardCornersSB(image, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return (image.shape[1], image.shape[0]), None
    return (image.shape[1], image.shape[0]), corners.reshape(-1, 2)


def check_anchor_span(corners, minimum, path):
    require(np.min(np.ptp(corners, axis=0)) >= minimum,
            f'Checkerboard is too small or edge-on in {path}')


def reorder_anchor(corners, pattern, labels, maximum_error, minimum_margin):
    marks = np.asarray(labels, dtype=float)
    require(marks.shape == (4, 2) and np.isfinite(marks).all(),
            'anchors_px requires four finite [u,v] labels in order O, +X, +Y, +XY')
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    alternatives = (grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1])
    keyed = []
    for oriented in alternatives:
        extreme = oriented[[0, 0, -1, -1], [0, -1, 0, -1]]
        error = np.linalg.norm(extreme - marks, axis=1)
        keyed.append((float(np.max(error)), oriented))
    keyed.sort(key=lambda pair: pair[0])
    closest, oriented = keyed[0]
    require(closest <= maximum_error and keyed[1][0] - closest >= minimum_margin,
            f'Missing or ambiguous board orientation labels (best {closest:.2f} px, next {keyed[1][0]:.2f} px)')
    return np.ascontiguousarray(oriented.reshape(-1, 2), dtype=np.float64)


def image_errors(points, pixels, rvec, tvec, K, distortion):
    projection, _ = cv2.projectPoints(points, rvec, tvec, K, distortion)
    return np.linalg.norm(projection.reshape(-1, 2) - pixels, axis=1)


def rms(errors):
    return float(np.sqrt(np.mean(errors**2)))


def calibrate_intrinsics(images, board, size, checks):
    require(len(images) >= checks['minimum_intrinsic_views'], 'Insufficient usable intrinsic checkerboard images')
    centroids = np.array([corners.mean(axis=0) for corners in images])
    spans = np.ptp(centroids, axis=0) / size
    require(np.all(spans >= checks['min_centroid_span_fraction']),
            f'Intrinsic checkerboard centroids cover too little of the frame: {spans.tolist()}')
    _, K, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [board.astype(np.float32)] * len(images), images, size, None, None)
    require(np.isfinite(K).all() and np.isfinite(distortion).all() and
            np.all((np.diag(K)[:2] > np.asarray(size) * 0.2) & (np.diag(K)[:2] < np.asarray(size) * 5)) and
            0 < K[0, 2] < size[0] and 0 < K[1, 2] < size[1], 'Implausible intrinsic solution')
    normals = np.array([cv2.Rodrigues(rvec)[0][:, 2] for rvec in rvecs])
    spread = float(np.degrees(np.arccos(np.clip(np.abs(normals @ normals.T).min(), 0, 1))))
    require(spread >= checks['min_board_tilt_span_deg'], f'Intrinsic checkerboard tilt range too small: {spread:.2f}°')
    errors = [image_errors(board, pixels, rvec, tvec, K, distortion)
              for pixels, rvec, tvec in zip(images, rvecs, tvecs)]
    frame_rms = [rms(error) for error in errors]
    overall = rms(np.concatenate(errors))
    require(overall <= checks['max_intrinsic_rms_px'] and max(frame_rms) <= checks['max_intrinsic_view_rms_px'],
            f'Intrinsic reprojection error too high (overall {overall:.3f} px, worst view {max(frame_rms):.3f} px)')
    return K, distortion.reshape(-1), overall, frame_rms, spans, spread


def pose_matrix(rvec, tvec):
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
    matrix[:3, 3] = tvec.reshape(3)
    return matrix


def proper_view(matrix, anchor):
    orient, translation = matrix[:3, :3], matrix[:3, 3]
    if np.any((anchor.world @ orient.T + translation)[:, 2] <= 0):
        return False
    center = -orient.T @ translation
    board_normal = anchor.board_to_world[:3, 2]
    return (center - anchor.board_to_world[:3, 3]) @ board_normal > 0


def pose_difference(first, second):
    relative = first[:3, :3] @ second[:3, :3].T
    degrees = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    return degrees, float(np.linalg.norm(first_center - second_center))


def anchored_pose(board, anchors, independent, K, distortion, checks):
    require(len(anchors) >= 1 and len(independent) >= 1,
            'At least one solve anchor and one independent check image are required')
    world = np.concatenate([anchor.world for anchor in anchors])
    pixels = np.concatenate([anchor.pixels for anchor in anchors])
    hypotheses = []
    for anchor in anchors:
        ok, local_rotations, local_translations, *_ = cv2.solvePnPGeneric(
            board, anchor.pixels, K, distortion, flags=cv2.SOLVEPNP_IPPE)
        require(ok, f'Unable to form a pose hypothesis from {anchor.source}')
        for local_r, local_t in zip(local_rotations, local_translations):
            seed = pose_matrix(local_r, local_t) @ np.linalg.inv(anchor.board_to_world)
            rvec = cv2.Rodrigues(seed[:3, :3])[0]
            tvec = seed[:3, 3].copy().reshape(3, 1)
            refined_r, refined_t = cv2.solvePnPRefineLM(world, pixels, K, distortion, rvec, tvec)
            matrix = pose_matrix(refined_r, refined_t)
            if all(proper_view(matrix, item) for item in anchors):
                error = rms(image_errors(world, pixels, refined_r, refined_t, K, distortion))
                hypotheses.append((error, matrix, refined_r, refined_t))
    require(bool(hypotheses), 'No positive-depth, printed-face world pose hypothesis')
    hypotheses.sort(key=lambda item: item[0])
    best, matrix, rvec, tvec = hypotheses[0]
    require(best <= checks['max_anchor_rms_px'], f'Anchor fit error is too high: {best:.3f} px')
    per_anchor = [rms(image_errors(a.world, a.pixels, rvec, tvec, K, distortion)) for a in anchors]
    require(max(per_anchor) <= checks['max_anchor_rms_px'], f'A solve anchor exceeds the error limit: {per_anchor}')
    require(all(proper_view(matrix, anchor) for anchor in independent),
            'World pose fails the printed-face/depth check on independent board images')
    check_errors = [rms(image_errors(a.world, a.pixels, rvec, tvec, K, distortion)) for a in independent]
    require(max(check_errors) <= checks['max_check_rms_px'],
            f'World pose fails independent reprojection: {check_errors}')
    distinct = [(error, runner) for error, runner, _, _ in hypotheses[1:]
                if any(delta > limit for delta, limit in zip(pose_difference(matrix, runner), (1.0, 0.01)))]
    if distinct:
        ambiguity_gap = distinct[0][0] - best
        require(ambiguity_gap >= checks['min_ambiguity_gap_px'],
                f'Ambiguous world pose: distinct solutions only {ambiguity_gap:.3f} px apart; use separated tilted anchors')
    else:
        ambiguity_gap = None
    return matrix, per_anchor, check_errors, ambiguity_gap


def quality_config(config):
    supplied = config.get('quality', {})
    defaults = {'min_centroid_span_fraction': 0.15, 'min_board_tilt_span_deg': 15.0,
                'max_intrinsic_rms_px': 0.7, 'max_intrinsic_view_rms_px': 1.5,
                'max_anchor_rms_px': 1.3, 'max_check_rms_px': 1.8,
                'max_label_error_px': 20.0, 'min_label_margin_px': 12.0, 'min_ambiguity_gap_px': 0.16,
                'min_anchor_span_px': 35.0,
                'max_pair_rms_px': 1.4, 'max_pair_check_rms_px': 2.5, 'min_pair_tilt_span_deg': 8.0,
                'max_graph_rotation_error_deg': 1.5, 'max_graph_center_error_m': 0.025}
    result = {key: number(supplied.get(key, value), key) for key, value in defaults.items()}
    result['minimum_intrinsic_views'] = integer(supplied.get('minimum_intrinsic_views', 8),
                                                'minimum_intrinsic_views', minimum=6)
    require(result['min_centroid_span_fraction'] < 1 and result['min_board_tilt_span_deg'] <= 90,
            'Invalid intrinsic diversity requirement')
    return result


def calibrate_camera(config, directory, pattern, board, checks):
    files = config['intrinsic_images']
    require(isinstance(files, list) and bool(files), 'intrinsic_images must list checkerboard image paths')
    paths = [(directory / file).resolve() for file in files]
    require(len(set(paths)) == len(paths), 'Duplicate intrinsic image path')
    images, accepted, rejected = [], [], []
    size = None
    for file, path in zip(files, paths):
        current, corners = read_corners(path, pattern)
        if size is None:
            size = current
        require(current == size, f'Image size mismatch in {file}: expected {size}, got {current}')
        if corners is None:
            rejected.append({'image': file, 'reason': 'checkerboard_not_found'})
        else:
            images.append(corners)
            accepted.append(file)
    K, distortion, overall, per_view, spans, spread = calibrate_intrinsics(images, board, size, checks)

    anchors = {'solve': [], 'check': []}
    entries = config.get('anchors', [])
    require(isinstance(entries, list), 'anchors must be a list')
    used = set(paths)
    for item in entries:
        file = item['image']
        path = (directory / file).resolve()
        role = item['role']
        require(role in anchors and path not in used, f'Unknown anchor role or duplicate calibration image: {file}')
        used.add(path)
        size_anchor, corners = read_corners(path, pattern)
        require(size_anchor == size and corners is not None, f'Missing checkerboard or inconsistent image size in {file}')
        check_anchor_span(corners, checks['min_anchor_span_px'], file)
        try:
            pixels = reorder_anchor(corners, pattern, item['anchors_px'],
                                    checks['max_label_error_px'], checks['min_label_margin_px'])
        except ValueError as error:
            raise ValueError(f'{file}: {error}') from error
        world_transform = rigid_matrix(item['T_world_board'], 'T_world_board')
        world = board @ world_transform[:3, :3].T + world_transform[:3, 3]
        anchors[role].append(Anchor(world, pixels, world_transform, file))
    quality = {'intrinsic_rms_px': overall, 'intrinsic_centroid_span_fraction': spans.tolist(),
               'intrinsic_tilt_span_deg': spread, 'intrinsic_rejected': rejected,
               'intrinsic_views': [dict(image=name, rms_px=error) for name, error in zip(accepted, per_view)]}
    result = {'image_size_px': list(size), 'K': K.tolist(), 'distortion': distortion.tolist(),
              'distortion_model': 'opencv_brown_k1_k2_p1_p2_k3',
              'slot': integer(config['slot'], 'slot', minimum=1), 'device_path': config['device_path'],
              'quality': quality}
    if entries:
        matrix, solve_errors, check_errors, gap = anchored_pose(
            board, anchors['solve'], anchors['check'], K, distortion, checks)
        result['T_camera_world'] = matrix.tolist()
        result['pose_via'] = []
        quality.update(solve_anchors=[dict(image=a.source, rms_px=e) for a, e in zip(anchors['solve'], solve_errors)],
                       check_anchors=[dict(image=a.source, rms_px=e) for a, e in zip(anchors['check'], check_errors)],
                       distinct_anchor_pose_rms_gap_px=gap)
    return result


def lens(camera):
    return np.array(camera['K']), np.array(camera['distortion'])


def local_board_pose(board, pixels, K, distortion, source):
    ok, rotations, translations, *_ = cv2.solvePnPGeneric(board, pixels, K, distortion,
                                                           flags=cv2.SOLVEPNP_IPPE)
    anchor = Anchor(board, pixels, np.eye(4), source)
    options = []
    if ok:
        for rvec, tvec in zip(rotations, translations):
            matrix = pose_matrix(rvec, tvec)
            if proper_view(matrix, anchor):
                options.append((rms(image_errors(board, pixels, rvec, tvec, K, distortion)), matrix))
    require(bool(options), f'No printed-face board pose in {source}')
    return min(options, key=lambda candidate: candidate[0])[1]


def pair_measurement(entry, cameras, directory, pattern, board, checks):
    source, target = entry['source'], entry['target']
    stereo = [cameras[source], cameras[target]]
    camera_lenses = [lens(camera) for camera in stereo]
    images, used = {'solve': [], 'check': []}, [set(), set()]
    for pair in entry['pairs']:
        role = pair['role']
        require(role in images, f'Unknown pair role {role}')
        observations, files = [], []
        for index, (key, camera) in enumerate(zip(('source', 'target'), stereo)):
            file = pair[f'{key}_image']
            path = (directory / file).resolve()
            require(path not in used[index], f'Duplicate shared-board image {file}')
            used[index].add(path)
            size, corners = read_corners(path, pattern)
            require(list(size) == camera['image_size_px'] and corners is not None,
                    f'Shared-board image {file}: wrong image size or missing checkerboard')
            check_anchor_span(corners, checks['min_anchor_span_px'], file)
            observations.append(reorder_anchor(corners, pattern, pair[f'{key}_anchors_px'],
                                              checks['max_label_error_px'], checks['min_label_margin_px']))
            files.append(file)
        images[role].append((observations, files))
    require(len(images['solve']) >= 2 and bool(images['check']),
            'A measured overlap needs at least two solve pairs and one held-out pair')
    for side in (0, 1):
        normals = np.array([local_board_pose(board, observations[side], *camera_lenses[side], files[side])[:3, 2]
                            for observations, files in images['solve']])
        spread = float(np.degrees(np.arccos(np.clip(np.abs(normals @ normals.T).min(), 0, 1))))
        require(spread >= checks['min_pair_tilt_span_deg'], f'Shared board needs more tilt diversity ({spread:.2f}°)')

    image_array = [[observations[side].astype(np.float32).reshape(-1, 1, 2)
                    for observations, _ in images['solve']] for side in (0, 1)]
    overall, _, _, _, _, rotation, translation, *_ = cv2.stereoCalibrate(
        [board.astype(np.float32)] * len(images['solve']), *image_array,
        *camera_lenses[0], *camera_lenses[1], tuple(stereo[0]['image_size_px']),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))
    require(overall <= checks['max_pair_rms_px'], f'Stereo reprojection fit is too high: {overall:.3f} px')
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation.reshape(3)
    rigid_matrix(transform, 'stereo transform')
    require(np.linalg.norm(translation) >= .03, 'Measured overlap baseline is below 30 mm')

    report = []
    for observations, files in images['check']:
        row = {'source_image': files[0], 'target_image': files[1]}
        for index, relative in enumerate((transform, np.linalg.inv(transform))):
            target_index = 1 - index
            source_pose = local_board_pose(board, observations[index], *camera_lenses[index], files[index])
            predicted = relative @ source_pose
            require(proper_view(predicted, Anchor(board, observations[target_index], np.eye(4), files[target_index])),
                    f'Overlap prediction fails depth/printed-face check in {files[target_index]}')
            error = rms(image_errors(board, observations[target_index], cv2.Rodrigues(predicted[:3, :3])[0],
                                     predicted[:3, 3], *camera_lenses[target_index]))
            require(error <= checks['max_pair_check_rms_px'],
                    f'Held-out overlap prediction error in {files[target_index]}: {error:.3f} px')
            row['target_rms_px' if index == 0 else 'source_rms_px'] = error
        report.append(row)
    return {'source': source, 'target': target, 'T_camera_target_camera_source': transform.tolist(),
            'baseline_m': float(np.linalg.norm(translation)), 'rms_px': overall,
            'solve_pairs': len(images['solve']), 'check_pairs': report}


def anchor_graph(cameras, edges, checks):
    require(any('T_camera_world' in camera for camera in cameras.values()), 'At least one anchored camera is required')
    if edges:
        connected = {next(iter(cameras))}
        while True:
            expanded = connected | {endpoint for edge in edges if connected.intersection((edge['source'], edge['target']))
                                    for endpoint in (edge['source'], edge['target'])}
            if expanded == connected:
                break
            connected = expanded
        require(connected == set(cameras), f'Overlap graph disconnected: {sorted(set(cameras) - connected)}')

    while True:
        added = False
        for edge in edges:
            source, target = edge['source'], edge['target']
            matrix = np.asarray(edge['T_camera_target_camera_source'])
            for known, unknown, relative in ((source, target, matrix), (target, source, np.linalg.inv(matrix))):
                first, second = cameras[known], cameras[unknown]
                if 'T_camera_world' in first and 'T_camera_world' not in second:
                    second['T_camera_world'] = (relative @ first['T_camera_world']).tolist()
                    second['pose_via'] = [*first['pose_via'], known]
                    added = True
        if not added:
            break

    for name, camera in cameras.items():
        require('T_camera_world' in camera, f'Camera {name} has no connected path to a world anchor')
        pose = np.asarray(camera['T_camera_world'])
        camera['center_world_m'] = (-pose[:3, :3].T @ pose[:3, 3]).tolist()
    for edge in edges:
        source, target = edge['source'], edge['target']
        predicted = np.asarray(edge['T_camera_target_camera_source']) @ cameras[source]['T_camera_world']
        degrees, distance = pose_difference(predicted, np.array(cameras[target]['T_camera_world']))
        require(degrees <= checks['max_graph_rotation_error_deg'] and distance <= checks['max_graph_center_error_m'],
                f'Inconsistent overlap cycle/direct anchors at {source}/{target}: {degrees:.2f}°, {distance:.3f} m')
        edge.update(graph_rotation_error_deg=degrees, graph_center_error_m=distance)


def calibrate(config, directory):
    cols = integer(config['board']['cols'], 'board.cols', minimum=3)
    rows = integer(config['board']['rows'], 'board.rows', minimum=3)
    require(cols != rows, 'Use a rectangular checkerboard with unequal counts for its X and Y directions')
    square = number(config['board']['square_m'], 'board.square_m', maximum=.5)
    board = np.zeros((rows * cols, 3), dtype=np.float64)
    board[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square
    checks = quality_config(config)
    require(isinstance(config['rig_id'], str) and bool(config['rig_id'].strip()), 'rig_id is required')
    items = config['cameras']
    require(isinstance(items, dict) and 2 <= len(items) <= 4, 'Specify two to four cameras')
    for field in ('slot', 'device_path'):
        values = [item[field] for item in items.values()]
        require(all(values) and len(set(values)) == len(values), f'Camera {field} values must be unique and nonempty')
    require(all(isinstance(item['device_path'], str) and item['device_path'].strip() for item in items.values()),
            'Each camera requires the recorder device path')
    cameras = {}
    for camera_id, item in items.items():
        try:
            cameras[camera_id] = calibrate_camera(item, directory, (cols, rows), board, checks)
        except (ValueError, OSError, KeyError, cv2.error) as error:
            raise ValueError(f'Camera {camera_id}: {error}') from error
    edges = []
    named_edges = set()
    for edge in config.get('overlaps', []):
        source, target = edge['source'], edge['target']
        pair = frozenset((source, target))
        require(source in cameras and target in cameras and len(pair) == 2 and pair not in named_edges,
                f'Unknown, repeated or self overlap {source}/{target}')
        named_edges.add(pair)
        try:
            edges.append(pair_measurement(edge, cameras, directory, (cols, rows), board, checks))
        except (ValueError, OSError, KeyError, cv2.error) as error:
            raise ValueError(f'Overlap {source}/{target}: {error}') from error
    anchor_graph(cameras, edges, checks)
    graph = []
    for source, first in cameras.items():
        for target, second in cameras.items():
            if source != target:
                baseline = float(np.linalg.norm(np.array(first['center_world_m']) - second['center_world_m']))
                require(baseline >= 0.03, f'Camera centers {source}/{target} are separated by less than 30 mm')
                transform = np.array(second['T_camera_world']) @ np.linalg.inv(first['T_camera_world'])
                graph.append({'source': source, 'target': target, 'baseline_m': baseline,
                              'T_camera_target_camera_source': transform.tolist()})
    return {'schema_version': 1, 'rig_id': config['rig_id'], 'coordinate_convention': CONVENTION,
            'board': config['board'], 'quality_limits': checks, 'cameras': cameras, 'camera_graph': graph,
            'measured_overlap_edges': edges}


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), f'Choose a new output path: {args.output}')
        result = calibrate(load_json(args.config), args.config.resolve().parent)
        fingerprint = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        result['calibration_id'] = hashlib.sha256(fingerprint).hexdigest()
        write_json(args.output, result)
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        parser.exit(2, f'Calibration rejected: {error}\n')
    print(f'Calibrated {len(result["cameras"])} cameras: {args.output}')


if __name__ == '__main__':
    main()

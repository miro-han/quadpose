"""Shared I/O and transform checks for the rig tools."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

CONVENTION = ('Right-handed world in metres: +X right, +Y away from the front camera, '
              '+Z up. OpenCV camera: +x right, +y down, +z forward. '
              'T_camera_world maps homogeneous world points into camera coordinates.')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='backslashreplace')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False)
            handle.write('\n')
        except Exception:
            handle.close()
            temporary.unlink()
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, minimum=0.0, maximum=None):
    fail = f'{name} must be numeric, finite, greater than {minimum}'
    require(isinstance(value, (int, float)) and not isinstance(value, bool), fail)
    require(np.isfinite(value) and value > minimum, fail)
    if maximum is not None:
        require(value <= maximum, f'{name} must be at most {maximum}')
    return float(value)


def integer(value, name, *, minimum):
    require(type(value) is int and value >= minimum, f'{name} must be an integer >= {minimum}')
    return value


def rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), f'{name} must be a finite 4x4 matrix')
    orient = matrix[:3, :3]
    require(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7) and
            np.allclose(orient.T @ orient, np.eye(3), atol=1e-6) and
            np.isclose(np.linalg.det(orient), 1, atol=1e-6), f'{name} must be a proper rigid transform')
    return matrix

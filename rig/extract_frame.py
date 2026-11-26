"""Extract an exact, unmodified recorder frame for checkerboard calibration."""

import argparse
from pathlib import Path

import cv2

from rig.common import configure_console, integer, load_json, require


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        integer(args.slot, 'slot', minimum=1)
        integer(args.frame, 'frame', minimum=0)
        require(args.output.suffix.lower() == '.png' and not args.output.exists(), 'Choose a new .png output path')
        session = load_json(args.session_dir / 'session.json')
        require(session['status'] == 'completed', 'Use a completed recorder session')
        require(args.frame < session['written_frames'][str(args.slot)], 'Frame index is outside the recorded stream')
        path = args.session_dir / f'camera_{args.slot}.avi'
        cap = cv2.VideoCapture(str(path))
        try:
            require(cap.isOpened(), f'Could not open {path}')
            for index in range(args.frame + 1):
                ok, image = cap.read()
                require(ok and image is not None, f'Could not decode recorded frame {index}')
        finally:
            cap.release()
        ok, data = cv2.imencode('.png', image)
        require(ok, 'Could not encode extracted PNG')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle:
            handle.write(data.tobytes())
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        parser.exit(2, f'Frame extraction rejected: {error}\n')
    print(f'{path} frame {args.frame} -> {args.output}')


if __name__ == '__main__':
    main()

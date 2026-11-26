"""Inspect Canny edges in a recorded video; object identification is not implemented.

Uses OpenCV's GaussianBlur/Canny implementations:
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
The video processing and comparison wrapper was written for this project.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def label(image, text):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(image, text, (9, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def resize(image, width):
    return cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))


def save_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=Path.home() / 'Pictures' /
                        'Camera Roll' / 'WIN_20261124_20_05_34_Pro.mp4')
    parser.add_argument('--low', type=int, default=30)
    parser.add_argument('--high', type=int, default=90)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent / 'outputs' / 'edges')
    args = parser.parse_args()
    if not 0 <= args.low < args.high:
        parser.error('Require 0 <= low < high')
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or total <= 0:
        cap.release()
        raise RuntimeError('Video must report valid FPS and frame count')
    args.output.mkdir(parents=True, exist_ok=True)
    samples = {round((total - 1) * x) for x in [0.02, 0.28, 0.41, 0.54, 0.67, 0.95]}
    rows, writer, count = [], None, 0
    width = height = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
            edges = cv2.Canny(blurred, args.low, args.high, apertureSize=3, L2gradient=True)
            overlay = frame.copy()
            overlay[edges != 0] = (0, 255, 0)
            preview = np.hstack([
                label(resize(frame, 640), f'Original | frame {count} | {count/fps:.2f}s'),
                label(resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 640),
                      f'Canny {args.low}/{args.high}'),
                label(resize(overlay, 640), 'Green = detected edges'),
            ])
            if writer is None:
                writer = cv2.VideoWriter(str(args.output / 'edge_preview.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (preview.shape[1], preview.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create output video')
            writer.write(preview)
            if count in samples:
                # Fixed manual viewing crop, not an automatically detected object region.
                ys, xs = slice(int(height*.20), int(height*.85)), slice(int(width*.40), int(width*.85))
                tiles = [label(resize(frame[ys, xs], 400), f'Original {count/fps:.2f}s')]
                for lo, hi in [(15, 45), (30, 90), (60, 150)]:
                    test = cv2.Canny(blurred, lo, hi, apertureSize=3, L2gradient=True)
                    tiles.append(label(resize(cv2.cvtColor(test[ys, xs], cv2.COLOR_GRAY2BGR), 400),
                                       f'Canny {lo}/{hi}'))
                tiles.append(label(resize(overlay[ys, xs], 400), f'Overlay {args.low}/{args.high}'))
                rows.append(np.hstack(tiles))
                save_image(args.output / f'frame_{count:04d}_original.png', frame)
                save_image(args.output / f'frame_{count:04d}_edges.png', edges)
                save_image(args.output / f'frame_{count:04d}_overlay.png', overlay)
            count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not count:
        raise RuntimeError('No frames decoded')
    if rows:
        save_image(args.output / 'threshold_comparison.jpg', np.vstack(rows))
    metadata = {
        'source': str(args.video), 'frames_processed': count, 'reported_frames': total,
        'source_size_px': [width, height], 'fps': fps,
        'approx_duration_seconds': count / fps,
        'processing': 'full-resolution grayscale, GaussianBlur 5x5 sigma=1.2, Canny L2gradient',
        'canny_thresholds': [args.low, args.high], 'coordinate_space': 'original_distorted_pixels',
        'comparison_thresholds': [[15, 45], [30, 90], [60, 150]],
        'comparison_crop_fraction_xyxy': [.40, .20, .85, .85],
        'note': 'Edge inspection only. No plate identification, corner detection, or camera synchronization. Output video has no audio; timing uses nominal FPS.'
    }
    (args.output / 'run_info.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f'Outputs: {args.output}')


if __name__ == '__main__':
    main()

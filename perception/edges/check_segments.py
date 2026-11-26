"""Compare Canny and OpenCV LSD; export per-view line candidates, NOT matches.

LSD: Grompone von Gioi et al., IPOL 2012, doi:10.5201/ipol.2012.gjmr-lsd.
https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html
ROI, length and edge-support screening below are project heuristics.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
try:
    from perception.edges.check_edges import label, resize, save_image
except ImportError:
    from check_edges import label, resize, save_image


def extract(gray, detector, bounds, min_length, min_support):
    edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 15, 45, L2gradient=True)
    distances = cv2.distanceTransform(255-edge, cv2.DIST_L2, 3)
    lines, widths, precision, nfa = detector.detect(gray)
    result = []
    if lines is None:
        return edge, result
    x0, y0, x1, y1 = bounds
    h, w = gray.shape
    for index, line in enumerate(lines.reshape(-1, 4)):
        a, b = line[:2], line[2:]
        length = float(np.linalg.norm(b-a))
        points = np.linspace(a, b, max(2, int(length)+1))
        px = np.clip(np.rint(points[:, 0]).astype(int), 0, w-1)
        py = np.clip(np.rint(points[:, 1]).astype(int), 0, h-1)
        support = float(np.mean(distances[py, px] <= 2.5))
        inside = bool(np.all((points[:, 0] >= x0) & (points[:, 0] <= x1)
                            & (points[:, 1] >= y0) & (points[:, 1] <= y1)))
        reasons = []
        if not inside:
            reasons.append('outside_manual_roi')
        if length < min_length:
            reasons.append('short_segment')
        if support < min_support:
            reasons.append('weak_canny_support')
        result.append(dict(local_id=index, endpoints_px=line.tolist(), length_px=length,
                           angle_deg=float(np.degrees(np.arctan2(*(b-a)[::-1])) % 180),
                           canny_support=support, width_px=float(widths.reshape(-1)[index]),
                           log_nfa=float(nfa.reshape(-1)[index]) if nfa is not None else None,
                           candidate=not reasons, rejection_reasons=reasons,
                           physical_edge_id=None))
    return edge, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--camera-id', default='camera_1')
    parser.add_argument('--roi', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='Manual normalized viewing region, not automatic object detection')
    parser.add_argument('--min-length', type=float, default=25)
    parser.add_argument('--min-support', type=float, default=.35)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roi = args.roi or [0, 0, 1, 1]
    if not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        parser.error('ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1')
    if args.min_length <= 0 or not 0 <= args.min_support <= 1:
        parser.error('Invalid length/support threshold')
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError('Invalid source FPS')
    args.output.mkdir(parents=True, exist_ok=True)
    samples = {round((total-1)*x) for x in [.02, .28, .41, .54, .67, .95]}
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    writer, count, selected_counts, sheets = None, 0, [], []
    try:
        with (args.output/'segments.jsonl').open('w', encoding='utf-8') as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                bounds = [roi[0]*w, roi[1]*h, roi[2]*w, roi[3]*h]
                edges, segments = extract(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector,
                                          bounds, args.min_length, args.min_support)
                all_lines, candidates = frame.copy(), frame.copy()
                selected = [s for s in segments if s['candidate']]
                selected_counts.append(len(selected))
                for s in segments:
                    a, b = np.rint(s['endpoints_px']).astype(int).reshape(2, 2)
                    cv2.line(all_lines, a, b, (0, 200, 255), 2, cv2.LINE_AA)
                    if s['candidate']:
                        cv2.line(candidates, a, b, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(candidates, str(s['local_id']), tuple((a+b)//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)
                if args.roi:
                    box = np.rint(bounds).astype(int)
                    cv2.rectangle(candidates, tuple(box[:2]), tuple(box[2:]), (255, 200, 0), 2)
                tiles = [label(resize(frame, 640), f'Original {count/fps:.2f}s'),
                         label(resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 640), 'Canny 15/45'),
                         label(resize(all_lines, 640), f'LSD all: {len(segments)}'),
                         label(resize(candidates, 640), f'Candidates: {len(selected)} | local IDs only')]
                preview = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output/'segment_preview.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (preview.shape[1], preview.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create output video')
                writer.write(preview)
                log.write(json.dumps(dict(camera_id=args.camera_id, frame_index=count,
                                          nominal_video_seconds=count/fps,
                                          coordinate_space='original_distorted_pixels',
                                          segments=segments), ensure_ascii=False)+'\n')
                if count in samples:
                    save_image(args.output/f'frame_{count:04d}_comparison.jpg', preview)
                    save_image(args.output/f'frame_{count:04d}_candidates.png', candidates)
                    sheets.append(resize(preview, 800))
                count += 1
                if count % 100 == 0:
                    print(f'{count}/{total}', flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
    if sheets:
        save_image(args.output/'comparison_sheet.jpg', np.vstack(sheets))
    summary = dict(source=str(args.video), processed_frames=count, camera_id=args.camera_id,
                   manual_roi=args.roi, min_length_px=args.min_length, min_support=args.min_support,
                   mean_candidates=float(np.mean(selected_counts)) if selected_counts else 0,
                   note='Heuristic line candidates, not confirmed box edges. Endpoints are not necessarily physical corners. IDs reset each frame. No cross-view matching. Nominal time is not synchronization.',
                   citation='Grompone von Gioi et al. LSD, IPOL 2012; OpenCV LineSegmentDetector')
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

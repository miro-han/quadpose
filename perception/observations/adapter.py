"""Convert one real source frame into unlabeled point/line observations."""
import argparse
import json
from pathlib import Path, PureWindowsPath
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frame_row(path, key, value):
    with path.open(encoding="utf-8-sig") as rows:
        for line in rows:
            row = json.loads(line)
            if row[key] == value:
                return row
    raise ValueError(f"{path.name} has no {key}={value}")


def binary_image(path, shape):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape:
        raise ValueError(f"Image size mismatch or unreadable: {path}")
    return image > 0


def transform_pixels(points, calibration):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if calibration is None or not len(points):
        return points.copy()
    K = np.asarray(calibration["K"], dtype=float)
    D = np.asarray(calibration["distortion"], dtype=float)
    return cv2.undistortPoints(points[:, None], K, D, P=K).reshape(-1, 2)


def corners_from_segments(lines, tolerance=10.0):
    """Bound both finite segment extents, reject near parallels, merge duplicates."""
    accepted = [r for r in lines if r["reason"] is None]
    corners = []
    for i, left in enumerate(accepted):
        a, b = np.array(left["xy"])
        for right in accepted[i+1:]:
            c, d = np.array(right["xy"])
            matrix = np.column_stack((b-a, c-d))
            lengths = np.linalg.norm(matrix, axis=0)
            if abs(np.linalg.det(matrix)) < 0.4 * np.prod(lengths):
                continue
            factors = np.linalg.solve(matrix, c-a)
            if np.any(np.minimum(factors, 1-factors)*lengths < -tolerance):
                continue
            point = a + factors[0]*(b-a)
            parents = [left["local_id"], right["local_id"]]
            duplicate = next((r for r in corners if np.linalg.norm(point-r["xy"]) < 4), None)
            if duplicate is not None:
                duplicate["parents"].append(parents)
            else:
                corners.append(dict(local_id=len(corners), xy=point.tolist(), parents=[parents],
                                    model_id=None, reason=None))
    return corners


def extract(video, session, lsd, source_frame, calibration=None, camera_id=None):
    meta = read_json(session / "session.json")
    lsd_meta = read_json(lsd / "summary.json")
    if any(PureWindowsPath(m["source"]).name != video.name for m in (meta, lsd_meta)):
        raise ValueError("Video basename differs from segmentation or LSD provenance")
    record = frame_row(session / "frames.jsonl", "source_frame", source_frame)
    lines = frame_row(lsd / "segments.jsonl", "frame_index", source_frame)
    if camera_id is not None and (lines["camera_id"]!=camera_id or lsd_meta["camera_id"]!=camera_id):
        raise ValueError("LSD camera ID differs from the requested camera")
    if lines["coordinate_space"] != "original_distorted_pixels":
        raise ValueError("Expected raw distorted LSD pixels")
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Cannot decode requested frame")
    finally:
        cap.release()
    h, w = frame.shape[:2]
    if calibration and list(calibration["image_size_px"]) != [w, h]:
        raise ValueError("Calibration resolution differs from source")
    name = f"{record['local_frame']:06d}.png"
    # Verify the exact source frame across moved handoffs and repeated camera_1.avi names.
    saved = cv2.imread(str(session / "input_frames" / name.replace(".png",".jpg")))
    if saved is None or saved.shape != frame.shape:
        raise ValueError("Missing SAM input frame or inconsistent dimensions")
    encoded_ok, encoded = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not encoded_ok or cv2.norm(cv2.imdecode(encoded,cv2.IMREAD_COLOR),saved,cv2.NORM_L1)/(h*w*3)>1:
        raise ValueError("Decoded frame differs from the archived SAM input")
    lsd_source=Path(lsd_meta["source"])
    if lsd_source.is_file() and lsd_source.resolve()!=video.resolve():
        raise ValueError("LSD belongs to a different accessible source file")
    mask = binary_image(session / "masks" / name, (h, w))
    edge = binary_image(session / "selected_edges" / name, (h, w))
    roi = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    edge_band = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8))
    if calibration:
        K, D = np.array(calibration["K"]), np.array(calibration["distortion"])
        frame = cv2.undistort(frame, K, D, None, K)
        roi_output = cv2.undistort(roi*255, K, D, None, K) > 127
    else:
        roi_output = roi > 0
    result = []
    for line in lines["segments"]:
        raw = np.array(line["endpoints_px"], dtype=float).reshape(2, 2)
        samples = np.linspace(*raw, 21)
        x, y = np.rint(samples).astype(int).T
        inside = (x>=0) & (x<w) & (y>=0) & (y<h)
        x, y = x[inside], y[inside]
        roi_fraction = float(np.sum(roi[y,x])/len(samples))
        support = float(np.sum(edge_band[y,x])/len(samples))
        rect = transform_pixels(samples, calibration)
        vector = rect[-1]-rect[0]
        length = np.linalg.norm(vector)
        normal = np.array([-vector[1], vector[0]]) / max(length, 1e-9)
        bend = float(np.max(np.abs((rect-rect[0]) @ normal)))
        reason = None
        if roi_fraction < 0.7:
            reason = "outside_roi"
        elif length < 15:
            reason = "short_segment"
        elif support < 0.35:
            reason = "weak_canny_support"
        elif bend > 2:
            reason = "rectified_curve"
        result.append(dict(local_id=line["local_id"], xy=rect[[0,-1]].tolist(),
                           reason=reason, roi_fraction=roi_fraction, edge_support=support,
                           legacy_candidate=line["candidate"], model_id=None))
    corners = corners_from_segments(result)
    for corner in corners:
        x, y = np.rint(corner["xy"]).astype(int)
        if not (0<=x<w and 0<=y<h and roi_output[y,x]):
            corner["reason"] = "outside_roi"
    return frame, corners, result


def observations(corners, segments):
    from pose.fusion import Detections
    return Detections(np.array([r["xy"] for r in corners if r["reason"] is None]).reshape(-1,2),
                      np.array([r["xy"] for r in segments if r["reason"] is None]).reshape(-1,2,2))


def apply_prior(corners, segments, camera_id, calibration, prior):
    from pose.fusion import BoxModel, Camera, EDGES, Pose, associate_view, project
    ext=np.asarray(calibration["T_camera_world"],float)
    transform=np.asarray(prior["T_world_object"],float)
    for T in (ext,transform):
        if (T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1])
                or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
                or not np.isclose(np.linalg.det(T[:3,:3]),1)):
            raise ValueError("Expected a proper SE(3) transform")
    camera=Camera(camera_id,np.asarray(calibration["K"]),ext[:3,:3],
                  -ext[:3,:3].T@ext[:3,3],tuple(calibration["image_size_px"]))
    pose=Pose(transform[:3,:3],transform[:3,3])
    box=BoxModel(np.asarray(prior["size_m"],float))
    points,lines=associate_view(pose,camera,observations(corners,segments),box)
    uv,_=project(box.vertices,pose,camera)
    for kind,rows,pairs in (("point",corners,points),("line",segments,lines)):
        accepted=[r for r in rows if r["reason"] is None]
        for row in accepted:
            row["assignment_reason"]="no_visible_match_within_gate_or_competing_detection"
        for model,obs in pairs:
            row=accepted[obs]
            if kind=="point":
                expected=uv[model]
                residual=np.linalg.norm(expected-row["xy"])
            else:
                expected=uv[EDGES[model]]
                a,b=np.asarray(row["xy"])
                delta=b-a
                normal=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
                residual=np.sqrt(np.mean(((expected-a)@normal)**2))
            row.update(model_id=int(model),residual_px=float(residual),
                       predicted_xy=expected.tolist(),assignment_reason="assigned")


def paint(frame, corners, segments):
    image = frame.copy()
    colors = {"outside_roi": (90,90,90), "short_segment": (100,75,75),
              "weak_canny_support": (120,70,120), "rectified_curve": (140,70,60)}
    for prefix, rows in (("E",segments),("V",corners)):
        for r in rows:
            coords = np.rint(r["xy"]).astype(int).reshape(-1,2)
            color = (80,255,80) if r["model_id"] is not None else colors.get(r["reason"],(0,175,255))
            if len(coords)==2:
                cv2.line(image,tuple(coords[0]),tuple(coords[1]),color,1,cv2.LINE_AA)
            else:
                cv2.circle(image,tuple(coords[0]),4,color,1)
            if r["model_id"] is not None:
                expected=np.rint(r["predicted_xy"]).astype(int).reshape(-1,2)
                cv2.line(image,tuple(expected[0]),tuple(expected[-1]),(255,195,65),2)
                cv2.line(image,tuple(expected.mean(axis=0).astype(int)),
                         tuple(coords.mean(axis=0).astype(int)),(255,195,65),1)
                cv2.putText(image,f"{prefix}{r['model_id']} {r['residual_px']:.1f}px",
                            tuple(coords.mean(axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    cv2.rectangle(image,(0,0),(790,34),(20,20,20),-1)
    cv2.putText(image,"green: assigned + residual | amber: unassigned | gray: ROI | purple: weak/short",
                (10,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video",type=Path,required=True)
    parser.add_argument("--session",type=Path,required=True)
    parser.add_argument("--lsd",type=Path,required=True)
    parser.add_argument("--frame",type=int,required=True)
    parser.add_argument("--rig",type=Path)
    parser.add_argument("--camera-id",required=True)
    parser.add_argument("--prior",type=Path,help="JSON with T_world_object and size_m")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.prior and not args.rig:
        parser.error("--prior requires --rig")
    rig=read_json(args.rig) if args.rig else None
    calibration=rig["cameras"][args.camera_id] if rig else None
    frame,corners,segments=extract(args.video,args.session,args.lsd,args.frame,calibration,args.camera_id)
    if args.prior:
        apply_prior(corners,segments,args.camera_id,calibration,read_json(args.prior))
    else:
        for row in corners+segments:
            if row["reason"] is None:
                row["assignment_reason"]="awaiting_calibrated_pose"
    report=dict(schema_version=1,video=args.video.name,video_path=str(args.video.resolve()),session=args.session.name,
                camera_id=args.camera_id,source_frame=args.frame,aligned_time_ns=None,
                pixel_space="undistorted_K_pixels" if rig else "raw_distorted_pixels",
                calibration_id=rig["calibration_id"] if rig else None,
                corners=corners,segments=segments)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"observations.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    if not cv2.imwrite(str(args.output/"overlay.jpg"),paint(frame,corners,segments)):
        raise OSError("Overlay write failed")


if __name__=="__main__":
    main()

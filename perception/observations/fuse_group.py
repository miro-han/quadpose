"""Run one verified recorder frame group through the estimated-only four-slot backend."""
import argparse
import json
from pathlib import Path
import numpy as np

from .adapter import read_json, observations
from pose.fusion import Pose, Camera, BoxModel, Detections, analyze_frame, choose_result
from pose.hypotheses import POLICIES as PHASE4_POLICIES, evaluate_hypotheses, select_pose
from pose.contracts import inference_packet

PHASE4_DEFAULT = "cross_verify"
PHASE4_ONLY_METHODS = ("cross", "cross_verify", "cross_alternating")


def run(config, directory):
    rig=read_json(directory/config["rig"])
    alignment=read_json(directory/config["alignment"])
    if alignment["status"]!="passed" or alignment["unverified_timing"]:
        raise ValueError("A flash-verified alignment is required for a measured pose")
    group=next(g for g in alignment["groups"] if g["group_id"]==config["group_id"])
    if not group["valid"]:
        raise ValueError(f"Invalid frame group: {group['reason']}")
    ids=config["camera_order"]
    if len(ids)!=4 or len(set(ids))!=4 or set(ids)!=set(rig["cameras"]):
        raise ValueError("Expected four distinct calibrated slots")
    recording=Path(alignment["recording_dir"])
    session=read_json(recording/"session.json")
    detections=[]
    cameras=[]
    for camera_id in ids:
        c=rig["cameras"][camera_id]
        ext=np.array(c["T_camera_world"])
        cameras.append(Camera(camera_id,np.array(c["K"]),ext[:3,:3],-ext[:3,:3].T@ext[:3,3],tuple(c["image_size_px"])))
        if camera_id not in group["frames"]:
            detections.append(Detections(np.empty((0,2)),np.empty((0,2,2))))
            continue
        r=read_json(directory/config["observations"][camera_id])
        slot=c["slot"]
        stream=alignment["streams"][camera_id]
        if (stream["slot"]!=slot or stream["device_path"]!=c["device_path"]
                or stream["image_size_px"]!=c["image_size_px"]
                or r["pixel_space"]!="undistorted_K_pixels" or r["camera_id"]!=camera_id
                or r["calibration_id"]!=rig["calibration_id"] or r["source_frame"]!=group["frames"][camera_id]["source_frame"]
                or Path(r["video_path"]).resolve()!=(recording/f"camera_{slot}.avi").resolve()
                or session["cameras"][str(slot)]["device"]["path"]!=c["device_path"]):
            raise ValueError(f"Mismatched observation, timestamps or device: {camera_id}")
        detections.append(observations(r["corners"],r["segments"]))
    prior=read_json(directory/config["prior"])
    T=np.array(prior["T_world_object"])
    prior_pose=Pose(T[:3,:3],T[:3,3])
    box=BoxModel(tuple(prior["size_m"]))
    method=config.get("method",PHASE4_DEFAULT)
    evidence=analyze_frame(prior_pose,cameras,detections,box)
    packet=inference_packet(evidence)
    diagnostics=None
    if method in PHASE4_ONLY_METHODS or (method in PHASE4_POLICIES and config.get("backend","phase4")=="phase4"):
        matrix=evaluate_hypotheses(prior_pose,cameras,detections,box)
        selection=select_pose(matrix,policy=method,assignment=config.get("assignment","greedy"))
        valid=bool(selection.valid and selection.pose is not None)
        pose=np.eye(4)
        if valid:
            pose[:3,:3],pose[:3,3]=selection.pose.rotation,selection.pose.translation
        excluded=[ids[i] for i,keep in enumerate(selection.support) if not keep]
        failure_reason=None if valid else selection.reason
        diagnostics={k:(list(v) if isinstance(v,tuple) else v) for k,v in selection.diagnostics.items()}
    else:
        estimate=choose_result(evidence,method)
        valid=bool(estimate.valid)
        pose=np.eye(4)
        pose[:3,:3],pose[:3,3]=estimate.pose.rotation,estimate.pose.translation
        excluded=[ids[i] for i in estimate.excluded]
        failure_reason=None if valid else "insufficient_independent_geometric_support"
    out=dict(session_key=alignment["session_key"],calibration_id=rig["calibration_id"],camera_order=ids,
             group_id=group["group_id"],aligned_time_ns=group["aligned_time_ns"],valid=valid,
             method=method,missing=group.get("missing",{}),
             excluded_camera_ids=excluded,
             failure_reason=failure_reason,
             hypothesis_order=["T123","T124","T134","T234"],
             T_world_object=pose.tolist() if valid else None,
             **{k:v.tolist() for k,v in packet.items()})
    if diagnostics is not None:
        out["diagnostics"]=diagnostics
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=run(read_json(args.config),args.config.resolve().parent)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,allow_nan=False)

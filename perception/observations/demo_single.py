"""Diagnostic single-view example. Its assumed K/size and manual pose are not calibration."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    out=parser.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    K=np.array([[1400.,0,960],[0,1400.,540],[0,0,1]])
    signs=np.array([[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1],[-1,-1,-1],[1,-1,-1]])
    clicks=np.array([[963,515],[1198,515],[940,348],[1161,345],[946,710],[1155,677]],dtype=float)
    size=np.array([.12,.12,.12])
    ok,r,t=cv2.solvePnP(signs*size/2,clicks,K,None)
    if not ok:
        raise RuntimeError("Manual example pose solve failed")
    ext=np.eye(4)
    ext[:3,:3]=cv2.Rodrigues(r)[0]
    ext[:3,3]=t.ravel()
    rig=dict(calibration_id="ILLUSTRATIVE_ONLY",cameras={"camera_1":dict(
        K=K.tolist(),distortion=[0.]*5,T_camera_world=ext.tolist(),image_size_px=[1920,1080])})
    prior=dict(size_m=size.tolist(),T_world_object=np.eye(4).tolist(),manual_pixels=clicks.tolist(),
               notes="Assumed cube dimensions, focal length and zero distortion; six manual clicks on this same frame. IDs and residuals illustrate conditional associations only.")
    for name,value in (("rig",rig),("prior",prior)):
        (out/f"{name}.json").write_text(json.dumps(value,indent=2),encoding="utf-8")
    subprocess.run([sys.executable,str(Path(__file__).with_name("adapter.py")),"--video",
        str(ROOT/"data/sample_videos/WIN_20261124_20_05_34_Pro.mp4"),"--session",
        str(ROOT/"outputs/segmentation/20261124_215508_e1cb28"),"--lsd",
        str(ROOT/"outputs/edges/lsd_baseline"),"--frame","11","--camera-id","camera_1",
        "--rig",str(out/"rig.json"),"--prior",str(out/"prior.json"),"--output",str(out/"frame")],check=True)


if __name__=="__main__":
    main()

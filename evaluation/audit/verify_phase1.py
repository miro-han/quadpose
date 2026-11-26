"""Independent projection, left perturbation and label-intervention checks."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments"/"fusion"))
import fusion_poc as old


def check():
    ids=np.arange(8)
    max_pixel=max_jac=0.
    for frame in (0,17,51):
        pose=old.trajectory(frame)
        for camera in old.cameras("opposite"):
            R=camera.rotation@pose.rotation
            t=camera.rotation@(pose.translation-camera.center)
            K=np.array([[old.FOCAL,0,old.CENTER[0]],[0,old.FOCAL,old.CENTER[1]],[0,0,1]])
            expected=cv2.projectPoints(old.VERTICES,cv2.Rodrigues(R)[0],t,K,None)[0][:,0,:]
            pixels,jac=old.project(pose,camera,ids,jacobian=True)
            max_pixel=max(max_pixel,float(np.max(np.abs(expected-pixels))))
            for i in range(6):
                step=np.eye(6)[i]*1e-6
                plus=old.project(old.increment(pose,step),camera,ids)[0]
                minus=old.project(old.increment(pose,-step),camera,ids)[0]
                max_jac=max(max_jac,float(np.max(np.abs((plus-minus)/2e-6-jac[:,:,i]))))
    assert max_pixel<1e-7 and max_jac<1e-4
    original=old.simulate_frame
    def scrambled(*args):
        return [(d,old.Labels(np.where(l.point_ids>=0,(l.point_ids+1)%8,-1),
                              np.where(l.edge_ids>=0,(l.edge_ids+1)%12,-1)))
                for d,l in original(*args)]
    args=(4,113,"baseline","opposite")
    original_rows=old.run_trial(*args)[0]
    with patch.object(old,"simulate_frame",scrambled):
        modified_rows=old.run_trial(*args)[0]
    equal=oracle_changed=0
    for a,b in zip(original_rows,modified_rows,strict=True):
        keys=("translation_mm","rotation_deg","solved","matches")
        unchanged=all(a[k]==b[k] for k in keys)
        if a["association"]=="estimated":
            assert unchanged
            equal+=1
        else:
            oracle_changed+=int(not unchanged)
    assert oracle_changed>0
    with patch.object(old,"trajectory",side_effect=AssertionError("GT used")):
        old.initial_pose()
    return dict(max_projection_delta_px=max_pixel,max_jacobian_delta=max_jac,
                unchanged_estimated_rows=equal,changed_oracle_rows=oracle_changed)


if __name__=="__main__":
    print(json.dumps(check(),indent=2))

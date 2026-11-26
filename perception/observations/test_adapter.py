import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from perception.observations.adapter import corners_from_segments, transform_pixels, extract, ROOT
from perception.observations.fuse_group import run
from pose.fusion import BoxModel, Camera, Pose, visible


class AdapterTests(unittest.TestCase):
    def test_partial_intersections(self):
        def seg(i,pts):
            return dict(local_id=i,xy=pts,reason=None)
        lines=[seg(1,[[20,10],[80,10]]),seg(2,[[82,14],[82,60]]),seg(3,[[150,11],[150,60]])]
        found=corners_from_segments(lines)
        self.assertEqual(len(found),1)
        np.testing.assert_allclose(found[0]["xy"],[82,10])

    def test_distortion(self):
        K=np.array([[900.,0,600],[0,910,380],[0,0,1.]])
        D=np.array([-.12,.02,.003,-.002,0.])
        points=np.array([[.31,-.12,1],[-.22,.21,1]])
        raw=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,D)[0]
        expected=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,None)[0]
        corrected=transform_pixels(raw,dict(K=K,distortion=D))
        np.testing.assert_allclose(corrected,expected[:,0],atol=1e-5)

    @unittest.skipUnless(
        (ROOT/"data/sample_videos/WIN_20261124_20_05_34_Pro.mp4").is_file()
        and (ROOT/"outputs/segmentation/20261124_215508_e1cb28").is_dir()
        and (ROOT/"outputs/edges/lsd_baseline").is_dir(),
        "Local sample video and generated segmentation/LSD outputs are not present",
    )
    def test_real_frame(self):
        _,corners,segments=extract(ROOT/"data/sample_videos/WIN_20261124_20_05_34_Pro.mp4",
            ROOT/"outputs/segmentation/20261124_215508_e1cb28",ROOT/"outputs/edges/lsd_baseline",11)
        self.assertTrue(corners)
        self.assertTrue(any(s["reason"]=="outside_roi" for s in segments))
        self.assertTrue(all(s["model_id"] is None for s in segments+corners))

    def test_group_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            def save(name,value):
                (directory/name).write_text(json.dumps(value),encoding="utf-8")
            ids=["front","side","top","diagonal"]
            centers=[[0,-1,0],[1,0,0],[0,0,1],[-.8,.8,-.8]]
            box=BoxModel((.19,.12,.085))
            pose=Pose(np.eye(3),np.zeros(3))
            config=dict(rig="rig.json",alignment="alignment.json",prior="prior.json",group_id=0,
                        observations={},camera_order=ids)
            rig=dict(calibration_id="test",cameras={})
            K=np.array([[850.,0,480],[0,850,360],[0,0,1]])
            for i,(name,center) in enumerate(zip(ids,centers),1):
                center=np.array(center,dtype=float)
                forward=-center/np.linalg.norm(center)
                right=np.cross(forward,[0,0,1] if name!="top" else [0,1,0])
                right/=np.linalg.norm(right)
                rotation=np.stack((right,np.cross(forward,right),forward))
                camera=Camera(name,K,rotation,center,(960,720))
                _,points,_,segments=visible(box,pose,camera)
                ext=np.eye(4)
                ext[:3,:3],ext[:3,3]=rotation,-rotation@center
                rig["cameras"][name]=dict(device_path=name,slot=i,K=K.tolist(),T_camera_world=ext.tolist(),image_size_px=[960,720])
                def rows(values):
                    return [dict(xy=v.tolist(),reason=None) for v in values]
                save(name+".json",dict(calibration_id="test",pixel_space="undistorted_K_pixels",camera_id=name,
                     source_frame=0,video_path=str(directory/f"camera_{i}.avi"),corners=rows(points),segments=rows(segments)))
                config["observations"][name]=name+".json"
            save("rig.json",rig)
            save("session.json",dict(cameras={str(i):dict(device=dict(path=name)) for i,name in enumerate(ids,1)}))
            alignment=dict(status="passed",unverified_timing=False,recording_dir=tmp,session_key="synthetic",
                           streams={name:dict(slot=i,device_path=name,image_size_px=[960,720]) for i,name in enumerate(ids,1)},
                           groups=[dict(group_id=0,valid=True,aligned_time_ns=100,frames={name:dict(source_frame=0) for name in ids})])
            save("alignment.json",alignment)
            save("prior.json",dict(size_m=list(box.size_m),T_world_object=np.eye(4).tolist()))
            result=run(config,directory)
            self.assertEqual(result["method"],"cross_verify")
            self.assertTrue(result["valid"])
            self.assertIn("diagnostics",result)
            np.testing.assert_allclose(result["T_world_object"],np.eye(4),atol=1e-5)
            self.assertEqual(np.array(result["subset_membership_mask"]).sum(),12)
            self.assertTrue(np.all(result["residual_valid_mask"]))
            legacy=run({**config,"method":"reliable"},directory)
            self.assertTrue(legacy["valid"])
            np.testing.assert_allclose(legacy["T_world_object"],np.eye(4),atol=1e-5)
            alignment["unverified_timing"]=True
            save("alignment.json",alignment)
            with self.assertRaisesRegex(ValueError,"flash"):
                run(config,directory)


if __name__=="__main__":
    unittest.main()

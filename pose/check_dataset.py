"""Validate paired archives; allow-list model inputs independently of evaluation labels."""
import argparse
from pathlib import Path
import numpy as np

INPUT_FIELDS=("camera_tokens","hypothesis_tokens","cross_view_residual_px","subset_membership_mask",
              "residual_valid_mask","camera_has_matches","hypothesis_valid","view_support")


def inputs_only(path):
    with np.load(path,allow_pickle=False) as data:
        if any(name.startswith("target_") for name in data):
            raise ValueError("Targets found in inference archive")
        return {name:data["packet_"+name] for name in INPUT_FIELDS}


def check(directory):
    inputs=inputs_only(directory/"tokens.npz")
    with np.load(directory/"tokens.npz",allow_pickle=False) as features, np.load(directory/"targets.npz",allow_pickle=False) as truth:
        keys=features["frame_key"]
        if len(np.unique(keys))!=len(keys) or not np.array_equal(keys,truth["frame_key"]):
            raise ValueError("Expected unique, identically ordered frame keys")
        if any(k.startswith("packet_") for k in truth):
            raise ValueError("Inference fields unexpectedly duplicated into label archive")
        n=len(keys)
        for k,v in inputs.items():
            assert len(v)==n and np.isfinite(v).all(),k
        membership=inputs["subset_membership_mask"]
        np.testing.assert_array_equal(membership,np.broadcast_to(np.eye(4,dtype=bool)[::-1]==False,(n,4,4)))
        valid=inputs["residual_valid_mask"]
        assert not np.any(valid & ~inputs["hypothesis_valid"][:,:,None])
        assert not np.any(inputs["view_support"] & ~valid)
        assert truth["target_packet_hypothesis_error"].shape==(n,4,2)
        assert truth["target_reliable_mask"].shape==(n,4)
        for trial in np.unique(features["trial_key"]):
            assert len(np.unique(features["split"][features["trial_key"]==trial]))==1
        return n


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    print("Validated frames:",check(parser.parse_args().directory))

"""A sparse Lucas–Kanade mask baseline with explicit tracking failures."""

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Observation:
    mask: np.ndarray | None
    reason: str
    matches: int = 0
    inliers: int = 0


class MaskTracker(Protocol):
    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> Observation: ...

    def update(self, frame: np.ndarray) -> Observation: ...


class LKMaskTracker:
    """Tracks a nearly planar foreground from a known initial mask."""

    def __init__(self):
        self.gray = None
        self.mask = None

    def initialize(self, frame, mask):
        if frame.shape[:2] != mask.shape or not np.any(mask):
            raise ValueError('Initial mask must be nonempty and match the frame')
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mask = np.asarray(mask, dtype=np.uint8).copy()
        return Observation(self.mask.astype(bool), 'initial_mask')

    def lose(self, reason, matches=0, inliers=0):
        self.gray = self.mask = None
        return Observation(None, reason, matches, inliers)

    def update(self, frame):
        if self.mask is None:
            return Observation(None, 'needs_reinitialization')
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if current.shape != self.gray.shape:
            return self.lose('frame_shape_changed')
        points = cv2.goodFeaturesToTrack(self.gray, maxCorners=120, qualityLevel=.006,
                                         minDistance=6, mask=self.mask, blockSize=7)
        if points is None or len(points) < 8:
            return self.lose('insufficient_features')

        forward, statuses, _ = cv2.calcOpticalFlowPyrLK(self.gray, current, points, None)
        if forward is None:
            return self.lose('forward_flow_failed')
        keep = statuses.ravel().astype(bool) & np.isfinite(forward).all(axis=(1, 2))
        source, target = points[keep], forward[keep]
        if len(source) < 8:
            return self.lose('insufficient_forward_tracks', len(source))
        backward, statuses, _ = cv2.calcOpticalFlowPyrLK(current, self.gray, target, None)
        if backward is None:
            return self.lose('backward_flow_failed')
        keep = statuses.ravel().astype(bool) & (np.linalg.norm(source - backward, axis=2).ravel() <= 1.5)
        source, target = source[keep, 0], target[keep, 0]
        if len(source) < 8:
            return self.lose('insufficient_consistent_tracks', len(source))

        homography, consensus = cv2.findHomography(source, target, cv2.RANSAC, 2.5)
        if homography is None or consensus is None or not np.isfinite(homography).all():
            return self.lose('invalid_homography', len(source))
        inlier_flags = consensus.ravel().astype(bool)
        inliers = int(inlier_flags.sum())
        if inliers < 8 or inliers / len(source) < .65:
            return self.lose('insufficient_homography_support', len(source), inliers)
        previous_area = cv2.countNonZero(self.mask)
        hull_area = cv2.contourArea(cv2.convexHull(source[inlier_flags]))
        if hull_area < .03 * previous_area:
            return self.lose('concentrated_keypoints', len(source), inliers)
        size = (current.shape[1], current.shape[0])
        warped = cv2.warpPerspective(self.mask, homography, size, flags=cv2.INTER_NEAREST)
        if not .65 <= cv2.countNonZero(warped) / previous_area <= 1.5:
            return self.lose('implausible_mask_area', len(source), inliers)
        self.gray = current
        self.mask = warped
        return Observation(warped.astype(bool), 'tracked', len(source), inliers)

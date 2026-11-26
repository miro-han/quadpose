"""Run the recovery benchmark with OpenCV's CPU thread count explicitly pinned."""

import cv2

from evaluation.pose_benchmark.benchmark import main


if __name__ == '__main__':
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads() != 1 or cv2.ocl.useOpenCL():
        raise RuntimeError('Recovery benchmark requires one OpenCV CPU thread and no OpenCL')
    main()

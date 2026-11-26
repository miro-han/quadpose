"""Paired OpenCV CPU/AMD-UMat timings on consecutive real HD frames."""
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import json
import os
from pathlib import Path
import statistics
import time
import uuid

from evaluation.resource_limits import limit_process
from experiments.amd.dxgi_identity import EngineCounter, dxgi_adapters, method

ROOT=Path(__file__).resolve().parents[3]


def process_video_memory():
    """IDXGIAdapter3::QueryVideoMemoryInfo for this process, no D3D device."""
    class Info(ct.Structure):
        _fields_=[(n,ct.c_uint64) for n in ("budget","usage","reservation_available","reservation")]
    def iid(value):
        return (ct.c_ubyte*16).from_buffer_copy(uuid.UUID(value).bytes_le)
    factory=ct.c_void_p()
    create=ct.WinDLL("dxgi").CreateDXGIFactory
    create.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p)]
    if create(iid("7b7166ec-21c7-44ae-b21a-c9ae321ae369"),ct.byref(factory)):
        raise RuntimeError("DXGI factory unavailable")
    result={}
    try:
        for desc in dxgi_adapters():
            adapter=ct.c_void_p()
            if method(factory,7,wt.LONG,wt.UINT,ct.POINTER(ct.c_void_p))(factory,desc["index"],ct.byref(adapter)):
                raise RuntimeError("Adapter unavailable")
            newer=ct.c_void_p()
            try:
                hr=method(adapter,0,wt.LONG,ct.c_void_p,ct.POINTER(ct.c_void_p))(
                    adapter,iid("645967a4-1392-4310-a798-8053ce3e93fd"),ct.byref(newer))
                if hr:
                    raise RuntimeError("IDXGIAdapter3 unavailable")
                values=[]
                for segment in (0,1):
                    info=Info()
                    hr=method(newer,14,wt.LONG,wt.UINT,ct.c_int,ct.c_void_p)(newer,0,segment,ct.byref(info))
                    if hr:
                        raise RuntimeError("QueryVideoMemoryInfo failed")
                    values.append(info.usage)
                result[desc["vendor"]]=values
            finally:
                if newer.value: method(newer,2,wt.ULONG)(newer)
                method(adapter,2,wt.ULONG)(adapter)
    finally:
        method(factory,2,wt.ULONG)(factory)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--memory-gib",type=float,default=4)
    p.add_argument("--repeats",type=int,default=15)
    args=p.parse_args()
    if args.output.exists() or args.repeats<5:
        p.error("Use a new output path and at least five repeats")
    limit_process(args.threads,args.memory_gib)
    drivers=list((Path(os.environ["WINDIR"])/"System32/DriverStore/FileRepository").glob("amdocl.inf*/amdocl64.dll"))
    if len(drivers)!=1:
        raise RuntimeError(f"Require exactly one AMD OpenCL runtime, got {drivers}")
    os.environ["OPENCV_OPENCL_RUNTIME"]=str(drivers[0])
    os.environ["OPENCV_OPENCL_DEVICE"]="AMD:GPU:gfx1035"
    os.environ["OPENCV_OPENCL_CACHE_ENABLE"]="0"
    mem={"start":process_video_memory()}
    import cv2
    import numpy as np
    import psutil
    cv2.setNumThreads(args.threads)
    cv2.ocl.setUseOpenCL(True)
    if not cv2.ocl.useOpenCL():
        raise RuntimeError("The explicit AMD OpenCL device is unavailable")
    device=cv2.ocl.Device_getDefault()
    if device.name()!="gfx1035" or not device.isAMD():
        raise RuntimeError("Unexpected OpenCL device")
    cap=cv2.VideoCapture(str(ROOT/"data/sample_videos/WIN_20261124_20_05_34_Pro.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,11)
    frames=[]
    for _ in range(2):
        ok,f=cap.read()
        if not ok: raise ValueError("Could not decode benchmark clip")
        frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
    cap.release()
    a,b=frames
    points=cv2.goodFeaturesToTrack(a,300,.01,7)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    ops={"blur":lambda a,b,p:cv2.GaussianBlur(a,(5,5),1.2),
         "canny":lambda a,b,p:cv2.Canny(a,15,45,L2gradient=True),
         "morph_close":lambda a,b,p:cv2.morphologyEx(a,cv2.MORPH_CLOSE,kernel),
         "pyr_lk":lambda a,b,p:cv2.calcOpticalFlowPyrLK(a,b,p,None),
         "chain":lambda a,b,p:cv2.morphologyEx(cv2.Canny(cv2.GaussianBlur(a,(5,5),1.2),15,45,
                                                     L2gradient=True),cv2.MORPH_CLOSE,kernel)}
    def as_host(data):
        return [v.get() if isinstance(v,cv2.UMat) else v for v in (data if isinstance(data,tuple) else (data,))]
    results={}
    engine_error=None
    try: counter=EngineCounter()
    except RuntimeError as exc: counter=None; engine_error=str(exc)
    for name,op in ops.items():
        cv2.ocl.setUseOpenCL(False)
        ref=as_host(op(a,b,points))
        row={}
        for mode in ("cpu","amd_resident","amd_roundtrip"):
            gpu=mode!="cpu"
            cv2.ocl.setUseOpenCL(gpu)
            if gpu and not cv2.ocl.useOpenCL(): raise RuntimeError("Lost AMD device")
            def upload():
                if name=="pyr_lk": return cv2.UMat(a),cv2.UMat(b),cv2.UMat(points)
                return cv2.UMat(a),None,None
            args_in=upload() if mode=="amd_resident" else (a,b,points)
            def once():
                output=op(*(upload() if mode=="amd_roundtrip" else args_in))
                if mode=="amd_roundtrip": output=tuple(as_host(output))
                if gpu: cv2.ocl.finish()
                return output
            for _ in range(3): once()
            ms=[]
            if counter: counter.sample()
            start=time.perf_counter()
            while len(ms)<args.repeats or time.perf_counter()-start<.8:
                t=time.perf_counter_ns()
                output=once()
                ms.append((time.perf_counter_ns()-t)/1e6)
            engines=counter.sample() if counter else {}
            data=as_host(output)
            metrics={}
            if name=="pyr_lk":
                common=(data[1][:,0]>0)&(ref[1][:,0]>0)
                error=np.linalg.norm(data[0][common]-ref[0][common],axis=2)
                metrics=dict(common_valid=int(common.sum()),status_agreement=float(np.mean(data[1]==ref[1])),
                             median_endpoint_delta_px=float(np.median(error)) if error.size else None)
            else:
                metrics=dict(pixel_agreement=float(np.mean(data[0]==ref[0])),
                             mean_abs_difference=float(np.mean(np.abs(data[0].astype(float)-ref[0]))))
            row[mode]=dict(repeats=len(ms),p50_ms=statistics.median(ms),p95_ms=float(np.percentile(ms,95)),
                           process_gpu_engines_pct=engines,**metrics)
            mem[name+"_"+mode]=process_video_memory()
        results[name]=row
    report=dict(pid=os.getpid(),cv=cv2.__version__,runtime=str(drivers[0]),opencl_device=device.name(),
                vendor=device.vendorName(),driver=device.driverVersion(),adapters=dxgi_adapters(),
                threads=args.threads,process_memory_cap_gib=args.memory_gib,
                peak_working_set_mib=psutil.Process().memory_info().peak_wset/2**20,
                process_local_nonlocal_bytes=mem,engine_counter_issue=engine_error,results=results,
                shape=a.shape,lk_points=len(points))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf8") as f: json.dump(report,f,indent=2,allow_nan=False)
    if counter: counter.close(counter.query)
    print(device.name(),report["peak_working_set_mib"],engine_error)


if __name__=="__main__":
    main()

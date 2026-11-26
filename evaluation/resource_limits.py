"""Apply Windows process commit cap before importing OpenCV, NumPy or torch."""
import ctypes as ct
from ctypes import wintypes as wt
import os

# Keep handle live until exit. This job applies only to the calling process.
_job=None


def limit_process(threads=1, memory_gib=3):
    global _job
    if not 1<=threads<=16 or not 0.25<=memory_gib<=24:
        raise ValueError("Invalid thread count or process memory budget")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
        os.environ[key]=str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    if os.name!="nt":
        import resource
        size=int(memory_gib*2**30)
        resource.setrlimit(resource.RLIMIT_AS,(size,size))
        return
    class Basic(ct.Structure):
        _fields_=[("per_process",ct.c_longlong),("per_job",ct.c_longlong),("flags",wt.DWORD),
                  ("min_ws",ct.c_size_t),("max_ws",ct.c_size_t),("active",wt.DWORD),
                  ("affinity",ct.c_size_t),("priority",wt.DWORD),("scheduling",wt.DWORD)]
    class Limits(ct.Structure):
        _fields_=[("basic",Basic),("io_counts",ct.c_ulonglong*6),("process_bytes",ct.c_size_t),
                  ("job_bytes",ct.c_size_t),("peak_process",ct.c_size_t),("peak_job",ct.c_size_t)]
    kernel=ct.WinDLL("kernel32",use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[ct.c_void_p,wt.LPCWSTR]
    kernel.CreateJobObjectW.restype=wt.HANDLE
    kernel.SetInformationJobObject.argtypes=[wt.HANDLE,ct.c_int,ct.c_void_p,wt.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[wt.HANDLE,wt.HANDLE]
    kernel.GetCurrentProcess.restype=wt.HANDLE
    _job=kernel.CreateJobObjectW(None,None)
    settings=Limits()
    settings.basic.flags=0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    settings.process_bytes=int(memory_gib*2**30)
    if not (_job and kernel.SetInformationJobObject(_job,9,ct.byref(settings),ct.sizeof(settings))
            and kernel.AssignProcessToJobObject(_job,kernel.GetCurrentProcess())):
        raise ct.WinError(ct.get_last_error())


if __name__=="__main__":
    import argparse
    import runpy
    import sys
    p=argparse.ArgumentParser(description="Run a CPU-only Python module under a per-process commit cap")
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--memory-gib",type=float,default=3)
    p.add_argument("module")
    p.add_argument("args",nargs=argparse.REMAINDER)
    args=p.parse_args()
    limit_process(args.threads,args.memory_gib)
    os.environ["OPENCV_OPENCL_RUNTIME"]="disabled"
    sys.argv=[args.module,*args.args]
    runpy.run_module(args.module,run_name="__main__",alter_sys=True)

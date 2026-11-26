"""Minimal bindings to the installed AMD vendor runtime, bypassing the ICD loader."""

import ctypes as ct
import os
from pathlib import Path


P, U, I, S, L = ct.c_void_p, ct.c_uint, ct.c_int, ct.c_size_t, ct.c_uint64


def check(status, operation):
    if status != 0:
        raise RuntimeError(f"AMD OpenCL {operation} failed ({status})")


class AMD:
    def __init__(self):
        paths = list((Path(os.environ["WINDIR"]) / "System32/DriverStore/FileRepository").glob("amdocl.inf*/amdocl64.dll"))
        if len(paths) != 1:
            raise RuntimeError("Expected exactly one AMD OpenCL vendor runtime")
        self.path = paths[0]
        self.library = ct.WinDLL(str(self.path))
        self._bind("GetPlatformIDs", I, U, P, P)
        self._bind("GetPlatformInfo", I, P, U, S, P, P)
        self._bind("GetDeviceIDs", I, P, L, U, P, P)
        self._bind("GetDeviceInfo", I, P, U, S, P, P)
        self._bind("CreateContext", P, P, U, P, P, P, P)
        self._bind("CreateCommandQueue", P, P, P, L, P)
        self._bind("CreateProgramWithSource", P, P, U, P, P, P)
        self._bind("BuildProgram", I, P, U, P, ct.c_char_p, P, P)
        self._bind("GetProgramBuildInfo", I, P, P, U, S, P, P)
        self._bind("CreateKernel", P, P, ct.c_char_p, P)
        self._bind("CreateBuffer", P, P, L, S, P, P)
        self._bind("SetKernelArg", I, P, U, S, P)
        self._bind("EnqueueNDRangeKernel", I, P, P, U, P, P, P, U, P, P)
        self._bind("EnqueueReadBuffer", I, P, P, U, S, S, P, U, P, P)
        self._bind("EnqueueWriteBuffer", I, P, P, U, S, S, P, U, P, P)
        self._bind("Finish", I, P)
        for name in ("Context", "CommandQueue", "Program", "Kernel", "MemObject"):
            self._bind(f"Release{name}", I, P)
        self.release = []
        self.platform = self._single(self.GetPlatformIDs)
        platform_vendor = self._string(self.GetPlatformInfo, self.platform, 0x0903)
        if platform_vendor != "Advanced Micro Devices, Inc.":
            raise RuntimeError(f"Unexpected OpenCL platform: {platform_vendor}")
        self.device = self._single(self.GetDeviceIDs, self.platform, 4)
        vendor_id = U()
        check(self.GetDeviceInfo(self.device, 0x1001, ct.sizeof(vendor_id), ct.byref(vendor_id), None), "vendor ID")
        self.name = self._string(self.GetDeviceInfo, self.device, 0x102B)
        self.vendor = self._string(self.GetDeviceInfo, self.device, 0x102C)
        self.driver = self._string(self.GetDeviceInfo, self.device, 0x102D)
        if vendor_id.value != 0x1002 or self.name != "gfx1035":
            raise RuntimeError(f"Unexpected AMD GPU: {self.name} / 0x{vendor_id.value:x}")
        device = (P * 1)(self.device)
        self.context = self._create("Context", self.CreateContext, None, 1, device, None, None)
        self.queue = self._create("CommandQueue", self.CreateCommandQueue, self.context, self.device, 0)

    def _bind(self, name, result, *arguments):
        function = getattr(self.library, f"cl{name}")
        function.restype = result
        function.argtypes = arguments
        setattr(self, name, function)

    @staticmethod
    def _single(query, *args):
        count, objects = U(), (P * 1)()
        check(query(*args, 0, None, ct.byref(count)), query.__name__)
        if count.value != 1:
            raise RuntimeError(f"Expected one AMD object, got {count.value}")
        check(query(*args, 1, objects, None), query.__name__)
        return objects[0]

    @staticmethod
    def _string(query, *args):
        size = S()
        check(query(*args, 0, None, ct.byref(size)), query.__name__)
        data = ct.create_string_buffer(size.value)
        check(query(*args, len(data), data, None), query.__name__)
        return data.value.decode("utf-8", "replace")

    def _create(self, kind, call, *args):
        status = I()
        handle = call(*args, ct.byref(status))
        check(status.value, f"create {kind}")
        self.release.append((getattr(self, f"Release{kind}"), handle))
        return handle

    def program(self, source, options):
        encoded = (ct.c_char_p * 1)(source)
        program = self._create("Program", self.CreateProgramWithSource, self.context, 1, encoded, None)
        devices = (P * 1)(self.device)
        status = self.BuildProgram(program, 1, devices, options.encode(), None, None)
        if status:
            raise RuntimeError(self._string(self.GetProgramBuildInfo, program, self.device, 0x1183))
        return program

    def kernel(self, program, name):
        return self._create("Kernel", self.CreateKernel, program, name.encode())

    def buffer(self, array):
        return self._create("MemObject", self.CreateBuffer, self.context, 1 | 32, array.nbytes, array.ctypes.data)

    def read(self, buffer, array):
        check(self.EnqueueReadBuffer(self.queue, buffer, 1, 0, array.nbytes, array.ctypes.data, 0, None, None), "read")
        return array

    def write(self, buffer, array):
        check(self.EnqueueWriteBuffer(self.queue, buffer, 1, 0, array.nbytes, array.ctypes.data, 0, None, None), "write")

    def set_args(self, kernel, *args):
        for index, argument in enumerate(args):
            check(self.SetKernelArg(kernel, index, ct.sizeof(argument), ct.byref(argument)), "kernel arguments")

    def dispatch(self, kernel, count):
        size = S(count)
        check(self.EnqueueNDRangeKernel(self.queue, kernel, 1, None, ct.byref(size), None, 0, None, None), "dispatch")

    def finish(self):
        check(self.Finish(self.queue), "finish")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for release, handle in reversed(self.release):
            release(handle)
        self.release.clear()

"""Enumerate DXGI adapter metadata and read WDDM process counters only."""

import ctypes as c
import json
import os
import re
import time
import uuid
from collections import defaultdict
from ctypes import wintypes as w


class AdapterDesc(c.Structure):
    _fields_ = [('description', w.WCHAR * 128), ('vendor', w.UINT), ('device', w.UINT),
                ('subsystem', w.UINT), ('revision', w.UINT), ('dedicated_video', c.c_size_t),
                ('dedicated_system', c.c_size_t), ('shared_system', c.c_size_t),
                ('luid_low', w.DWORD), ('luid_high', w.LONG)]


def method(pointer, index, result, *arguments):
    table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
    return c.WINFUNCTYPE(result, c.c_void_p, *arguments)(table[index])


def dxgi_adapters():
    """Use EnumAdapters, the same ordering used by the ORT DML device_id option."""
    create = c.WinDLL('dxgi', use_last_error=True).CreateDXGIFactory
    create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
    create.restype = w.LONG
    factory = c.c_void_p()
    interface = (c.c_ubyte * 16).from_buffer_copy(uuid.UUID('7b7166ec-21c7-44ae-b21a-c9ae321ae369').bytes_le)
    if create(interface, c.byref(factory)) != 0:
        raise RuntimeError('CreateDXGIFactory failed')
    result = []
    try:
        enum = method(factory, 7, w.LONG, w.UINT, c.POINTER(c.c_void_p))
        for index in range(32):
            adapter = c.c_void_p()
            status = enum(factory, index, c.byref(adapter)) & 0xffffffff
            if status == 0x887A0002:
                return result
            if status:
                raise RuntimeError(f'EnumAdapters returned 0x{status:08x}')
            try:
                info = AdapterDesc()
                if method(adapter, 8, w.LONG, c.POINTER(AdapterDesc))(adapter, c.byref(info)) != 0:
                    raise RuntimeError('IDXGIAdapter::GetDesc failed')
                result.append({'index': index, 'description': info.description,
                               'vendor': f'0x{info.vendor:04x}', 'device': f'0x{info.device:04x}',
                               'luid': f'0x{info.luid_high & 0xffffffff:08x}_0x{info.luid_low:08x}',
                               'dedicated_video_mib': info.dedicated_video / 2**20,
                               'shared_system_mib': info.shared_system / 2**20})
            finally:
                method(adapter, 2, w.ULONG)(adapter)
    finally:
        method(factory, 2, w.ULONG)(factory)
    raise RuntimeError('Too many DXGI adapters')


def amd_adapter():
    adapters = dxgi_adapters()
    matches = [adapter for adapter in adapters if adapter['vendor'] == '0x1002'
               and 'radeon' in adapter['description'].casefold()]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one AMD Radeon adapter: {adapters}')
    return matches[0]


class PdhValue(c.Structure):
    _fields_ = [('status', w.DWORD), ('value', c.c_double)]


class PdhItem(c.Structure):
    _fields_ = [('name', w.LPWSTR), ('formatted', PdhValue)]


GPU_ENGINE_NAME = re.compile(r'pid_(\d+)_luid_(0x[\da-f]+_0x[\da-f]+)_.*_engtype_(.+)', re.I)


def typed(library, name, *args):
    function = getattr(library, name)
    function.argtypes = args
    function.restype = w.LONG
    return function


def check(value):
    if value:
        raise RuntimeError(f'PDH status 0x{value & 0xffffffff:08x}')


class EngineCounter:
    """Only queries utilization; it never creates a D3D device or GPU workload."""
    def __init__(self, pid=None):
        self.pid = os.getpid() if pid is None else pid
        pdh = c.WinDLL('pdh')
        self.query = c.c_void_p()
        self.counter = c.c_void_p()
        open_query = typed(pdh, 'PdhOpenQueryW', w.LPCWSTR, c.c_size_t, c.POINTER(c.c_void_p))
        add = typed(pdh, 'PdhAddEnglishCounterW', c.c_void_p, w.LPCWSTR, c.c_size_t, c.POINTER(c.c_void_p))
        self.collect = typed(pdh, 'PdhCollectQueryData', c.c_void_p)
        self.read_array = typed(pdh, 'PdhGetFormattedCounterArrayW', c.c_void_p, w.DWORD,
                                c.POINTER(w.DWORD), c.POINTER(w.DWORD), c.c_void_p)
        self.close = typed(pdh, 'PdhCloseQuery', c.c_void_p)
        check(open_query(None, 0, c.byref(self.query)))
        try:
            check(add(self.query, r'\GPU Engine(*)\Utilization Percentage', 0, c.byref(self.counter)))
            check(self.collect(self.query))
        except RuntimeError:
            self.close(self.query)
            raise

    def sample(self):
        check(self.collect(self.query))
        size, count = w.DWORD(), w.DWORD()
        status = self.read_array(self.counter, 0x200, c.byref(size), c.byref(count), None)
        if status & 0xffffffff != 0x800007D2:
            check(status)
        if not size.value:
            return {}
        buffer = c.create_string_buffer(size.value)
        check(self.read_array(self.counter, 0x200, c.byref(size), c.byref(count), buffer))
        matches = defaultdict(dict)
        array = c.cast(buffer, c.POINTER(PdhItem))
        for i in range(count.value):
            item = array[i]
            match = GPU_ENGINE_NAME.fullmatch(item.name)
            if match and int(match[1]) == self.pid and item.formatted.status in (0, 1):
                name = match[3].casefold()
                luid = match[2].lower()
                matches[luid][name] = max(matches[luid].get(name, 0.), item.formatted.value)
        return dict(matches)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close(self.query)


if __name__ == '__main__':
    print(json.dumps(dxgi_adapters(), indent=2))
    try:
        with EngineCounter() as counter:
            time.sleep(.2)
            print(f'Current process WDDM engines: {counter.sample()}')
    except RuntimeError as exc:
        print(f'WDDM counters unavailable: {exc}')

"""Fail-closed AMD selection and per-process WDDM verification."""

from collections import Counter
import ctypes as c
from importlib.metadata import distribution
from pathlib import Path
import json
import os
import subprocess
from threading import Event, Thread
from time import perf_counter

from evaluation.resource_limits import limit_process

from .dxgi_identity import EngineCounter, amd_adapter, dxgi_adapters
from .opencl.measure import process_video_memory

limit_process(threads=2, memory_gib=4)

import numpy as np


# The experiment interpreter uses a CPU-only torch wheel and this AMD-only guard.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'


def nvidia_snapshot():
    def query(*flags):
        return subprocess.run(['nvidia-smi', *flags, '--format=csv,noheader,nounits'],
                              text=True, capture_output=True, timeout=5, check=True).stdout.strip()

    info = query('--query-gpu=memory.used,memory.total,utilization.gpu')
    contexts = query('--query-compute-apps=pid')
    pids = {int(line.strip()) for line in contexts.splitlines() if line.strip().isdigit()}
    return {'system_memory_and_utilization': info, 'experiment_pid_listed': os.getpid() in pids}


class HardwareWatch:
    def __init__(self, amd):
        self.amd = amd
        self.stop = Event()
        self.forbidden = Event()
        self.peaks = {}
        self.errors = []
        self.samples = 0
        self.memory_peaks = {}
        self.guard_trigger = None
        self.nvidia_luids = {entry['luid'] for entry in dxgi_adapters() if entry['vendor'] == '0x10de'}

    def __enter__(self):
        self.before = nvidia_snapshot()
        if self.before['experiment_pid_listed']:
            raise RuntimeError('Experiment PID was already listed on NVIDIA')
        self.sample_memory()
        self.require_safe()
        self.counter = EngineCounter()
        self.thread = Thread(target=self.sample_loop, name='WDDM read-only monitor', daemon=True)
        self.thread.start()
        return self

    def sample_loop(self):
        while not self.stop.wait(.2):
            try:
                reading = self.counter.sample()
                self.samples += 1
                for luid, engines in reading.items():
                    for engine, utilization in engines.items():
                        key = f'{luid}/{engine}'
                        self.peaks[key] = max(self.peaks.get(key, 0.), utilization)
                    if luid in self.nvidia_luids:
                        self.guard_trigger = {'kind': 'NVIDIA_engine_instance', 'luid': luid, 'engines': engines}
                        self.forbidden.set()
                self.sample_memory()
            except RuntimeError as exc:
                self.errors.append(str(exc))

    def require_safe(self):
        if self.forbidden.is_set():
            raise RuntimeError(f'The experiment PID was reported on NVIDIA: {self.guard_trigger}')

    def sample_memory(self):
        usage = process_video_memory()
        for vendor, pair in usage.items():
            previous = self.memory_peaks.get(vendor, [0, 0])
            self.memory_peaks[vendor] = [max(a, b) for a, b in zip(previous, pair, strict=True)]
        if any(usage.get('0x10de', ())):
            self.guard_trigger = {'kind': 'NVIDIA_process_memory', 'local_nonlocal_bytes': usage['0x10de']}
            self.forbidden.set()

    def verify_window(self, run):
        """Exercise the actual workload long enough to obtain process/adapter samples."""
        deadline = perf_counter() + 1.5
        while perf_counter() < deadline:
            self.require_safe()
            run()
        self.require_safe()

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.counter.__exit__()
        self.sample_memory()
        self.after = nvidia_snapshot()
        self.require_safe()
        if self.after['experiment_pid_listed']:
            raise RuntimeError('Experiment PID was listed on NVIDIA after the run')

    def summary(self):
        amd_prefix = self.amd['luid'] + '/'
        return {'pid': os.getpid(), 'pre_nvidia': self.before, 'post_nvidia': self.after,
                'wddm_samples': self.samples, 'wddm_process_peak_percent': self.peaks,
                'dxgi_process_peak_local_and_nonlocal_bytes': self.memory_peaks,
                'wddm_errors': sorted(set(self.errors)),
                'amd_positive_engine_sample': any(key.startswith(amd_prefix) and value > 0
                                                  for key, value in self.peaks.items())}


def torch_cpu():
    import torch

    if torch.version.cuda is not None:
        raise RuntimeError('Use the isolated CPU-PyTorch DirectML environment')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return torch


def compare(left, right):
    pairs = list(zip(left, right, strict=True))
    if any(a.shape != b.shape for a, b in pairs):
        raise RuntimeError('Output shapes differ')
    if any(not np.isfinite(value).all() for pair in pairs for value in pair):
        raise RuntimeError('Nonfinite output')
    return {'max_abs': max(float(np.max(np.abs(a - b))) for a, b in pairs),
            'rmse': float(np.sqrt(np.mean([np.mean(np.square(a - b)) for a, b in pairs])))}


class Tensor(c.Structure):
    _fields_ = [('name', c.c_char_p), ('data', c.POINTER(c.c_float)), ('shape', c.POINTER(c.c_int64)),
                ('rank', c.c_size_t), ('elements', c.c_size_t)]


class ExplicitDMLSession:
    def __init__(self, model, amd, directory, strict):
        dll = Path(__file__).parent / '.venv' / 'explicit_dml.dll'
        if not dll.is_file():
            raise RuntimeError('Build the AMD-only DirectML bridge with build_native.cmd')
        bridge = self.bridge = c.CDLL(str(dll))
        bridge.dml_error.restype = c.c_char_p
        bridge.dml_open.argtypes = [c.c_wchar_p, c.c_wchar_p, c.c_char_p, c.c_size_t, c.c_uint, c.c_uint,
                                    c.c_uint32, c.c_int32, c.c_wchar_p, c.c_int]
        bridge.dml_open.restype = c.c_void_p
        bridge.dml_run.argtypes = [c.c_void_p, c.POINTER(Tensor), c.c_size_t, c.POINTER(Tensor), c.c_size_t]
        bridge.dml_run.restype = c.c_int
        bridge.dml_stop_profile.argtypes = [c.c_void_p]
        bridge.dml_stop_profile.restype = c.c_char_p
        bridge.dml_close.argtypes = [c.c_void_p]
        bridge.dml_close.restype = None
        package = distribution('onnxruntime-directml')
        ort_path = package.locate_file('onnxruntime/capi/onnxruntime.dll')
        dml_path = package.locate_file('onnxruntime/capi/DirectML.dll')
        high, low = (int(part, 16) for part in amd['luid'].split('_'))
        prefix = str(Path(directory).resolve() / 'dml')
        self.handle = bridge.dml_open(str(ort_path), str(dml_path), model, len(model), amd['index'],
                                      int(amd['device'], 16), low, high, prefix, int(strict))
        if not self.handle:
            raise RuntimeError(bridge.dml_error().decode('utf-8'))

    def bind(self, input_values, output_names, compare_outputs):
        def desc(name, value):
            if value.dtype != np.float32 or not value.flags.c_contiguous:
                raise ValueError('Native bridge accepts contiguous FP32 tensors')
            return Tensor(name.encode(), value.ctypes.data_as(c.POINTER(c.c_float)),
                          (c.c_int64 * value.ndim)(*value.shape), value.ndim, value.size)

        inputs = (Tensor * len(input_values))(*(desc(name, value) for name, value in input_values.items()))
        outputs = [np.empty_like(reference) for reference in compare_outputs]
        targets = (Tensor * len(outputs))(*(desc(name, value) for name, value in zip(output_names, outputs)))

        def run():
            if self.bridge.dml_run(self.handle, inputs, len(inputs), targets, len(targets)):
                raise RuntimeError(self.bridge.dml_error().decode('utf-8'))
            return outputs

        return run

    def stop_profiling(self):
        result = self.bridge.dml_stop_profile(self.handle)
        if not result:
            raise RuntimeError(self.bridge.dml_error().decode('utf-8'))
        return Path(result.decode('utf-8'))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.bridge.dml_close(self.handle)


def ort_inference(model, sample, amd, directory, watch, *, strict=True, repeat=20, verify=True):
    import onnxruntime as ort

    def time_run(run):
        for _ in range(3):
            output = run()
        durations = []
        for _ in range(repeat):
            watch.require_safe()
            start = perf_counter()
            output = run()
            durations.append((perf_counter() - start)*1000)
        return output, durations

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    cpu = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
    cpu.disable_fallback()
    expected, cpu_ms = time_run(lambda: cpu.run(None, sample))
    current_amd = amd_adapter()
    if current_amd != amd:
        raise RuntimeError('DXGI AMD identity changed between enumeration and ORT session')
    watch.require_safe()
    try:
        with ExplicitDMLSession(model, amd, directory, strict) as session:
            watch.sample_memory()
            watch.require_safe()
            bindings = session.bind(sample, [output.name for output in cpu.get_outputs()], expected)
            bindings()
            trace = session.stop_profiling()
            events = json.loads(trace.read_text(encoding='utf-8'))
            executed = Counter((entry['args']['provider'], entry['args'].get('op_name', '?'))
                               for entry in events if entry.get('cat') == 'Node'
                               and entry.get('args', {}).get('provider'))
            provider_counts = Counter()
            for (provider, _), count in executed.items():
                provider_counts[provider] += count
            if not provider_counts['DmlExecutionProvider'] or (strict and len(provider_counts) != 1):
                raise RuntimeError(f'Unexpected provider dispatch: {dict(provider_counts)}')
            actual, dml_ms = time_run(bindings)
            if verify:
                watch.verify_window(bindings)
        return {'status': 'ok', 'strict': strict, 'bridge': 'DML1 explicit D3D12 device and queue',
                'requested_dxgi_index': amd['index'], 'requested_adapter_luid': amd['luid'],
                'provider_node_events': dict(provider_counts),
                'executed_ops': {f'{provider}/{op}': count for (provider, op), count in sorted(executed.items())},
                'cpu_ms': cpu_ms, 'dml_ms': dml_ms, 'cpu_comparison': compare(expected, actual)}
    except (RuntimeError, ValueError) as exc:
        watch.require_safe()
        return {'status': 'directml_failed', 'strict': strict, 'requested_dxgi_index': amd['index'],
                'reason': str(exc)[:4000], 'cpu_ms': cpu_ms}

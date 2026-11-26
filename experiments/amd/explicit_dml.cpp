// ONNX Runtime session for an explicitly verified AMD D3D12 adapter.
#define ORT_API_MANUAL_INIT
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
thread_local std::string error;
struct Session {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDMLDevice> dml;
    Ort::Env environment{nullptr};
    Ort::Session session{nullptr};
    std::string profile_path;
};

void check(HRESULT status, const char* step) {
    if (FAILED(status)) {
        throw std::runtime_error(std::string(step) + " failed with HRESULT " + std::to_string(status));
    }
}

HMODULE library(const wchar_t* path) {
    auto module = LoadLibraryExW(path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) throw std::runtime_error("Could not load a package DLL by absolute path");
    return module; // Keep package DLLs resident for the test process lifetime.
}

template<typename T>
T symbol(HMODULE module, const char* name) {
    auto address = GetProcAddress(module, name);
    if (!address) throw std::runtime_error(std::string("Missing package export: ") + name);
    return reinterpret_cast<T>(address);
}
}

struct Tensor {
    const char* name;
    float* data;
    const int64_t* shape;
    size_t rank;
    size_t elements;
};

extern "C" __declspec(dllexport) const char* dml_error() { return error.c_str(); }

extern "C" __declspec(dllexport) void* dml_open(
    const wchar_t* ort_path, const wchar_t* dml_path, const void* model, size_t model_bytes,
    UINT index, UINT expected_device, DWORD luid_low, LONG luid_high,
    const wchar_t* profile_prefix, int strict) {
    try {
        ComPtr<IDXGIFactory1> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> adapter;
        check(factory->EnumAdapters1(index, &adapter), "EnumAdapters1 selected index");
        DXGI_ADAPTER_DESC1 info{};
        check(adapter->GetDesc1(&info), "GetDesc1");
        if (info.VendorId != 0x1002 || info.DeviceId != expected_device ||
            info.AdapterLuid.LowPart != luid_low || info.AdapterLuid.HighPart != luid_high ||
            (info.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            throw std::runtime_error("DXGI index, vendor, device and LUID failed AMD identity check");
        }
        auto result = std::make_unique<Session>();
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&result->device)),
              "AMD D3D12CreateDevice");
        auto actual = result->device->GetAdapterLuid();
        if (actual.LowPart != luid_low || actual.HighPart != luid_high) {
            throw std::runtime_error("D3D12 device LUID differs from the AMD adapter");
        }
        D3D12_COMMAND_QUEUE_DESC queue{};
        queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(result->device->CreateCommandQueue(&queue, IID_PPV_ARGS(&result->queue)), "AMD CreateCommandQueue");

        auto create_dml = symbol<decltype(&DMLCreateDevice)>(library(dml_path), "DMLCreateDevice");
        check(create_dml(result->device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&result->dml)),
              "DMLCreateDevice on AMD device");
        auto api_base = symbol<decltype(&OrtGetApiBase)>(library(ort_path), "OrtGetApiBase")();
        auto api = api_base->GetApi(ORT_API_VERSION);
        if (!api) throw std::runtime_error("ONNX Runtime ABI version mismatch");
        Ort::InitApi(api);
        const void* ep_api = nullptr;
        Ort::ThrowOnError(api->GetExecutionProviderApi("DML", ORT_API_VERSION, &ep_api));
        const auto* dml_api = static_cast<const OrtDmlApi*>(ep_api);
        result->environment = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "amd-explicit-dml");
        Ort::SessionOptions options;
        options.DisableMemPattern();
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetIntraOpNumThreads(2);
        options.SetInterOpNumThreads(1);
        options.EnableProfiling(profile_prefix);
        if (strict) options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        Ort::ThrowOnError(dml_api->SessionOptionsAppendExecutionProvider_DML1(
            options, result->dml.Get(), result->queue.Get()));
        result->session = Ort::Session(result->environment, model, model_bytes, options);
        return result.release();
    } catch (const std::exception& failure) {
        error = failure.what();
        return nullptr;
    }
}

extern "C" __declspec(dllexport) int dml_run(
    void* handle, const Tensor* inputs, size_t num_inputs, const Tensor* outputs, size_t num_outputs) {
    try {
        auto& session = static_cast<Session*>(handle)->session;
        std::vector<Ort::Value> input_values;
        std::vector<const char*> input_names, output_names;
        auto allocator = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        for (size_t i = 0; i < num_inputs; ++i) {
            const auto& input = inputs[i];
            input_names.push_back(input.name);
            input_values.push_back(Ort::Value::CreateTensor<float>(
                allocator, input.data, input.elements, input.shape, input.rank));
        }
        for (size_t i = 0; i < num_outputs; ++i) output_names.push_back(outputs[i].name);
        auto result = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                                  input_values.size(), output_names.data(), output_names.size());
        for (size_t i = 0; i < num_outputs; ++i) {
            auto info = result[i].GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                result[i].GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU ||
                shape.size() != outputs[i].rank ||
                !std::equal(shape.begin(), shape.end(), outputs[i].shape) ||
                info.GetElementCount() != outputs[i].elements) {
                throw std::runtime_error("Unexpected DML output shape or non-FP32 dtype");
            }
            std::copy_n(result[i].GetTensorData<float>(), outputs[i].elements, outputs[i].data);
        }
        return 0;
    } catch (const std::exception& failure) {
        error = failure.what();
        return -1;
    }
}

extern "C" __declspec(dllexport) const char* dml_stop_profile(void* handle) {
    try {
        auto owner = static_cast<Session*>(handle);
        Ort::AllocatorWithDefaultOptions allocator;
        owner->profile_path = owner->session.EndProfilingAllocated(allocator).get();
        return owner->profile_path.c_str();
    } catch (const std::exception& failure) {
        error = failure.what();
        return nullptr;
    }
}

extern "C" __declspec(dllexport) void dml_close(void* handle) { delete static_cast<Session*>(handle); }

"""Compare small four-view fusion models on CPU and the explicitly chosen AMD iGPU."""

import argparse
import copy
import io
import json
from pathlib import Path
import statistics
import tempfile
from time import perf_counter
from .amd_runtime import HardwareWatch, amd_adapter, ort_inference, torch_cpu


def make_models(torch):
    nn = torch.nn

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(16, 48), nn.ReLU(), nn.Linear(48, 48), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 6))

        def forward(self, views):
            return self.rho(self.phi(views).mean(dim=1))

    class CrossAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, 1, 32) * .1)
            self.keys = nn.Linear(16, 32)
            self.values = nn.Linear(16, 32)
            self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 6))

        def forward(self, views):
            query = self.query.expand(views.size(0), 1, 32)
            weights = (query @ self.keys(views).transpose(-2, -1) * (32 ** -.5)).softmax(dim=-1)
            return self.head((weights @ self.values(views)).squeeze(1))

    return {'deep_sets': DeepSets(), 'cross_attention': CrossAttention()}


def train_model(torch, original, views, expected, watch):
    model = copy.deepcopy(original)
    optimizer = torch.optim.SGD(model.parameters(), lr=.08, foreach=False)

    def step():
        watch.require_safe()
        optimizer.zero_grad(set_to_none=True)
        error = (model(views) - expected).square().mean()
        error.backward()
        optimizer.step()
        return float(error.detach().cpu())

    losses, durations = [], []
    for _ in range(3):
        step()
    for _ in range(30):
        start = perf_counter()
        losses.append(step())
        durations.append((perf_counter() - start) * 1000)
    updates = sum(float((param.detach().cpu() - start).abs().max()) for param, start
                  in zip(model.parameters(), original.parameters(), strict=True))
    if not (losses[-1] < losses[0] and updates > 0):
        raise RuntimeError(f'Training did not update and improve: {losses[0]} -> {losses[-1]}')
    return {'step_ms': durations, 'median_ms': statistics.median(durations),
            'first_mse': losses[0], 'last_mse': losses[-1], 'parameter_delta_sum_max': updates}


def training(torch, models, views, target, watch):
    return {name: {'cpu': train_model(torch, model, views, target, watch),
                   'amd_dml': {'status': 'stopped_at_native_preflight', 'workload_started': False}}
            for name, model in models.items()}


def inference(torch, models, views, amd, watch, directory):
    result = {}
    for name, model in models.items():
        model.eval()
        result[name] = {}
        for batch in (1, len(views)):
            sample = views[:batch]
            buffer = io.BytesIO()
            torch.onnx.export(model, (sample,), buffer, input_names=['views'], output_names=['pose'],
                              opset_version=17)
            feed = {'views': sample.numpy()}
            strict = ort_inference(buffer.getvalue(), feed, amd, directory, watch, repeat=30)
            result[name][str(batch)] = {'strict': strict}
            if strict['status'] != 'ok':
                result[name][str(batch)]['mixed_fallback_diagnostic'] = ort_inference(
                    buffer.getvalue(), feed, amd, directory, watch, strict=False, repeat=30)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'inference'))
    args = parser.parse_args()
    root = Path(__file__).parent / 'results'
    root.mkdir(exist_ok=True)
    amd = amd_adapter()
    torch = torch_cpu()
    torch.manual_seed(492)
    models = make_models(torch)
    views = torch.randn(32, 4, 16)
    target = views.mean(dim=1)[:, :6] * .4
    with HardwareWatch(amd) as watch, tempfile.TemporaryDirectory(dir=root, prefix='fusion-') as directory:
        if args.mode == 'train':
            output = training(torch, models, views, target, watch)
            api_identity = {'torch_directml_imported': False, 'training_device': 'cpu'}
        else:
            output = inference(torch, models, views, amd, watch, directory)
            api_identity = {'explicit_dxgi_index': amd['index'], 'provider_api': 'DML1'}
    report = {'amd_adapter': amd, 'torch': torch.__version__, 'torch_cuda_version': torch.version.cuda,
              'model_shapes': {'batch': 32, 'views': 4, 'inputs_per_view': 16, 'regression_outputs': 6},
              'telemetry': watch.summary(), 'api_identity': api_identity, 'models': output}
    path = root / f'fusion_{args.mode}.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {path}', flush=True)


if __name__ == '__main__':
    main()

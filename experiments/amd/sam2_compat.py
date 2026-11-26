"""Check the full SAM video path on CPU and checkpoint subgraphs on explicit AMD DML1."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch
import warnings

from .amd_runtime import HardwareWatch, amd_adapter, ort_inference, torch_cpu


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'models' / 'sam2.1_hiera_tiny.pt'
SOURCE = ROOT / 'third_party' / 'sam2'
CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'
FRAMES = ROOT / 'outputs/segmentation/20261124_215925_8f211c/input_frames'
BOX = [831.5766738660907, 272.9157667386609, 1237.451403887689, 706.7818574514039]


def normalize(torch, load, index, side):
    image, height, width = load(str(FRAMES / f'{index:06d}.jpg'), side)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (image.float() - mean) / std, height, width


def video_cpu(torch, model, load):
    images = [normalize(torch, load, index, model.image_size) for index in range(2)]
    frames = torch.stack([item[0] for item in images])
    _, height, width = images[0]

    def preloaded(**_):
        return frames, height, width

    started = perf_counter()
    with torch.inference_mode(), patch('sam2.sam2_video_predictor.load_video_frames', new=preloaded):
        state = model.init_state('preloaded', offload_video_to_cpu=True, offload_state_to_cpu=True)
        model.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=BOX)
        mask_areas = [int((scores[0, 0] > 0).sum()) for _, _, scores in model.propagate_in_video(state)]
    if len(mask_areas) != 2 or not all(mask_areas):
        raise RuntimeError(f'Unexpected whole video CPU mask areas: {mask_areas}')
    return {'status': 'two_nonempty_masks', 'mask_areas_pixels': mask_areas,
            'init_prompt_and_propagate_ms': (perf_counter() - started) * 1000}


def subgraphs(torch, model, load):
    class Encoder(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, image):
            return tuple(self.predictor.forward_image(image)['backbone_fpn'])

    class Decoder(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, main, fine, middle, position, sparse, dense):
            scores, _, _, _ = self.decoder(main, position, sparse, dense, multimask_output=False,
                                           repeat_image=False, high_res_features=[fine, middle])
            return scores

    small, _, _ = normalize(torch, load, 0, 256)
    yield 'image_backbone_256', Encoder(model), (small.unsqueeze(0),), ['image'], ['fine', 'mid', 'coarse']

    image, height, width = normalize(torch, load, 0, model.image_size)
    coord = torch.tensor(BOX).reshape(1, 2, 2)
    coord /= torch.tensor([width, height])
    coord *= model.image_size
    labels = torch.tensor([[2, 3]], dtype=torch.int32)
    with torch.no_grad():
        fine, middle, coarse = model.forward_image(image.unsqueeze(0))['backbone_fpn']
        position = model.sam_prompt_encoder.get_dense_pe()
        sparse, dense = model.sam_prompt_encoder(points=(coord, labels), boxes=None, masks=None)
    yield ('mask_decoder_1024_features', Decoder(model.sam_mask_decoder),
           (coarse, fine, middle, position, sparse, dense),
           ['main', 'fine', 'middle', 'position', 'sparse', 'dense'], ['mask_logits'])


def check_component(torch, name, module, inputs, names, outputs, amd, directory, watch):
    result = {'input_shapes': {key: list(value.shape) for key, value in zip(names, inputs)}}
    buffer = io.BytesIO()
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with torch.no_grad():
                torch.onnx.export(module.eval(), inputs, buffer, input_names=names, output_names=outputs,
                                  opset_version=17)
            if captured:
                result['export_warnings'] = sorted({str(item.message).splitlines()[0] for item in captured})
        payload = buffer.getvalue()
        feed = {key: value.detach().contiguous().numpy() for key, value in zip(names, inputs)}
        result['onnx_bytes'] = len(payload)
        result['strict'] = ort_inference(payload, feed, amd, directory, watch, repeat=3)
        if result['strict']['status'] != 'ok':
            result['mixed_fallback_diagnostic'] = ort_inference(payload, feed, amd, directory, watch,
                                                                 strict=False, repeat=3)
    except Exception as exc:
        watch.require_safe()
        result.update(status='partial_export_or_runtime_failed', error=f'{type(exc).__name__}: {exc}'[:3500])
    return name, result


def main():
    torch = torch_cpu()
    amd = amd_adapter()
    sys.path.insert(0, str(SOURCE))  # Read bundled code without installing the incompatible package requirements.
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.misc import _load_img_as_tensor

    root = Path(__file__).parent / 'results'
    root.mkdir(exist_ok=True)
    with HardwareWatch(amd) as watch, tempfile.TemporaryDirectory(dir=root, prefix='sam-components-') as folder:
        model = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device='cpu', apply_postprocessing=False)
        with contextlib.redirect_stderr(sys.stdout):
            whole = video_cpu(torch, model, _load_img_as_tensor)
        parts = dict(check_component(torch, *component, amd, folder, watch)
                     for component in subgraphs(torch, model, _load_img_as_tensor))
    with CHECKPOINT.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    values = {'amd_adapter': amd, 'telemetry': watch.summary(), 'checkpoint_sha256': sha,
              'torch': torch.__version__, 'source_commit': (SOURCE.parent / 'sam2_commit.txt').read_text().strip(),
              'sam_package_minimum': {'torch': '2.5.1', 'torchvision': '0.20.1'},
              'torch_directml_wheel_requires': {'torch': '2.4.1', 'torchvision': '0.19.1'},
              'video_cpu': whole, 'video_torch_dml': 'stopped_at_torch_directml_preflight',
              'video_onnx_dml': 'not_exported_stateful_video', 'static_onnx_components': parts}
    path = root / 'sam2_compat.json'
    path.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {path}', flush=True)


if __name__ == '__main__':
    main()

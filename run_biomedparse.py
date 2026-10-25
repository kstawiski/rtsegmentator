"""One supported BiomedParse v2 target, using the pinned authors' 3D pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

from biomedparse import (SEED, WINDOWS, binding, require_device, result_record, validate_prompt)


def preprocess(image, modality, window=None):
    """README v2: CT W/L, otherwise 0.5/99.5 percentiles, then [0,255]."""
    import numpy as np
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or not np.isfinite(image).all() or min(image.shape) < 1:
        raise ValueError('BiomedParse needs a finite scalar 3D image')
    if image.max() == image.min():
        raise ValueError('BiomedParse cannot normalize a constant image')
    if modality == 'CT':
        window = window or 'soft_tissue'
        width, level = WINDOWS[window]
        lower, upper = level - width / 2, level + width / 2
        method = {'method': 'CT_window', 'window': window, 'width': width, 'level': level}
    else:
        lower, upper = np.percentile(image, [0.5, 99.5])
        method = {'method': 'percentile', 'percentiles': [0.5, 99.5]}
    image = np.clip(image, lower, upper)
    # CT uses the specified W/L endpoints; other modalities use clipped min/max.
    if upper <= lower:
        raise ValueError('BiomedParse cannot normalize an image with collapsed percentile bounds')
    image = (image - lower) / (upper - lower) * 255
    # Truncate positive values. See docs/BIOMEDPARSE.md for the source evidence
    # and the unresolved conversion provenance of the shipped v2 examples.
    return image.astype(np.uint8), {**method, 'bounds': [float(lower), float(upper)]}


def load_model(source, checkpoint, tokenizer):
    import torch
    import hydra
    import ftfy  # Required upstream CLIP text cleanup; forbid the optional fallback.
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    sys.path.insert(0, str(Path(source).resolve()))
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(Path(source).resolve() / 'configs/model'),
                                     job_name='example_prediction', version_base=None):
        cfg = compose(config_name='biomedparse_3D')
    cfg.sem_seg_head.predictor.language_encoder.tokenizer.pretrained_model_name_or_path = str(Path(tokenizer).resolve())
    model = hydra.utils.instantiate(cfg, _convert_='object')
    model.load_pretrained(str(checkpoint))
    return model.to(torch.device('cuda:0')).eval()


def infer(model, image, text):
    """Keep the README/notebook operations, ordering, thresholds and batch size."""
    import torch
    import torch.nn.functional as F
    from utils import process_input, process_output
    from inference import postprocess, merge_multiclass_masks
    imgs, pad_width, padded_size, valid_axis = process_input(image, 512)
    imgs = imgs.to('cuda:0').int()
    with torch.no_grad():
        output = model({'image': imgs.unsqueeze(0), 'text': [text]}, mode='eval', slice_batch_size=4)
    predictions = output['predictions']
    masks = F.interpolate(predictions['pred_gmasks'], size=(512, 512), mode='bicubic',
                          align_corners=False, antialias=True)
    masks = postprocess(masks, predictions['object_existence'])
    masks = merge_multiclass_masks(masks, [1])
    masks = process_output(masks, pad_width, padded_size, valid_axis)
    scores = predictions['object_existence'].sigmoid()[0].detach().cpu().tolist()
    return masks, scores, valid_axis


def seed_inference():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path)
    group.add_argument('--npz', type=Path, help='Upstream preprocessed public reference example')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sidecar', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--modality', choices=['CT', 'MR', 'PT'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--bundle-root', type=Path, required=True)
    a = p.parse_args()
    raw = json.loads(a.prompt.read_text())
    prompt = validate_prompt(raw.get('text'), a.modality, raw.get('ct_window'))
    # No cached hash is used for execution. Verify before loading pickled weights.
    binding(a.bundle_root)
    from biomedparse import SOURCE, CHECKPOINT, TOKENIZER
    if any(given.resolve() != (a.bundle_root / relative).resolve() for given, relative in
           [(a.source, SOURCE), (a.checkpoint, CHECKPOINT), (a.tokenizer, TOKENIZER)]):
        raise ValueError('BiomedParse runtime paths must match the verified bundle')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import nibabel as nib
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('BiomedParse v2 requires a GPU acceptance run; CPU inference is disabled')
    props = torch.cuda.get_device_properties(0)
    require_device({'compute_capability': list(torch.cuda.get_device_capability(0)), 'total_memory': props.total_memory})
    if a.input:
        reference = nib.load(a.input)
        image, preprocessing = preprocess(np.asarray(reference.dataobj).transpose(2, 1, 0), a.modality, prompt['ct_window'])
    else:
        with np.load(a.npz, allow_pickle=False) as data:
            image = data['imgs']
        if image.ndim != 3 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
            raise ValueError('The upstream NPZ example must already be 3D in [0,255]')
        preprocessing = {'method': 'upstream_preprocessed_npz'}
    torch.cuda.reset_peak_memory_stats()
    seed_inference()
    model = load_model(a.source, a.checkpoint, a.tokenizer)
    masks, scores, axis = infer(model, image, prompt['text'])
    if a.input:
        nib.save(nib.Nifti1Image(masks.transpose(2, 1, 0).astype(np.uint8), reference.affine, reference.header), a.output)
    else:
        np.savez_compressed(a.output, segs=masks)
    record = result_record(prompt, scores, int(np.count_nonzero(masks)), slice_axis=axis, preprocessing=preprocessing)
    record['gpu_peak_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
    record['gpu_peak_memory_reserved_bytes'] = torch.cuda.max_memory_reserved()
    a.sidecar.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'message': record['message'], 'voxel_count': record['voxel_count']}))


if __name__ == '__main__':
    main()

"""Offline NV-Reason-CT subprocess; invoked only with its isolated runtime."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import warnings

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pydicom

from nv_reason_ct_report import (TASK, REVISION, REGION_ORGANS, detect_cbct, hu_check, padding_values_hu,
    mask_stats, covered_regions, check_consistency, thumbnail, rtstruct_summary, write_report, wait_for_free_memory)


def _stream_sha256(stream):
    # hashlib.file_digest needs Python 3.11; the portal image runs 3.10.
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


class Refusal(Exception):
    """Expected input/resource refusal; never fail a segmentation job."""


def wait_for_gpu(torch, timeout, required_gib=14):
    def probe():
        return torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
    free = wait_for_free_memory(probe, timeout, required_gib)
    if free is None:
        raise Refusal(f'Insufficient free GPU memory: at least {required_gib} GiB and CUDA required after bounded wait')
    return free


def region_mask_paths(masks, region):
    return [masks / (name + '.nii.gz') for name in REGION_ORGANS[region]
            if (masks / (name + '.nii.gz')).is_file()]


def mask_geometry(paths, crop_affine=None, crop_shape=None):
    """Count native mask voxels inside the crop in world coordinates; avoid resampling bias."""
    count = inside = 0.0; lower = np.full(3, np.inf); upper = np.full(3, -np.inf)
    inv = np.linalg.inv(crop_affine) if crop_affine is not None else None
    for path in paths:
        img = nib.load(str(path)); data = np.asarray(img.dataobj) > 0
        volume = abs(np.linalg.det(img.affine[:3, :3]))
        for z in range(data.shape[2]):
            xy = np.argwhere(data[:, :, z])
            if not len(xy): continue
            ijk = np.column_stack([xy, np.full(len(xy), z)])
            world = nib.affines.apply_affine(img.affine, ijk)
            lower = np.minimum(lower, world.min(axis=0)); upper = np.maximum(upper, world.max(axis=0))
            count += len(xy) * volume
            if inv is not None:
                vox = nib.affines.apply_affine(inv, world)
                inside += np.all((vox >= -0.5) & (vox < np.asarray(crop_shape) - 0.5), axis=1).sum() * volume
    if not count: raise RuntimeError('Covered region has no organ-mask voxels')
    return {'organ_fraction_inside': float(inside / count) if inv is not None else None,
            'bounds_ras_mm': [lower.tolist(), upper.tolist()]}


def crop_qc(volume, affine, shape, paths, fallback):
    return {'shape_zyx': list(volume.shape),
            'frac_padding_exact_minus1000': round(float(np.isclose(volume, -1000).mean()), 6),
            'padding_measure': 'Exact -1000 HU fraction; includes native air at -1000 HU',
            'frac_body_gt_minus500': round(float((volume > -500).mean()), 6),
            'frac_lung_minus950_minus600': round(float(((volume > -950) & (volume < -600)).mean()), 6),
            'fallback_crop': fallback, **mask_geometry(paths, affine, shape)}


def prepare_crop(processor, source, masks, region, work, prompt):
    """Retain the publisher crop affine before its tensor transpose; verify actual inputs.

    Guided crops are exactly 192³ at 2 mm in LPS, centred on the world-space organ
    bounding box. On this grid the publisher's padding and centered crop are no-ops.
    """
    loader = processor.image_processor_3d
    transforms = loader.get_transforms().transforms
    names = [type(t).__name__ for t in transforms]
    if names != ['LoadImaged', 'EnsureTyped', 'Orientationd', 'Spacingd', 'SpatialPadd',
                 'AnatomySpatialCropd', 'Transposed', 'ToTensord']:
        raise RuntimeError('Publisher preprocessing changed; crop geometry must be re-audited')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        d = {'image': str(source), 'anatomy_region': region}
        for transform in transforms[:6]: d = transform(d)
    image = d['image']; affine = image.affine.detach().cpu().numpy()
    expected = image[0].detach().cpu().numpy().transpose(2, 1, 0).copy()
    fallback = any('could not determine' in str(w.message) for w in caught)
    paths = region_mask_paths(masks, region)
    initial = crop_qc(expected, affine, image.shape[1:], paths, fallback)
    crop_path = 'publisher_anatomy'; selected = source; anatomy = region
    if fallback or initial['organ_fraction_inside'] < .9:
        bounds = np.asarray(initial['bounds_ras_mm']); center = bounds.mean(axis=0)
        affine = np.diag([-2., -2., 2., 1.])
        affine[:3, 3] = center - affine[:3, :3] @ np.full(3, 95.5)
        img = resample_from_to(nib.load(str(source)), ((192, 192, 192), affine),
                               order=1, mode='constant', cval=-1000.)
        selected = work / (region + '-guided.nii.gz'); nib.save(img, str(selected))
        expected = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 1, 0).copy()
        crop_path = 'segmentation_guided'; anatomy = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        inputs = processor(text=prompt, images3d=[str(selected)], anatomy_region=anatomy, return_tensors='pt')
    actual = inputs['pixel_values'].detach().cpu().numpy()[0, 0]
    normalized = np.clip(expected, -1000, 1000) / 1000
    error = float(np.max(np.abs(actual - normalized)))
    matches = bool(np.allclose(actual, normalized, rtol=0, atol=1e-5))
    if not matches: raise RuntimeError(f'Model input differs from expected crop (max error {error})')
    final = crop_qc(expected, affine, expected.shape[::-1], paths,
                   any('could not determine' in str(w.message) for w in caught))
    final.update(tensor_matches_expected=matches, tensor_max_absolute_error=error,
                 tolerance=1e-5, normalization='clip HU to [-1000,1000] / 1000',
                 affine_ras_mm=affine.tolist())
    # Both raw HU and model-normalized thumbnail planes have identical soft-tissue values.
    return inputs, {'crop_path': crop_path, 'crop_qc': {'initial': initial, 'final': final},
                    'thumbnail_png': thumbnail(actual * 1000)}


def headers_and_summary(job, report):
    headers = []
    for path in sorted((job / 'input').iterdir()):
        try: ds = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError: continue
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'): headers.append(ds)
    if not headers: raise Refusal('No DICOM images in the selected series')
    report['cbct_check'] = next((check for ds in headers if (check := detect_cbct(ds))['is_cbct']), detect_cbct(headers[0]))
    ds = headers[0]
    report['inputs'] = {'modality': str(getattr(ds, 'Modality', '')), 'slices': len(headers),
                        'rows': int(ds.Rows), 'columns': int(ds.Columns)}
    # Padding identified by the headers: None if no header identifies it; inconsistent headers authorise none.
    paddings = {None if (v := padding_values_hu(x)) is None else tuple(v) for x in headers}
    report['inputs']['padding_values_hu'] = (None if paddings == {None} else
                                             list(paddings.pop()) if len(paddings) == 1 else [])
    orientation = np.asarray(getattr(ds, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0]), dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    positions = sorted({float(np.dot(np.asarray(x.ImagePositionPatient, dtype=float), normal))
                        for x in headers if hasattr(x, 'ImagePositionPatient')})
    spacing = float(np.median(np.diff(positions))) if len(positions) > 1 else float(getattr(ds, 'SliceThickness', 1))
    report['segmentation_summary'] = [rtstruct_summary(p, spacing, positions, normal)
        for p in sorted((job / 'output').glob('*_RTSTRUCT.dcm'))]
    if any(str(getattr(x, 'Modality', '')) != 'CT' for x in headers):
        raise Refusal('Only diagnostic CT is supported')
    if report['cbct_check']['is_cbct']: raise Refusal('CBCT: diagnostic CT is required')
    if any(str(getattr(x, 'RescaleType', 'HU')).upper() not in ('HU', '') for x in headers):
        raise Refusal('Non-HU DICOM rescale units')


def coverage_segmentation(source, masks, work, worker_python, mode='fast'):
    if mode not in {'fast', 'full'}: raise ValueError('Coverage mode must be fast or full')
    worker_python = Path(worker_python)
    seg_env = os.environ.copy()
    original_home = Path(seg_env.get('TOTALSEG_HOME_DIR', Path.home() / '.totalsegmentator'))
    weights = Path(seg_env.get('TOTALSEG_WEIGHTS_PATH', original_home / 'nnunet/results'))
    # Refuse missing local checkpoints before TotalSegmentator can try a download.
    required = ('Dataset297_',) if mode == 'fast' else ('Dataset291_', 'Dataset292_', 'Dataset293_', 'Dataset294_', 'Dataset295_')
    if any(not any(weights.glob(prefix + '*')) for prefix in required):
        raise RuntimeError(f'{mode} TotalSegmentator total weights are missing; configure TOTALSEG_WEIGHTS_PATH')
    seg_home = work / 'totalsegmentator'; seg_home.mkdir(exist_ok=True)
    (seg_home / 'config.json').write_text(json.dumps({'totalseg_id': 'rtseg-report',
        'send_usage_stats': False, 'prediction_counter': 0, 'statistics_disclaimer_shown': True}))
    seg_env.update(TOTALSEG_HOME_DIR=str(seg_home), TOTALSEG_WEIGHTS_PATH=str(weights), PYTHONDONTWRITEBYTECODE='1')
    version = subprocess.run([str(worker_python), '-c',
        'import importlib.metadata; print(importlib.metadata.version("TotalSegmentator"))'],
        check=True, capture_output=True, text=True, env=seg_env).stdout.strip()
    binary = worker_python.parent / 'TotalSegmentator'
    executable = str(binary) if binary.is_file() else shutil.which('TotalSegmentator') or str(binary)
    command = [executable, '-i', str(source), '-o', str(masks), '--task', 'total',
               '--device', 'gpu', '--nr_thr_saving', '1']
    if mode == 'fast': command.append('--fast')
    subprocess.run(command, check=True, env=seg_env)
    coverage = {}
    for path in sorted(masks.glob('*.nii.gz')):
        mask = nib.as_closest_canonical(nib.load(str(path)))
        coverage[path.name.removesuffix('.nii.gz')] = mask_stats(
            np.asarray(mask.dataobj) > 0, abs(np.linalg.det(mask.affine[:3, :3])))
    return coverage, version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--worker-python', type=Path, required=True)
    ap.add_argument('--coverage-mode', choices=['fast', 'full'],
                    default=os.environ.get('RTSEG_NV_REASON_CT_COVERAGE_MODE', 'fast'))
    ap.add_argument('--gpu-wait-seconds', type=float, default=120)
    args = ap.parse_args()
    if not 0 <= args.gpu_wait_seconds <= 600: ap.error('GPU wait must be between 0 and 600 seconds')
    if args.coverage_mode not in {'fast', 'full'}: ap.error('Coverage mode must be fast or full')
    started = time.perf_counter(); job = args.job; work = job / 'external' / TASK
    work.mkdir(parents=True, exist_ok=True)
    # Set before importing torch/transformers. No Hub lookup is allowed.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    report = {'schema_version': 1, 'status': 'not_generated', 'reason': '', 'inputs': {},
              'model': 'nvidia/NV-Reason-CT', 'revision': REVISION, 'sha256': {},
              'package_versions': {'python': platform.python_version()}, 'cbct_check': {},
              'coverage': {}, 'covered_regions': [], 'regions': [], 'flags': [],
              'segmentation_summary': [], 'timings': {}, 'peak_vram_gib': 0,
              'peak_vram_scope': 'NV-Reason-CT process allocations; coverage subprocess excluded',
              'coverage_segmentation': {'task': 'total', 'fast': args.coverage_mode == 'fast', 'roi_subset': False},
              'decoding': {'enable_thinking': True, 'do_sample': False, 'max_new_tokens': 4096, 'use_cache': True}}
    model = torch = None
    try:
        for package in ('torch', 'transformers', 'monai', 'nibabel', 'numpy', 'pydicom', 'Pillow',
                        'dynamic-network-architectures', 'flash-linear-attention'):
            report['package_versions'][package] = importlib.metadata.version(package)
        for name in ('model.safetensors', 'model.py', 'processor.py'):
            with (args.model_dir / name).open('rb') as f: report['sha256'][name] = _stream_sha256(f)
        headers_and_summary(job, report)
        nifti = work / 'input.nii.gz'; masks = work / 'coverage'
        step = time.perf_counter()
        subprocess.run([str(args.worker_python), str(Path(__file__).parent / 'dicom_series_to_nifti.py'),
                        str(job / 'input'), str(nifti)], check=True)
        img = nib.load(str(nifti)); data = np.asarray(img.dataobj, dtype=np.float32)
        report['inputs'].update(shape=list(img.shape), spacing_mm=list(map(float, img.header.get_zooms()[:3])),
                                hu_check=hu_check(data, padding_values=report['inputs']['padding_values_hu']))
        del data
        report['timings']['convert_s'] = round(time.perf_counter() - step, 3)
        if not report['inputs']['hu_check']['looks_like_hu']: raise Refusal('Non-HU or unsupported intensity distribution')
        import torch as torch_module
        torch = torch_module
        report['free_vram_before_segmentation_gib'] = wait_for_gpu(torch, args.gpu_wait_seconds)
        if torch.cuda.get_device_capability()[0] < 8:
            raise Refusal('NV-Reason-CT requires a CUDA sm_80-class or newer GPU')
        step = time.perf_counter()
        report['coverage'], version = coverage_segmentation(nifti, masks, work, args.worker_python, args.coverage_mode)
        report['package_versions']['TotalSegmentator'] = version
        if not report['coverage']: raise RuntimeError('TotalSegmentator created no coverage masks')
        report['coverage_slice_convention'] = 'Zero based, inferior to superior in closest-canonical mask grid'
        report['timings']['coverage_s'] = round(time.perf_counter() - step, 3)
        report['covered_regions'] = covered_regions(report['coverage'])
        if not report['covered_regions']: raise Refusal('No supported region in the scanned volume by segmentation coverage rules')
        report['free_vram_before_model_gib'] = wait_for_gpu(torch, args.gpu_wait_seconds)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        step = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForImageTextToText.from_pretrained(str(args.model_dir), local_files_only=True,
            trust_remote_code=True, dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
        processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True, local_files_only=True)
        report['timings']['load_s'] = round(time.perf_counter() - step, 3)
        report['vram_after_load_gib'] = round(torch.cuda.memory_allocated() / 2**30, 3)
        for region in report['covered_regions']:
            text = f'Write a structured {region} CT report.'
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            step = time.perf_counter()
            inputs, crop = prepare_crop(processor, nifti, masks, region, work, prompt)
            preprocess_s = time.perf_counter() - step
            inputs = inputs.to(model.device)
            free_gib = wait_for_gpu(torch, args.gpu_wait_seconds, required_gib=4)
            step = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=4096, do_sample=False, use_cache=True)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - step
            tokens = output[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            record = {'region': region, 'prompt': text, 'raw_text': raw, **crop,
                      **check_consistency(raw, report['coverage'], region),
                      'preprocess_s': round(preprocess_s, 3), 'generate_s': round(elapsed, 3),
                      'free_vram_before_generation_gib': free_gib,
                      'new_tokens': int(tokens.shape[1]), 'input_tokens': int(inputs.input_ids.shape[1]),
                      'finished_before_limit': bool(tokens.shape[1] < 4096 or int(tokens[0, -1]) in eos),
                      'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3)}
            if record['crop_qc']['final']['organ_fraction_inside'] < .9:
                record['flags'].append({'status': 'crop_incomplete', 'reason': '384 mm cube contains less than 90% of regional organ volume'})
            if not record['finished_before_limit']: record['flags'].append({'status': 'generation_truncated'})
            report['regions'].append(record)
            report['flags'].extend({'region': region, **f} for f in record['flags'])
            del inputs, output, tokens
        report['status'] = 'generated'
    except Refusal as exc:
        report['reason'] = str(exc)
    finally:
        if torch is not None and torch.cuda.is_available():
            report['peak_vram_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            del model
            gc.collect(); torch.cuda.empty_cache()
        shutil.rmtree(work)
    report['timings']['total_s'] = round(time.perf_counter() - started, 3)
    write_report(job / 'output' / 'report', report)
    print(json.dumps({'report_status': report['status'], 'reason': report['reason'],
                      'regions': report['covered_regions'], 'timings': report['timings'],
                      'peak_vram_gib': report['peak_vram_gib']}), flush=True)


if __name__ == '__main__': main()

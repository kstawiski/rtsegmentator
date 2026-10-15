#!/usr/bin/env python3
"""Run the pinned LNCTVSeg nnU-Net v1 release in the existing v1 runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil

from lnctvseg_bundle import validate_assets


def job_output(path, root):
    target = Path(path).resolve()
    if target.is_relative_to(Path(root).resolve()):
        raise ValueError('LNCTVSeg output and work directories must be outside the read-only model bundle')
    return target


def stage_model(root, work, release):
    """Give v1 its expected all/ layout without changing the flat author release."""
    work = job_output(work, root)
    model = work / 'model'
    model.mkdir(parents=True, exist_ok=False)
    source = Path(root) / release['model_directory']
    fold = model / 'all'
    fold.mkdir()
    shutil.copy2(source / 'plans.pkl', model / 'plans.pkl')
    checkpoint = release['checkpoint']
    (fold / f'{checkpoint}.model').symlink_to((source / f'{checkpoint}.model').resolve())
    # restore_model instantiates the trainer with the publisher's init tuple.
    # Rewrite only paths in this job's copy; preserve the plans and architecture.
    with (source / f'{checkpoint}.model.pkl').open('rb') as stream:
        metadata = pickle.load(stream)
    init = list(metadata['init'])
    init[0] = str(model / 'plans.pkl')
    init[2] = str(model)
    init[3] = str(work / 'preprocessed')
    metadata['init'] = tuple(init)
    with (fold / f'{checkpoint}.model.pkl').open('wb') as stream:
        pickle.dump(metadata, stream)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--disable-tta', action='store_true',
                        help='Smoke checks only: disable the default mirror augmentation')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    spec = validate_assets(args.model_root)
    release = spec['nnunet']
    if args.threads < 1:
        parser.error('--threads must be positive')
    work = job_output(args.work, args.model_root)
    output = job_output(args.output, args.model_root)
    model = stage_model(args.model_root, work, release)
    os.environ['nnUNet_raw_data_base'] = str(work / 'raw')
    os.environ['nnUNet_preprocessed'] = str(work / 'preprocessed')
    os.environ['RESULTS_FOLDER'] = str(work / 'results')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    gpu = torch.cuda.is_available()
    from nnunet.inference.predict import predict_from_folder
    predict_from_folder(str(model), args.input, str(output), release['folds'],
                        save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1,
                        lowres_segmentations=None, part_id=0, num_parts=1,
                        tta=release['tta'] and not args.disable_tta, mixed_precision=gpu,
                        overwrite_existing=True, mode='normal', overwrite_all_in_gpu=None,
                        step_size=0.5, checkpoint_name=release['checkpoint'])
    print(json.dumps({'task': 'lnctvseg_nnunet_v1', 'source_commit': spec['source_commit'],
                      'checkpoint': release['checkpoint'], 'folds': release['folds'],
                      'device': 'cuda' if gpu else 'cpu',
                      'tta': release['tta'] and not args.disable_tta,
                      'output_label_ids': release['output_label_ids'],
                      'label_contract': spec['label_contract']['status']}))


if __name__ == '__main__':
    main()

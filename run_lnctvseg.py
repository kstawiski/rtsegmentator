#!/usr/bin/env python3
"""Run the pinned LNCTVSeg nnU-Net v1 release in the existing v1 runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from lnctvseg_bundle import validate_assets


def stage_file(source, target):
    if source.endswith('.model'):
        os.symlink(os.path.abspath(source), target)
    else:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--model-root', required=True)
    args = parser.parse_args()
    spec = validate_assets(args.model_root)
    release = spec['nnunet']
    work = Path(args.work)
    model = work / 'model'
    if model.exists():
        raise ValueError('Use a fresh LNCTVSeg inference work directory')
    source = Path(args.model_root) / release['model_directory']
    # Copy only small metadata; symlink immutable checkpoint files. Every output
    # directory remains writable without modifying the installed model bundle.
    shutil.copytree(source, model, copy_function=stage_file)
    os.environ['nnUNet_raw_data_base'] = str(work / 'raw')
    os.environ['nnUNet_preprocessed'] = str(work / 'preprocessed')
    os.environ['RESULTS_FOLDER'] = str(work / 'results')
    from nnunet.inference.predict import predict_from_folder
    predict_from_folder(str(model), args.input, args.output, release['folds'],
                        save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1,
                        lowres_segmentations=None, part_id=0, num_parts=1,
                        tta=release.get('tta', True), mixed_precision=True,
                        overwrite_existing=True, mode='normal', overwrite_all_in_gpu=None,
                        step_size=0.5, checkpoint_name=release['checkpoint'])
    print(json.dumps({'task': 'lnctvseg_nnunet_v1', 'source_commit': spec['source_commit'],
                      'checkpoint': release['checkpoint'], 'folds': release['folds']}))


if __name__ == '__main__':
    main()

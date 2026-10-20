#!/usr/bin/env python3
"""PI-CAI baseline five-fold nnU-Net v1 inference; no training or registration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

from prostate_models import CHANNEL_ROLES, TASKS, TRAINER, model_folder, runtime_provenance
from prostate_postprocess import assert_same_grid, extract_picai, write_array, write_json
from picai_predict import CUDNN_POLICY


def mirror_files(source, target):
    """Materialise writable directories, linking only immutable regular files."""
    shutil.copytree(source, target, copy_function=lambda a, b: os.symlink(Path(a).resolve(), b),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def prepare_trainer_overlay(source, target):
    import nnunet
    mirror_files(Path(nnunet.__file__).parent, target / 'nnunet')
    training = target / 'nnunet/training/network_training'
    for name, parent in (
        ('nnUNetTrainerV2_focalLoss.py', training / 'nnUNet_variants/loss_function'),
        ('nnUNetTrainerV2_Loss_CE_checkpoints.py', training),
        ('nnUNetTrainerV2_Loss_FL_and_CE.py', training),
    ):
        destination = parent / name
        destination.unlink(missing_ok=True)
        shutil.copyfile(source / name, destination)


def prediction_command(input_dir, output_dir, task):
    number = '2201' if task == 'picai_nnunet_supervised_bpmri' else '2203'
    return [sys.executable, str(Path(__file__).with_name('picai_predict.py')), '-t', f'Task{number}_picai_baseline',
            '-i', str(input_dir), '-o', str(output_dir), '-m', '3d_fullres', '-tr', TRAINER,
            '--num_threads_preprocessing', '2', '--num_threads_nifti_save', '1',
            '-f', '0', '1', '2', '3', '4', '-chk', 'model_best', '--save_npz']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=[t for t in TASKS if t.startswith('picai_')])
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    work, output = args.work, args.output
    model = TASKS[args.task][0]
    from picai_prep.preprocessing import Sample
    from picai_baseline.nnunet.softmax_export import save_softmax_nifti_from_softmax
    provenance = runtime_provenance(args.task, args.model_root)
    provenance.update(json.loads((work / 'context.json').read_text()))
    provenance.update({'weights_licence': 'CC BY-NC-SA 4.0', 'noncommercial': True,
                       'checkpoint': 'model_best', 'folds': list(range(5)), 'trainer': TRAINER,
                       'cudnn': dict(CUDNN_POLICY),
                       'preprocessing': 'picai-prep 2.0.0 Sample.preprocess(); no extra registration',
                       'adc_units': 'DICOM stored values with converter rescale; no inferred unit conversion'})
    scans = [sitk.ReadImage(str(work / 'native' / f'{role}.nii.gz')) for role in CHANNEL_ROLES]
    sample = Sample(scans=scans)
    sample.preprocess()
    inputs, predictions = work / 'input', work / 'prediction'
    inputs.mkdir()
    predictions.mkdir()
    for index, scan in enumerate(sample.scans):
        sitk.WriteImage(scan, str(inputs / f'case_{index:04d}.nii.gz'))
    assert_same_grid(sample.scans[0], scans[0])
    overlay = work / 'python'
    prepare_trainer_overlay(args.model_root / 'sources' / model, overlay)
    relative = model_folder(args.task)
    target = work / 'results' / relative.relative_to(Path('weights') / model / 'results')
    mirror_files(args.model_root / relative, target)
    env = os.environ.copy()
    env.update(RESULTS_FOLDER=str(work / 'results'), nnUNet_raw_data_base=str(work / 'raw'),
               nnUNet_preprocessed=str(work / 'preprocessed'), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([str(overlay), env.get('PYTHONPATH', '')]))
    subprocess.run(prediction_command(inputs, predictions, args.task), env=env, check=True)
    with np.load(predictions / 'case.npz') as data:
        foreground = data['softmax'][1].astype(np.float32)
    with (predictions / 'case.pkl').open('rb') as stream:
        properties = pickle.load(stream)
    probability_path = Path(str(output) + '.probability.nii.gz')
    save_softmax_nifti_from_softmax(np.stack([1 - foreground, foreground]), str(probability_path), properties)
    probability = sitk.ReadImage(str(probability_path))
    assert_same_grid(probability, scans[0])
    detection, masks, report = extract_picai(sitk.GetArrayFromImage(probability), scans[0].GetSpacing(),
                                            args.task, provenance)
    detection_path = Path(str(output) + '.detection.nii.gz')
    write_array(detection, scans[0], detection_path)
    report['artifacts'] = {'probability_map': probability_path.name, 'detection_map': detection_path.name}
    write_json(str(output) + '.lesions.json', report)
    mask_dir = work / 'masks'
    mask_dir.mkdir()
    mappings = {}
    for name, mask in masks.items():
        path = mask_dir / f'{name}.nii.gz'
        write_array(mask.astype(np.uint8), scans[0], path)
        mappings[name] = str(path)
    write_json(work / 'masks.json', mappings)
    write_json(work / 'result.json', {'empty': not bool(masks)})


if __name__ == '__main__':
    main()

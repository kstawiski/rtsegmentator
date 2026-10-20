#!/usr/bin/env python3
"""Native Umeå Dataset077 five-fold nnU-Net v2 probability export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

from prostate_models import model_folder, runtime_provenance, validate_umu_dataset
from prostate_postprocess import assert_same_grid, umu_postprocess, write_array, write_json

TASK = 'umu_prostate_zones_urethra_t2'


def prediction_command(input_dir, output_dir, folder):
    return [str(Path(sys.executable).parent / 'nnUNetv2_predict_from_modelfolder'),
            '-i', str(input_dir), '-o', str(output_dir), '-m', str(folder),
            '-f', '0', '1', '2', '3', '4', '-chk', 'checkpoint_best.pth',
            '-npp', '1', '-nps', '1', '-device', 'cuda', '--save_probabilities']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=[TASK])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work, output = args.work, args.output
    folder = args.model_root / model_folder(TASK)
    validate_umu_dataset(json.loads((folder / 'dataset.json').read_text()))
    provenance = runtime_provenance(TASK, args.model_root)
    provenance.update(json.loads((work / 'context.json').read_text()))
    provenance.update({'weights_licence': 'Weight licence not stated', 'code_licence': 'MIT',
                       'checkpoint': 'checkpoint_best.pth', 'folds': list(range(5)),
                       'urethra_radius_mm': 3.0, 'urethra_is_lumen': False,
                       'postprocessing': 'Probability-weighted quadratic centreline, 18-connected zone components, nearest-zone fill',
                       'inplane_spacing': 'Equal x/y spacing required by published circle construction',
                       'empty_zone_distance': 'Published SciPy distance transform, including absent zones',
                       'zone_postprocessor': 'Unchanged fill_empty_voxels from pinned MIT source'})
    reference = sitk.ReadImage(str(work / 'native/t2w.nii.gz'))
    if not np.isclose(*reference.GetSpacing()[:2], rtol=1e-5, atol=1e-6):
        raise ValueError('Published Umeå urethra construction requires equal in-plane spacing')
    inputs, predictions = work / 'input', work / 'prediction'
    inputs.mkdir()
    predictions.mkdir()
    sitk.WriteImage(reference, str(inputs / 'case_0000.nrrd'))
    env = os.environ.copy()
    env.update(nnUNet_raw=str(work / 'raw'), nnUNet_preprocessed=str(work / 'preprocessed'),
               nnUNet_results=str(work / 'results'), nnUNet_compile='false', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(prediction_command(inputs, predictions, folder), env=env, check=True)
    prediction = sitk.ReadImage(str(predictions / 'case.nrrd'))
    assert_same_grid(prediction, reference)
    with np.load(predictions / 'case.npz') as archive:
        probabilities = archive['probabilities']
    if tuple(probabilities.shape[1:]) != tuple(reversed(reference.GetSize())):
        raise ValueError('Umeå probabilities have not been restored to the original grid')
    labels = umu_postprocess(probabilities, reference.GetSpacing())
    write_array(labels, reference, work / 'restored.nii.gz')
    write_array(labels, reference, str(output) + '.segmentation.nii.gz')
    # Retain full spatial probabilities with their original-grid metadata.
    probability_path = Path(str(output) + '.probabilities.npz')
    np.savez_compressed(probability_path, probabilities=probabilities,
                        spacing_xyz=reference.GetSpacing(), origin_xyz=reference.GetOrigin(),
                        direction=reference.GetDirection())
    provenance['probability_map'] = probability_path.name
    provenance['status'] = 'nonempty' if labels.any() else 'empty'
    write_json(str(output) + '.provenance.json', provenance)
    write_json(work / 'result.json', {'empty': not bool(labels.any())})


if __name__ == '__main__':
    main()

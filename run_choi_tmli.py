"""Run the Choi release in a private workspace; output retains numeric labels."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

MODEL_RELATIVE = Path('nnUNet/3d_fullres/Task001_LN/nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1')


def prepare_input(source, destination):
    import nibabel as nib
    import numpy as np
    image = nib.load(source)
    if len(image.shape) != 3:
        raise ValueError('Choi requires one 3D CT in HU')
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('CT contains non-finite intensities')
    shifted = nib.Nifti1Image(values + 1024.0, image.affine, image.header.copy())
    shifted.set_data_dtype(np.float32)
    nib.save(shifted, destination)


def prepare_model(source, results, folds):
    target = Path(results) / MODEL_RELATIVE
    target.mkdir(parents=True, exist_ok=False)
    # Only checkpoints are linked. Metadata that inference might copy is private.
    shutil.copy2(Path(source) / 'plans.pkl', target / 'plans.pkl')
    for fold in folds:
        folder = target / f'fold_{fold}'
        folder.mkdir()
        for name in ('model_final_checkpoint.model', 'model_final_checkpoint.model.pkl'):
            original = (Path(source) / f'fold_{fold}' / name).resolve(strict=True)
            if name.endswith('.pkl'):
                shutil.copy2(original, folder / name)
            else:
                (folder / name).symlink_to(original)
    return target


def prediction_command(executable, inputs, outputs, folds):
    return [str(executable), '-i', str(inputs), '-o', str(outputs), '-t', 'Task001_LN',
            '-m', '3d_fullres', '-tr', 'nnUNetTrainerV2_noMirroring', '-p', 'nnUNetPlansv2.1',
            '-chk', 'model_final_checkpoint', '-f', *map(str, folds), '--disable_tta',
            '--num_threads_preprocessing', '1', '--num_threads_nifti_save', '1']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='CT NIfTI in HU, before the release-specific offset')
    p.add_argument('--model', type=Path, required=True, help='Released trainer folder containing plans.pkl')
    p.add_argument('--work', type=Path, required=True, help='New private work directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predict', type=Path, default=Path(sys.executable).parent / 'nnUNet_predict')
    p.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    args = p.parse_args(argv)
    if len(set(args.folds)) != len(args.folds):
        p.error('Duplicate folds')
    args.work.mkdir(parents=True, exist_ok=False)
    inputs, outputs, results = (args.work / name for name in ('input', 'prediction', 'results'))
    inputs.mkdir(); outputs.mkdir()
    prepare_input(args.input, inputs / 'case_0000.nii.gz')
    prepare_model(args.model, results, args.folds)
    env = os.environ.copy()
    env.update(RESULTS_FOLDER=str(results), nnUNet_raw_data_base=str(args.work / 'raw'),
               nnUNet_preprocessed=str(args.work / 'preprocessed'), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(prediction_command(args.predict, inputs, outputs, args.folds), env=env, check=True)
    prediction = outputs / 'case.nii.gz'
    if not prediction.is_file():
        raise RuntimeError('Choi inference produced no prediction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prediction, args.output)


if __name__ == '__main__':
    main()

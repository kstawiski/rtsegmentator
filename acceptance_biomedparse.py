"""GPU-only acceptance on the pinned public AMOS example; never run on patient data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from biomedparse import (TASK, SOURCE, ENV, CHECKPOINT, TOKENIZER, SEED, CHECKS, artifact_names,
                        binding, probe_device, require_device, validate_marker, vocabulary, roi_name)
from prostate_models import file_sha256

EXAMPLE = 'CT_AMOS_amos_0018.npz'
PROMPT = 'CT imaging of the spleen within the abdomen'
ORIGIN = {
    'repository_example': 'https://github.com/microsoft/BiomedParse/blob/e02096c03af0d79c6994ffc2d60a49eeb0361e1f/examples/imgs/CT_AMOS_amos_0018.npz',
    'upstream_notebook': 'https://github.com/microsoft/BiomedParse/blob/e02096c03af0d79c6994ffc2d60a49eeb0361e1f/inference_example_3D.ipynb',
    'public_challenge_data': 'https://huggingface.co/datasets/junma/CVPR-BiomedSegFM',
    'AMOS': 'https://amos22.grand-challenge.org/',
    'dicom_note': 'Public upstream AMOS pixels encoded as synthetic HU on synthetic DICOM geometry with new identifiers. Exact raw transport test; the original AMOS preprocessing window and uint8 rule are unverified.'}


def run(command, env, cwd=None):
    cp = subprocess.run(list(map(str, command)), env=env, cwd=cwd, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(cp.stdout, end='', flush=True)
    if cp.returncode:
        raise RuntimeError(f'Acceptance subprocess failed with exit {cp.returncode}: {command[0:2]}')
    return cp.stdout


def build_dicom(image, output):
    """Encode uint8 pixels as HU at soft-tissue quantization-bin centres.

    Stored value 2*p+1, slope 200/255 and intercept -160 yield
    (HU+160)/400*255 = p+0.5 before clipping/truncation. The half-bin
    margin avoids float32 round-off at integer boundaries in the real converter.
    This inverts our raw normalization, not unverified AMOS preprocessing.
    """
    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    from pydicom.valuerep import format_number_as_ds
    output.mkdir(parents=True)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for z, plane in enumerate(image):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = output / f'{z:04}.dcm'
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.PatientName, ds.PatientID = 'PUBLIC_AMOS^DERIVED', 'PUBLIC_AMOS_DERIVED'
        ds.Modality, ds.StudyDate, ds.StudyTime, ds.StudyID = 'CT', '20000101', '120000', 'PUBLIC'
        ds.SeriesDescription = 'Public AMOS; synthetic HU and geometry'
        ds.SeriesNumber, ds.InstanceNumber = 1, z + 1
        ds.Rows, ds.Columns = plane.shape
        # This is the portal HFS LPS storage contract: no voxel permutation.
        ds.ImageOrientationPatient, ds.ImagePositionPatient = [1, 0, 0, 0, 1, 0], [0, 0, z * 2]
        ds.PixelSpacing, ds.SliceThickness = [1, 1], 2
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = format_number_as_ds(200 / 255), -160
        ds.PixelData = (plane.astype('<i2') * 2 + 1).tobytes()
        ds.save_as(path, enforce_file_format=True)


def reference(args):
    """Execute upstream inference.main; relocate only config/checkpoint/tokenizer paths."""
    import numpy as np
    import torch
    import hydra
    from run_biomedparse import seed_inference
    source = (args.bundle_root / SOURCE).resolve()
    sys.path.insert(0, str(source))
    import inference
    seed_inference()
    # Upstream CLI has stale config and checkpoint locations; these are path substitutions,
    # with model operations, dtypes, RNG ordering, slice batching and thresholds untouched.
    hydra.initialize = lambda **kw: hydra.initialize_config_dir(
        config_dir=str(source / 'configs/model'), job_name=kw['job_name'], version_base=None)
    instantiate = hydra.utils.instantiate
    def local_instantiate(cfg, **kw):
        cfg.sem_seg_head.predictor.language_encoder.tokenizer.pretrained_model_name_or_path = str(args.bundle_root / TOKENIZER)
        model = instantiate(cfg, **kw)
        original_load = model.load_pretrained
        model.load_pretrained = lambda unused: original_load(str(args.bundle_root / CHECKPOINT))
        def capture(module, inputs, output):
            scores = output['predictions']['object_existence'].sigmoid()[0].detach().cpu().tolist()
            (args.work / 'reference-scores.json').write_text(json.dumps(scores))
        model.register_forward_hook(capture)
        return model
    hydra.utils.instantiate = local_instantiate
    # Call the installed unmodified main, not our infer wrapper.
    inference.main(argparse.Namespace(input_dir=str(args.work / 'reference-input'),
                                      output_dir=str(args.work / 'reference-output')))


def acceptance(args):
    root = args.bundle_root.resolve()
    app = Path(__file__).resolve().parent
    # Invalidate the explicitly requested device even when its initial probe fails.
    markers = [root / folder / f'{TASK}-{args.device}.json'
               for folder in ('validation', 'orientation-validation')]
    passed = False
    try:
        for p in markers:
            p.unlink(missing_ok=True)
            p.with_suffix('.json.tmp').unlink(missing_ok=True)
        import nibabel as nib
        import numpy as np
        import pydicom
        gpu = probe_device(root / ENV / 'bin/python')
        if require_device(gpu) != args.device:
            raise ValueError('Requested GPU device does not match this host')
        identity = binding(root)
        example = root / SOURCE / 'examples/imgs' / EXAMPLE
        if file_sha256(example) != vocabulary()['examples'][EXAMPLE]['sha256']:
            raise ValueError('Public upstream example checksum mismatch')
        args.work.mkdir(parents=True, exist_ok=False)
        with np.load(example, allow_pickle=False) as data:
            image = data['imgs']
        if image.shape != (63, 512, 512) or image.dtype != np.uint8:
            raise ValueError('Unexpected public AMOS example')
        prompt = args.work / 'prompt.json'
        prompt.write_text(json.dumps({'text': PROMPT}))
        refinput = args.work / 'reference-input'
        refinput.mkdir()
        np.savez_compressed(refinput / 'case.npz', imgs=image,
                            text_prompts={'1': PROMPT, 'instance_label': 0})
        env = os.environ.copy()
        env.update(RTSEG_APP_ROOT=str(app), RTSEG_MODEL_ROOT=str(root), RTSEG_RUNTIME_ROOT=str(root),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', RTSEG_BIOMEDPARSE_ACCEPTANCE_AUDIT='1')
        # No multiprocessing is used by this v2 inference path. Keep bulk temporary data here.
        env['TMPDIR'] = str(args.work / 'tmp')
        Path(env['TMPDIR']).mkdir()
        model_python = root / ENV / 'bin/python'
        run([model_python, app / 'acceptance_biomedparse.py', '--reference', '--bundle-root', root,
             '--work', args.work], env)
        ref = np.load(args.work / 'reference-output/case.npz')['segs']
        ref_scores = json.loads((args.work / 'reference-scores.json').read_text())
        results = []
        for i in range(2):
            output, sidecar = args.work / f'run{i}.npz', args.work / f'run{i}.json'
            run([model_python, app / 'run_biomedparse.py', '--npz', example, '--output', output,
                 '--sidecar', sidecar, '--prompt', prompt, '--modality', 'CT', '--source', root / SOURCE,
                 '--checkpoint', root / CHECKPOINT, '--tokenizer', root / TOKENIZER, '--bundle-root', root], env)
            result = np.load(output)['segs']
            scores = json.loads(sidecar.read_text())['existence']['scores_per_slice']
            if not np.array_equal(result, ref) or scores != ref_scores:
                raise ValueError('Reference/repeatability comparison failed; exact agreement is required')
            results.append(result)
        portal_runs = []
        for i in range(2):
            job = args.work / f'job{i}'
            (job / 'output').mkdir(parents=True)
            build_dicom(image, job / 'input')
            # Exercise the real raw conversion and fail before GPU dispatch if
            # float/dtype changes prevent exact reconstruction of the NPZ pixels.
            raw_input = job / 'raw-roundtrip.nii.gz'
            run([args.worker_python, app / 'dicom_series_to_nifti.py', job / 'input', raw_input], env)
            from run_biomedparse import preprocess
            raw_pixels, raw_info = preprocess(
                np.asarray(nib.load(raw_input).dataobj).transpose(2, 1, 0), 'CT', 'soft_tissue')
            if not np.array_equal(raw_pixels, image) or raw_info['method'] != 'CT_window':
                raise ValueError('Raw HU fixture does not exactly reproduce the public NPZ pixels')
            shutil.copyfile(prompt, job / 'prompt.json')
            (job / 'job.json').write_text(json.dumps({'models': [TASK], 'prompt': {'text': PROMPT}}))
            log = run([args.worker_python, app / 'run_task.py', job, TASK], env)
            (job / 'worker.log').write_text(log)
            validation = [json.loads(line) for line in log.splitlines() if line.startswith('{"valid":')]
            if not validation or validation[-1].get('valid') is not True or validation[-1]['rois'] != 1:
                raise ValueError('Missing strict RTSTRUCT converter validation')
            dcm = job / 'output' / f'{TASK}_RTSTRUCT.dcm'
            ds = pydicom.dcmread(dcm)
            if len(ds.StructureSetROISequence) != 1 or ds.StructureSetROISequence[0].ROIName != roi_name(PROMPT):
                raise ValueError('Portal output ROI contract failed')
            prediction = nib.load(job / 'external' / TASK / 'prediction.nii.gz')
            if not np.array_equal(np.asarray(prediction.dataobj).transpose(2, 1, 0), ref):
                raise ValueError('Portal mask differs from public NPZ reference')
            sidecar = json.loads((job / 'output' / artifact_names(TASK)[0]).read_text())
            if sidecar['preprocessing'] != raw_info:
                raise ValueError('Portal sidecar does not confirm raw CT normalization')
            if sidecar['existence']['scores_per_slice'] != ref_scores or sidecar['voxel_count'] != int(ref.sum()):
                raise ValueError('Portal sidecar differs from reference')
            portal_runs.append(validation[-1])
        digest = file_sha256(args.work / 'reference-output/case.npz')
        record = {'schema': 1, 'task': TASK, 'result': 'passed', 'checks': CHECKS, 'seed': SEED,
                  'criterion': 'exact', 'mask_voxel_differences': 0, 'device': args.device, 'gpu': gpu,
                  'public_origin': ORIGIN, 'example_sha256': file_sha256(example),
                  'reference': {'passed': True, 'runs': 1, 'result_sha256': digest},
                  'repeatability': {'passed': True, 'runs': 2, 'result_sha256': digest},
                  'dicom': {'passed': True, 'runs': 2, 'rois': 1, 'validation': portal_runs,
                            'raw_roundtrip_pixel_differences': 0, 'preprocessing': raw_info}, **identity}
        validate_marker(record, root, gpu)
        for p in markers:
            p.parent.mkdir(parents=True, exist_ok=True)
            temporary = p.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            os.replace(temporary, p)
        passed = True
        print(json.dumps({'passed': True, 'device': args.device, 'voxel_count': int(ref.sum()),
                          'markers': list(map(str, markers))}))
    finally:
        if not passed:
            for p in markers:
                p.unlink(missing_ok=True)
                p.with_suffix('.json.tmp').unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='New, empty authorized acceptance directory')
    p.add_argument('--worker-python', default=sys.executable)
    p.add_argument('--device', choices=['sm90', 'sm61'])
    p.add_argument('--reference', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.reference:
        reference(args)
    else:
        if not args.device:
            p.error('--device is required')
        acceptance(args)


if __name__ == '__main__':
    main()

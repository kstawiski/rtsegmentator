"""Synthetic DICOM -> positional model -> RTSTRUCT -> independent viewer evidence."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from dicom_series import ordered_dicom_files
from orientation_contract import (exact_orientation, map_prompt, normalize_image,
                                  restore_image, sidecar_path)
from test_rtstruct_geometry import convert
from test_viewer import rasterize
import viewer

ROOT = Path(__file__).parents[1]
PLANES = {
    'axial': ([1, 0, 0, 0, 1, 0], 'LPS'),
    'coronal': ([1, 0, 0, 0, 0, -1], 'LIP'),
    'sagittal': ([0, 1, 0, 0, 0, -1], 'PIR'),
    'oblique': ([.8, 0, .6, 0, 1, 0], 'LPS'),
}
POSITIONS = {'HFS': (1, 1, 1), 'HFP': (-1, -1, 1),
             'FFS': (-1, 1, -1), 'FFP': (1, -1, -1)}


def command(*args):
    result = subprocess.run([sys.executable, *map(str, args)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def base_image(plane):
    iop, _ = PLANES[plane]
    axes = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
    affine = np.eye(4)
    affine[:3, :3] = axes @ np.diag([.7, 1.3, 2.1])
    shape = (24, 20, 12)
    affine[:3, 3] = -affine[:3, :3] @ ((np.array(shape) - 1) / 2)
    affine[:2] *= -1
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape) - 1000
    return nib.Nifti1Image(data, affine)


def storage_variant(image, position):
    # Re-export exactly the same physical volume: change storage, not anatomy.
    directions = nib.orientations.io_orientation(image.affine)[:, 0].astype(int)
    signs = np.array(POSITIONS[position])[directions]
    return image.as_reoriented(np.column_stack((range(3), signs)))


def read_stored(source, output):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ordered_dicom_files(source))
    sitk.WriteImage(reader.Execute(), str(output), True)
    return nib.load(output)


def positional_model(image):
    """Label the high-index half on the HFS plane's left/right array axis."""
    _, codes = next((p, c) for p, (_, c) in PLANES.items()
                    if c == ''.join(nib.aff2axcodes(normalize_image(image)[0].affine)))
    axis = next(i for i, code in enumerate(codes) if code in 'LR')
    idx = np.indices(image.shape)[axis]
    # PIR's third axis points right; the low-index half is left there.
    mask = idx >= image.shape[axis] // 2 if codes[axis] == 'L' else idx < image.shape[axis] // 2
    return nib.Nifti1Image(mask.astype(np.uint8), image.affine)


def viewer_volume(source, output):
    series = viewer.index_series(sorted(source.iterdir()))
    _, mapped = viewer.map_structure(viewer.read_structure(output), output, output.parent, series)
    first = pydicom.dcmread(next(source.iterdir()), stop_before_pixels=True)
    volume = np.zeros((int(first.Columns), int(first.Rows), len(series.slices)), dtype=np.uint8)
    for k, rois in mapped.items():
        volume[:, :, k] = rasterize(rois[0]['paths'], (int(first.Rows), int(first.Columns))).T
    return volume


def physical_left_centroid(source, volume):
    # Independent patient-space measurement from original DICOM headers; no
    # contract, NIfTI affine or writer geometry helper participates.
    series = viewer.index_series(sorted(source.iterdir()))
    left_sum, count = 0., 0
    for k, item in enumerate(series.slices):
        ds = pydicom.dcmread(item.path, stop_before_pixels=True)
        indices = np.argwhere(volume[:, :, k] > 0)
        left = (float(ds.ImagePositionPatient[0])
                + indices[:, 0] * float(ds.ImageOrientationPatient[0]) * float(ds.PixelSpacing[1])
                + indices[:, 1] * float(ds.ImageOrientationPatient[3]) * float(ds.PixelSpacing[0]))
        left_sum += left.sum()
        count += len(left)
    return float(left_sum / count)


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
@pytest.mark.parametrize('position', POSITIONS)
@pytest.mark.parametrize('plane', PLANES)
def test_physical_side_matrix(tmp_path, plane, position, independent):
    base = base_image(plane)
    fixture = tmp_path / 'fixture.nii.gz'
    nib.save(storage_variant(base, position), fixture)
    source = tmp_path / 'dicom'
    command(ROOT / 'nifti_to_reference_dicom.py', fixture, source, '--modality', 'CT',
            '--preserve-storage', '--patient-position', position)
    stored = read_stored(source, tmp_path / 'stored.nii.gz')
    model_path = tmp_path / 'input.nii.gz'
    command(ROOT / 'dicom_series_to_nifti.py', source, model_path)
    model = nib.load(model_path)
    assert ''.join(nib.aff2axcodes(model.affine)) == PLANES[plane][1]
    expected_model = np.asanyarray(base.dataobj)
    assert np.array_equal(np.asanyarray(model.dataobj), expected_model)
    assert np.allclose(model.affine, base.affine, atol=1e-4, rtol=0)
    pred = positional_model(model)
    prediction = tmp_path / 'prediction.nii.gz'
    nib.save(pred, prediction)
    result, output = convert(source, tmp_path, model_path, prediction, independent, {'1': 'Fake Left'})
    assert result.returncode == 0, result.stdout + result.stderr
    validation = json.loads(result.stdout.splitlines()[-1])
    assert validation['physical_roundtrip']['Fake Left']['roundtrip_dice'] == 1
    reconstructed = viewer_volume(source, output)
    left_mm = physical_left_centroid(source, reconstructed)
    assert left_mm > 0
    reconstructed = nib.Nifti1Image(reconstructed, stored.affine)
    rebuilt, _ = restore_image(reconstructed, base.shape, base.affine)
    expected = np.asanyarray(positional_model(base).dataobj)
    assert np.array_equal(np.asanyarray(rebuilt.dataobj), expected)
    # Baseline actually goes through the same writers and independent viewer,
    # with stored-order input and the identical positional fake model.
    baseline_path = tmp_path / 'baseline.nii.gz'
    nib.save(positional_model(stored), baseline_path)
    baseline_root = tmp_path / 'baseline'
    baseline_root.mkdir()
    baseline_result, baseline_output = convert(source, baseline_root, tmp_path / 'stored.nii.gz',
                                               baseline_path, independent, {'1': 'Fake Left'})
    assert baseline_result.returncode == 0, baseline_result.stdout + baseline_result.stderr
    baseline_volume = viewer_volume(source, baseline_output)
    baseline_left_mm = physical_left_centroid(source, baseline_volume)
    assert (baseline_left_mm > 0) == (position in ('HFS', 'FFP'))
    raw_baseline = nib.Nifti1Image(baseline_volume, stored.affine)
    baseline, _ = restore_image(raw_baseline, base.shape, base.affine)
    baseline_mask = np.asanyarray(baseline.dataobj)
    dice = 2 * np.count_nonzero(baseline_mask & expected) / (baseline_mask.sum() + expected.sum())
    expected_dice = 1 if position in ('HFS', 'FFP') else 0
    assert dice == expected_dice
    assert sidecar_path(output).is_file()
    print('MATRIX ' + json.dumps(dict(plane=plane, position=position, writer='masks' if independent else 'labels',
                                    contract_dice=1.0, baseline_dice=float(dice), differing_voxels=0,
                                    contract_left_centroid_mm=left_mm, baseline_left_centroid_mm=baseline_left_mm)))


@pytest.mark.parametrize('plane', PLANES)
def test_hfs_sitk_byte_identity(tmp_path, plane):
    fixture = tmp_path / 'fixture.nii.gz'
    nib.save(base_image(plane), fixture)
    source = tmp_path / 'dicom'
    command(ROOT / 'nifti_to_reference_dicom.py', fixture, source, '--modality', 'CT',
            '--preserve-storage', '--patient-position', 'HFS')
    old = tmp_path / 'old.nii.gz'
    original = read_stored(source, old)
    new = tmp_path / 'new.nii.gz'
    command(ROOT / 'dicom_series_to_nifti.py', source, new)
    actual = nib.load(new)
    assert ''.join(nib.aff2axcodes(original.affine)) == PLANES[plane][1]
    assert original.affine.tobytes() == actual.affine.tobytes()
    assert np.asanyarray(original.dataobj).tobytes() == np.asanyarray(actual.dataobj).tobytes()
    assert gzip.decompress(old.read_bytes()) == gzip.decompress(new.read_bytes())
    assert old.read_bytes() == new.read_bytes()
    assert not json.loads(sidecar_path(new).read_text())['changed']
    print('HFS_BYTE_IDENTITY ' + json.dumps(dict(plane=plane, axcodes=''.join(nib.aff2axcodes(original.affine)),
        arrays_equal=True, affines_equal=True, nifti_payload_equal=True, gzip_equal=True,
        sha256=hashlib.sha256(old.read_bytes()).hexdigest())))


@pytest.mark.parametrize('permutation', list(itertools.permutations(range(3))))
@pytest.mark.parametrize('signs', list(itertools.product((-1, 1), repeat=3)))
def test_all_48_exact_inverses_and_prompts(permutation, signs):
    source = base_image('oblique')
    orientation = np.column_stack((permutation, signs))
    changed = source.as_reoriented(orientation)
    restored, _ = restore_image(changed, source.shape, source.affine)
    assert restored.affine.tobytes() == source.affine.tobytes()
    assert np.asanyarray(restored.dataobj).tobytes() == np.asanyarray(source.dataobj).tobytes()
    normalized, record = normalize_image(changed)
    reversed_image, _ = restore_image(normalized, changed.shape, changed.affine)
    assert np.array_equal(np.asanyarray(reversed_image.dataobj), np.asanyarray(changed.dataobj))
    prompt = {'positive_points': [[2, 3, 4]], 'negative_points': [[5, 6, 7]],
              'box': [[1, 2, 3], [4, 5, 3]], 'text': 'target'}
    mapped = map_prompt(prompt, record)
    for key in ('positive_points', 'negative_points'):
        assert np.allclose(nib.affines.apply_affine(changed.affine, prompt[key]),
                           nib.affines.apply_affine(normalized.affine, mapped[key]))
    assert mapped['text'] == prompt['text']
    assert np.all(np.asarray(mapped['box'][0]) <= np.asarray(mapped['box'][1]))


@pytest.mark.parametrize('defect', ['translation', 'scale', 'shear', 'crop', 'nan'])
def test_refuse_nonexact_grid(defect):
    image = base_image('axial')
    affine = image.affine.copy()
    shape = image.shape
    if defect == 'translation': affine[0, 3] += .01
    if defect == 'scale': affine[0, 0] *= 1.01
    if defect == 'shear': affine[0, 1] += .01
    if defect == 'crop': shape = (image.shape[0] - 1, *image.shape[1:])
    if defect == 'nan': affine[0, 0] = np.nan
    with pytest.raises(ValueError):
        exact_orientation(shape, affine, image.shape, image.affine)


def test_pet_suv_and_crop_restore_before_dicom_export(synthetic_ct, tmp_path):
    """PET intensity scaling and CT-reference/crop helpers preserve the contract."""
    from pydicom.dataset import Dataset
    for path in synthetic_ct.iterdir():
        ds = pydicom.dcmread(path)
        ds.ImageOrientationPatient = [-1, 0, 0, 0, -1, 0]
        ds.Modality = 'PT'
        ds.Units, ds.PatientWeight = 'BQML', 70
        ds.SeriesDate, ds.SeriesTime = '20000101', '120000'
        radio = Dataset()
        radio.RadionuclideTotalDose, radio.RadionuclideHalfLife = 70000000, 3600
        radio.RadiopharmaceuticalStartTime = '120000'
        ds.RadiopharmaceuticalInformationSequence = [radio]
        ds.save_as(path, enforce_file_format=True)
    model_path, suv, registered = (tmp_path / name for name in ('input.nii.gz', 'suv.nii.gz', 'on-ct.nii.gz'))
    command(ROOT / 'dicom_series_to_nifti.py', synthetic_ct, model_path)
    command(ROOT / 'pet_bqml_to_suv1000.py', model_path, synthetic_ct, suv)
    command(ROOT / 'resample_image_to_reference.py', suv, model_path, registered)
    model = nib.load(model_path)
    assert np.array_equal(nib.load(suv).affine, model.affine)
    scaled = nib.load(suv)
    # The existing SUV writer keeps the source integer header and encodes the
    # float values with NIfTI scaling. Test its quantization bound separately
    # from the exact geometry/label restoration required by this contract.
    assert np.allclose(scaled.get_fdata(), model.get_fdata(), rtol=0,
                       atol=abs(scaled.dataobj.slope) / 2 + 2e-6)  # factor * 1000 = 1
    assert np.allclose(nib.load(registered).get_fdata(), scaled.get_fdata(), rtol=0, atol=2e-6)
    mask = np.zeros(model.shape, np.uint8)
    mask[20:28, 3:10, 4:8] = 1
    prediction = tmp_path / 'prediction.nii.gz'
    nib.save(nib.Nifti1Image(mask, model.affine), prediction)
    crop, restored = tmp_path / 'crop.nii.gz', tmp_path / 'restored.nii.gz'
    command(ROOT / 'crop_nifti_indices.py', prediction, crop, 18, 30, 2, 12, 3, 9)
    cropped = nib.load(crop)
    with pytest.raises(ValueError, match='exact permutation/flip'):
        restore_image(cropped, model.shape, model.affine)
    command(ROOT / 'resample_label_to_reference.py', crop, model_path, restored)
    assert np.array_equal(nib.load(restored).get_fdata(), mask)
    result, output = convert(synthetic_ct, tmp_path, model_path, restored)
    assert result.returncode == 0, result.stdout + result.stderr
    from rtstruct_geometry import dicom_grid
    _, affine, _ = dicom_grid(synthetic_ct, model)
    seen = nib.Nifti1Image(viewer_volume(synthetic_ct, output), affine)
    seen, _ = restore_image(seen, model.shape, model.affine)
    assert np.array_equal(seen.get_fdata(), mask)


def test_manager_fixture_builder_and_comparison(synthetic_ct, tmp_path):
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location('contract_gpu', ROOT / 'scripts/run-orientation-contract-gpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_path = module.prepare_variants(synthetic_ct, tmp_path / 'variants', sys.executable, os.environ.copy())
    evidence = json.loads((reference_path.parent / 'variants.json').read_text())
    assert set(evidence) == {'HFS', 'HFP', 'FFS', 'FFP'}
    assert all(item['exact_ct_voxels'] for item in evidence.values())
    assert len(module.TASKS) == 4
    assert {'hnlnl_2d3d', 'lnctvseg_nnunet_v1'} <= set(module.TASKS)
    reference = nib.load(reference_path)
    prediction = positional_model(reference)
    changed = prediction.as_reoriented(np.array([[2, -1], [0, 1], [1, -1]]))
    labels = {'1': 'Left', '2': 'Empty'}
    prediction_path = tmp_path / 'prediction.nii.gz'
    nib.save(prediction, prediction_path)
    result, rtstruct = convert(synthetic_ct, tmp_path, reference_path, prediction_path, labels=labels)
    assert result.returncode == 0, result.stdout + result.stderr
    exported = module.exported_prediction(synthetic_ct, rtstruct, reference, labels)
    assert np.array_equal(exported.get_fdata(), prediction.get_fdata())
    rows = module.physical_dice(prediction, changed, labels)
    assert rows[0]['dice'] == 1
    assert rows[1]['dice'] is None and rows[1]['status'] == 'both_empty'


def test_prompt_dispatch_preserves_clicked_source_voxel(synthetic_ct, tmp_path, monkeypatch):
    import os
    import runpy
    import signal
    import tempfile
    # Feet-first storage also reverses slice indices.
    for path in synthetic_ct.iterdir():
        ds = pydicom.dcmread(path)
        ds.ImageOrientationPatient = [-1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, -float(ds.InstanceNumber - 1) * 2]
        ds.save_as(path, enforce_file_format=True)
    job = tmp_path / 'prompt-job'
    (job / 'output').mkdir(parents=True)
    (job / 'input').symlink_to(synthetic_ct)
    original = {'positive_points': [[3, 8, 2]], 'negative_points': [], 'modality': 'CT'}
    (job / 'prompt.json').write_text(json.dumps(original))
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('RTSEG_SCRATCH_ROOT', str(tmp_path / 'scratch'))
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(job), 'sam_med2d_prompted'])
    monkeypatch.setattr(signal, 'signal', lambda *args: None)
    actual_popen = subprocess.Popen
    seen = []

    class Finished:
        pid = -1
        def wait(self): return 0
        def poll(self): return 0

    def popen(args, **kwargs):
        if Path(args[1]).name != 'run_sam_med2d.py':
            return actual_popen(args, pass_fds=(fd,), **kwargs)
        image = nib.load(args[args.index('--input') + 1])
        prompt = json.loads(Path(args[args.index('--prompt') + 1]).read_text())
        seen.append(prompt)
        point = np.asarray(prompt['positive_points'][0], dtype=int)
        data = np.zeros(image.shape, np.uint8)
        data[tuple(point)] = 1
        nib.save(nib.Nifti1Image(data, image.affine), args[args.index('--output') + 1])
        return Finished()

    monkeypatch.setattr(subprocess, 'Popen', popen)
    # Keep the runner's short socket path within the authorized synthetic job.
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, 'tempdir', f'/proc/self/fd/{fd}')
        runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    finally:
        os.close(fd)
    assert seen[0]['positive_points'] == [[28, 8, 9]]
    assert json.loads((job / 'prompt.json').read_text()) == original
    output = job / 'output/sam_med2d_prompted_RTSTRUCT.dcm'
    volume = viewer_volume(synthetic_ct, output)
    assert np.argwhere(volume).tolist() == [[3, 8, 2]]
    assert sidecar_path(output).is_file()


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_swapped_dicom_axes_and_permuted_prediction(tmp_path, independent):
    base = base_image('axial')
    stored_variant = base.as_reoriented(np.array([[1, 1], [0, 1], [2, -1]]))
    fixture, source = tmp_path / 'fixture.nii.gz', tmp_path / 'dicom'
    nib.save(stored_variant, fixture)
    command(ROOT / 'nifti_to_reference_dicom.py', fixture, source, '--modality', 'CT', '--preserve-storage')
    model_path = tmp_path / 'input.nii.gz'
    command(ROOT / 'dicom_series_to_nifti.py', source, model_path)
    model = nib.load(model_path)
    assert model.shape == base.shape != stored_variant.shape
    assert np.array_equal(model.get_fdata(), base.get_fdata())
    # A runner may return another lossless storage order. This must be accepted,
    # whereas a resampled prediction is refused by the same converter boundary.
    prediction = positional_model(model).as_reoriented(np.array([[2, -1], [0, 1], [1, -1]]))
    path = tmp_path / 'prediction.nii.gz'
    nib.save(prediction, path)
    result, output = convert(source, tmp_path, model_path, path, independent)
    assert result.returncode == 0, result.stdout + result.stderr
    reconstructed = viewer_volume(source, output)
    assert physical_left_centroid(source, reconstructed) > 0
    raw = read_stored(source, tmp_path / 'stored.nii.gz')
    rebuilt, _ = restore_image(nib.Nifti1Image(reconstructed, raw.affine), base.shape, base.affine)
    assert np.array_equal(rebuilt.get_fdata(), positional_model(base).get_fdata())
    shifted = prediction.affine.copy()
    shifted[0, 3] += .01
    nib.save(nib.Nifti1Image(prediction.get_fdata(), shifted), path)
    rejected, _ = convert(source, tmp_path, model_path, path, independent)
    assert rejected.returncode != 0
    assert 'not an exact permutation/flip' in rejected.stderr


def test_colab_and_download_archive_retain_orientation_record(tmp_path, portal, monkeypatch):
    import zipfile
    job = tmp_path / 'archive-job'
    (job / 'input').mkdir(parents=True)
    (job / 'output').mkdir()
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'models': ['total']}))
    name = 'total_RTSTRUCT.dcm'
    record = json.dumps({'version': 1, 'inverse': {'prediction_to_dicom_orientation': [[0, -1], [1, -1], [2, 1]]}})

    def remote(command, **kwargs):
        if 'download' in command:
            target = Path(command[-1])
            if command[-2].endswith('rtseg-status.json'):
                target.write_text(json.dumps({'status': 'complete'}))
            elif command[-2].endswith('rtseg-results.zip'):
                with zipfile.ZipFile(target, 'w') as archive:
                    archive.writestr(name, b'synthetic result')
                    archive.writestr(name + '.orientation.json', record)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(portal.subprocess, 'run', remote)
    portal.run_colab_job(job)
    assert (job / 'output' / (name + '.orientation.json')).read_text() == record
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as archive:
        assert archive.read(name + '.orientation.json').decode() == record
        assert archive.read(name) == b'synthetic result'

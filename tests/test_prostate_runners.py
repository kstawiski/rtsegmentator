"""Synthetic CPU contracts; no checkpoints, datasets, model forward pass or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import pytest
import SimpleITK as sitk

import prostate_models as models
from prostate_postprocess import (assert_same_grid, extract_picai, fill_umu_zones,
                                 largest_component, restore_to_reference, roi_name,
                                 umu_postprocess, write_array)
from scripts import build_structure_catalog as builder

ROOT = Path(__file__).resolve().parents[1]


def make_series(directory, study, frame, value, *, b_value=None, series=None):
    directory.mkdir(parents=True)
    series = series or generate_uid()
    for index in range(5):
        path = directory / f'{index:03d}.dcm'
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.PatientName, ds.PatientID = 'SYNTHETIC^PROSTATE', 'SYNTHETIC'
        ds.StudyDate, ds.StudyTime, ds.StudyID = '20000101', '120000', 'SYNTHETIC'
        ds.Modality, ds.SeriesDescription = 'MR', 'Synthetic prostate runner test'
        ds.Rows, ds.Columns, ds.InstanceNumber = 32, 40, index + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [12, -8, index * 3]
        ds.PixelSpacing, ds.SliceThickness = [0.8, 0.8], 3
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.RescaleSlope, ds.RescaleIntercept = 2, 10
        if b_value is not None:
            ds.DiffusionBValue = b_value
        ds.PixelData = np.full((32, 40), value + index, np.uint16).tobytes()
        ds.save_as(path, enforce_file_format=True)
    return series


def update_manifest(job, data):
    data['input_manifest'] = {entry['dir']: [{'name': p.name, 'size': p.stat().st_size}
                              for p in sorted((job / entry['dir']).iterdir())]
                              for entry in data['input_roles'].values()}
    (job / 'job.json').write_text(json.dumps(data))


@pytest.fixture
def role_job(tmp_path):
    study, frame = generate_uid(), generate_uid()
    records = {}
    # Deliberately create dictionaries in reverse order; channel order is fixed.
    for index, role in enumerate(reversed(models.CHANNEL_ROLES)):
        dirname = 'input' if role == 't2w' else f'input_{role}'
        uid = make_series(tmp_path / dirname, study, frame, 10 * (index + 1),
                          b_value=1400 if role == 'hbv' else None)
        records[role] = {'dir': dirname, 'series_uid': uid}
        if role == 'hbv':
            records[role].update(b_value=1400, b_value_source='(0018,9087)')
    data = {'input_roles': records}
    update_manifest(tmp_path, data)
    return tmp_path, data


def test_role_order_and_dicom_contract(role_job):
    job, data = role_job
    dirs = models.role_directories(job)
    assert list(dirs) == ['t2w', 'adc', 'hbv']
    assert [p.name for p in dirs.values()] == ['input', 'input_adc', 'input_hbv']
    assert models.validate_dicom_roles(dirs, data['input_roles']) == []


@pytest.mark.parametrize('defect', ['missing_role', 'redirect', 'missing_file', 'changed_size', 'missing_manifest', 'directory', 'symlink'])
def test_role_manifest_rejects_partial_or_redirected_input(role_job, defect):
    job, data = role_job
    if defect == 'missing_role':
        del data['input_roles']['adc']
    elif defect == 'redirect':
        data['input_roles']['adc']['dir'] = '../outside'
    elif defect == 'missing_file':
        (job / 'input_adc/000.dcm').unlink()
    elif defect == 'changed_size':
        with (job / 'input_adc/000.dcm').open('ab') as f:
            f.write(b'bad')
    elif defect == 'missing_manifest':
        del data['input_manifest']
    elif defect == 'directory':
        (job / 'input_adc/extra').mkdir()
    elif defect == 'symlink':
        (job / 'input_adc/000.dcm').unlink()
        (job / 'input_adc/000.dcm').symlink_to(job / 'input/000.dcm')
    (job / 'job.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        models.role_directories(job)


@pytest.mark.parametrize('defect', ['study', 'series', 'mixed_b', 'duplicate_position', 'multiframe', 'modality', 'unknown_b'])
def test_dicom_role_validation_rejects_wrong_volumes(role_job, defect):
    job, data = role_job
    path = job / 'input_hbv/000.dcm'
    ds = pydicom.dcmread(path)
    if defect == 'study': ds.StudyInstanceUID = generate_uid()
    if defect == 'series': ds.SeriesInstanceUID = generate_uid()
    if defect == 'mixed_b': ds.DiffusionBValue = 800
    if defect == 'duplicate_position': ds.ImagePositionPatient = [12, -8, 3]
    if defect == 'multiframe': ds.NumberOfFrames = 2
    if defect == 'modality': ds.Modality = 'CT'
    if defect == 'unknown_b': data['input_roles']['hbv'].pop('b_value_source')
    ds.save_as(path, enforce_file_format=True)
    update_manifest(job, data)
    with pytest.raises(ValueError):
        models.validate_dicom_roles(models.role_directories(job), data['input_roles'])


def test_different_frame_is_warning_and_single_series_needs_no_roles(role_job):
    job, data = role_job
    uid = generate_uid()
    for path in (job / 'input_adc').iterdir():
        ds = pydicom.dcmread(path)
        ds.FrameOfReferenceUID = uid
        ds.save_as(path, enforce_file_format=True)
    update_manifest(job, data)
    assert len(models.validate_dicom_roles(models.role_directories(job), data['input_roles'])) == 1
    (job / 'job.json').write_text('{}')
    assert list(models.role_directories(job, multisequence=False)) == ['t2w']


def test_conversion_preserves_dicom_rescale_and_reference_grid(role_job):
    job, _ = role_job
    converted, native = job / 'converted/t2w.nii.gz', job / 'native/t2w.nii.gz'
    subprocess.run([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(job / 'input'), str(converted)], check=True)
    models.native_input(converted, native)
    image = sitk.ReadImage(str(native))
    assert image.GetSize() == (40, 32, 5)
    assert image.GetOrigin() == pytest.approx((12, -8, 0))
    assert image.GetSpacing() == pytest.approx((0.8, 0.8, 3))
    assert sitk.GetArrayFromImage(image)[:, 0, 0].tolist() == [70, 72, 74, 76, 78]


def test_native_input_reverses_storage_flip_exactly(tmp_path):
    from orientation_contract import normalize_file
    path, native = tmp_path / 'converted/input.nii.gz', tmp_path / 'native/input.nii.gz'
    path.parent.mkdir()
    values = np.arange(5 * 7 * 4, dtype=np.int16).reshape(5, 7, 4)
    affine = np.diag([0.7, 0.7, 3, 1])
    affine[:3, 3] = [4, -7, 13]
    nib.save(nib.Nifti1Image(values, affine), path)
    assert normalize_file(path)['changed']
    models.native_input(path, native)
    restored = nib.load(native)
    np.testing.assert_array_equal(restored.dataobj, values)
    np.testing.assert_allclose(restored.affine, affine)


def require_rga():
    pytest.importorskip('report_guided_annotation')
    pytest.importorskip('picai_prep')


def test_picai_matches_pinned_rga_and_keeps_equal_score_candidates():
    require_rga()
    from report_guided_annotation import extract_lesion_candidates
    from picai_prep.preprocessing import crop_or_pad
    probability = np.zeros((24, 60, 60), np.float32)
    probability[0:2, 1:5, 1:5] = 0.99  # Outside the published 20-slice voxel crop.
    probability[7:10, 10:14, 10:14] = 0.73123455
    probability[7:10, 40:44, 40:44] = 0.73123455
    detection, masks, report = extract_picai(probability, (0.5, 0.7, 3), 'synthetic')
    expected, scores, indexed = extract_lesion_candidates(
        crop_or_pad(crop_or_pad(probability, (20, 384, 384)), probability.shape), threshold='dynamic')
    np.testing.assert_array_equal(detection, expected)
    assert len(masks) == 2
    assert list(masks) == ['csPCa_L01_p0.73', 'csPCa_L02_p0.73']
    for lesion, (candidate_id, score) in zip(report['lesions'], scores):
        np.testing.assert_array_equal(masks[lesion['roi_name']], indexed == candidate_id)
        assert lesion['likelihood'] == float(score)
        assert lesion['voxel_count'] == 48
        assert lesion['volume_mm3'] == pytest.approx(50.4)
        assert lesion['volume_ml'] == pytest.approx(0.0504)
    assert report['case_likelihood'] == float(expected.max())
    assert report['extraction']['crop_units'] == 'voxels'
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize('tiny', [False, True])
def test_picai_empty_and_small_peak_have_no_invented_contour(tiny):
    require_rga()
    probability = np.zeros((6, 16, 16), np.float32)
    if tiny: probability[3, 4, 4] = 0.9
    detection, masks, report = extract_picai(probability, (1, 1, 3), 'synthetic')
    assert not detection.any() and not masks
    assert report['status'] == 'empty' and report['case_likelihood'] == 0
    assert report['lesions'] == []


def test_picai_candidate_limit_and_roi_names():
    require_rga()
    probability = np.zeros((10, 80, 80), np.float32)
    for index in range(7):
        probability[3:6, 8 * index:8 * index + 3, 30:33] = 0.9 - index / 10
    _, masks, report = extract_picai(probability, (1, 1, 1), 'synthetic')
    assert len(masks) == len(report['lesions']) == 5
    assert all(len(name) <= 64 for name in masks)
    for bad in (float('nan'), -0.1, 1.1, 0):
        with pytest.raises(ValueError): roi_name(1, bad)


def umu_probabilities():
    # Non-square volume, with all four zones and a central urethra peak.
    labels = np.zeros((7, 49, 55), np.uint8)
    labels[:, 4:45, 4:51] = 1
    labels[:, 4:24, 28:51] = 2
    labels[:, 24:45, 4:28] = 3
    labels[:, 24:45, 28:51] = 4
    probabilities = np.eye(6, dtype=np.float32)[labels].transpose(3, 0, 1, 2) * 0.8 + 0.01
    probabilities[5, :, 24, 27] = 0.95
    return probabilities


def test_umu_urethra_radius_with_anisotropic_slice_spacing():
    probabilities = umu_probabilities()
    result = umu_postprocess(probabilities, (0.5, 0.5, 3))
    y, x = np.ogrid[:49, :55]
    circle = ((x - 27) * 0.5) ** 2 + ((y - 24) * 0.5) ** 2 <= 9
    for plane in result:
        np.testing.assert_array_equal(plane == 5, circle)
    assert result[3, 24, 33] == 5 and result[3, 24, 34] != 5
    assert set(np.unique(result)) == set(range(6))
    assert np.all(result[:, 4:45, 4:51] != 0)
    with pytest.raises(ValueError, match='equal in-plane'):
        umu_postprocess(probabilities, (0.5, 0.8, 3))


def test_umu_nearest_zone_uses_physical_distance():
    gland = np.ones((9, 9, 3), bool)
    labels = np.zeros(gland.shape, np.uint8)
    labels[1, 4, 1] = 1  # Three x voxels (6 mm).
    labels[4, 0, 1] = 2  # Four y voxels (4 mm).
    result = fill_umu_zones(gland, labels, np.zeros_like(gland), (2, 1, 3))
    assert result[4, 4, 1] == 2
    # The published empty-zone distance transform can assign the absent TZ.
    assert set(np.unique(result)) == {1, 2, 3}
    # Upstream assigns ties to PZ even when every zone is absent.
    assert np.all(fill_umu_zones(gland, labels * 0, np.zeros_like(gland), (2, 1, 3)) == 1)


def test_umu_clips_urethra_and_handles_empty_prostate():
    probabilities = umu_probabilities()
    probabilities[:, :, :24, :] = 0
    probabilities[0, :, :24, :] = 1
    result = umu_postprocess(probabilities, (0.5, 0.5, 3))
    assert not (result[:, :24] == 5).any()
    empty = np.zeros((6, 4, 9, 9), np.float32)
    empty[0] = 1
    assert not umu_postprocess(empty, (0.5, 0.5, 3)).any()


def test_umu_matches_published_postprocessing_for_supported_grids():
    # Optional direct comparison with the supplied pinned source (no data files).
    source = os.environ.get('RTSEG_UMU_SOURCE')
    if not source:
        pytest.skip('Set RTSEG_UMU_SOURCE to compare the pinned postprocessor')
    import runpy
    upstream = runpy.run_path(str(Path(source) / 'nnUNet/utils_postprocess_nnUNet.py'))
    probabilities = umu_probabilities()
    p = probabilities.transpose(3, 2, 1, 0)
    labels = p.argmax(3)
    prostate = upstream['getLargestCC'](~upstream['getLargestCC'](labels == 0))
    u = p[..., 5]
    peaks = [upstream['findIndices_allSlices'](u[..., k:k + 1]) for k in range(u.shape[2])]
    z = np.arange(u.shape[2]); peaks = np.asarray(peaks)
    x = np.round(np.polyval(np.polyfit(z, peaks[:, 0], 2, w=peaks[:, 2]), z))
    y = np.round(np.polyval(np.polyfit(z, peaks[:, 1], 2, w=peaks[:, 2]), z))
    urethra = np.stack([upstream['drawUrethra'](u[..., k].shape, [x[k], y[k]], 3, 0.5)
                        for k in z], axis=2).astype(bool) & prostate
    expected = upstream['fill_empty_voxels'](~prostate, *[(labels == k) & prostate for k in range(1, 5)],
                                             urethra, (0.5, 0.5, 3)).argmax(3).transpose(2, 1, 0)
    np.testing.assert_array_equal(umu_postprocess(probabilities, (0.5, 0.5, 3)), expected)


def test_prostate158_largest_component_and_connectivity():
    mask = np.zeros((20, 20, 20), bool)
    mask[3:7, 3:7, 3:7] = True
    mask[15:17, 15:17, 15:17] = True
    mask[7, 7, 7] = True  # Diagonal neighbour retained by MONAI full connectivity.
    result = largest_component(mask)
    assert result.sum() == 65 and result[7, 7, 7]
    assert not result[15:17, 15:17, 15:17].any()
    assert not largest_component(mask * 0).any()


def test_geometry_restoration_uses_physical_grid_and_label_interpolation():
    reference = sitk.Image([15, 19, 7], sitk.sitkUInt8)
    reference.SetSpacing((0.5, 0.8, 3))
    reference.SetOrigin((12, -7, 21))
    reference.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    values = np.zeros((7, 19, 15), np.uint8)
    values[2:5, 5:12, 4:9] = 4
    original = sitk.GetImageFromArray(values); original.CopyInformation(reference)
    flipped = sitk.Flip(original, [True, True, False])
    restored = restore_to_reference(flipped, reference)
    assert_same_grid(restored, reference)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), values)
    assert set(np.unique(sitk.GetArrayFromImage(restored))) == {0, 4}


def test_bad_header_copy_is_rejected(tmp_path):
    reference = sitk.Image([10, 20, 30], sitk.sitkUInt8)
    with pytest.raises(ValueError, match='Array shape'):
        write_array(np.zeros((10, 20, 30), np.uint8), reference, tmp_path / 'bad.nii.gz')


def test_catalogues_label_order_and_variable_contract(tmp_path):
    catalog = json.loads((ROOT / 'catalog/prostate.json').read_text())
    structures = json.loads((ROOT / 'catalog/structures.json').read_text())
    assert set(catalog) == set(models.TASKS)
    for task, spec in catalog.items():
        if task != 'umu_prostate_zones_urethra_t2':
            assert [r['role'] for r in spec['input_roles']] == list(models.CHANNEL_ROLES)
            assert spec['reference_role'] == 't2w'
        else:
            assert 'input_roles' not in spec
            assert spec['license'] == 'Weight licence not stated'
        if task.startswith('picai_'):
            assert spec['noncommercial'] is True
            assert 'model card lists no performance' in spec['description']
            assert structures[task]['kind'] == 'variable'
            assert 'maximum 5' in structures[task]['note']
        else:
            assert structures[task]['kind'] == 'fixed'
    assert structures['umu_prostate_zones_urethra_t2']['names'] == [
        'Prostate_PZ', 'Prostate_CZ', 'Prostate_TZ', 'Prostate_AFS', 'Urethra_6mm_model']
    metadata = tmp_path / models.model_folder('umu_prostate_zones_urethra_t2') / 'dataset.json'
    metadata.parent.mkdir(parents=True)
    data = {'labels': models.UMU_LABELS, 'channel_names': {'0': 'T2'}, 'file_ending': '.nrrd'}
    metadata.write_text(json.dumps(data))
    entry, _ = builder.resolve('umu_prostate_zones_urethra_t2', {}, {}, tmp_path, {}, Path('unused'))
    assert entry == structures['umu_prostate_zones_urethra_t2']
    data['labels'] = {**models.UMU_LABELS, 'CZ': 3, 'TZ': 2}
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='label contract'):
        builder.resolve('umu_prostate_zones_urethra_t2', {}, {}, tmp_path, {}, Path('unused'))


def test_commands_require_five_best_checkpoints():
    from run_picai import prediction_command as picai
    from run_umu_prostate import prediction_command as umu
    for task in (t for t in models.TASKS if t.startswith('picai_')):
        command = picai(Path('in'), Path('out'), task)
        assert command[command.index('-f') + 1:command.index('-chk')] == ['0', '1', '2', '3', '4']
        assert command[command.index('-chk') + 1] == 'model_best'
        assert models.TRAINER in command and '--save_npz' in command
    command = umu(Path('in'), Path('out'), Path('model'))
    assert '--save_probabilities' in command
    assert command[command.index('-chk') + 1] == 'checkpoint_best.pth'
    assert command[command.index('-f') + 1:command.index('-chk')] == ['0', '1', '2', '3', '4']


@pytest.mark.parametrize('rotated', [False, True])
def test_prostate158_resamples_physical_coordinates_before_transforms(tmp_path, monkeypatch, rotated):
    from types import SimpleNamespace
    import run_prostate158 as runner
    reference = sitk.Image([11, 10, 8], sitk.sitkFloat32)
    reference.SetSpacing((0.5, 0.5, 1))
    reference.SetOrigin((-3, -2, -1))
    if rotated:
        reference.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    paths = {role: tmp_path / f'{role}.nii.gz' for role in models.CHANNEL_ROLES}
    sitk.WriteImage(reference, str(paths['t2w']))
    sources = {}
    for role, spacing in [('adc', (2, 2, 2)), ('hbv', (1, 2, 2))]:
        source = sitk.Image([3 if role == 'adc' else 6, 6, 5], sitk.sitkInt16)
        source.SetOrigin((-4, -3, -2))
        source.SetSpacing(spacing)
        for z, y, x in np.ndindex(tuple(reversed(source.GetSize()))):
            px, py, pz = source.TransformIndexToPhysicalPoint((x, y, z))
            source[x, y, z] = int(100 + px + 3 * py + 5 * pz)
        if role == 'hbv':
            source = sitk.Flip(source, [True, False, False])
        sources[role] = source
        sitk.WriteImage(source, str(paths[role]))
    original = {role: path.read_bytes() for role, path in paths.items()}
    seen = []
    def published_transforms(aligned):
        seen.append(aligned)
        assert aligned['t2w'] == str(paths['t2w'])
        for role in ('adc', 'hbv'):
            image = sitk.ReadImage(aligned[role])
            assert_same_grid(image, reference)
            assert image.GetPixelID() == sitk.sitkFloat32
            interior = padding = 0
            for z, y, x in np.ndindex(tuple(reversed(reference.GetSize()))):
                point = reference.TransformIndexToPhysicalPoint((x, y, z))
                index = np.array(sources[role].TransformPhysicalPointToContinuousIndex(point))
                size = np.array(sources[role].GetSize())
                if np.all((index >= 0) & (index <= size - 1)):
                    assert image[x, y, z] == pytest.approx(100 + np.dot((1, 3, 5), point))
                    interior += 1
                elif np.any((index < -0.5) | (index >= size - 0.5)):
                    assert image[x, y, z] == 0
                    padding += 1
            assert interior > 0 and padding > 0
        return {role: SimpleNamespace(shape=(1, 11, 10, 8), affine=np.eye(4)) for role in paths}
    monkeypatch.setattr(runner, 'preprocessing', lambda: published_transforms)
    provenance = {}
    runner.prepare_channels(paths, provenance)
    assert len(seen) == 1
    record = provenance['input_grid_resampling']
    assert record['transform'] == 'identity' and record['registration'] is False
    assert record['interpolation'] == 'linear' and record['default_value'] == 0
    assert record['reference_grid']['size_xyz'] == [11, 10, 8]
    for role in ('adc', 'hbv'):
        assert record['channels'][role]['output_grid'] == record['reference_grid']
        assert record['channels'][role]['source_grid'] != record['reference_grid']
        assert not Path(seen[0][role]).exists()
    assert original == {role: path.read_bytes() for role, path in paths.items()}
    assert not list(tmp_path.glob('aligned-*'))


def test_prostate158_removes_resampled_intermediates_after_transform_failure(tmp_path, monkeypatch):
    import run_prostate158 as runner
    paths = {role: tmp_path / f'{role}.nii.gz' for role in models.CHANNEL_ROLES}
    for path in paths.values():
        sitk.WriteImage(sitk.Image([3, 4, 5], sitk.sitkUInt16), str(path))
    def fail(data):
        assert all(Path(path).is_file() for path in data.values())
        raise RuntimeError('synthetic transform failure')
    monkeypatch.setattr(runner, 'preprocessing', lambda: fail)
    with pytest.raises(RuntimeError, match='synthetic transform failure'):
        runner.prepare_channels(paths)
    assert not list(tmp_path.glob('aligned-*'))


def test_dispatch_empty_result_retains_sidecar_and_uses_existing_empty_marker(role_job):
    job, _ = role_job
    task = 'picai_nnunet_supervised_bpmri'
    output = job / 'output' / f'{task}_RTSTRUCT.dcm'
    output.parent.mkdir()
    output.write_bytes(b'stale result')
    calls = []
    def run(command):
        calls.append(command)
        if command[1].endswith('dicom_series_to_nifti.py'):
            subprocess.run(command, check=True)
        else:
            work = Path(command[command.index('--work') + 1])
            (work / 'result.json').write_text('{"empty": true}')
            Path(str(output) + '.lesions.json').write_text('{"status":"empty","lesions":[]}')
    with pytest.raises(SystemExit, match='empty prediction: no RTSTRUCT created'):
        models.run_task(task, job, ROOT, Path('/models'), sys.executable, run)
    assert [Path(c[2]).name for c in calls[:3]] == ['input', 'input_adc', 'input_hbv']
    assert not output.exists()
    assert Path(str(output) + '.lesions.json').is_file()
    assert not any('convert_masks' in str(c) for c in calls)


def test_lesion_sidecar_schema_positive_and_empty():
    require_rga()
    import jsonschema
    schema = json.loads((ROOT / 'catalog/prostate-lesions.schema.json').read_text())
    probability = np.zeros((8, 20, 20), np.float32)
    for positive in (False, True):
        if positive: probability[2:5, 4:8, 4:8] = 0.73123455
        _, _, report = extract_picai(probability, (0.5, 0.5, 3), 'synthetic')
        jsonschema.validate(report, schema)
        broken = json.loads(json.dumps(report))
        broken['case_likelihood'] = 1.5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)


@pytest.mark.parametrize('task', list(models.TASKS))
def test_dispatch_positive_synthetic_result_uses_reference_dicom(role_job, task):
    job, _ = role_job
    calls = []
    def run(command):
        calls.append(command)
        script = Path(command[1]).name
        if script.startswith('run_'):
            work = Path(command[command.index('--work') + 1])
            reference = sitk.ReadImage(str(work / 'native/t2w.nii.gz'))
            values = np.zeros((5, 32, 40), np.uint8)
            values[1:4, 8:24, 10:28] = 1
            prediction = work / 'restored.nii.gz'
            write_array(values, reference, prediction)
            (work / 'result.json').write_text('{"empty":false}')
            if task.startswith('picai_'):
                (work / 'masks.json').write_text(json.dumps({'csPCa_L01_p0.73': str(prediction)}))
        else:
            subprocess.run(command, check=True)
    models.run_task(task, job, ROOT, Path('/models'), sys.executable, run)
    output = job / 'output' / f'{task}_RTSTRUCT.dcm'
    assert output.is_file()
    ds = pydicom.dcmread(output)
    expected_name = ('csPCa_L01_p0.73' if task.startswith('picai_') else
                     'Prostate_PZ' if task.startswith('umu_') else 'Prostate158_Tumour')
    assert [roi.ROIName for roi in ds.StructureSetROISequence] == [expected_name]
    references = {str(image.ReferencedSOPInstanceUID) for roi in ds.ROIContourSequence
                  for contour in roi.ContourSequence for image in contour.ContourImageSequence}
    source = {str(pydicom.dcmread(path).SOPInstanceUID) for path in (job / 'input').iterdir()}
    assert references and references <= source
    assert len([c for c in calls if Path(c[1]).name == 'dicom_series_to_nifti.py']) == (1 if task.startswith('umu_') else 3)

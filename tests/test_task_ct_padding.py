"""CPU-only dispatch checks: stop at the model command, without weights or inference imports."""
import ast
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import ct_padding
import dicom_series
import lnctvseg_bundle
import model_preflight
import monai_bundle
import pydicom
from nv_reason_ct_report import hu_check
from test_ct_padding import AFFINE, go_sim_ct

ROOT = Path(__file__).parents[1]
EXPECTED = {
    'bamf_liver_lesions_ct', 'lnctvseg_nnunet_v1', 'medical_sam3_3d_prompted',
    'medsam_vit_b_box_2d', 'monai_whole_body_ct', 'myeloma_spine_verse20_early_ct',
    'myeloma_spine_verse20_mm_ct', 'myeloma_spine_verse20_stable_ct',
    'nninteractive_v1_prompted', 'sam_med2d_prompted', 'segvol_prompted_ct',
    'voxtell_v1_1_text_prompted',
}


def test_declared_set_and_preflight_requirements():
    assert ct_padding.CT_PADDING_TASKS == EXPECTED
    for task in EXPECTED:
        assert model_preflight.REQUIREMENTS[task].count('@ct_padding.py') == 1
    for task in {'pam_prompted', 'sam_med3d_turbo_prompted'}:
        assert task not in ct_padding.CT_PADDING_TASKS
        assert '@ct_padding.py' not in model_preflight.REQUIREMENTS[task]


@pytest.mark.parametrize('task', sorted(EXPECTED))
@pytest.mark.parametrize('padded', [True, False])
def test_dispatch_model_receives_only_exterior_replacement(tmp_path, monkeypatch, capsys, task, padded):
    """Execute each real dispatch branch, replacing child processes with a conversion fixture and model probe."""
    vol = go_sim_ct(size=64, slices=6).astype(np.int16)
    exterior = vol == -8192
    if not padded:
        vol[exterior] = -1000
    vol[28:32, 28:32, 2:4] = -8192  # interior low values never become air
    expected = vol.copy()
    if padded:
        expected[exterior] = -1024
    fixture = tmp_path / 'fixture.nii.gz'
    nib.save(nib.Nifti1Image(vol, AFFINE), fixture)
    original_bytes = fixture.read_bytes()
    job = tmp_path / 'job'
    (job / 'input').mkdir(parents=True)
    source = job / 'input/synthetic.dcm'
    source.write_bytes(b'synthetic source sentinel')
    metadata = tmp_path / 'no-models/weights/monai-bundles/wholeBody_ct_segmentation/configs/metadata.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({'network_data_format': {'outputs': {'pred': {'channel_def': {'0': 'background', '1': 'synthetic'}}}}}))
    (job / 'prompt.json').write_text('{"positive_points": [[30, 30, 3]]}')
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(tmp_path / 'no-models'))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(tmp_path / 'no-runtimes'))
    monkeypatch.setenv('RTSEG_SCRATCH_ROOT', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', [str(ROOT / 'run_task.py'), str(job), task,
                                     *(['--nodal-acceptance-audit'] if task == lnctvseg_bundle.TASK else [])])
    monkeypatch.setattr(pydicom, 'dcmread', lambda *a, **k: SimpleNamespace(Modality='CT'))
    monkeypatch.setattr(dicom_series, 'ordered_dicom_files', lambda *a: [str(source)])
    monkeypatch.setattr(monai_bundle, 'compatible_config', lambda *a: tmp_path / 'config.json')
    seen = {}

    class ModelReached(Exception):
        pass

    class Process:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            if cmd[1] == str(ROOT / 'dicom_series_to_nifti.py'):
                path = Path(cmd[3]); shutil.copyfile(fixture, path)
                sidecar = Path(str(path) + '.orientation.json')
                sidecar.write_bytes(b'synthetic orientation sentinel')
                seen.update(path=path, sidecar=sidecar, stat=path.stat())
            elif cmd[1] == str(ROOT / 'orientation_contract.py'):
                assert cmd[cmd.index('--model-input') + 1] == str(seen['path'])
            else:
                if '--input-dir' in cmd:
                    path = Path(cmd[cmd.index('--input-dir') + 1]) / 'case_0000.nii.gz'
                elif '--input' in cmd:
                    path = Path(cmd[cmd.index('--input') + 1])
                    if cmd[1] == str(ROOT / 'run_lnctvseg.py'):
                        path /= 'case_0000.nii.gz'
                elif '--datalist' in cmd:
                    entries = ast.literal_eval(cmd[cmd.index('--datalist') + 1])
                    path = Path(entries[0]['image'] if isinstance(entries[0], dict) else entries[0])
                else:
                    pytest.fail(f'Unexpected command before model: {cmd}')
                assert path == seen['path']
                image = nib.load(path)
                assert np.array_equal(np.asanyarray(image.dataobj), expected)
                assert image.shape == vol.shape and np.array_equal(image.affine, AFFINE)
                assert nib.aff2axcodes(image.affine) == nib.aff2axcodes(AFFINE)
                assert seen['sidecar'].read_bytes() == b'synthetic orientation sentinel'
                if not padded:
                    assert path.read_bytes() == original_bytes
                    assert path.stat().st_mtime_ns == seen['stat'].st_mtime_ns
                seen['model'] = cmd
                raise ModelReached

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr('subprocess.Popen', Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # A short scratch path inside this fixture, even when its parent path is long.
        monkeypatch.setattr(tempfile, 'tempdir', f'/proc/self/fd/{fd}')
        with pytest.raises(ModelReached):
            runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    finally:
        os.close(fd)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"task":')]
    assert records == [{'task': task, 'ct_padding_replaced_voxels': int(exterior.sum()) if padded else 0,
                        'ct_padding_residual_low_voxels': 0, 'ct_padding_fill_hu': -1024}]
    assert 'model' in seen
    assert source.read_bytes() == b'synthetic source sentinel'
    assert fixture.read_bytes() == original_bytes


def dispatch_run(namespace):
    """Load the real command wrapper alone so unrelated task branches need no model fixtures."""
    namespace.setdefault('sys', sys)
    tree = ast.parse((ROOT / 'run_task.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'run_task.py'), 'exec'), namespace)
    return namespace['run']


# Exercise every preflight task outside the extension, including CT, MR, PET and both class-D tasks.
@pytest.mark.parametrize('task', sorted(set(model_preflight.REQUIREMENTS) - EXPECTED))
def test_unlisted_dispatch_never_reads_or_writes_input(tmp_path, monkeypatch, task):
    class Process:
        def __init__(self, *args, **kwargs):
            pass
        def wait(self):
            return 0
    def forbidden(*args, **kwargs):
        pytest.fail('Unlisted task accessed the padding writer')
    path = tmp_path / 'input.nii.gz'
    path.write_bytes(b'byte-identical CT/MR/PET sentinel')
    before = path.stat()
    namespace = dict(os=os, subprocess=SimpleNamespace(Popen=Process), TEMP_ROOT=tmp_path,
                     command_stage=lambda cmd: 'conversion', progress=SimpleNamespace(emit=lambda *a, **k: None),
                     task=task, CT_PADDING_TASKS=ct_padding.CT_PADDING_TASKS, APP_ROOT=ROOT,
                     acquired_ct_model_input=forbidden)
    monkeypatch.setattr(nib, 'load', forbidden)
    dispatch_run(namespace)([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), 'source', str(path)])
    assert path.read_bytes() == b'byte-identical CT/MR/PET sentinel'
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_failed_conversion_never_runs_padding(tmp_path):
    namespace = dict(os=os, subprocess=SimpleNamespace(Popen=lambda *a, **k: SimpleNamespace(wait=lambda: 9)),
                     TEMP_ROOT=tmp_path, command_stage=lambda cmd: 'conversion',
                     progress=SimpleNamespace(emit=lambda *a, **k: None))
    with pytest.raises(SystemExit) as error:
        dispatch_run(namespace)([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), 'source', 'missing'])
    assert error.value.code == 9


@pytest.mark.parametrize('kind', ['fully_padded', 'interior_corrupt', 'lost_slice'])
def test_invalid_low_values_keep_existing_hu_refusal(tmp_path, kind):
    vol = go_sim_ct(size=64, slices=6).astype(np.int16)
    if kind == 'fully_padded':
        vol[:] = -8192
    elif kind == 'interior_corrupt':
        vol[20:44, 20:44, :] = -8192
    else:
        vol[:, :, 2] = -8192
    assert not hu_check(vol)['looks_like_hu']
    path = tmp_path / 'input.nii.gz'
    count = ct_padding.ct_model_input(nib.Nifti1Image(vol, AFFINE), path)
    got = np.asanyarray(nib.load(path).dataobj) if count else vol
    assert not hu_check(got)['looks_like_hu']
    if kind == 'fully_padded':
        assert count == 0 and not path.exists()
    elif kind == 'interior_corrupt':
        assert np.all(got[20:44, 20:44, :] == -8192)
    else:
        assert np.all(got[:, :, 2] == -8192)


def test_nv_segment_uses_shared_implementation():
    import run_nv_segment
    assert run_nv_segment.ct_model_input is ct_padding.ct_model_input
    assert run_nv_segment.axial_axis is ct_padding.axial_axis


def test_corrupt_nifti_is_refused_without_rewriting(tmp_path, capsys, monkeypatch):
    path = tmp_path / 'broken.nii.gz'
    path.write_bytes(b'corrupt synthetic NIfTI')
    monkeypatch.setattr(pydicom, 'dcmread', lambda *a, **k: SimpleNamespace(Modality='CT'))
    monkeypatch.setattr(dicom_series, 'ordered_dicom_files', lambda *a: ['synthetic.dcm'])
    before = path.stat()
    namespace = dict(os=os, subprocess=SimpleNamespace(Popen=lambda *a, **k: SimpleNamespace(wait=lambda: 0)),
                     TEMP_ROOT=tmp_path, command_stage=lambda cmd: 'conversion',
                     progress=SimpleNamespace(emit=lambda *a, **k: None),
                     task='monai_whole_body_ct', CT_PADDING_TASKS=ct_padding.CT_PADDING_TASKS,
                     CT_AIR_HU=ct_padding.CT_AIR_HU, acquired_ct_model_input=ct_padding.acquired_ct_model_input,
                     APP_ROOT=ROOT, json=json, pydicom=pydicom)
    with pytest.raises(nib.filebasedimages.ImageFileError):
        dispatch_run(namespace)([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), 'source', str(path)])
    assert path.read_bytes() == b'corrupt synthetic NIfTI'
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert not capsys.readouterr().out


@pytest.mark.parametrize('task', sorted(EXPECTED))
@pytest.mark.parametrize('modality', ['MR', 'PT'])
def test_listed_task_non_ct_series_is_byte_identical(tmp_path, monkeypatch, capsys, task, modality):
    path = tmp_path / 'input.nii.gz'
    nib.save(nib.Nifti1Image(go_sim_ct(size=64, slices=6).astype(np.int16), AFFINE), path)
    original, stat = path.read_bytes(), path.stat()
    def forbidden(*args, **kwargs):
        pytest.fail('Non-CT input accessed the padding writer')
    monkeypatch.setattr(pydicom, 'dcmread', lambda *a, **k: SimpleNamespace(Modality=modality))
    monkeypatch.setattr(dicom_series, 'ordered_dicom_files', lambda *a: ['synthetic.dcm'])
    monkeypatch.setattr(nib, 'load', forbidden)
    namespace = dict(os=os, subprocess=SimpleNamespace(Popen=lambda *a, **k: SimpleNamespace(wait=lambda: 0)),
                     TEMP_ROOT=tmp_path, command_stage=lambda cmd: 'conversion',
                     progress=SimpleNamespace(emit=lambda *a, **k: None), task=task,
                     CT_PADDING_TASKS=ct_padding.CT_PADDING_TASKS, acquired_ct_model_input=forbidden,
                     APP_ROOT=ROOT, pydicom=pydicom)
    dispatch_run(namespace)([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), 'source', str(path)])
    assert path.read_bytes() == original and path.stat().st_mtime_ns == stat.st_mtime_ns
    assert not capsys.readouterr().out


@pytest.mark.parametrize('acquisition,slices', [('axial-artefact', 6), ('oblique', 40)] +
                         [(plane, n) for plane in ('coronal', 'sagittal') for n in (6, 24, 40, 96)])
def test_real_dicom_dispatch_preserves_nonfill_and_handles_acquisition_planes(
        synthetic_ct, tmp_path, capsys, acquisition, slices):
    """Run actual conversion and the dispatch wrapper; no model subprocess is reached."""
    import copy
    import subprocess
    from pydicom.uid import generate_uid

    y, x = np.mgrid[:256, :256] - 127.5
    exterior = np.hypot(x, y) > 124
    volume = np.repeat(np.where(exterior, -8192, 40).astype(np.int16)[:, :, None], slices, axis=2)
    if acquisition == 'axial-artefact':
        volume[127, 4, :] = -2000  # six in-field voxels adjoining exterior fill
    orientation, step = {
        'axial-artefact': ([1, 0, 0, 0, 1, 0], [0, 0, 2]),
        'coronal': ([1, 0, 0, 0, 0, -1], [0, 2, 0]),
        'sagittal': ([0, 1, 0, 0, 0, 1], [2, 0, 0]),
        'oblique': ([1, 0, 0, 0, .5, -.866025403784], [0, 1.732050807568, 1]),
    }[acquisition]
    source = tmp_path / 'acquisition'
    source.mkdir()
    template = pydicom.dcmread(next(synthetic_ct.iterdir()))
    template.SeriesInstanceUID = generate_uid()
    template.Rows = template.Columns = 256
    template.ImageOrientationPatient = orientation
    template.PixelPaddingValue = -8192
    template['PixelPaddingValue'].VR = 'SS'
    for k in range(volume.shape[2]):
        ds = copy.deepcopy(template)
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.ImagePositionPatient = (np.asarray(step) * k).tolist()
        ds.InstanceNumber = k + 1
        ds.PixelData = volume[:, :, k].T.tobytes()
        ds.save_as(source / f'{k}.dcm', enforce_file_format=True)
    output = tmp_path / 'converted.nii.gz'
    namespace = dict(os=os, subprocess=subprocess, TEMP_ROOT=tmp_path,
                     command_stage=lambda _: 'conversion', progress=SimpleNamespace(emit=lambda *a, **k: None),
                     task='monai_whole_body_ct', CT_PADDING_TASKS=ct_padding.CT_PADDING_TASKS,
                     CT_AIR_HU=ct_padding.CT_AIR_HU, acquired_ct_model_input=ct_padding.acquired_ct_model_input,
                     APP_ROOT=ROOT, json=json, pydicom=pydicom)
    dispatch_run(namespace)([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(source), str(output)])
    saved = nib.load(output)
    actual = np.asanyarray(saved.dataobj)
    assert np.count_nonzero(actual == -8192) == 0
    assert np.count_nonzero(actual == -1024) == int(exterior.sum()) * slices
    assert np.count_nonzero(actual == -2000) == (slices if acquisition == 'axial-artefact' else 0)
    assert np.count_nonzero(actual == 40) == np.count_nonzero(volume == 40)
    if acquisition == 'coronal':
        assert ''.join(nib.aff2axcodes(saved.affine)) == 'LIP'
        assert ct_padding.axial_axis(saved.affine) == 1  # acquired slices remain on axis 2
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"task":')]
    assert records == [{'task': 'monai_whole_body_ct', 'ct_padding_replaced_voxels': int(exterior.sum()) * slices,
                        'ct_padding_residual_low_voxels': slices if acquisition == 'axial-artefact' else 0, 'ct_padding_fill_hu': -1024}]


@pytest.mark.parametrize('kind', ['two-fills', 'float-per-slice', 'ellipse', 'letterbox', 'no-fill', 'two-values-per-slice'])
def test_real_dicom_dispatch_per_slice_values_and_false_detections(synthetic_ct, tmp_path, capsys, kind):
    import copy
    import subprocess
    from pydicom.uid import generate_uid

    yy, xx = np.mgrid[:256, :256] - 127.5
    distance = np.hypot(xx, yy)
    exterior = distance > 124
    plane = np.where(exterior, -8192, -1000).astype(np.int16)
    plane[100:156, 100:156] = 40
    if kind == 'ellipse':
        plane = np.where((xx / 100) ** 2 + (yy / 70) ** 2 <= 1, 40, -3024).astype(np.int16)
    if kind == 'letterbox':
        plane[:] = -1000; plane[100:156, 100:156] = 40
        plane[:10] = plane[-10:] = -2000
    if kind == 'no-fill':
        plane[exterior] = -1000
    ring = exterior & (distance < 135)
    if kind == 'two-values-per-slice':
        plane[ring] = -3024
    source = tmp_path / 'source'; source.mkdir()
    template = pydicom.dcmread(next(synthetic_ct.iterdir()))
    template.Rows = template.Columns = 256
    template.SeriesInstanceUID = generate_uid()
    template.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    for k in range(20):
        ds = copy.deepcopy(template)
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.ImagePositionPatient = [0, 0, 2 * k]; ds.InstanceNumber = k + 1
        pixels = plane.copy()
        if kind == 'two-fills' and k >= 12:
            pixels[exterior] = -3024
        ds.RescaleSlope = 1 + 1e-4 * k if kind == 'float-per-slice' else 1
        ds.RescaleIntercept = 0
        ds.PixelData = pixels.T.tobytes()
        ds.save_as(source / f'{k}.dcm', enforce_file_format=True)
    output = tmp_path / 'input.nii.gz'
    seen = {}
    def prepare(image, path, slice_axis):
        assert slice_axis == 2 and image.shape[2] == 20
        before, mtime = Path(path).read_bytes(), Path(path).stat().st_mtime_ns
        result = ct_padding.acquired_ct_model_input(image, path, slice_axis)
        if kind in {'ellipse', 'letterbox', 'no-fill'}:
            assert Path(path).read_bytes() == before and Path(path).stat().st_mtime_ns == mtime
        seen['result'] = result
        return result
    namespace = dict(os=os, subprocess=subprocess, TEMP_ROOT=tmp_path,
                     command_stage=lambda _: 'conversion', progress=SimpleNamespace(emit=lambda *a, **k: None),
                     task='monai_whole_body_ct', CT_PADDING_TASKS=ct_padding.CT_PADDING_TASKS,
                     CT_AIR_HU=ct_padding.CT_AIR_HU, acquired_ct_model_input=prepare,
                     APP_ROOT=ROOT, json=json, pydicom=pydicom)
    dispatch_run(namespace)([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(source), str(output)])
    got = np.asanyarray(nib.load(output).dataobj)
    captured = capsys.readouterr()
    if kind in {'two-fills', 'float-per-slice'}:
        assert seen['result'] == (int(exterior.sum()) * 20, 0)
        assert not (got < -1500).any()
    elif kind == 'two-values-per-slice':
        residual = int(ring.sum()) * 20
        assert seen['result'] == (int((exterior & ~ring).sum()) * 20, residual)
        assert (got == -3024).sum() == residual
        assert f'WARNING: ct_padding_residual_low_voxels={residual}' in captured.err
    else:
        assert seen['result'] == (0, 0)
    records = [json.loads(line) for line in captured.out.splitlines() if line.startswith('{"task":')]
    assert records[0]['ct_padding_residual_low_voxels'] == seen['result'][1]


@pytest.mark.parametrize('coronal', [False, True])
def test_nv_segment_replaced_counts_equal_base(tmp_path, coronal):
    yy, xx = np.mgrid[:256, :256] - 127.5
    exterior = np.hypot(xx, yy) > 124
    depth = 200 if coronal else 20
    data = np.repeat(np.where(exterior, -8192, -1000).astype(np.int16)[:, :, None], depth, 2)
    affine = AFFINE
    if coronal:
        affine = np.array([[-1., 0, 0, 0], [0, 0, -2., 0], [0, -1., 0, 0], [0, 0, 0, 1.]])
    else:
        data[:, :, 12:][np.broadcast_to(exterior[:, :, None], data[:, :, 12:].shape)] = -3024
    output = tmp_path / 'nv.nii.gz'
    # Counts measured with the base NV-Segment function on these same volumes.
    assert ct_padding.ct_model_input(nib.Nifti1Image(data, affine), output) == (0 if coronal else 344320)
    if coronal:
        assert not output.exists()
    else:
        assert not (np.asanyarray(nib.load(output).dataobj) < -1500).any()

"""CT reconstruction-field padding in the NV-Segment model input (no FastAPI needed)."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import run_nv_segment as runner
import ct_padding
import nv_reason_ct_report
from nv_segment_logic import MODES

ROOT = Path(__file__).parents[1]
AFFINE = np.diag([-1.171875, -1.171875, 1.5, 1.0])


def go_sim_ct(size=96, slices=12, air_sd=0.0, offset=0.0, seed=0):
    """Axial slices like SOMATOM go.Sim: -8192 outside the reconstruction circle, (noisy) air, a body with bone."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    r = np.hypot(x, y)
    vol = np.repeat(np.where(r < size * 0.35, 40.0, -1000.0)[:, :, None], slices, axis=2)
    vol[np.repeat((r < size * 0.08)[:, :, None], slices, axis=2)] = 900.0
    vol += rng.normal(0, air_sd, vol.shape) * (vol == -1000.0) + offset
    vol[np.repeat((r > size / 2 - 0.5)[:, :, None], slices, axis=2)] = -8192.0
    return np.rint(vol)


# ---------------------------------------------------------------- NV-Segment CT model input
def test_only_exterior_padding_becomes_air(tmp_path):
    vol = go_sim_ct(air_sd=55)
    vol[40:46, 40:46, 4:8] = -8192  # interior corruption stays visible to the model
    image = nib.Nifti1Image(vol.astype(np.int16), AFFINE)
    exterior = int((vol == -8192).sum()) - 6 * 6 * 4
    out = tmp_path / 'ct_model_input.nii.gz'
    assert runner.ct_model_input(image, out) == exterior
    saved = nib.load(out)
    data = np.asanyarray(saved.dataobj)
    assert saved.shape == vol.shape and np.array_equal(saved.affine, image.affine)
    assert saved.get_data_dtype() == np.int16
    changed = data != vol
    assert np.all(vol[changed] == -8192) and np.all(data[changed] == runner.CT_AIR_HU) and changed.sum() == exterior
    assert np.all(data[40:46, 40:46, 4:8] == -8192)
    assert (data < -1100).sum() == ((vol < -1100) & (vol > -1500)).sum() + 6 * 6 * 4  # air noise untouched


def test_nothing_written_without_exterior_padding(tmp_path):
    vol = go_sim_ct(air_sd=55)
    vol[vol == -8192] = -1000
    out = tmp_path / 'ct_model_input.nii.gz'
    assert runner.ct_model_input(nib.Nifti1Image(vol.astype(np.int16), AFFINE), out) == 0
    assert not out.exists()


def test_scaled_unsigned_storage_and_large_values(tmp_path):
    vol = go_sim_ct()
    stored = (vol + 8192).astype(np.uint16)  # unsigned storage, padding stored as 0, intercept -8192
    src = nib.Nifti1Image(stored, AFFINE); src.header.set_slope_inter(1, -8192)
    nib.save(src, tmp_path / 'scaled.nii.gz')
    out = tmp_path / 'a.nii.gz'
    assert runner.ct_model_input(nib.load(tmp_path / 'scaled.nii.gz'), out) == int((vol == -8192).sum())
    data = np.asanyarray(nib.load(out).dataobj)
    assert np.array_equal(data, np.where(vol == -8192, runner.CT_AIR_HU, vol))
    big = vol.copy(); big[48, 48, 3] = 40000
    out2 = tmp_path / 'b.nii.gz'
    runner.ct_model_input(nib.Nifti1Image(big.astype(np.float64), AFFINE), out2)
    assert nib.load(out2).get_data_dtype() == np.float64 and np.asanyarray(nib.load(out2).dataobj).max() == 40000


def test_image_class_and_grid_preserved(tmp_path):
    affine = np.array([[-0.9765625, 0, 0, 249.51171875], [0, -0.9765625, 0, 381.51171875],
                       [0, 0, 1.25, -612.375], [0, 0, 0, 1]])
    nib.save(nib.Nifti2Image(go_sim_ct().astype(np.int16), affine), tmp_path / 'n2.nii.gz')
    out = tmp_path / 'out.nii.gz'
    image = nib.load(tmp_path / 'n2.nii.gz')
    assert runner.ct_model_input(image, out) > 0
    saved = nib.load(out)
    assert isinstance(saved, nib.Nifti2Image) and np.array_equal(saved.affine, image.affine)


@pytest.mark.parametrize('mode,load_only', [('nv_segment_ctmr_ct_body', False), ('nv_segment_ct_auto', False),
                                            ('nv_segment_ctmr_ct_body', True), ('nv_segment_ctmr_mri_body', False)])
def test_runner_hands_the_bundle_the_replaced_copy(tmp_path, monkeypatch, mode, load_only):
    vol = go_sim_ct(air_sd=55)
    input_path, model = tmp_path / 'input.nii.gz', tmp_path / 'models'
    model.mkdir()
    nib.save(nib.Nifti1Image(vol.astype(np.int16), AFFINE), input_path)
    output = tmp_path / 'job/prediction.nii.gz'
    labels = json.loads((ROOT / 'labels' / f'{mode}.json').read_text())
    indices = json.loads((ROOT / 'nv_segment_manifest.json').read_text())['modes'][mode]['indices']
    seen = {}
    monkeypatch.setattr(runner, 'generate', lambda *args: (labels, indices))
    monkeypatch.setattr(runner, 'prepare_bundle', lambda source, ckpt, work, path, *rest: seen.update(
        path=Path(path), data=np.asanyarray(nib.load(path).dataobj).copy()))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(model))
    def run(command, *, cwd, env):
        (cwd / 'bundle_metrics.json').write_text(json.dumps({'load_only': load_only}))
        if not load_only:
            (cwd / 'output').mkdir()
            pred = np.zeros(vol.shape, np.float32); pred[40:50, 40:50, 2:6] = indices[0]
            nib.save(nib.Nifti1Image(pred, AFFINE), cwd / 'output' / 'case_trans.nii.gz')
    monkeypatch.setattr(runner, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['run_nv_segment.py', '--mode', mode, '--input', str(input_path),
        '--output', str(output), '--labels-out', str(output.parent / 'labels.json'),
        '--model-root', str(model), '--device', 'cpu', *(['--load-only'] if load_only else [])])
    runner.main()
    metrics = json.loads(Path(str(output) + '.metrics.json').read_text())
    padding = int((vol == -8192).sum())
    if MODES[mode]['modality'] == 'CT' and not load_only:
        assert seen['path'].name == 'ct_model_input.nii.gz' and seen['data'].min() > -1500
        assert metrics['ct_padding_replaced_voxels'] == padding and metrics['ct_padding_fill_hu'] == -1024
    else:
        assert seen['path'] == input_path.resolve() and seen['data'].min() == -8192
        assert 'ct_padding_replaced_voxels' not in metrics
    assert not list(output.parent.glob('nv-segment-*'))


# ---------------------------------------------------------------- shared detector
def test_report_uses_the_same_detector():
    assert nv_reason_ct_report.exterior_padding is ct_padding.exterior_padding
    assert nv_reason_ct_report.PADDING_BELOW_HU == ct_padding.PADDING_BELOW_HU == -1500


# ---------------------------------------------------------------- axis order, aliasing, precision
@pytest.mark.parametrize('order', [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_axis_order_of_a_physically_equal_image_does_not_matter(tmp_path, order):
    vol = go_sim_ct(size=64, slices=10, air_sd=55).astype(np.int16)
    data = np.transpose(vol, order)
    affine = np.eye(4); affine[:3, :3] = AFFINE[:3, :3][:, list(order)]
    out = tmp_path / 'p.nii.gz'
    assert runner.axial_axis(affine) == order.index(2)
    assert runner.ct_model_input(nib.Nifti1Image(data, affine), out) == int((vol == -8192).sum())
    back = np.transpose(np.asanyarray(nib.load(out).dataobj), np.argsort(order))
    assert np.array_equal(back, np.where(vol == -8192, runner.CT_AIR_HU, vol))


def test_caller_array_is_not_modified(tmp_path):
    a = go_sim_ct().astype(np.int16)
    before = a.copy()
    image = nib.Nifti1Image(a, AFFINE)
    assert runner.ct_model_input(image, tmp_path / 'p.nii.gz') > 0
    assert np.array_equal(a, before) and np.array_equal(np.asanyarray(image.dataobj), before)


def test_fractional_and_scaled_values_are_kept_exactly(tmp_path):
    vol = go_sim_ct(air_sd=55)
    frac = vol + np.where(vol == -8192, 0, 0.123456789)
    out = tmp_path / 'f.nii.gz'
    runner.ct_model_input(nib.Nifti1Image(frac, AFFINE), out)
    got = np.asanyarray(nib.load(out).dataobj)
    keep = vol != -8192
    assert got.dtype == np.float64 and np.array_equal(got[keep], frac[keep]) and np.all(got[~keep] == runner.CT_AIR_HU)
    stored = np.where(vol == -8192, 0, np.rint((vol + 2048) * 10)).astype(np.uint16)
    src = nib.Nifti1Image(stored, AFFINE); src.header.set_slope_inter(0.1, -2048)
    nib.save(src, tmp_path / 's.nii.gz')
    proxy = nib.load(tmp_path / 's.nii.gz')
    expected = np.asanyarray(proxy.dataobj)
    out2 = tmp_path / 's_out.nii.gz'
    runner.ct_model_input(proxy, out2)
    got2 = np.asanyarray(nib.load(out2).dataobj)
    pad = stored == 0
    assert np.array_equal(got2[~pad], expected[~pad]) and np.all(got2[pad] == runner.CT_AIR_HU)


@pytest.mark.parametrize('extra,dtype', [(40000, np.int32), (4294967303, np.int64)])
def test_integer_values_beyond_int16_keep_their_value(tmp_path, extra, dtype):
    vol = go_sim_ct().astype(np.int64)
    vol[48, 48, 3] = extra
    out = tmp_path / 'i.nii.gz'
    assert runner.ct_model_input(nib.Nifti1Image(vol, AFFINE, dtype=np.int64), out) == int((vol == -8192).sum())
    got = np.asanyarray(nib.load(out).dataobj)
    assert got.dtype == dtype and got[48, 48, 3] == extra
    assert np.array_equal(got, np.where(vol == -8192, runner.CT_AIR_HU, vol))


@pytest.mark.parametrize('pair,single', [(nib.Nifti1Pair, nib.Nifti1Image), (nib.Nifti2Pair, nib.Nifti2Image)])
def test_paired_nifti_becomes_its_single_file_counterpart(tmp_path, pair, single):
    vol = go_sim_ct().astype(np.int16)
    nib.save(pair(vol, AFFINE), tmp_path / 'p.img')
    image = nib.load(tmp_path / 'p.img')
    out = tmp_path / 'ct_model_input.nii.gz'
    assert runner.ct_model_input(image, out) == int((vol == -8192).sum())
    saved = nib.load(out)
    assert type(saved) is single and np.array_equal(saved.affine, image.affine)
    assert np.array_equal(np.asanyarray(saved.dataobj), np.where(vol == -8192, runner.CT_AIR_HU, vol))

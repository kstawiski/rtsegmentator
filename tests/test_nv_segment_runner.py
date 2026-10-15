"""Exercise copied-bundle patches and export checks without inference libraries."""
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import run_nv_segment as runner
from nv_segment_logic import MODES

ROOT = Path(__file__).parents[1]


def tiny_bundle(tmp_path, monkeypatch):
    source, work = tmp_path / 'source', tmp_path / 'work'
    (source / 'configs').mkdir(parents=True)
    (source / 'scripts').mkdir()
    work.mkdir()
    config = dict(initialize=['$prepare_huggingface_checkpoint()', '$network', '$evaluator'],
                  checkpointloader={'strict': False}, evaluator={'amp': True},
                  postprocessing={'transforms': [{'_target_': 'Lambdad',
                      'func': '$lambda x: torch.nan_to_num(x, nan=255)'}]})
    (source / 'configs/inference.json').write_text(json.dumps(config))
    (source / 'scripts/evaluator.py').write_text('points = torch.from_numpy(points).to(inputs.device)\n')
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'synthetic checkpoint')
    mode = 'nv_segment_ctmr_mri_brain'
    monkeypatch.setitem(MODES, mode, {**MODES[mode], 'sha256': runner.sha256(checkpoint)})
    manifest = {'source_files': {str(f.relative_to(source)): runner.sha256(f)
                                for f in source.rglob('*') if f.is_file()}}
    return source, work, checkpoint, mode, manifest


def test_prepare_bundle_preserves_source_and_patches_only_job(tmp_path, monkeypatch):
    source, work, checkpoint, mode, manifest = tiny_bundle(tmp_path, monkeypatch)
    before = {str(f): f.read_bytes() for f in source.rglob('*') if f.is_file()}
    cfg = runner.prepare_bundle(source, checkpoint, work, tmp_path / 'image.nii.gz', mode, 'cpu', manifest)
    assert {str(f): f.read_bytes() for f in source.rglob('*') if f.is_file()} == before
    assert cfg['modality'] == 'MRI_BRAIN' and cfg['checkpointloader']['strict']
    assert not cfg['evaluator']['amp'] and cfg['device'] == "$torch.device('cpu')"
    assert cfg['output_dtype'] == '$np.uint16'
    assert 'nan=0' in cfg['postprocessing']['transforms'][0]['func']
    assert not any('huggingface' in x for x in cfg['initialize'])
    assert 'torch.as_tensor' in (work / 'scripts/evaluator.py').read_text()
    assert (work / 'models/model.pt').resolve() == checkpoint.resolve()


@pytest.mark.parametrize('tamper', ['source', 'checkpoint'])
def test_prepare_rejects_modified_pins_before_copying(tmp_path, monkeypatch, tamper):
    source, work, checkpoint, mode, manifest = tiny_bundle(tmp_path, monkeypatch)
    target = checkpoint if tamper == 'checkpoint' else source / 'scripts/evaluator.py'
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        runner.prepare_bundle(source, checkpoint, work, tmp_path / 'image.nii.gz', mode, 'cuda', manifest)
    assert not list(work.iterdir())


def test_caches_and_home_are_job_local(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/unrelated/overlay')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    env = runner.job_environment(tmp_path, 'cuda')
    for key in ['HOME', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR',
                'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'NUMBA_CACHE_DIR', 'TMPDIR', 'TEMP', 'TMP']:
        assert Path(env[key]).is_relative_to(tmp_path)
    assert 'PYTHONPATH' not in env and env['CUDA_VISIBLE_DEVICES'] == '2'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == env['PYTHONDONTWRITEBYTECODE'] == '1'
    assert runner.job_environment(tmp_path, 'cpu')['CUDA_VISIBLE_DEVICES'] == ''


def test_reject_bundle_write_including_symlink_ancestors(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (tmp_path / 'alias').symlink_to(bundle)
    for path in [bundle / 'new/output.nii.gz', tmp_path / 'alias/out.nii.gz']:
        with pytest.raises(ValueError, match='outside'):
            runner.check_write_paths([path], [bundle])
    runner.check_write_paths([tmp_path / 'job/out.nii.gz'], [bundle])


@pytest.mark.parametrize('failure', [None, 'grid', 'fractional', 'unknown', 'empty'])
def test_runner_export_and_cleanup_without_inference(tmp_path, monkeypatch, failure):
    # Actual runner main/export; publisher execution is replaced with one small synthetic volume.
    import sys
    mode = 'nv_segment_ctmr_mri_brain'
    model, output = tmp_path / 'models', tmp_path / 'job/prediction.nii.gz'
    model.mkdir()
    input_path = tmp_path / 'input.nii.gz'
    affine = np.diag([1., 2., 3., 1.])
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.uint8), affine), input_path)
    labels = json.loads((ROOT / 'labels' / f'{mode}.json').read_text())
    manifest = json.loads((ROOT / 'nv_segment_manifest.json').read_text())['modes'][mode]
    # Use the brain label set but bypass its separate registration subprocesses;
    # inverse-grid patch execution is tested below.
    monkeypatch.setitem(MODES, mode, {**MODES[mode], 'upstream_mode': 'MRI_BODY'})
    monkeypatch.setattr(runner, 'generate', lambda *args: (labels, manifest['indices']))
    monkeypatch.setattr(runner, 'prepare_bundle', lambda *args: None)
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(model))
    def run(command, *, cwd, env):
        (cwd / 'bundle_metrics.json').write_text('{"load_only":false,"cuda_available":false}')
        out = cwd / 'output'
        out.mkdir()
        data = np.zeros((4, 5, 6), np.float32)
        data[1:3, 1:4, 1:3] = 255
        data[1:3, 1:4, 3:5] = 345
        if failure == 'fractional': data[1, 1, 1] = 254.3
        if failure == 'unknown': data[1, 1, 1] = 1
        if failure == 'empty': data[:] = 0
        prediction_affine = affine.copy()
        if failure == 'grid': prediction_affine[0, 3] += 1
        nib.save(nib.Nifti1Image(data, prediction_affine), out / 'case_trans.nii.gz')
    monkeypatch.setattr(runner, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['run_nv_segment.py', '--mode', mode, '--input', str(input_path),
        '--output', str(output), '--labels-out', str(output.parent / 'labels.json'),
        '--model-root', str(model), '--device', 'cpu'])
    if failure:
        with pytest.raises(ValueError): runner.main()
        assert not output.exists()
    else:
        runner.main()
        image = nib.load(output)
        assert image.get_data_dtype() == np.dtype('uint16')
        assert np.array_equal(image.affine, affine)
        assert set(np.unique(np.asarray(image.dataobj))) == {0, 255, 345}
        assert json.loads(Path(str(output) + '.metrics.json').read_text())['exact_input_grid'] is True
    assert not list(output.parent.glob('nv-segment-*'))


def test_inverse_patch_preserves_high_labels_with_uint8_header(tmp_path):
    # Run the exact patched writer statements on a uint8 reference header.
    text = '''if True:
        reverted_mask_npy = resampling(inv_mask_npy, tar_pixdim, original_pixdim, order=0)
        mask_out_header = out_header.copy()
        out_mask_nib = nib.Nifti1Image(reverted_mask_nib.get_fdata(), original_affine, header=mask_out_header)
'''
    patched = runner.patch_brain_revert_text(text)
    original = nib.Nifti1Image(np.zeros((3, 4, 5), np.uint8), np.eye(4))
    data = np.zeros(original.shape, np.uint16)
    data[0, 0, 0], data[1, 1, 1] = 255, 345
    calls = []
    def zoom(array, factors, **kwargs):
        calls.append((factors, kwargs))
        return array
    ns = dict(np=np, nib=nib, zoom=zoom, inv_mask_npy=data, original_reoriented=original,
              out_header=original.header, original_affine=original.affine,
              reverted_mask_nib=nib.Nifti1Image(data, original.affine))
    exec(patched, ns)
    assert calls[0][1]['order'] == 0 and calls[0][1]['prefilter'] is False
    output = tmp_path / 'labels.nii.gz'
    nib.save(ns['out_mask_nib'], output)
    saved = nib.load(output)
    assert saved.get_data_dtype() == np.dtype('uint16')
    assert np.array_equal(np.asarray(saved.dataobj), data)
    with pytest.raises(ValueError, match='anchor mismatch'):
        runner.patch_brain_revert_text('unrecognised source')

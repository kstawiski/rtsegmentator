"""CPU-only configuration/command regressions; no model assets or torch imports."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from monai_bundle import bundle_command, compatible_config


@pytest.mark.parametrize('extension', ['json', 'yaml'])
@pytest.mark.parametrize('cpu', [None, False, True])
@pytest.mark.parametrize('remove_size', [False, True])
def test_job_local_config_preserves_inference_and_official_file(tmp_path, extension, cpu, remove_size):
    bundle = tmp_path / 'bundle'
    (bundle / 'configs').mkdir(parents=True)
    source = bundle / f'configs/inference.{extension}'
    original = {'device': "$torch.device('cuda:0')", 'network': '$@network_def.to(@device)',
                'network_def': {'_target_': 'SwinUNETR', 'img_size': 96, 'out_channels': 14},
                'inferer': {'_target_': 'SlidingWindowInferer', 'device': '@device', 'sw_batch_size': 4},
                'postprocessing': {'_target_': 'Compose', 'transforms': [
                    {'_target_': 'Activationsd', 'keys': 'pred', 'softmax': True},
                    {'_target_': 'Invertd', 'keys': 'pred', 'transform': '@preprocessing',
                     'nearest_interp': False, 'to_tensor': True, 'device': '@device'},
                    {'_target_': 'AsDiscreted', 'keys': 'pred', 'argmax': True},
                    {'_target_': 'SaveImaged', 'keys': 'pred', 'output_dir': '@output_dir'}]}}
    source.write_text(json.dumps(original) if extension == 'json' else yaml.safe_dump(original))
    before = source.read_bytes()
    spec = {'config': str(source.relative_to(bundle)), 'remove_network_img_size': remove_size}
    if cpu is not None:
        spec['postprocess_on_cpu'] = cpu
    work = tmp_path / 'job'
    target = compatible_config(bundle, spec, work)
    expected = copy.deepcopy(original)
    if remove_size:
        del expected['network_def']['img_size']
    if cpu:
        expected['postprocessing']['transforms'][1]['device'] = 'cpu'
        expected['postprocessing']['transforms'].insert(1,
            {'_target_': 'EnsureTyped', 'keys': 'pred', 'device': 'cpu'})
    assert yaml.safe_load(target.read_text()) == expected
    assert source.read_bytes() == before
    if cpu or remove_size:
        assert target == work / 'bundle-overlay/configs/inference-compatible.json'
    else:
        assert target == source and not work.exists()


def test_cpu_flag_refuses_changed_pipeline_without_inversion(tmp_path):
    source = tmp_path / 'inference.json'
    source.write_text(json.dumps({'postprocessing': {'transforms': [
        {'_target_': 'AsDiscreted', 'keys': 'pred', 'argmax': True}]}}))
    with pytest.raises(ValueError, match='inversion/resampling'):
        compatible_config(tmp_path, {'config': source.name, 'postprocess_on_cpu': True}, tmp_path / 'job')
    assert not (tmp_path / 'job').exists()


def test_cpu_transfer_precedes_first_resampling_and_accepts_qualified_target(tmp_path):
    source = tmp_path / 'inference.json'
    source.write_text(json.dumps({'postprocessing': {'transforms': [
        {'_target_': 'monai.transforms.Spacingd', 'keys': ['pred'], 'pixdim': [1, 1, 1]},
        {'_target_': 'monai.transforms.Invertd', 'keys': ['pred'], 'device': 'cuda:0'}]}}))
    output = compatible_config(tmp_path, {'config': source.name, 'postprocess_on_cpu': True}, tmp_path / 'job')
    transforms = json.loads(output.read_text())['postprocessing']['transforms']
    assert transforms[0] == {'_target_': 'EnsureTyped', 'keys': 'pred', 'device': 'cpu'}
    assert transforms[1] == {'_target_': 'monai.transforms.Spacingd', 'keys': ['pred'], 'pixdim': [1, 1, 1]}
    assert transforms[2]['device'] == 'cpu'


@pytest.mark.parametrize('records', [False, True])
def test_command_is_unchanged_apart_from_config_path(records):
    spec = {'datalist_records': records, 'trusted_architecture_pickle': True, 'run_id': 'evaluating',
            'overrides': {'displayable_configs#highres': False, 'n': 2, 'values': [1, 2], 'word': 'example'}}
    for cpu in [False, True]:
        config = Path('/job/inference-compatible.json' if cpu else '/bundle/configs/inference.yaml')
        command = bundle_command('/runtime/python', Path('/bundle'), {**spec, 'postprocess_on_cpu': cpu},
                                 config, Path('/job/input.nii.gz'), Path('/job/inference'))
        assert command == ['/runtime/python', '-m', 'monai.bundle', 'run', '--config_file', str(config),
            '--bundle_root', '/bundle', '--datalist', repr([{'image': '/job/input.nii.gz'}] if records else ['/job/input.nii.gz']),
            '--output_dir', '/job/inference', '--dataloader#num_workers', '0', '--arch_ckpt',
            "$torch.load(@arch_ckpt_path, map_location=torch.device('cuda'), weights_only=False)",
            '--run_id', 'evaluating', '--displayable_configs#highres', 'False', '--n', '2',
            '--values', '[1, 2]', '--word', 'example']


def test_catalog_cpu_decisions():
    catalog = json.loads((Path(__file__).parents[1] / 'catalog/monai-bundles.json').read_text())
    enabled = {'monai_spleen_ct', 'monai_multi_organ_ct', 'monai_pancreas_dints_ct',
               'monai_pediatric_abdominal_ct', 'monai_renal_unest_ct', 'monai_swin_unetr_btcv_ct'}
    assert {name for name, spec in catalog.items() if spec.get('postprocess_on_cpu')} == enabled
    assert {name for name, spec in catalog.items() if not spec.get('postprocess_on_cpu')} == {
        'monai_whole_body_ct', 'monai_prostate_anatomy_mr',
        'monai_ventricular_short_axis_mr', 'monai_whole_brain_unest_mr'}


def test_gpu_comparison_label_metrics_and_empty_labels():
    import numpy as np
    from scripts.compare_monai_postprocessing import compare_arrays
    result = compare_arrays(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), {'1': 'A', '2': 'B', '3': 'Absent'})
    assert result['exact_voxel_agreement'] == .75
    labels = {r['label']: r for r in result['labels']}
    assert labels[1]['dice'] == pytest.approx(2 / 3)
    assert labels[1]['voxel_agreement'] == .75
    assert labels[1]['voxels_off'] == 2 and labels[1]['voxels_on'] == 1
    assert labels[3]['dice'] == labels[3]['voxel_agreement'] == 1
    with pytest.raises(ValueError, match='integer labels'):
        compare_arrays(np.array([0, .5]), np.array([0, 1]), {})


def test_gpu_comparison_rejects_different_physical_grids(tmp_path):
    import nibabel as nib
    import numpy as np
    from scripts.compare_monai_postprocessing import compare_files
    a, b = tmp_path / 'off.nii.gz', tmp_path / 'on.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), np.eye(4)), a)
    affine = np.eye(4)
    affine[0, 3] = 1
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), affine), b)
    with pytest.raises(ValueError, match='affine mismatch'):
        compare_files(a, b, {})


@pytest.mark.parametrize('baseline_fails', [False, True])
def test_gpu_harness_runs_both_flags_and_keeps_failure_evidence(tmp_path, monkeypatch, baseline_fails):
    """Exercise manager orchestration with subprocess doubles, never a real GPU."""
    import sys
    from types import SimpleNamespace
    import nibabel as nib
    import numpy as np
    from scripts import compare_monai_postprocessing as compare
    bundle = tmp_path / 'models/weights/monai-bundles/synthetic'
    (bundle / 'configs').mkdir(parents=True)
    source = bundle / 'configs/inference.json'
    source.write_text(json.dumps({'postprocessing': {'transforms': [
        {'_target_': 'Invertd', 'keys': 'pred', 'transform': '@preprocessing'},
        {'_target_': 'AsDiscreted', 'keys': 'pred', 'argmax': True}]}}))
    original_bytes = source.read_bytes()
    catalog_dir = tmp_path / 'app/catalog'
    catalog_dir.mkdir(parents=True)
    (catalog_dir / 'monai-bundles.json').write_text(json.dumps({'synthetic': {
        'bundle': 'synthetic', 'config': 'configs/inference.json',
        'postprocess_on_cpu': True, 'labels': {'1': 'Example'}}}))
    monkeypatch.setattr(compare, 'APP_ROOT', catalog_dir.parent)
    dicom = tmp_path / 'dicom'
    dicom.mkdir()
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['compare', '--dicom', str(dicom), '--model-root', str(tmp_path / 'models'),
                                    '--runtime-python', sys.executable, '--output', str(output)])
    requests = []
    def run(command, **kwargs):
        if '--worker' not in command:
            Path(command[-1]).touch()  # Synthetic conversion double.
            return SimpleNamespace(returncode=0)
        request = json.loads(Path(command[-1]).read_text())
        requests.append(request)
        mode = Path(request['metrics']).parent.name
        failed = baseline_fails and mode == 'off'
        Path(request['metrics']).write_text(json.dumps({'status': 'failed' if failed else 'ok',
            'peak_allocated_mib': 64, 'peak_reserved_mib': 128, 'bundle_wall_seconds': 1}))
        if not failed:
            pred = Path(request['metrics']).parent / 'inference/input/input_trans.nii.gz'
            pred.parent.mkdir()
            nib.save(nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), np.eye(4)), pred)
        return SimpleNamespace(returncode=1 if failed else 0)
    monkeypatch.setattr(compare.subprocess, 'run', run)
    assert compare.main() == int(baseline_fails)
    assert len(requests) == 2
    off, on = [r['command'] for r in requests]
    # Identical flags except the config and output path specific to each run.
    for key in ('--config_file', '--output_dir'):
        off[off.index(key) + 1] = on[on.index(key) + 1] = '<run-specific>'
    assert off == on
    summary = json.loads((output / 'summary.json').read_text())
    result = summary['tasks'][0]
    assert result['runs']['on']['status'] == 'ok'
    if baseline_fails:
        assert result['comparison']['status'] == 'unavailable'
        assert result['runs']['off']['returncode'] == 1
    else:
        assert result['comparison']['exact_voxel_agreement'] == 1
        assert all(r['dice'] == 1 for r in result['comparison']['labels'])
    assert source.read_bytes() == original_bytes


def test_overlay_keeps_bundle_code_and_metadata_importable(tmp_path):
    bundle = tmp_path / 'bundle'
    (bundle / 'configs').mkdir(parents=True)
    (bundle / 'scripts/networks').mkdir(parents=True)
    (bundle / 'scripts/__init__.py').write_text('')
    (bundle / 'scripts/networks/__init__.py').write_text('VALUE = 7\n')
    (bundle / 'configs/metadata.json').write_text('{}')
    (bundle / 'configs/inference.json').write_text(json.dumps({'postprocessing': {'transforms': [
        {'_target_': 'Invertd', 'keys': 'pred'}]}}))
    target = compatible_config(bundle, {'config': 'configs/inference.json', 'postprocess_on_cpu': True},
                               tmp_path / 'job')
    # MONAI adds the parent of a directory named "configs" to sys.path.
    assert target.parent.name == 'configs'
    root = target.parent.parent
    assert (root / 'scripts').resolve() == (bundle / 'scripts').resolve()
    assert (target.parent / 'metadata.json').resolve() == (bundle / 'configs/metadata.json').resolve()
    assert (target.parent / 'inference.json').resolve() == (bundle / 'configs/inference.json').resolve()
    import subprocess, sys
    probe = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(1, sys.argv[1]); '
                            'import scripts.networks as n; print(n.VALUE)', str(root)],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == '7'

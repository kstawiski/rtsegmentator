"""Synthetic CPU tests for the physical/dummy-grid boundary."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from run_choi_tmli import (AXIS_VARIANTS, TARGET_SPACING, axis_variant, main,
                           network_affine, prepare_input, restore_prediction, target_geometry)


def save(path, values, affine):
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    return path


def source_affine(position, shape):
    directions = {'HFS': np.diag([1., 1., 1.]),
                  'FFS': np.diag([-1., 1., -1.]),
                  'HFP': np.diag([-1., -1., 1.])}
    angle = np.deg2rad(23)
    # Tilt both axial and sagittal planes, not just an in-plane rotation.
    rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    directions['oblique'] = rz @ rx
    affine = np.eye(4)
    affine[:3, :3] = np.diag([-1., -1., 1.]) @ directions[position] @ np.diag(TARGET_SPACING)
    affine[:3, 3] = -affine[:3, :3] @ ((np.array(shape)-1)/2)
    return affine


@pytest.mark.parametrize('variant', AXIS_VARIANTS)
@pytest.mark.parametrize('position', ['HFS', 'FFS', 'HFP', 'oblique'])
def test_lateral_half_model_returns_correct_physical_side_and_exact_source_grid(tmp_path, position, variant):
    shape = (14, 18, 4)
    affine = source_affine(position, shape)
    source = save(tmp_path/'source.nii.gz', np.zeros(shape, dtype=np.float32), affine)
    # The model selects the first storage-x half, independent of header/world axes.
    geometry = prepare_input(source, tmp_path/'network.nii.gz', variant)
    network = nib.load(tmp_path/'network.nii.gz')
    np.testing.assert_array_equal(network.affine, network_affine())
    assert network.header.get_zooms() == (1., 1., 1.)
    assert network.header.get_xyzt_units()[0] == 'mm'
    model = np.zeros(network.shape, dtype=np.uint8)
    model[:network.shape[0]//2] = 1
    save(tmp_path/'prediction.nii.gz', model, network_affine())
    restore_prediction(tmp_path/'prediction.nii.gz', source, geometry, tmp_path/'restored.nii.gz')
    restored = nib.load(tmp_path/'restored.nii.gz')
    assert restored.shape == shape
    np.testing.assert_array_equal(restored.affine, nib.load(source).affine)
    points = nib.affines.apply_affine(nib.load(source).affine, np.indices(shape).reshape(3, -1).T)
    # LPS identity x increases toward patient left. First half is right unless flipped.
    expected = points[:, 0] > 0 if variant[0] == '0' else points[:, 0] < 0
    np.testing.assert_array_equal(np.asanyarray(restored.dataobj).ravel(), expected.astype(np.uint8))
    # Every source voxel face is within the true target voxel-face box.
    corners = np.array(list(itertools.product(*[(-.5, n-.5) for n in shape])))
    world = nib.affines.apply_affine(nib.load(source).affine, corners)
    target_voxels = nib.affines.apply_affine(np.linalg.inv(geometry['target_affine_ras_mm']), world)
    assert np.all(target_voxels >= -.50001)
    assert np.all(target_voxels <= np.array(network.shape)-.5 + 1e-5)


@pytest.mark.parametrize('variant', AXIS_VARIANTS)
def test_all_storage_axis_flips_inverse_preserves_asymmetric_labels(tmp_path, variant):
    shape = (14, 18, 4)
    affine = source_affine('HFS', shape)
    values = np.ones(shape, dtype=np.float32)
    values[2:6, 3:8, 1:3] = 7
    values[9:12, 12:15, 2:4] = 13
    source = save(tmp_path/'ct.nii.gz', values-1024, affine)
    geometry = prepare_input(source, tmp_path/'network.nii.gz', variant)
    network = nib.load(tmp_path/'network.nii.gz')
    save(tmp_path/'prediction.nii.gz', np.rint(network.get_fdata()).astype(np.uint8), network_affine())
    restore_prediction(tmp_path/'prediction.nii.gz', source, geometry, tmp_path/'restored.nii.gz')
    np.testing.assert_array_equal(nib.load(tmp_path/'restored.nii.gz').get_fdata(), values)
    np.testing.assert_array_equal(nib.load(source).get_fdata(), values-1024)
    assert network.get_fdata()[0, 0, 0] == 0  # -1024 HU padding + offset


def test_target_size_padding_large_fov_and_no_axis_default(monkeypatch, tmp_path):
    image = SimpleNamespace(shape=(512, 512, 735), affine=np.diag([-.977, -.977, 1.5, 1]))
    geometry = target_geometry(image, '000')
    assert geometry['target_shape_xyz'] == [512, 512, 221]
    image = nib.Nifti1Image(np.zeros((620, 4, 3), dtype=np.uint8), np.diag([-1.113, -1.113, 5., 1.]))
    assert target_geometry(image, '111')['target_shape_xyz'] == [620, 512, 3]
    monkeypatch.delenv('CHOI_AXIS_VARIANT', raising=False)
    with pytest.raises(ValueError, match='unconfirmed'):
        axis_variant()
    with pytest.raises(SystemExit) as exc:
        main(['--input', 'missing', '--model', 'missing', '--work', str(tmp_path/'work'), '--output', 'missing'])
    assert exc.value.code == 2 and not (tmp_path/'work').exists()
    monkeypatch.setenv('CHOI_AXIS_VARIANT', '101')
    assert axis_variant() == '101' and axis_variant('010') == '010'


def test_restore_rejects_wrong_grid_and_fractional_labels(tmp_path):
    source = save(tmp_path/'ct.nii.gz', np.zeros((4, 4, 2), dtype=np.float32), np.eye(4))
    geometry = prepare_input(source, tmp_path/'network.nii.gz', '000')
    shape = geometry['target_shape_xyz']
    pred = save(tmp_path/'wrong.nii.gz', np.ones(shape, dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match='network grid'):
        restore_prediction(pred, source, geometry, tmp_path/'out.nii.gz')
    save(pred, np.full(shape, .5, dtype=np.float32), network_affine())
    with pytest.raises(ValueError, match='integer'):
        restore_prediction(pred, source, geometry, tmp_path/'out.nii.gz')


def test_restored_prediction_passes_actual_converter(synthetic_ct, tmp_path):
    affine = np.diag([-1., -1., 2., 1.])
    source = save(tmp_path/'ct.nii.gz', np.zeros((32, 32, 12), dtype=np.int16), affine)
    geometry = prepare_input(source, tmp_path/'network.nii.gz', '101')
    grid = np.indices(geometry['target_shape_xyz']).reshape(3, -1).T
    points = nib.affines.apply_affine(geometry['target_affine_ras_mm'], grid)
    labels = ((points[:, 0] < -8) & (points[:, 0] > -23) & (points[:, 1] < -5) &
              (points[:, 1] > -20) & (points[:, 2] > 4) & (points[:, 2] < 18))
    save(tmp_path/'model.nii.gz', labels.reshape(geometry['target_shape_xyz']).astype(np.uint8), network_affine())
    prediction = tmp_path/'restored.nii.gz'
    restore_prediction(tmp_path/'model.nii.gz', source, geometry, prediction)
    mapping = tmp_path/'labels.json'; mapping.write_text('{"1":"synthetic_target"}')
    result = subprocess.run([sys.executable, str(Path(__file__).parents[1]/'convert_validate_rtstruct.py'),
                             '--prediction', str(prediction), '--model-input', str(source),
                             '--dicom', str(synthetic_ct), '--labels', str(mapping),
                             '--output', str(tmp_path/'rtstruct.dcm')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['valid'] and report['physical_roundtrip']['synthetic_target']['roundtrip_dice'] == 1.0


def test_cli_contract_reaches_predictor_and_restores_without_gpu(tmp_path, monkeypatch):
    import pickle
    import run_choi_tmli as runner
    shape=(14,18,4);affine=source_affine('HFP',shape)
    source=save(tmp_path/'ct.nii.gz',np.zeros(shape,dtype=np.float32),affine)
    model=tmp_path/'model';model.mkdir()
    plans={'plans_per_stage':{0:{'current_spacing':[1.,1.,1.]}},'transpose_forward':[0,1,2]}
    (model/'plans.pkl').write_bytes(pickle.dumps(plans))
    fold=model/'fold_0';fold.mkdir()
    for name in ('model_final_checkpoint.model','model_final_checkpoint.model.pkl'):
        (fold/name).write_bytes(b'synthetic')
    calls=[]
    def predict(command,env,check):
        calls.append(command)
        input_dir=Path(command[command.index('-i')+1]);output_dir=Path(command[command.index('-o')+1])
        image=nib.load(input_dir/'case_0000.nii.gz')
        data=np.zeros(image.shape,dtype=np.uint8);data[:image.shape[0]//2]=1
        save(output_dir/'case.nii.gz',data,image.affine)
        assert command[command.index('--mode')+1]=='normal' and '--disable_tta' in command
        assert Path(env['RESULTS_FOLDER']).is_relative_to(tmp_path/'work')
    monkeypatch.setattr(runner.subprocess,'run',predict)
    monkeypatch.setenv('CHOI_AXIS_VARIANT','110')
    output=tmp_path/'output.nii.gz'
    runner.main(['--input',str(source),'--model',str(model),'--work',str(tmp_path/'work'),
                 '--output',str(output),'--folds','0'])
    assert len(calls)==1 and output.exists()
    np.testing.assert_array_equal(nib.load(output).affine,nib.load(source).affine)
    assert json.loads((tmp_path/'work/geometry.json').read_text())['axis_variant']=='110'
    plans['plans_per_stage'][0]['current_spacing']=[1.,1.,2.]
    (model/'plans.pkl').write_bytes(pickle.dumps(plans))
    with pytest.raises(ValueError,match='full-resolution'):
        runner.validate_plans(model)


def test_linear_hu_sampling_and_exact_air_outside_voxel_support():
    from run_choi_tmli import resample_hu
    grid=np.indices((4,5,6),dtype=float)
    values=(grid[0]+2*grid[1]+3*grid[2]-100).astype(np.float32)
    transform=np.eye(4);transform[:3,:3]=np.diag([.5,.5,.5]);transform[:3,3]=[-.75,-.25,.25]
    output=resample_hu(values,transform,(10,12,14))
    target=np.indices(output.shape).reshape(3,-1).T
    coordinates=nib.affines.apply_affine(transform,target)
    valid=np.all((coordinates>=-.5) & (coordinates<np.array(values.shape)-.5),axis=1)
    clamped=np.clip(coordinates,0,np.array(values.shape)-1)
    expected=clamped@np.array([1.,2.,3.])-100
    expected[~valid]=-1024
    np.testing.assert_allclose(output.ravel(),expected,atol=1e-6,rtol=0)

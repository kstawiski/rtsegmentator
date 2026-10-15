import nibabel as nib
import numpy as np
import pytest

from audit_nodal_laterality import check_laterality


def fixture_images():
    ct = np.full((80, 60, 20), -1000, dtype=np.int16)
    ct[10:70, 10:50, :] = 30
    pred = np.zeros_like(ct)
    pred[18:28, 20:30, 5:15] = 2  # RAS negative X: patient's left.
    pred[52:62, 20:30, 5:15] = 1
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = [-80, -60, 0]
    return nib.Nifti1Image(ct, affine), nib.Nifti1Image(pred, affine)


LABELS = {'1': 'LN CTV R II-III-Va', '2': 'LN CTV L II-III-Va'}


def test_correct_and_reversed_laterality():
    ct, pred = fixture_images()
    result = check_laterality(ct, pred, LABELS)
    assert result['result'] == 'passed'
    assert len(result['components']) == 2
    assert all(x['volume_ml'] == pytest.approx(8) for x in result['components'])
    result = check_laterality(ct, pred, {'1': LABELS['2'], '2': LABELS['1']})
    assert result['result'] == 'failed'
    assert len(result['failures']) == 2


def test_reoriented_storage_preserves_patient_laterality():
    ct, pred = fixture_images()
    transform = np.array([[2, -1], [0, 1], [1, -1]])
    assert check_laterality(ct.as_reoriented(transform), pred.as_reoriented(transform), LABELS)['result'] == 'passed'


def test_ignore_tiny_opposite_component_but_never_accept_empty():
    ct, pred = fixture_images()
    data = np.asarray(pred.dataobj).copy()
    data[65, 35, 8] = 2  # 0.008 mL on the wrong side.
    result = check_laterality(ct, nib.Nifti1Image(data, pred.affine), LABELS)
    assert result['result'] == 'passed' and result['ignored_components'] == 1
    empty = nib.Nifti1Image(np.zeros(ct.shape, dtype=np.int16), ct.affine)
    assert check_laterality(ct, empty, LABELS)['result'] == 'failed'


def test_reject_one_side_only_and_noninteger_or_wrong_grid():
    ct, pred = fixture_images()
    data = np.asarray(pred.dataobj).copy()
    data[data == 1] = 0
    assert check_laterality(ct, nib.Nifti1Image(data, pred.affine), LABELS)['result'] == 'failed'
    fractional = data.astype(np.float32); fractional[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match='noninteger'):
        check_laterality(ct, nib.Nifti1Image(fractional, pred.affine), LABELS)
    affine = pred.affine.copy(); affine[0, 3] += 1
    with pytest.raises(ValueError, match='grids differ'):
        check_laterality(ct, nib.Nifti1Image(data, affine), LABELS)

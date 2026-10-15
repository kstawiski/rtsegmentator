"""Physical-grid and raw-class checks for development-fixture overlap evidence."""
import nibabel as nib
import numpy as np
import pytest

from audit_lnctvseg_dice import compare


def images(tmp_path):
    ref = np.zeros((10, 10, 10), dtype=np.uint8)
    for label in range(1, 7):
        ref[label, 2:4, 2:4] = label
    affine = np.diag([2., 2., 3., 1.])
    reference = tmp_path / 'reference.nii.gz'
    nib.save(nib.Nifti1Image(ref, affine), reference)
    return ref, affine, reference, tmp_path / 'prediction.nii.gz'


def test_reoriented_storage_matches_and_unmapped_id_is_retained(tmp_path):
    ref, affine, reference, prediction = images(tmp_path)
    pred = ref.copy()
    pred[0, 0, 0] = 7
    image = nib.Nifti1Image(pred, affine)
    nib.save(image.as_reoriented(np.array([[0, -1], [1, -1], [2, 1]])), prediction)
    result = compare(prediction, reference)
    assert [row['dice'] for row in result['labels']] == [1.] * 6
    assert result['unmapped_label_7_voxels'] == 1
    assert result['unmapped_label_7_ml'] == pytest.approx(.012)
    assert result['prediction_label_ids'] == list(range(8))


def test_missing_prediction_label_has_zero_dice(tmp_path):
    ref, affine, reference, prediction = images(tmp_path)
    ref[ref == 3] = 0
    nib.save(nib.Nifti1Image(ref, affine), prediction)
    assert compare(prediction, reference)['labels'][2]['dice'] == 0


@pytest.mark.parametrize('invalid', ['grid', 'fractional', 'unexpected', 'nan', 'missing_reference'])
def test_reject_invalid_comparisons(tmp_path, invalid):
    ref, affine, reference, prediction = images(tmp_path)
    pred = ref.astype(np.float32)
    if invalid == 'grid':
        affine[0, 3] += 2
    elif invalid == 'missing_reference':
        ref[ref == 6] = 0
        nib.save(nib.Nifti1Image(ref, affine), reference)
    else:
        pred[0, 0, 0] = {'fractional': .5, 'unexpected': 8, 'nan': np.nan}[invalid]
    nib.save(nib.Nifti1Image(pred, affine), prediction)
    with pytest.raises(ValueError):
        compare(prediction, reference)

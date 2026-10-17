"""CPU equivalence and bounded-memory checks against the original validator."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pydicom
import pytest
from rtstruct_geometry import add_roi, create_rtstruct, dicom_grid
from orientation_contract import restore_image
from rtstruct_roundtrip import crop_mask, mask_box, data_box, validate_roi_roundtrip
from roundtrip_reference import full_volume_reference
from test_viewer import GEOMETRIES, make_series
from test_rtstruct_geometry import inputs

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('iop,spacing,reverse,gaps,orientation', GEOMETRIES)
def test_cropped_reports_equal_full_reference(synthetic_ct, tmp_path, iop, spacing, reverse, gaps, orientation):
    source = tmp_path / 'source'
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    image_path, _, _ = inputs(source, tmp_path)
    image = nib.load(image_path)
    shape, affine, _ = dicom_grid(source, image)
    image, _ = restore_image(image, shape, affine)
    rt = create_rtstruct(source, image)
    # Including a hole, a point and a thin line; independent overlapping ROIs.
    for index in range(4):
        mask = np.zeros((48, 32, 6), bool)
        if index < 2:
            mask[5+index:26, 7:24, 1:5] = 1
            mask[10:17, 12:18, 2:4] = 0
        elif index == 2: mask[40, 24, 2] = 1
        else: mask[1:16, 29, 0] = 1
        add_roi(rt, mask, f'ROI {index}', [100, 120, 140])
        roi = rt.ds.ROIContourSequence[-1]
        cropped = crop_mask(mask)
        assert cropped.data.nbytes < mask.nbytes
        add_roi(rt, cropped.data, f'cropped {index}', [100, 120, 140], offset=cropped.offset)
        rebuilt = rt.ds.ROIContourSequence[-1]
        assert [c.to_json_dict() for c in rebuilt.ContourSequence] == [c.to_json_dict() for c in roi.ContourSequence]
        for threshold in (0, .95):
            assert validate_roi_roundtrip(roi, image, cropped, threshold, source_slices=rt.series_data) == \
                full_volume_reference(roi, image, mask, threshold, source_slices=rt.series_data)
        # Count a disjoint contour outside the expected ROI box, rather than
        # clipping it away and incorrectly improving Dice.
        foreign = copy.deepcopy(roi.ContourSequence[-1])
        origin = np.array(rt.series_data[0].ImagePositionPatient, float)
        col, row = np.array(rt.series_data[0].ImageOrientationPatient, float).reshape(2, 3)
        xy = np.array([[33, 2], [36, 2], [36, 4], [33, 4]])
        foreign.ContourData = (origin + xy[:, :1]*col*spacing[1] + xy[:, 1:]*row*spacing[0]).ravel().tolist()
        foreign.NumberOfContourPoints = 4
        foreign.ContourImageSequence[0].ReferencedSOPInstanceUID = rt.series_data[0].SOPInstanceUID
        roi.ContourSequence.append(foreign)
        assert validate_roi_roundtrip(roi, image, cropped, 0, source_slices=rt.series_data) == \
            full_volume_reference(roi, image, mask, 0, source_slices=rt.series_data)


def test_memory_scales_with_one_roi_not_roi_count():
    output = subprocess.check_output([sys.executable, str(ROOT / 'scripts/benchmark-rtstruct-memory.py'),
                                      '--compare', '--shape', '128', '128', '160', '--rois', '24'], text=True)
    result = json.loads(output)
    assert result['reports_equal']
    before, after = result['results']
    assert after['peak_rss_mib'] < before['peak_rss_mib'] - 20, result
    print(output)


@pytest.mark.parametrize('order', ['C', 'F', 'flipped'])
def test_slab_boxes_equal_full_mask_boxes(order):
    data = np.zeros((27, 35, 41), dtype=np.int16, order='F' if order == 'F' else 'C')
    data[3:10, 7:16, 11:20] = 1
    data[0, -1, -1] = 2
    data[-1, 0, 0] = 2
    if order == 'flipped': data = data[::-1, ::-1, ::-1]
    for label in (1, 2, 3):
        assert data_box(data, label=label) == mask_box(data == label)
    assert data_box(data, threshold=1.5) == mask_box(data > 1.5)

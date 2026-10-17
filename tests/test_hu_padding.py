"""Synthetic HU gate checks after excluding reconstruction padding."""
from types import SimpleNamespace

import numpy as np
import pytest

from nv_reason_ct_report import hu_check


@pytest.mark.parametrize('padding', [-8192, -3024])
def test_padded_ct_ring_is_accepted(padding):
    data = np.full((64, 64, 8), padding, np.float32)
    data[8:-8, 8:-8] = -1000
    data[20:44, 20:44] = 30
    data[28:36, 28:36] = 900
    result = hu_check(data)
    assert result['looks_like_hu']
    assert result['padding_fraction'] == .4375
    assert result['hu_min'] == -1000 and result['hu_max'] == 900
    assert '-1500 HU' in result['rule'] and 'PixelPaddingRangeLimit' in result['rule']


@pytest.mark.parametrize('slope', [2, -2])
def test_dicom_padding_range_is_converted_to_hu(slope):
    # Declared padding in the otherwise plausible air range must be excluded.
    header = SimpleNamespace(PixelPaddingValue=0, PixelPaddingRangeLimit=10,
                             RescaleSlope=slope, RescaleIntercept=-1200)
    padding = np.arange(11) * slope - 1200
    valid = np.array([-1000] * 2000 + [30] * 1000 + [700] * 100)
    result = hu_check(np.concatenate([padding, valid]), dicom_headers=[header])
    assert result['looks_like_hu']
    assert result['padding_fraction'] == round(11 / 3111, 6)
    assert result['hu_min'] == -1000


@pytest.mark.parametrize('data', [np.arange(4001), np.full((4, 4), -8192),
    np.full((4, 4), -3024), np.array([0, 30, 500]), np.array([-1000, 0, 299]),
    np.array([-8192, 0, 500]), np.array([-1000, np.nan]), np.array([])])
def test_invalid_intensities_still_refused(data):
    assert not hu_check(data)['looks_like_hu']


def test_all_dicom_padding_refused():
    result = hu_check(np.full(100, -1000), dicom_headers=[SimpleNamespace(PixelPaddingValue=-1000)])
    assert not result['looks_like_hu'] and result['padding_fraction'] == 1
    assert result['reason'] == 'All intensities are padding'


def test_padding_metadata_is_returned_by_header_gate(synthetic_ct, tmp_path):
    import pydicom
    from run_nv_reason_ct import headers_and_summary
    job = tmp_path / 'job'; job.mkdir()
    synthetic_ct.rename(job / 'input')
    (job / 'output').mkdir()
    path = next((job / 'input').iterdir())
    header = pydicom.dcmread(path)
    header.add_new(0x00280120, 'SS', 0)
    header.add_new(0x00280121, 'SS', 10)
    header.RescaleSlope, header.RescaleIntercept = 2, -1200
    header.save_as(path)
    report = {}
    headers = headers_and_summary(job, report)
    data = np.array([-1200] * 100 + [-1000] * 200 + [30] * 100 + [700] * 10)
    result = hu_check(data, dicom_headers=headers)
    assert result['looks_like_hu'] and result['padding_fraction'] == round(100 / 410, 6)
    assert result['hu_min'] == -1000

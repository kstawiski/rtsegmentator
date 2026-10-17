"""Synthetic byte equivalence and strict saved-contour decoding contracts."""
import copy
import io
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataelem import RawDataElement
from pydicom.filebase import DicomBytesIO
from pydicom.filewriter import write_data_element

from rtstruct_geometry import add_roi, create_rtstruct, dicom_grid
from rtstruct_roundtrip import (CroppedMask, label_boxes, mask_box, read_rtstruct,
                                validate_roi_roundtrip)
from roundtrip_reference import full_volume_reference
import writer_reference
from test_viewer import GEOMETRIES, make_series


def element_bytes(contour):
    stream = DicomBytesIO()
    stream.is_little_endian = True
    stream.is_implicit_VR = True
    write_data_element(stream, contour.get_item(0x30060050))
    return stream.getvalue()


@pytest.mark.parametrize('iop,spacing,reverse,gaps,orientation', GEOMETRIES)
@pytest.mark.parametrize('pattern', ['holes-islands', 'dense', 'tiny'])
def test_exact_saved_dataset_and_reports(synthetic_ct, tmp_path, iop, spacing, reverse, gaps, orientation, pattern):
    source = tmp_path / 'source'
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    data = np.zeros((48, 32, 6), np.uint16)
    data[2:23, 4:26, 1:5] = 1
    data[8:15, 9:19, 2:4] = 0  # holes
    data[29:38, 7:16, 1:5] = 1  # islands on the same slices
    data[0, 0, 0] = 2
    data[-1, -1, -1] = 2
    data[3:12, 30, 0] = 3  # tiny/degenerate contours
    if pattern == 'dense':
        data[:] = 1
        data[5:24, 6:26, 1:5] = 2
        data[-1, -1, -1] = 3
    elif pattern == 'tiny':
        data[:] = 0
        data[0, 0, 0] = 1
        data[-1, -1, -1] = 2
        data[1:10, 1, 2] = 3
    reference = nib.Nifti1Image(data, np.eye(4))
    shape, affine, slices = dicom_grid(source, reference)
    image = nib.Nifti1Image(data, affine)
    new = create_rtstruct(source, image)
    old = SimpleNamespace(ds=copy.deepcopy(new.ds), series_data=slices)
    boxes = label_boxes(data)
    for label, box in boxes.items():
        full = data == label
        expected_bounds = writer_reference.mask_bounds(full, slices)
        assert writer_reference.add_roi(old, full, f'ROI {label}', [111, 123, 245])
        bounds = []
        assert add_roi(new, data[box] == label, f'ROI {label}', [111, 123, 245],
                       offset=tuple(s.start for s in box), bounds_out=bounds)
        assert np.array_equal(bounds, expected_bounds)
        assert isinstance(new.ds.ROIContourSequence[-1].ContourSequence[0].get_item(0x30060050), RawDataElement)
        for a, b in zip(old.ds.ROIContourSequence[-1].ContourSequence,
                        new.ds.ROIContourSequence[-1].ContourSequence):
            assert element_bytes(a) == element_bytes(b)
    old_path, new_path = tmp_path / 'old.dcm', tmp_path / 'new.dcm'
    old.ds.save_as(old_path)
    new.ds.save_as(new_path)
    assert isinstance(new.ds.ROIContourSequence[0].ContourSequence[0].get_item(0x30060050), RawDataElement)
    assert old_path.read_bytes() == new_path.read_bytes()
    previous = pydicom.config.use_DS_numpy
    saved = read_rtstruct(new_path)
    assert pydicom.config.use_DS_numpy == previous
    normal = pydicom.dcmread(old_path)
    assert [x.ROIName for x in saved.StructureSetROISequence] == [x.ROIName for x in normal.StructureSetROISequence]
    for label, roi, original in zip(boxes, saved.ROIContourSequence, normal.ROIContourSequence):
        assert roi.ROIDisplayColor == original.ROIDisplayColor
        for a, b in zip(original.ContourSequence, roi.ContourSequence):
            assert isinstance(b.ContourData, np.ndarray)
            assert np.array_equal(np.asarray(a.ContourData), b.ContourData)
            assert a.ContourImageSequence == b.ContourImageSequence
        box = boxes[label]
        expected = CroppedMask(data[box] == label, tuple(s.start for s in box), shape)
        assert validate_roi_roundtrip(roi, image, expected, source_slices=slices) == \
            full_volume_reference(original, image, data == label, source_slices=slices)


@pytest.mark.parametrize('order', ['C', 'F', 'reversed'])
def test_one_pass_boxes(order):
    data = np.zeros((20, 30, 40), np.int16, order='F' if order == 'F' else 'C')
    data[2:7, 4:9, 9:12] = 1
    data[0, 0, -1] = 8
    data[-1, -1, 0] = 8
    if order == 'reversed': data = data[::-1, ::-1, ::-1]
    assert label_boxes(data) == {label: mask_box(data == label) for label in (1, 8)}
    assert label_boxes(np.zeros((2, 2, 2), np.uint8)) == {}
    assert label_boxes(np.full((2, 2, 2), -1, np.int16)) == {}


@pytest.mark.parametrize('values', [[8], [-1], [8, -1], [0]])
def test_label_refusal_messages_match_frozen_converter(tmp_path, monkeypatch, values):
    import json
    import sys
    import converter_reference
    import convert_validate_rtstruct
    data = np.zeros((3, 3, 2), np.int16)
    data.ravel()[:len(values)] = values
    image = nib.Nifti1Image(data, np.eye(4))
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'1': 'ROI 1'}))
    monkeypatch.setattr(nib, 'load', lambda path: image)
    monkeypatch.setattr(sys, 'argv', ['converter', '--prediction', 'synthetic.nii.gz',
        '--model-input', 'synthetic.nii.gz', '--dicom', 'synthetic',
        '--output', str(tmp_path / 'unused.dcm'), '--labels', str(mapping)])
    messages = []
    for module, main in ((converter_reference, converter_reference.labels_main),
                         (convert_validate_rtstruct, convert_validate_rtstruct.main)):
        monkeypatch.setattr(module, 'dicom_grid', lambda *args: (image.shape, image.affine, []))
        with pytest.raises(SystemExit) as error:
            main()
        messages.append(str(error.value))
    assert messages[0] == messages[1]


@pytest.mark.parametrize('coordinates,error', [
    ([], 'empty contour coordinates'),
    ([0, 0, float('nan')], 'invalid contour coordinates'),
    ([0, 0, 0, 1, 1, 0], 'contour point count does not match coordinates'),
])
def test_fast_read_preserves_geometry_errors(tmp_path, coordinates, error):
    ds = pydicom.dataset.Dataset()
    ds.is_implicit_VR, ds.is_little_endian = True, True
    contour = pydicom.dataset.Dataset()
    contour.ContourGeometricType = 'CLOSED_PLANAR'
    contour.NumberOfContourPoints = 1
    contour.ContourData = coordinates
    roi = pydicom.dataset.Dataset(); roi.ContourSequence = [contour]
    ds.ROIContourSequence = [roi]
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, ds)
    stream.seek(0)
    # A file with a DICOM header is needed for the production read function.
    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta = meta
    path = tmp_path / 'corrupt.dcm'
    ds.save_as(path, enforce_file_format=True)
    saved = read_rtstruct(path)
    image = nib.Nifti1Image(np.zeros((3, 3, 2), np.uint8), np.eye(4))
    with pytest.raises(ValueError, match=error):
        validate_roi_roundtrip(saved.ROIContourSequence[0], image, np.zeros(image.shape, bool))


def test_fast_read_failure_restores_config(tmp_path):
    path = tmp_path / 'invalid.dcm'
    path.write_bytes(b'invalid synthetic DICOM')
    previous = pydicom.config.use_DS_numpy
    with pytest.raises(pydicom.errors.InvalidDicomError):
        read_rtstruct(path)
    assert pydicom.config.use_DS_numpy == previous


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
@pytest.mark.parametrize('iop,spacing,reverse,gaps,orientation', GEOMETRIES)
def test_converter_json_and_file_equal_frozen_3600f83(synthetic_ct, tmp_path, monkeypatch, capsys,
                                                    independent, iop, spacing, reverse, gaps, orientation):
    import json
    import sys
    import converter_reference
    import convert_validate_rtstruct
    import convert_masks_validate_rtstruct
    source = tmp_path / 'source'
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    data = np.zeros((48, 32, 6), np.uint8)
    data[2:23, 4:26, 1:5] = 1
    data[8:15, 9:19, 2:4] = 0
    data[29:38, 7:16, 1:5] = 1
    data[0, 0, 0] = 2
    data[3:12, 30, 0] = 2
    data[-1, -1, -1] = 2
    shape, affine, _ = dicom_grid(source, nib.Nifti1Image(data, np.eye(4)))
    image = nib.Nifti1Image(data, affine)
    image_path = tmp_path / 'input.nii.gz'; nib.save(image, image_path)
    template = create_rtstruct(source, image)
    module = convert_masks_validate_rtstruct if independent else convert_validate_rtstruct
    for target in (module, converter_reference):
        monkeypatch.setattr(target, 'create_rtstruct', lambda *args: copy.deepcopy(template))
    output = tmp_path / 'model_RTSTRUCT.dcm'
    mapping = tmp_path / 'mapping.json'
    if independent:
        paths = {}
        for i in (1, 2):
            path = tmp_path / f'mask-{i}.nii.gz'
            nib.save(nib.Nifti1Image((data == i).astype(np.uint8), affine), path)
            paths[f'ROI {i}'] = str(path)
        mapping.write_text(json.dumps(paths))
        extra = ['--masks-json', str(mapping)]
        before = converter_reference.masks_main
    else:
        mapping.write_text(json.dumps({'1': 'ROI 1', '2': 'ROI 2'}))
        extra = ['--prediction', str(image_path), '--labels', str(mapping)]
        before = converter_reference.labels_main
    monkeypatch.setattr(sys, 'argv', ['converter', '--model-input', str(image_path),
                                     '--dicom', str(source), '--output', str(output), *extra])
    before()
    report_before = json.loads(capsys.readouterr().out.splitlines()[-1])
    bytes_before = output.read_bytes()
    module.main()
    report_after = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report_after == report_before
    assert output.read_bytes() == bytes_before


@pytest.mark.parametrize('xy', [
    [[-8, -8], [5, -8], [5, 5], [-8, 5]],
    [[-5, 5], [45, 8], [20, 48]],
    [[35, 44], [40, 44], [40, 50], [35, 50]],
])
def test_contours_outside_expected_box_and_grid_keep_dice(xy):
    mask = np.zeros((32, 40, 4), bool)
    mask[12:17, 15:20, 1] = True
    image = nib.Nifti1Image(mask.astype(np.uint8), np.diag([-1., -1., 1., 1.]))
    contour = pydicom.dataset.Dataset()
    contour.ContourData = np.column_stack((xy, np.ones(len(xy)))).ravel().tolist()
    contour.NumberOfContourPoints = len(xy)
    contour.ContourGeometricType = 'CLOSED_PLANAR'
    roi = pydicom.dataset.Dataset(); roi.ContourSequence = [contour]
    box = mask_box(mask)
    expected = CroppedMask(mask[box], tuple(s.start for s in box), mask.shape)
    assert validate_roi_roundtrip(roi, image, expected, 0) == full_volume_reference(roi, image, mask, 0)
    with pytest.raises(ValueError) as before:
        full_volume_reference(roi, image, mask)
    with pytest.raises(ValueError) as after:
        validate_roi_roundtrip(roi, image, expected)
    assert str(after.value) == str(before.value)


@pytest.mark.parametrize('token', [b'1.2.3', b'1e+', b'+', b'1 2'])
def test_fast_read_never_discards_malformed_coordinate_tail(tmp_path, token):
    from pydicom.tag import Tag
    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds = pydicom.dataset.FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
    contour = pydicom.dataset.Dataset()
    contour.set_original_encoding(True, True, contour._character_set)
    contour.ContourGeometricType = 'CLOSED_PLANAR'
    contour.NumberOfContourPoints = 3
    encoded = b'0\\0\\0\\1\\0\\0\\1\\1\\0\\' + token
    if len(encoded) % 2: encoded += b' '
    tag = Tag(0x30060050)
    contour[tag] = RawDataElement(tag, 'DS', len(encoded), encoded, 0, True, True)
    roi = pydicom.dataset.Dataset(); roi.ContourSequence = [contour]
    ds.ROIContourSequence = [roi]
    path = tmp_path / 'malformed.dcm'; ds.save_as(path, enforce_file_format=True)
    image = nib.Nifti1Image(np.zeros((3, 3, 2), np.uint8), np.eye(4))
    with pytest.raises(ValueError):
        saved = read_rtstruct(path)
        validate_roi_roundtrip(saved.ROIContourSequence[0], image, np.zeros(image.shape, bool), 0)

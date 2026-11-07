"""Synthetic saved-file corruption and empty-structure reporting contracts."""
import copy
import json
from pathlib import Path
import sys
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import pytest

import convert_validate_rtstruct as labels_writer
import convert_masks_validate_rtstruct as masks_writer
from rtstruct_geometry import dicom_grid, slice_geometry, validate_reference_identity
from orientation_contract import sidecar_path
from rtstruct_roundtrip import validate_contour_extent, validate_roi_roundtrip, validate_rtstruct_extent, read_rtstruct


@pytest.fixture
def export_case(synthetic_ct, tmp_path, monkeypatch):
    data = np.zeros((32, 32, 12), np.uint8)
    data[0:20, 5:20, 2:8] = 1
    data[25:27, 25:27, 3] = 2
    _, affine, slices = dicom_grid(synthetic_ct, nib.Nifti1Image(data, np.eye(4)))
    image = tmp_path / 'input.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), image)

    def run(independent, mutate=None, empty=False, output=None, all_empty=False):
        output = output or tmp_path / 'test_RTSTRUCT.dcm'
        if all_empty:
            data.fill(0)
            nib.save(nib.Nifti1Image(data, affine), image)
        mapping = tmp_path / 'mapping.json'
        names = {'1': 'Large', '2': 'Small', **({'3': 'Empty'} if empty else {})}
        module = masks_writer if independent else labels_writer
        if independent:
            paths = {}
            for label, name in names.items():
                path = tmp_path / f'mask-{label}.nii.gz'
                nib.save(nib.Nifti1Image((data == int(label)).astype(np.uint8), affine), path)
                paths[name] = str(path)
            mapping.write_text(json.dumps(paths))
            extra = ['--masks-json', str(mapping)]
        else:
            mapping.write_text(json.dumps(names))
            extra = ['--prediction', str(image), '--labels', str(mapping)]
        monkeypatch.setattr(sys, 'argv', ['converter', '--model-input', str(image),
            '--dicom', str(synthetic_ct), '--output', str(output), *extra])
        original = module.read_rtstruct
        if mutate:
            def injected(path):
                saved = pydicom.dcmread(path)
                mutate(saved)
                saved.save_as(path)
                return original(path)
            monkeypatch.setattr(module, 'read_rtstruct', injected)
        module.main()
        return output
    return run, slices, data, affine


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
@pytest.mark.parametrize('location', ['top-frame', 'sequence-frame', 'roi-frame', 'series',
                                     'extra-frame', 'extra-series', 'missing-frame', 'missing-series',
                                     'study', 'extra-study', 'hierarchy-image', 'hierarchy-class',
                                     'contour-class', 'missing-hierarchy-image', 'top-study'])
def test_saved_reference_identity(export_case, independent, location, tmp_path):
    run, _, _, _ = export_case
    def mutate(ds):
        frame = ds.ReferencedFrameOfReferenceSequence[0]
        study = frame.RTReferencedStudySequence[0]
        if location == 'top-study': ds.StudyInstanceUID = '2.25.123'
        if location == 'top-frame': ds.FrameOfReferenceUID = '2.25.123'
        if location == 'sequence-frame': frame.FrameOfReferenceUID = '2.25.123'
        if location == 'roi-frame': ds.StructureSetROISequence[-1].ReferencedFrameOfReferenceUID = '2.25.123'
        if location == 'series': study.RTReferencedSeriesSequence[0].SeriesInstanceUID = '2.25.123'
        if location == 'extra-frame':
            extra = copy.deepcopy(frame); extra.FrameOfReferenceUID = '2.25.123'
            ds.ReferencedFrameOfReferenceSequence.append(extra)
        if location == 'extra-series':
            extra = copy.deepcopy(study.RTReferencedSeriesSequence[0]); extra.SeriesInstanceUID = '2.25.123'
            study.RTReferencedSeriesSequence.append(extra)
        if location == 'study': study.ReferencedSOPInstanceUID = '2.25.123'
        if location == 'extra-study':
            extra = copy.deepcopy(study); extra.ReferencedSOPInstanceUID = '2.25.123'
            frame.RTReferencedStudySequence.append(extra)
        if location == 'hierarchy-image':
            study.RTReferencedSeriesSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID = '2.25.123'
        if location == 'hierarchy-class':
            study.RTReferencedSeriesSequence[0].ContourImageSequence[0].ReferencedSOPClassUID = pydicom.uid.MRImageStorage
        if location == 'contour-class':
            ds.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPClassUID = pydicom.uid.MRImageStorage
        if location == 'missing-hierarchy-image':
            uid = ds.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID
            images = study.RTReferencedSeriesSequence[0].ContourImageSequence
            study.RTReferencedSeriesSequence[0].ContourImageSequence = [im for im in images if im.ReferencedSOPInstanceUID != uid]
        if location == 'missing-frame': del ds.ReferencedFrameOfReferenceSequence
        if location == 'missing-series': del study.RTReferencedSeriesSequence
    with pytest.raises(ValueError, match='identity mismatch|missing .*references|missing or foreign|SOP class mismatch|missing from series hierarchy'):
        run(independent, mutate)
    assert not (tmp_path / 'test_RTSTRUCT.dcm').exists()
    assert not sidecar_path(tmp_path / 'test_RTSTRUCT.dcm').exists()


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
@pytest.mark.parametrize('mutation', ['delete-small', 'empty-small', 'duplicate-number'])
def test_saved_roi_completeness(export_case, independent, mutation):
    run, _, _, _ = export_case
    def mutate(ds):
        if mutation == 'delete-small':
            del ds.StructureSetROISequence[-1]
            del ds.ROIContourSequence[-1]
        if mutation == 'empty-small': ds.ROIContourSequence[-1].ContourSequence = []
        if mutation == 'duplicate-number': ds.ROIContourSequence[-1].ReferencedROINumber = ds.ROIContourSequence[0].ReferencedROINumber
    with pytest.raises((ValueError, SystemExit), match='ROI count|missing contour geometry'):
        run(independent, mutate)


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_exterior_fragment_cannot_be_clipped(export_case, independent):
    run, _, _, _ = export_case
    def mutate(ds):
        contours = ds.ROIContourSequence[0].ContourSequence
        extra = copy.deepcopy(contours[0])
        z = float(extra.ContourData[2])
        extra.NumberOfContourPoints = 3
        extra.ContourData = [-2, 10, z, -1, 10, z, -1, 11, z]
        contours.append(extra)
    with pytest.raises(ValueError, match='outside source image extent'):
        run(independent, mutate)


def test_roundtrip_rejects_exterior_fragment_even_with_zero_dice_threshold(export_case):
    run, slices, data, affine = export_case
    ds = pydicom.dcmread(run(False))
    roi = ds.ROIContourSequence[0]
    extra = copy.deepcopy(roi.ContourSequence[0])
    z = float(extra.ContourData[2])
    extra.NumberOfContourPoints = 3
    extra.ContourData = [-2, 10, z, -1, 10, z, -1, 11, z]
    roi.ContourSequence.append(extra)
    with pytest.raises(ValueError, match='outside source image extent'):
        validate_roi_roundtrip(roi, nib.Nifti1Image(data, affine), data == 1,
                               minimum_dice=0, source_slices=slices)


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_nonempty_input_cannot_be_silently_omitted(export_case, independent, monkeypatch):
    run, _, _, _ = export_case
    module = masks_writer if independent else labels_writer
    original = module.add_roi
    def drop_small(rt, mask, name, *args, **kwargs):
        return False if name == 'Small' else original(rt, mask, name, *args, **kwargs)
    monkeypatch.setattr(module, 'add_roi', drop_small)
    with pytest.raises((ValueError, SystemExit), match='ROI count'):
        run(independent)


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_empty_structure_report_reaches_manifest(export_case, independent, portal, capsys):
    run, _, _, _ = export_case
    job = portal.BASE / 'synthetic-empty'
    (job / 'output').mkdir(parents=True)
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'models': ['total']}))
    with pytest.warns(UserWarning, match='empty structure: Empty'):
        output = run(independent, empty=True, output=job / 'output/total_RTSTRUCT.dcm')
    captured = capsys.readouterr()
    report = json.loads(captured.out.splitlines()[-1])
    assert report['valid'] and report['rois'] == 2
    assert report['omitted_rois'] == [{'name': 'Empty', 'voxels': 0,
        'reason': 'empty input mask' if independent else 'empty input label'}]
    (job / 'worker.log').write_text(captured.err)
    portal.finalize_results(job)
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as package:
        manifest = json.loads(package.read('rtsegmentator_manifest.json'))
    assert 'total: not present in the prediction: Empty.' in manifest['notes']
    assert [str(roi.ROIName) for roi in pydicom.dcmread(output).StructureSetROISequence] == ['Large', 'Small']


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_valid_reference_and_border_extent(export_case, independent):
    run, slices, data, affine = export_case
    output = run(independent)
    ds = pydicom.dcmread(output)
    validate_reference_identity(ds, slices)
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    points = np.asarray(contour.ContourData, float).reshape(-1, 3)
    # Valid edge convention places border vertices half a voxel beyond centres.
    points[points[:, 0] == 0, 0] = -0.5
    contour.ContourData = points.ravel().tolist()
    validate_rtstruct_extent(ds, slices)
    validate_roi_roundtrip(ds.ROIContourSequence[0], nib.Nifti1Image(data, affine),
                           data == 1, source_slices=slices)


def test_single_image_without_thickness_accepts_contours_on_its_plane(export_case):
    run, slices, _, _ = export_case
    ds = pydicom.dcmread(run(False))
    roi = ds.ROIContourSequence[0]
    roi.ContourSequence = [roi.ContourSequence[0]]
    ds.ROIContourSequence = [roi]
    uid = roi.ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID
    image = copy.deepcopy(next(image for image in slices if image.SOPInstanceUID == uid))
    del image.SliceThickness
    validate_rtstruct_extent(ds, [image])


@pytest.mark.parametrize('xy', [[[-.5, -.5], [31.5, 31.5]], [[-.50009, 2]]])
def test_voxel_edges_and_numeric_tolerance(xy):
    validate_contour_extent(np.asarray(xy), (32, 32))


@pytest.mark.parametrize('xy', [[[-.50011, 2]], [[31.50011, 2]], [[2, -.50011]], [[2, 31.50011]]])
def test_every_vertex_is_checked(xy):
    with pytest.raises(ValueError, match='outside source image extent'):
        validate_contour_extent(np.asarray(xy), (32, 32))


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_empty_rerun_removes_final_artifacts(export_case, independent, tmp_path):
    run, _, _, _ = export_case
    output = run(independent)
    assert output.exists() and sidecar_path(output).exists()
    with pytest.warns(UserWarning, match='empty structure'), pytest.raises(SystemExit, match='empty.*no RTSTRUCT'):
        run(independent, all_empty=True)
    assert not output.exists() and not sidecar_path(output).exists()


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_early_input_failure_removes_final_artifacts(export_case, independent, monkeypatch):
    run, _, _, _ = export_case
    output = run(independent)
    module = masks_writer if independent else labels_writer
    def fail(*args, **kwargs):
        raise ValueError('synthetic input failure')
    monkeypatch.setattr(module.nib, 'load', fail)
    with pytest.raises(ValueError, match='^synthetic input failure$'):
        run(independent)
    assert not output.exists() and not sidecar_path(output).exists()


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
@pytest.mark.parametrize('failure', [ValueError, SystemExit])
def test_saved_failure_removes_file_and_sidecar(export_case, independent, monkeypatch, tmp_path, failure):
    run, _, _, _ = export_case
    module = masks_writer if independent else labels_writer
    def fail(path):
        sidecar_path(path).write_text('synthetic stale sidecar')
        raise failure('synthetic saved validation failure')
    monkeypatch.setattr(module, 'read_rtstruct', fail)
    with pytest.raises(failure, match='^synthetic saved validation failure$'):
        run(independent)
    output = tmp_path / 'test_RTSTRUCT.dcm'
    assert not output.exists() and not sidecar_path(output).exists()


def test_all_empty_label_notes_reach_manifest(export_case, portal, capsys):
    run, _, _, _ = export_case
    job = portal.BASE / 'all-empty-labels'
    (job / 'output').mkdir(parents=True)
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'models': ['total'],
                                            'no_structures_found': ['total']}))
    output = job / 'output/total_RTSTRUCT.dcm'
    with pytest.warns(UserWarning, match='empty structure'), pytest.raises(SystemExit, match='empty prediction'):
        run(False, empty=True, all_empty=True, output=output)
    captured = capsys.readouterr()
    notes = [json.loads(line.removeprefix('RTSTRUCT_CONVERSION_NOTE '))
             for line in captured.err.splitlines() if line.startswith('RTSTRUCT_CONVERSION_NOTE ')]
    assert [item['name'] for item in notes[0]['omitted_rois']] == ['Large', 'Small', 'Empty']
    assert all(item['voxels'] == 0 for item in notes[0]['omitted_rois'])
    (job / 'worker.log').write_text(captured.err)
    portal.finalize_results(job, empty=['total'])
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as package:
        manifest = json.loads(package.read('rtsegmentator_manifest.json'))
        assert output.name not in package.namelist()
    assert 'total: not present in the prediction: Large, Small, Empty.' in manifest['notes']
    assert not output.exists() and not sidecar_path(output).exists()


def test_saved_rounded_oblique_border(export_case, tmp_path):
    run, slices, _, _ = export_case
    output = run(False)
    ds = pydicom.dcmread(output)
    image = copy.deepcopy(slices[0])
    image.Rows = image.Columns = 512
    image.ImageOrientationPatient = [.707107, -.707107, 0, .707107, .707107, 0]
    origin, u, v = slice_geometry(image)
    xy = np.array([[-.5, -.5], [511.5, -.5], [511.5, 511.5], [-.5, 511.5]])
    points = origin + xy[:, :1] * u + xy[:, 1:] * v
    roi = ds.ROIContourSequence[0]
    roi.ContourSequence = [roi.ContourSequence[0]]
    contour = roi.ContourSequence[0]
    contour.NumberOfContourPoints = 4
    contour.ContourData = points.ravel().tolist()
    contour.ContourImageSequence[0].ReferencedSOPInstanceUID = image.SOPInstanceUID
    ds.ROIContourSequence = [roi]
    ds.save_as(output)
    saved = read_rtstruct(output)
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1], affine[:3, 2], affine[:3, 3] = u, v, [0, 0, 2], origin
    affine[:2] *= -1
    full = np.ones((512, 512, 1), np.uint8)
    validate_rtstruct_extent(saved, [image])
    report = validate_roi_roundtrip(saved.ROIContourSequence[0], nib.Nifti1Image(full, affine), full,
                                    source_slices=[image])
    assert report['roundtrip_dice'] == 1.0
    assert report['intersection_voxels'] == 262144


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
@pytest.mark.parametrize('reference_class', [pydicom.uid.CTImageStorage, pydicom.uid.EnhancedCTImageStorage])
def test_source_dataset_and_file_meta_classes_both_accepted(
        export_case, independent, reference_class, monkeypatch):
    run, _, _, _ = export_case
    module = masks_writer if independent else labels_writer
    original = module.create_rtstruct
    def create(*args):
        rt = original(*args)
        for image in rt.series_data:
            image.file_meta.MediaStorageSOPClassUID = pydicom.uid.EnhancedCTImageStorage
        return rt
    monkeypatch.setattr(module, 'create_rtstruct', create)
    def mutate(saved):
        for frame in saved.ReferencedFrameOfReferenceSequence:
            for study in frame.RTReferencedStudySequence:
                for series in study.RTReferencedSeriesSequence:
                    for image in series.ContourImageSequence:
                        image.ReferencedSOPClassUID = reference_class
        for roi in saved.ROIContourSequence:
            for contour in roi.ContourSequence:
                contour.ContourImageSequence[0].ReferencedSOPClassUID = reference_class
    # Both converters must accept the saved file through all their normal checks.
    assert run(independent, mutate).exists()


@pytest.mark.parametrize('independent', [False, True], ids=['labels', 'masks'])
def test_patient_names_outside_latin1_still_export(export_case, synthetic_ct, independent):
    """rt-utils saves under ISO_IR 100, so these names gain replacement characters and must not fail a check."""
    run, _, _, _ = export_case
    for path in sorted(synthetic_ct.glob('*.dcm')):
        ds = pydicom.dcmread(path)
        ds.SpecificCharacterSet = 'ISO_IR 192'
        ds.PatientName = 'Łódź^Żółć'
        ds.PatientID = 'ID-Ł1'
        ds.save_as(path, enforce_file_format=True)
    assert run(independent).exists()

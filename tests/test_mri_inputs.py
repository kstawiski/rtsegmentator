"""Synthetic MRI only. No model code, GPU, patient images or network transport."""
import json
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

from mri_inputs import B_VALUE_SOURCE, role_candidates, suggest_role


ROLES = [
    {'role': 't2w', 'modality': 'MR', 'required': True, 'label': 'Axial T2-weighted'},
    {'role': 'adc', 'modality': 'MR', 'required': True, 'label': 'ADC map'},
    {'role': 'hbv', 'modality': 'MR', 'required': True, 'label': 'High b-value DWI (one b-value)'},
]
MODEL = {'name': 'synthetic_bpmri', 'modality': 'MR', 'enabled': True,
         'input_roles': ROLES, 'reference_role': 't2w'}


@pytest.fixture
def mri(portal, monkeypatch):
    root = portal.BASE / 'upload-synthetic'
    (root / 'input').mkdir(parents=True)
    study, frame = generate_uid(), generate_uid()
    enqueued = []
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': [MODEL, {'name': 'total_mr', 'modality': 'MR', 'enabled': True}]})
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda jobs, **kwargs: enqueued.extend(jobs))

    def add(name, values=(None, None, None), image_type=None, **changes):
        series_uid = generate_uid()
        for k, value in enumerate(values):
            path = root / 'input' / f'{name}-{k}.dcm'
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
            ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series_uid, frame
            ds.Modality, ds.PatientName, ds.PatientID = 'MR', 'SYNTHETIC^MRI', 'SYNTHETIC'
            ds.StudyDate, ds.AcquisitionTime = '20000101', '121314.5'
            ds.SeriesDescription, ds.ProtocolName = name, 'Synthetic prostate MRI'
            ds.ImageType = image_type or ['ORIGINAL', 'PRIMARY']
            ds.Rows = ds.Columns = 8
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, 3 * (k % 3)]
            ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [0.5, 0.5], 3, 3
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.RescaleSlope, ds.RescaleIntercept = 0.1, 2
            ds.PixelData = np.full((8, 8), k + 10, dtype=np.uint16).tobytes()
            if value is not None:
                ds.DiffusionBValue = value
            for key, change in changes.items():
                setattr(ds, key, change)
            ds.save_as(path, enforce_file_format=True)
        return f'{series_uid}|{changes.get("Modality", "MR")}'

    def assess():
        report = portal.inspect_upload(root / 'input')
        (root / 'upload.json').write_text(json.dumps(report))
        return {s['description']: s for s in report['series']}

    def submit(keys, roles=None, models=None):
        fields = {'series_keys': keys, 'models': models or ['synthetic_bpmri']}
        if roles is not None:
            fields['roles'] = json.dumps(roles)
        return TestClient(portal.app).post(f'/api/v1/uploads/{root.name}/jobs', data=fields)

    return SimpleNamespace(portal=portal, root=root, add=add, assess=assess, submit=submit,
                           enqueued=enqueued, study=study, frame=frame)


def standard_exam(mri, second=False):
    t2 = mri.add('T2 axial')
    t2b = mri.add('T2 repeat') if second else None
    adc = mri.add('Diffusion map', image_type=['DERIVED', 'PRIMARY', 'ADC'])
    hbv = mri.add('DWI', (0, 0, 0, 800, 800, 800, 1400, 1400, 1400))
    rows = mri.assess()
    return t2, adc, hbv, t2b, rows


@pytest.mark.parametrize('role,name', [('t2w', 'T2 axial'), ('adc', 'Diffusion map'), ('hbv', 'DWI')])
@pytest.mark.parametrize('slice_index', [0, 1, 2])
def test_review_h1_rejected_slice_blocks_series_before_manifest(mri, role, name, slice_index):
    t2, adc, hbv, _, _ = standard_exam(mri)
    if role == 'hbv':
        slice_index *= 4  # First, middle and last instance of the mixed-b acquisition.
    path = mri.root / 'input' / f'{name}-{slice_index}.dcm'
    ds = pydicom.dcmread(path)
    ds.NumberOfFrames = 0
    ds.save_as(path, enforce_file_format=True)
    rows = mri.assess()
    response = mri.submit([t2], {t2: {'adc': adc, 'hbv': {'series_key': hbv, 'b_value': 1400}}})
    assert response.status_code == 400, response.text
    assert 'assessment' in response.text.lower()
    assert 'Invalid NumberOfFrames' in response.text
    assert rows[name]['unavailable_reason']
    assert not list(mri.portal.BASE.glob('*/job.json'))
    assert not mri.enqueued


@pytest.mark.parametrize('phase', ['before_manifest', 'during_copy'])
@pytest.mark.parametrize('role,name', [('t2w', 'T2 axial'), ('adc', 'Diffusion map'), ('hbv', 'DWI')])
@pytest.mark.parametrize('slice_index', [0, 1, 2])
def test_review_h1_rechecks_sources_and_staged_roles(mri, monkeypatch, phase, role, name, slice_index):
    t2, _, _, _, _ = standard_exam(mri)
    if role == 'hbv':
        slice_index += 6  # First, middle and final slice of the selected b=1400 volume.
    path = mri.root / 'input' / f'{name}-{slice_index}.dcm'
    if phase == 'before_manifest':
        ds = pydicom.dcmread(path)
        ds.NumberOfFrames = 0
        ds.save_as(path, enforce_file_format=True)
    else:
        # Change a byte without changing size: the existing size manifest cannot detect this.
        original = mri.portal.shutil.move if role == 't2w' else mri.portal.shutil.copy2
        def corrupt(source, destination, *args, **kwargs):
            result = original(source, destination, *args, **kwargs)
            if source == path:
                raw = destination.read_bytes()
                old = b'\x28\x00\x10\x00US\x02\x00\x08\x00'  # Rows = 8
                assert old in raw
                destination.write_bytes(raw.replace(old, old[:-2] + b'\x00\x00'))
            return result
        monkeypatch.setattr(mri.portal.shutil, 'move' if role == 't2w' else 'copy2', corrupt)
    try:
        response = mri.submit([t2])
        assert response.status_code == 400, response.text
    except ValueError as exc:
        assert 'assessment' in str(exc).lower()
    assert not mri.enqueued
    assert not list(mri.portal.BASE.glob('*/job.json'))


@pytest.mark.parametrize('name,values', [('DWI for ADC calculation', (1400,) * 3),
                                        ('ADC map', (None,) * 3)])
@pytest.mark.parametrize('explicit', [False, True])
def test_review_h2_description_requires_explicit_choice(mri, name, values, explicit):
    t2 = mri.add('T2 axial')
    adc = mri.add(name, values)
    mri.add('High b DWI', (2000,) * 3)
    mri.assess()
    response = mri.submit([t2], {t2: {'adc': adc}} if explicit else None)
    assert response.status_code == (202 if explicit else 400), response.text
    if explicit:
        data = mri.portal.metadata(mri.enqueued[0])
        assert data['input_roles']['adc']['selection_method'] == 'user'
    else:
        assert 'ImageType' in response.text
        assert not list(mri.portal.BASE.glob('*/job.json'))


def test_review_h2_image_type_method_is_retained(mri):
    t2, _, _, _, _ = standard_exam(mri)
    assert mri.submit([t2]).status_code == 202
    records = mri.portal.metadata(mri.enqueued[0])['input_roles']
    assert records['adc']['selection_method'] == 'image_type'
    assert records['t2w']['selection_method'] == 'user'
    assert records['hbv']['selection_method'] == 'b_value'


@pytest.mark.parametrize('readable_identity', [False, True])
def test_review_h1_unreadable_instance_cannot_disappear(mri, monkeypatch, readable_identity):
    t2, adc, _, _, _ = standard_exam(mri)
    target = mri.root / 'input/Diffusion map-2.dcm'
    read = mri.portal.pydicom.dcmread
    def unreadable(path, *args, **kwargs):
        if path == target and (not readable_identity or not kwargs.get('specific_tags')):
            raise pydicom.errors.InvalidDicomError('synthetic unreadable metadata')
        return read(path, *args, **kwargs)
    monkeypatch.setattr(mri.portal.pydicom, 'dcmread', unreadable)
    rows = mri.assess()
    assert rows['Diffusion map']['slices'] == 2
    assert rows['Diffusion map']['unavailable_reason']
    assert mri.submit([t2], {t2: {'adc': adc}}).status_code == 400
    assert not list(mri.portal.BASE.glob('*/job.json'))


def test_review_h1_completely_rejected_series_remains_visible(mri):
    mri.add('Rejected ADC', image_type=['ADC'], NumberOfFrames=0)
    series = mri.assess()['Rejected ADC']
    assert series['slices'] == 0 and len(series['assessment_errors']) == 3
    assert series['unavailable_reason']
    public = mri.portal.public_assessment({'series': [series]})['series'][0]
    assert all('file' not in error for error in public['assessment_errors'])


@pytest.mark.parametrize('modality', [None, 'OT'])
def test_review_h1_rejected_modality_remains_attached_to_series_uid(mri, modality):
    t2, adc, _, _, _ = standard_exam(mri)
    path = mri.root / 'input/Diffusion map-2.dcm'
    ds = pydicom.dcmread(path)
    if modality is None:
        del ds.Modality
    else:
        ds.Modality = modality
    ds.save_as(path, enforce_file_format=True)
    mri.assess()
    response = mri.submit([t2], {t2: {'adc': adc}})
    assert response.status_code == 400, response.text
    assert 'assessment' in response.text and 'Modality' in response.text
    assert not mri.enqueued and not list(mri.portal.BASE.glob('*/job.json'))


@pytest.mark.parametrize('phase', ['assessment', 'before_manifest'])
@pytest.mark.parametrize('name,offset', [('T2 axial', 0), ('Diffusion map', 0), ('DWI', 6)])
@pytest.mark.parametrize('slice_index', [0, 1, 2])
def test_round2_h1_missing_identity_blocks_every_mr_role(mri, phase, name, offset, slice_index):
    t2, adc, hbv, _, _ = standard_exam(mri)
    path = mri.root / 'input' / f'{name}-{offset + slice_index}.dcm'
    ds = pydicom.dcmread(path)
    del ds.Modality, ds.SeriesInstanceUID
    ds.save_as(path, enforce_file_format=True)
    if phase == 'assessment':
        rows = mri.assess()
        assert all(row['unavailable_reason'] for row in rows.values())
        assert all(any(error['file'] == path.name for error in row['assessment_errors']) for row in rows.values())
    response = mri.submit([t2], {t2: {'adc': adc, 'hbv': {'series_key': hbv, 'b_value': 1400}}})
    assert response.status_code == 400, response.text
    assert 'assessment' in response.text.lower()
    assert not mri.enqueued and not list(mri.portal.BASE.glob('*/job.json'))


@pytest.mark.parametrize('evidence', ['preamble', 'sop_class', 'pixel_data'])
def test_round2_h1_image_evidence_without_identity_or_filename(mri, evidence):
    t2, _, _, _, _ = standard_exam(mri)
    path = mri.root / 'input/Diffusion map-2.dcm'
    ds = pydicom.dcmread(path)
    del ds.Modality, ds.SeriesInstanceUID
    if evidence != 'preamble':
        ds.preamble = None
    if evidence != 'sop_class':
        del ds.SOPClassUID, ds.file_meta.MediaStorageSOPClassUID
    if evidence != 'pixel_data':
        del ds.PixelData
    path.unlink()
    ds.save_as(path.with_suffix('.bin'), enforce_file_format=False)
    assert all(row['unavailable_reason'] for row in mri.assess().values())
    assert mri.submit([t2]).status_code == 400
    assert not mri.enqueued


def test_round2_h1_unrelated_non_dicom_file_is_ignored(mri):
    t2, _, _, _, _ = standard_exam(mri)
    (mri.root / 'input/readme.txt').write_text('Unrelated upload notes')
    rows = mri.assess()
    assert not any(row.get('unavailable_reason') for row in rows.values())
    assert json.loads((mri.root / 'upload.json').read_text())['ignored'] == 1
    assert mri.submit([t2]).status_code == 202
    job = mri.enqueued[0]
    assert mri.portal.input_manifest_complete(job, mri.portal.metadata(job))


def test_review_h2_explicit_tagged_adc_override_records_user(mri):
    t2, adc, hbv, _, _ = standard_exam(mri)
    assert mri.submit([t2], {t2: {'adc': adc, 'hbv': {'series_key': hbv, 'b_value': 800}}}).status_code == 202
    records = mri.portal.metadata(mri.enqueued[0])['input_roles']
    assert records['adc']['selection_method'] == records['hbv']['selection_method'] == 'user'


def test_assessment_standard_b_values_adc_geometry_and_public_fields(mri):
    _, _, _, _, rows = standard_exam(mri)
    adc, dwi = rows['Diffusion map'], rows['DWI']
    assert adc['is_adc'] and not dwi['is_adc']
    assert dwi['b_values'] == [0, 800, 1400]
    assert dwi['b_value_sources'] == [B_VALUE_SOURCE]
    assert dwi['dimensions'] == [8, 8, 9]
    assert sorted(i['b_value'] for i in dwi['instances']) == [0, 0, 0, 800, 800, 800, 1400, 1400, 1400]
    assert {o['b_value']: len(o['files']) for o in dwi['hbv_options']} == {1400: 3, 800: 3}
    public = mri.portal.series_metadata(dwi)
    assert public['study_uid'] == mri.study and public['frame_uid'] == mri.frame
    assert public['orientation'] == [1, 0, 0, 0, 1, 0]
    assert public['pixel_spacing'] == [0.5, 0.5]
    assert public['acquisition_time'] == '121314.5'
    assert public['protocol_name'] == 'Synthetic prostate MRI'
    assert 'files' not in public and all('file' not in i for i in public['instances'])
    assessment = mri.portal.public_assessment({'series': list(rows.values())})
    assert all('files' not in o for s in assessment['series'] for o in s.get('hbv_options', []))


def test_single_b_unknown_and_private_tags_are_not_guessed(mri):
    mri.add('Single high b', (1000, 1000, 1000))
    mri.add('Mixed missing', (1400, None, 1400))
    mri.add('Private b1400')
    for path in (mri.root / 'input').glob('Private*'):
        ds = pydicom.dcmread(path)
        ds.add_new((0x0019, 0x0010), 'LO', 'SYNTHETIC_PRIVATE')
        ds.add_new((0x0019, 0x100C), 'IS', '1400')
        ds.save_as(path, enforce_file_format=True)
    rows = mri.assess()
    assert rows['Single high b']['b_values'] == [1000]
    assert len(rows['Single high b']['hbv_options']) == 1
    for name in ['Mixed missing', 'Private b1400']:
        assert rows[name]['hbv_options'] == []
        assert 'missing' in rows[name]['hbv_unavailable_reason']


@pytest.mark.parametrize('shared,values,available', [
    (False, [0, 1400], False), (False, [1400, 1400], True),
    (False, [1400, None], False), (True, [1400, 1400], True)])
def test_enhanced_per_frame_and_shared_b_values(mri, shared, values, available):
    mri.add('Enhanced', (None,))
    path = next((mri.root / 'input').iterdir())
    ds = pydicom.dcmread(path)
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
    ds.NumberOfFrames = 2
    ds.PerFrameFunctionalGroupsSequence = []
    for index, value in enumerate(values):
        frame = Dataset()
        position = Dataset(); position.ImagePositionPatient = [0, 0, index * 3]
        frame.PlanePositionSequence = [position]
        if not shared:
            diffusion = Dataset()
            if value is not None: diffusion.DiffusionBValue = value
            frame.MRDiffusionSequence = [diffusion]
        ds.PerFrameFunctionalGroupsSequence.append(frame)
    if shared:
        group, diffusion = Dataset(), Dataset()
        diffusion.DiffusionBValue = 1400
        group.MRDiffusionSequence = [diffusion]
        ds.SharedFunctionalGroupsSequence = [group]
    ds.PixelData = np.zeros((2, 8, 8), dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    row = mri.assess()['Enhanced']
    assert row['b_values'] == sorted({v for v in values if v is not None})
    assert [f['b_value'] for f in row['instances'][0]['frames']] == values
    assert bool(row['hbv_options']) is available
    assert row['dimensions'] == [8, 8, 2]
    assert ('Shared' if shared else 'PerFrame') in row['instances'][0]['frames'][0]['b_value_path']


def test_suggestions_rank_adc_tags_above_description_and_highest_b(mri):
    _, adc, hbv, _, rows = standard_exam(mri)
    mri.add('ADC description')
    mri.add('Other study ADC', image_type=['ADC'], StudyInstanceUID=generate_uid())
    mri.add('Other study DWI', (2000, 2000, 2000), StudyInstanceUID=generate_uid())
    rows = mri.assess(); primary = rows['T2 axial']; all_series = list(rows.values())
    assert suggest_role(primary, all_series, ROLES[1]) == ({'series_key': adc}, '')
    assert suggest_role(primary, all_series, ROLES[2]) == ({'series_key': hbv, 'b_value': 1400}, '')
    assert all(c['series_key'] != rows['Other study ADC']['key'] for c in role_candidates(primary, all_series, ROLES[1]))
    mri.add('Another tagged ADC', image_type=['ADC'])
    rows = mri.assess()
    assert 'Ambiguous' in suggest_role(rows['T2 axial'], list(rows.values()), ROLES[1])[1]


def test_multib_staging_shared_companions_and_unchanged_adc_bytes(mri):
    t2, adc, hbv, t2b, rows = standard_exam(mri, second=True)
    originals = {p.name: p.read_bytes() for p in (mri.root / 'input').iterdir()}
    response = mri.submit([t2, t2b], {key: {'adc': adc, 'hbv': {'series_key': hbv, 'b_value': 800}} for key in [t2, t2b]}, ['total_mr', 'synthetic_bpmri'])
    assert response.status_code == 202, response.text
    assert len(mri.enqueued) == 2 and not mri.root.exists()
    for job in mri.enqueued:
        data = mri.portal.metadata(job)
        assert data['models'] == ['total_mr', 'synthetic_bpmri']
        assert data['input_roles']['t2w']['dir'] == 'input'
        assert data['input_roles']['hbv']['b_value'] == 800
        assert data['input_roles']['hbv']['b_value_source'] == B_VALUE_SOURCE
        assert set(data['input_manifest']) == {'input', 'input_adc', 'input_hbv'}
        assert mri.portal.input_manifest_complete(job, data)
        assert mri.portal.job_input_manifest(job) == data['input_manifest']
        assert len(list((job / 'input_hbv').iterdir())) == 3
        assert {float(pydicom.dcmread(p).DiffusionBValue) for p in (job / 'input_hbv').iterdir()} == {800}
        for path in (job / 'input_adc').iterdir():
            assert path.read_bytes() == originals[path.name]


def test_ambiguous_role_blocks_only_multisequence_models(mri):
    t2, _, _, _, _ = standard_exam(mri)
    mri.add('Second ADC', image_type=['ADC']); mri.assess()
    response = mri.submit([t2], models=['total_mr', 'synthetic_bpmri'])
    assert response.status_code == 202, response.text
    record = response.json()['jobs'][0]
    assert record['models'] == ['total_mr']
    assert 'Ambiguous ADC' in record['skipped_models']['synthetic_bpmri']
    data = mri.portal.metadata(mri.enqueued[0])
    assert 'input_roles' not in data and set(data['input_manifest']) == {'input'}


def test_explicit_adc_choice_and_frame_warning(mri):
    t2 = mri.add('T2 axial')
    adc = mri.add('Unlabelled map', FrameOfReferenceUID=generate_uid())
    hbv = mri.add('DWI', (1400, 1400, 1400)); mri.assess()
    response = mri.submit([t2], {t2: {'adc': adc, 'hbv': hbv}})
    assert response.status_code == 202, response.text
    record = mri.portal.metadata(mri.enqueued[0])['input_roles']['adc']
    assert record['frame_of_reference_differs']
    assert record['study_uid'] == mri.study


@pytest.mark.parametrize('case,message', [
    ('study', 'StudyInstanceUID'), ('modality', 'requires MR'), ('b', 'Invalid HBV'),
    ('low_b', 'Invalid HBV'), ('unknown', 'Unknown series'), ('role', 'not a companion role'),
    ('primary', 'companion series'), ('bad_shape', 'object'), ('bad_primary', 'primary series'),
    ('null_hbv', 'Missing High'), ('adc_b', 'only supported'), ('bool_b', 'explicit documented'),
    ('path', 'not a companion role')])
def test_validation_is_atomic(mri, case, message):
    t2, adc, hbv, _, _ = standard_exam(mri)
    foreign = mri.add('Foreign ADC', StudyInstanceUID=generate_uid())
    ct = mri.add('CT', Modality='CT')
    mri.assess()
    roles = {t2: {'adc': adc, 'hbv': {'series_key': hbv, 'b_value': 1400}}}
    if case == 'study': roles[t2]['adc'] = foreign
    if case == 'modality': roles[t2]['adc'] = ct
    if case in {'b', 'low_b', 'bool_b'}: roles[t2]['hbv']['b_value'] = {'b': 999, 'low_b': 0, 'bool_b': True}[case]
    if case == 'unknown': roles[t2]['adc'] = 'missing'
    if case == 'role': roles[t2]['other'] = adc
    if case == 'primary': roles[t2]['adc'] = t2
    if case == 'bad_shape': roles[t2] = []
    if case == 'bad_primary': roles['missing'] = roles.pop(t2)
    if case == 'null_hbv': roles[t2]['hbv'] = None
    if case == 'adc_b': roles[t2]['adc'] = {'series_key': adc, 'b_value': 1400}
    if case == 'path': roles[t2]['../../outside'] = adc
    response = mri.submit([t2], roles)
    assert response.status_code == 400, response.text
    assert message in response.json()['detail']
    assert not list(mri.portal.BASE.glob('*/job.json')) and not mri.enqueued
    assert len(list((mri.root / 'input').iterdir())) == 21


@pytest.mark.parametrize('change', ['missing_dir', 'missing_file', 'size', 'identity', 'manifest', 'unsafe_dir'])
def test_partial_or_changed_roles_never_retry_or_recover(mri, change, monkeypatch):
    t2, _, _, _, _ = standard_exam(mri)
    assert mri.submit([t2]).status_code == 202
    job = mri.enqueued[0]
    data = mri.portal.metadata(job)
    if change == 'missing_dir': shutil.rmtree(job / 'input_adc')
    if change == 'missing_file': next((job / 'input_hbv').iterdir()).unlink()
    if change == 'size': next((job / 'input_adc').iterdir()).write_bytes(b'partial')
    if change == 'identity': data['input_roles']['hbv']['b_value'] = 800
    if change == 'manifest': del data['input_manifest']['input_adc']
    if change == 'unsafe_dir': data['input_roles']['adc']['dir'] = '../input_adc'
    data['status'] = 'failed'
    (job / 'job.json').write_text(json.dumps(data))
    assert not mri.portal.input_manifest_complete(job, data)
    response = TestClient(mri.portal.app).post(f'/api/v1/jobs/{job.name}/retry')
    assert response.status_code == 409
    mri.enqueued.clear()
    mri.portal.save_metadata(job, status='queued')
    monkeypatch.setattr(mri.portal.threading, 'Thread', lambda **kw: pytest.fail('Partial input was dispatched'))
    mri.portal.recover_interrupted_jobs()
    assert mri.portal.metadata(job)['status'] == 'failed'
    assert not mri.enqueued


def test_valid_retry_preserves_role_selection(mri):
    t2, _, _, _, _ = standard_exam(mri)
    assert mri.submit([t2]).status_code == 202
    job = mri.enqueued[0]; original = mri.portal.metadata(job)
    mri.portal.save_metadata(job, status='failed')
    response = TestClient(mri.portal.app).post(f'/api/v1/jobs/{job.name}/retry')
    assert response.status_code == 202
    after = mri.portal.metadata(job)
    assert after['input_identity'] == original['input_identity']
    assert after['input_roles'] == original['input_roles']


def test_copy_failure_rolls_back_all_jobs_and_preserves_upload(mri, monkeypatch):
    t2, _, _, t2b, _ = standard_exam(mri, second=True)
    originals = {p.name: p.read_bytes() for p in (mri.root / 'input').iterdir()}
    copy2 = mri.portal.shutil.copy2
    count = 0
    def fail(source, destination):
        nonlocal count
        count += 1
        if count == 8: raise OSError('Synthetic copy interruption')
        return copy2(source, destination)
    monkeypatch.setattr(mri.portal.shutil, 'copy2', fail)
    with pytest.raises(OSError, match='interruption'):
        mri.portal.create_assessed_jobs(mri.root.name, [t2, t2b], ['synthetic_bpmri'])
    assert not list(mri.portal.BASE.glob('*/job.json')) and not mri.enqueued
    assert {p.name: p.read_bytes() for p in (mri.root / 'input').iterdir()} == originals


def test_crash_during_role_copy_recovers_batch_as_partial(mri, monkeypatch):
    t2, _, _, t2b, _ = standard_exam(mri, second=True)
    copy2 = mri.portal.shutil.copy2
    count = 0
    def crash(source, destination):
        nonlocal count
        count += 1
        if count == 2: raise KeyboardInterrupt('Synthetic process exit')
        return copy2(source, destination)
    monkeypatch.setattr(mri.portal.shutil, 'copy2', crash)
    with pytest.raises(KeyboardInterrupt):
        mri.portal.create_assessed_jobs(mri.root.name, [t2, t2b], ['synthetic_bpmri'])
    jobs = [p.parent for p in mri.portal.BASE.glob('*/job.json')]
    assert len(jobs) == 2
    mri.portal.recover_interrupted_jobs()
    for job in jobs:
        assert mri.portal.metadata(job)['input_preparation_interrupted']
        assert not (job / 'input_adc').exists()
    assert len(list((mri.root / 'input').iterdir())) == 18
    assert not mri.enqueued


def test_remote_archive_and_result_sidecar_cleanup(mri, monkeypatch):
    t2, _, _, _, _ = standard_exam(mri)
    assert mri.submit([t2]).status_code == 202
    job = mri.enqueued[0]
    actual_run = mri.portal.subprocess.run
    members = []
    def run(command, **kwargs):
        if command[0] == 'tar': return actual_run(command, **kwargs)
        return SimpleNamespace(returncode=0)
    def transfer(job, archive, remote):
        with tarfile.open(archive) as stream:
            members.extend(stream.getnames())
            data = json.load(stream.extractfile('job.json'))
            assert data['input_roles']['hbv']['b_value'] == 1400
    monkeypatch.setattr(mri.portal.subprocess, 'run', run)
    monkeypatch.setattr(mri.portal, 'transfer_to_worker', transfer)
    mri.portal.prepare_worker_job(job, '/synthetic/job', ['synthetic_bpmri'])
    assert {'input', 'input_adc', 'input_hbv', 'job.json'} <= set(members)
    assert len([m for m in members if m.startswith('input_hbv/')]) == 3
    assert not (job / 'selected-series.tar').exists()
    (job / 'output/synthetic_bpmri_RTSTRUCT.dcm').write_bytes(b'SYNTHETIC RESULT')
    sidecar = job / 'output/synthetic_bpmri_RTSTRUCT.dcm.lesions.json'
    sidecar.write_text('{"lesions": []}')
    mri.portal.finalize_results(job)
    with mri.portal.zipfile.ZipFile(job / 'rtstruct-results.zip') as archive:
        assert sidecar.name in archive.namelist()
    assert (job / 'input').is_dir() and not (job / 'input_adc').exists() and not (job / 'input_hbv').exists()


def test_direct_routes_require_assessment_for_role_models(mri):
    with pytest.raises(mri.portal.HTTPException, match='Sequence roles'):
        mri.portal.validate_job_selection(['synthetic_bpmri'], {'tasks': [MODEL]})


def test_partial_roles_stop_execution_and_transport(mri, monkeypatch):
    t2, _, _, _, _ = standard_exam(mri)
    assert mri.submit([t2]).status_code == 202
    job = mri.enqueued[0]
    shutil.rmtree(job / 'input_hbv')
    monkeypatch.setattr(mri.portal, 'execute_job_task', lambda *a, **kw: pytest.fail('Incomplete MRI executed'))
    monkeypatch.setattr(mri.portal.subprocess, 'run', lambda *a, **kw: pytest.fail('Incomplete MRI transferred'))
    with pytest.raises(ValueError, match='preparing input'):
        mri.portal.prepare_worker_job(job, '/synthetic/job', ['synthetic_bpmri'])
    mri.portal.process(job)
    assert mri.portal.metadata(job)['input_preparation_interrupted']


def test_colab_refuses_roles_before_launching_legacy_runners(mri, monkeypatch):
    t2, _, _, _, _ = standard_exam(mri)
    assert mri.submit([t2]).status_code == 202
    job = mri.enqueued[0]
    monkeypatch.setattr(mri.portal.subprocess, 'run', lambda *a, **kw: pytest.fail('Unexpected Colab launch'))
    monkeypatch.setattr(mri.portal.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Unexpected Colab launch'))
    for runner in [mri.portal.run_colab_job, mri.portal.run_colab_job_via_worker, mri.portal.run_colab_job_on_nfs_worker]:
        with pytest.raises(RuntimeError, match='Colab role transport is not supported'):
            runner(job)


def test_legacy_single_series_manifest_still_works(mri):
    t2 = mri.add('T2 only'); mri.assess()
    response = mri.submit([t2], models=['total_mr'])
    assert response.status_code == 202
    job = mri.enqueued[0]; data = mri.portal.metadata(job)
    assert 'input_roles' not in data and set(data['input_manifest']) == {'input'}
    assert mri.portal.input_manifest_complete(job, data)
    for key in ['input_manifest', 'staging_upload']:
        data.pop(key)
    assert mri.portal.input_manifest_complete(job, data)

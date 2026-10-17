"""Batch planning and FIFO regression tests: synthetic data, no model execution."""
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def series(key, modality='CT', **extra):
    return dict(key=key, series_uid='synthetic-'+key, modality=modality,
                description='Synthetic '+key, slices=24, rows=32, columns=32,
                study_date='20000101', study_uid='synthetic-study', frame_uid='synthetic-frame',
                cbct=False, files=[key+'.dcm'], **extra)


@pytest.fixture
def batch(portal, monkeypatch):
    catalog = [{'name': name, 'modality': modality, 'enabled': True, **extra} for name, modality, extra in [
        ('total', 'CT', {}), ('total_mr', 'MR', {}),
        ('nv_segment_ctmr_ct_body', 'CT', {}), ('nv_segment_ctmr_mri_body', 'MR', {}),
        ('nv_reason_ct_report', 'CT', {'output': 'report'}),
        ('pet_only', 'PT', {}), ('pet_pair', 'PT', {'requires_ct': True}),
        ('synthetic_prompt', 'CT', {'prompt': 'points'}),
        ('unavailable', 'CT', {'enabled': False})]]
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': catalog})
    enqueued = []
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda jobs, **kwargs: enqueued.extend(jobs))
    def stage(rows):
        uid = 'upload-synthetic'
        root = portal.BASE / uid
        (root / 'input').mkdir(parents=True)
        for row in rows:
            for name in row['files']:
                (root / 'input' / name).write_bytes(('SYNTHETIC '+name).encode())
        (root / 'upload.json').write_text(json.dumps({'series': rows}))
        return uid
    client = TestClient(portal.app)
    return SimpleNamespace(portal=portal, client=client, stage=stage, enqueued=enqueued, catalog=catalog)


def submit(batch, rows, keys=None, models=None, **extra):
    uid = batch.stage(rows)
    return batch.client.post(f'/api/v1/uploads/{uid}/jobs', data={
        'series_keys': keys if keys is not None else [s['key'] for s in rows],
        'models': models if models is not None else ['total', 'total_mr'], **extra})


def test_batch_creates_series_order_subsets_and_public_metadata(batch):
    rows = [series('ct'), series('mr', 'MR')]
    response = submit(batch, rows, keys=['mr', 'ct'], models=[t['name'] for t in batch.catalog[:5]])
    assert response.status_code == 202, response.text
    payload = response.json()
    assert [j['series_key'] for j in payload['jobs']] == ['mr', 'ct']
    assert [j['models'] for j in payload['jobs']] == [
        ['total_mr', 'nv_segment_ctmr_mri_body'],
        ['total', 'nv_segment_ctmr_ct_body', 'nv_reason_ct_report']]
    assert payload['ids'] == [job.name for job in batch.enqueued]
    assert payload['ids'] == sorted(payload['ids'])
    for index, jid in enumerate(payload['ids'], 1):
        job = batch.client.get('/api/v1/jobs/'+jid).json()
        assert (job['batch_id'], job['batch_index'], job['batch_size']) == (payload['batch_id'], index, 2)
        assert job['report_status'] == ('pending' if index == 2 else None)
        assert set(job['series']) == {'key', 'series_uid', 'modality', 'description', 'slices', 'rows', 'columns', 'study_date', 'cbct'}
        assert 'files' not in job['series']
        assert len(list((batch.portal.BASE/jid/'input').iterdir())) == 1
    assert not (batch.portal.BASE/'upload-synthetic').exists()
    summaries = batch.client.get('/api/v1/jobs').json()['jobs']
    assert all(j['batch_id'] == payload['batch_id'] and j['series'] for j in summaries)


@pytest.mark.parametrize('keys,models,extra,message', [
    (['ct', 'ct'], ['total'], {}, 'Duplicate series keys: ct'),
    (['missing'], ['total'], {}, 'Unknown series keys: missing'),
    (['ct'], ['unavailable'], {}, 'unavailable'),
    (['ct'], ['not_in_catalog'], {}, 'not_in_catalog'),
    (['ct'], ['total', 'total_mr'], {}, 'total_mr'),
    (['ct', 'mr'], ['total'], {}, 'Synthetic mr'),
    (['ct', 'mr'], ['total', 'total_mr', 'synthetic_prompt'], {'prompt_json': '{}'}, 'single series'),
    (['ct'], ['synthetic_prompt'], {'prompt_json': '[]'}, 'Prompt JSON must be an object'),
    (['ct'], ['synthetic_prompt'], {'prompt_json': '{"positive_points":[[40,0,0]]}'}, 'outside'),
    (['mr'], ['total_mr'], {'postprocess': 'prostate_sbrt_crop'}, 'requires the total'),
    (['ct'], ['total'], {'postprocess': 'unknown'}, 'Unknown postprocess'),
])
def test_invalid_batch_is_atomic(batch, keys, models, extra, message):
    response = submit(batch, [series('ct'), series('mr', 'MR')], keys=keys, models=models, **extra)
    assert response.status_code == 400, response.text
    assert message.lower() in response.json()['detail'].lower()
    assert not list(batch.portal.BASE.glob('*/job.json'))
    assert sorted(path.name for path in batch.portal.BASE.iterdir()) == ['upload-synthetic']
    assert len(list((batch.portal.BASE/'upload-synthetic/input').iterdir())) == 2
    assert not batch.enqueued


def test_empty_series_and_models_rejected_before_creation(batch):
    uid = batch.stage([series('ct')])
    for keys, models in [([], ['total']), (['ct'], [])]:
        with pytest.raises(batch.portal.HTTPException) as error:
            batch.portal.create_assessed_jobs(uid, keys, models)
        assert error.value.status_code == 400
    assert not list(batch.portal.BASE.glob('*/job.json'))


@pytest.mark.parametrize('keys', [['ct', 'pet-a', 'pet-b'], ['pet-a', 'ct', 'pet-b'], ['pet-a', 'pet-b']])
def test_pet_ct_is_copied_before_selected_ct_moves(batch, monkeypatch, keys):
    moved = []
    original = batch.portal.shutil.move
    def move(source, target):
        moved.append(str(target))
        return original(source, target)
    monkeypatch.setattr(batch.portal.shutil, 'move', move)
    response = submit(batch, [series('ct'), series('pet-a', 'PT'), series('pet-b', 'PT')], keys=keys,
                      models=['total', 'pet_pair'] if 'ct' in keys else ['pet_pair'])
    assert response.status_code == 202, response.text
    for job in batch.enqueued:
        metadata = batch.portal.metadata(job)
        if metadata['series']['modality'] == 'PT':
            assert (job/'input_ct/ct.dcm').read_bytes() == b'SYNTHETIC ct.dcm'
            assert all('input_ct' not in name for name in moved)
        else:
            assert (job/'input/ct.dcm').read_bytes() == b'SYNTHETIC ct.dcm'
    assert len(moved) == len(keys)


@pytest.mark.parametrize('change', [{'frame_uid': 'other'}, {'study_uid': ''}, {'frame_uid': ''}])
def test_missing_pet_pair_rejects_whole_batch(batch, change):
    pet = series('pet', 'PT'); pet.update(change)
    response = submit(batch, [series('ct'), pet], models=['total', 'pet_pair'])
    assert response.status_code == 400
    assert 'Synthetic pet' in response.json()['detail'] and 'pet_pair' in response.json()['detail']
    assert not list(batch.portal.BASE.glob('*/job.json'))


def test_reports_skip_cbct_and_mr(batch):
    cbct = series('cbct'); cbct['cbct'] = True
    response = submit(batch, [series('ct'), cbct, series('mr', 'MR')], models=['total', 'total_mr', 'nv_reason_ct_report'])
    assert response.status_code == 202, response.text
    jobs = [batch.portal.metadata(job) for job in batch.enqueued]
    assert [j['report_status'] for j in jobs] == ['pending', None, None]
    assert [j['models'] for j in jobs] == [['total', 'nv_reason_ct_report'], ['total'], ['total_mr']]


@pytest.mark.parametrize('models,message', [(['nv_reason_ct_report'], 'no compatible'), (['total', 'nv_reason_ct_report'], 'apply to no')])
def test_report_only_on_cbct_is_atomic(batch, models, message):
    cbct = series('cbct'); cbct['cbct'] = True
    response = submit(batch, [cbct], models=models)
    assert response.status_code == 400
    assert message in response.json()['detail']
    assert not list(batch.portal.BASE.glob('*/job.json'))


def test_postprocess_applies_only_to_ct_job(batch):
    response = submit(batch, [series('ct'), series('mr', 'MR')], postprocess='prostate_sbrt_crop')
    assert response.status_code == 202
    assert [batch.portal.metadata(j)['postprocess'] for j in batch.enqueued] == [['prostate_sbrt_crop'], []]


def test_remote_postprocess_rejected_atomically(batch, monkeypatch):
    monkeypatch.setattr(batch.portal, 'WORKER_MODE', 'ssh')
    response = submit(batch, [series('ct')], models=['total'], postprocess='prostate_sbrt_crop')
    assert response.status_code == 400
    assert 'local total' in response.json()['detail']
    assert not list(batch.portal.BASE.glob('*/job.json'))


def test_legacy_route_keeps_single_id_and_strict_validation(batch):
    uid = batch.stage([series('ct'), series('mr', 'MR')])
    response = batch.client.post(f'/api/uploads/{uid}/start', data={'series_key': 'ct', 'models': 'total_mr'})
    assert response.status_code == 400
    response = batch.client.post(f'/api/uploads/{uid}/start', data={'series_key': 'ct', 'models': 'total'})
    assert response.status_code == 200, response.text
    assert set(response.json()) == {'id'}
    metadata = batch.portal.metadata(batch.portal.BASE/response.json()['id'])
    assert metadata['batch_size'] == metadata['batch_index'] == 1 and metadata['batch_id']
    assert metadata['series']['key'] == 'ct'
    assert batch.client.post(f'/api/uploads/{uid}/start', data={'series_key': 'mr', 'models': 'total_mr'}).status_code == 404


def test_single_prompt_saved_in_job(batch):
    prompt = {'positive_points': [[3, 4, 1]]}
    response = submit(batch, [series('ct')], models=['synthetic_prompt'], prompt_json=json.dumps(prompt))
    assert response.status_code == 202, response.text
    assert json.loads((batch.enqueued[0]/'prompt.json').read_text()) == prompt


def test_move_failure_restores_stage_and_removes_jobs(batch, monkeypatch):
    uid = batch.stage([series('ct'), series('mr', 'MR')])
    move = batch.portal.shutil.move
    calls = 0
    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('Synthetic move failure')
        return move(source, target)
    monkeypatch.setattr(batch.portal.shutil, 'move', fail_second)
    with pytest.raises(OSError, match='Synthetic move failure'):
        batch.portal.create_assessed_jobs(uid, ['ct', 'mr'], ['total', 'total_mr'])
    assert not list(batch.portal.BASE.glob('*/job.json'))
    assert sorted(p.name for p in (batch.portal.BASE/uid/'input').iterdir()) == ['ct.dcm', 'mr.dcm']
    assert not batch.enqueued


def make_job(portal, jid):
    job = portal.BASE/jid
    (job/'input').mkdir(parents=True)
    (job/'input/synthetic.txt').write_text('Synthetic only')
    (job/'output').mkdir()
    (job/'job.json').write_text(json.dumps({'id': jid, 'status': 'queued', 'created': 1, 'models': ['total'],
                                              'input_manifest': portal.job_input_manifest(job)}))
    return job


def test_fifo_reserves_batch_and_updates_waiting_positions(portal, monkeypatch):
    jobs = [make_job(portal, f'20260101-00000{i}-synthetic') for i in range(4)]
    entered = [threading.Event() for _ in jobs]
    release = [threading.Event() for _ in jobs]
    drained = threading.Event()
    order = []
    def run(job):
        index = jobs.index(job)
        order.append(index)
        entered[index].set()
        assert release[index].wait(30)
        portal.save_metadata(job, status='complete')
    original_drain = portal.drain_jobs
    def drain():
        try:
            original_drain()
        finally:
            drained.set()
    monkeypatch.setattr(portal, 'process', run)
    monkeypatch.setattr(portal, 'drain_jobs', drain)
    try:
        portal.enqueue_jobs(jobs[:1]); assert entered[0].wait(30)
        portal.enqueue_jobs(jobs[1:3]); portal.enqueue_jobs(jobs[3:])
        assert portal.metadata(jobs[1])['detail'] == 'Queued behind 1 earlier job'
        assert portal.metadata(jobs[2])['detail'] == 'Queued behind 2 earlier jobs'
        assert portal.metadata(jobs[3])['detail'] == 'Queued behind 3 earlier jobs'
        release[0].set(); assert entered[1].wait(30)
        assert portal.metadata(jobs[2])['detail'] == 'Queued behind 1 earlier job'
        assert portal.metadata(jobs[3])['detail'] == 'Queued behind 2 earlier jobs'
        release[1].set(); assert entered[2].wait(30)
        assert portal.metadata(jobs[3])['detail'] == 'Queued behind 1 earlier job'
    finally:
        for event in release:
            event.set()
        assert drained.wait(30)
    assert order == [0, 1, 2, 3] and not portal.queue_running


def test_recovery_requeues_in_creation_order(portal, monkeypatch):
    jobs = [make_job(portal, jid) for jid in ['20260103-synthetic', '20260101-synthetic', '20260102-synthetic']]
    queued = []
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda batch, **kwargs: queued.extend(batch))
    portal.recover_interrupted_jobs()
    assert queued == sorted(jobs)
    assert all(portal.metadata(job)['recovery_count'] == 1 for job in queued)


def test_jobs_list_keeps_batches_together_newest_batch_first(batch):
    jobs = [make_job(batch.portal, jid) for jid in ['synthetic-first', 'synthetic-middle', 'synthetic-second']]
    for job, values in zip(jobs, [dict(created=2, batch_id='synthetic-batch', batch_index=1, batch_size=2),
                                dict(created=3), dict(created=4, batch_id='synthetic-batch', batch_index=2, batch_size=2)]):
        batch.portal.save_metadata(job, **values)
    assert [j['id'] for j in batch.client.get('/api/v1/jobs').json()['jobs']] == [
        'synthetic-first', 'synthetic-second', 'synthetic-middle']
    assert [j['id'] for j in batch.client.get('/api/v1/jobs?batch_id=synthetic-batch').json()['jobs']] == [
        'synthetic-first', 'synthetic-second']


def test_assessment_image_type_supports_localiser_preselection(portal, synthetic_ct):
    import pydicom
    path = next(synthetic_ct.iterdir())
    ds = pydicom.dcmread(path)
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'LOCALIZER']
    ds.save_as(path, enforce_file_format=True)
    # Inspect only this fully synthetic image; image_type is public inspection metadata.
    response = TestClient(portal.app).post('/api/uploads', files={'files': (path.name, path.read_bytes())})
    assert response.status_code == 200
    assert response.json()['series'][0]['image_type'] == ['ORIGINAL', 'PRIMARY', 'LOCALIZER']


def test_real_dicom_assessment_to_mixed_batch(batch, synthetic_ct):
    import io
    import pydicom
    from pydicom.uid import MRImageStorage, generate_uid
    ct_path = next(synthetic_ct.iterdir())
    mr = pydicom.dcmread(ct_path)
    mr.Modality = 'MR'
    mr.SeriesInstanceUID = generate_uid()
    mr.SOPClassUID = mr.file_meta.MediaStorageSOPClassUID = MRImageStorage
    mr.SOPInstanceUID = mr.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    mr.SeriesDescription = 'Synthetic MR batch test'
    stream = io.BytesIO()
    mr.save_as(stream, enforce_file_format=True)
    response = batch.client.post('/api/uploads', files=[
        ('files', ('synthetic-ct.dcm', ct_path.read_bytes())),
        ('files', ('synthetic-mr.dcm', stream.getvalue()))])
    assert response.status_code == 200, response.text
    upload = response.json()
    assert [s['modality'] for s in upload['series']] == ['CT', 'MR']
    response = batch.client.post(f"/api/v1/uploads/{upload['upload_id']}/jobs", data={
        'series_keys': [s['key'] for s in upload['series']], 'models': ['total', 'total_mr']})
    assert response.status_code == 202, response.text
    assert [j['models'] for j in response.json()['jobs']] == [['total'], ['total_mr']]
    assert [batch.portal.metadata(j)['series']['modality'] for j in batch.enqueued] == ['CT', 'MR']
    assert pydicom.dcmread(next((batch.enqueued[1]/'input').iterdir())).SeriesInstanceUID == mr.SeriesInstanceUID


def test_recovery_uses_creation_time_for_older_random_id_suffixes(portal, monkeypatch):
    early = make_job(portal, '20260101-000000-ffff')
    late = make_job(portal, '20260101-000000-aaaa')
    portal.save_metadata(early, created=1)
    portal.save_metadata(late, created=2)
    queued = []
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda jobs, **kwargs: queued.extend(jobs))
    portal.recover_interrupted_jobs()
    assert queued == [early, late]


def test_failed_or_expired_queued_jobs_do_not_stall_dispatcher(portal, monkeypatch):
    import shutil
    jobs = [make_job(portal, f'synthetic-dispatch-{i}') for i in range(3)]
    shutil.rmtree(jobs[1])
    order = []
    def run(job):
        order.append(job)
        if job == jobs[0]:
            raise RuntimeError('Synthetic task failure')
        portal.save_metadata(job, status='complete')
    monkeypatch.setattr(portal, 'process', run)
    monkeypatch.setattr(portal, 'threading', SimpleNamespace(Thread=lambda target, args, **kwargs:
        SimpleNamespace(start=lambda: target(*args), is_alive=lambda: False)))
    portal.enqueue_jobs(jobs)
    assert order == [jobs[0], jobs[2]]
    assert portal.metadata(jobs[0])['status'] == 'failed'
    assert portal.metadata(jobs[2])['status'] == 'complete'
    assert not portal.queue_running and not portal.pending_jobs


def test_session_stages_input_while_retaining_fifo_position(portal, monkeypatch, synthetic_ct):
    first = make_job(portal, 'synthetic-earlier-job')
    first_entered, release, staged, drained = (threading.Event() for _ in range(4))
    order = []
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': [{'name': 'total', 'modality': 'CT', 'enabled': True}]})
    original_stage, original_drain = portal.stage_session_files, portal.drain_jobs
    def stage(*args):
        original_stage(*args)
        staged.set()
    def drain():
        try:
            original_drain()
        finally:
            drained.set()
    def run(job):
        order.append(job)
        if job == first:
            first_entered.set()
            assert release.wait(30)
        portal.save_metadata(job, status='complete')
    monkeypatch.setattr(portal, 'stage_session_files', stage)
    monkeypatch.setattr(portal, 'drain_jobs', drain)
    monkeypatch.setattr(portal, 'process', run)
    client = TestClient(portal.app)
    try:
        portal.enqueue_jobs([first])
        assert first_entered.wait(30)
        sid = client.post('/api/v1/upload-sessions').json()['session_id']
        payload = next(synthetic_ct.iterdir()).read_bytes()
        assert client.put(f'/api/v1/upload-sessions/{sid}/files/0',
            params={'name': 'synthetic.dcm', 'offset': 0, 'size': len(payload)}, content=payload).status_code == 200
        response = client.post(f'/api/v1/upload-sessions/{sid}/jobs', data={'models': 'total'})
        assert response.status_code == 202, response.text
        job = portal.BASE/response.json()['id']
        assert staged.wait(30)
        assert any((job/'input').iterdir()), 'Input must be retained before the earlier job finishes'
        assert portal.metadata(job)['detail'] == 'Queued behind 1 earlier job'
        assert order == [first]
    finally:
        release.set()
        assert drained.wait(30)
    assert order == [first, job]
    data = portal.metadata(job)
    assert data['batch_id'] and data['batch_index'] == data['batch_size'] == 1
    assert data['series']['modality'] == 'CT'


def test_direct_submission_cannot_pass_batch_during_companion_copy(batch, monkeypatch):
    copy_started, release, direct_attempted = (threading.Event() for _ in range(3))
    original_copy = batch.portal.shutil.copy2
    errors = []
    def copy(source, target):
        copy_started.set()
        assert release.wait(30)
        return original_copy(source, target)
    monkeypatch.setattr(batch.portal.shutil, 'copy2', copy)
    uid = batch.stage([series('ct'), series('pet', 'PT')])
    direct = batch.portal.BASE/'synthetic-direct'
    (direct/'input').mkdir(parents=True)
    (direct/'output').mkdir()
    def start_batch():
        try:
            batch.portal.create_assessed_jobs(uid, ['pet'], ['pet_pair'])
        except Exception as exc:
            errors.append(exc)
    def start_direct():
        direct_attempted.set()
        try:
            batch.portal.queue_ready_job(direct, {'id': direct.name, 'models': ['total'], 'status': 'queued'})
        except Exception as exc:
            errors.append(exc)
    batch_thread = threading.Thread(target=start_batch)
    direct_thread = threading.Thread(target=start_direct)
    batch_thread.start()
    try:
        assert copy_started.wait(30)
        direct_thread.start()
        assert direct_attempted.wait(30)
        assert not (direct/'job.json').exists()
        assert not batch.enqueued
    finally:
        release.set()
        batch_thread.join(30)
        if direct_thread.ident is not None:
            direct_thread.join(30)
    assert not errors and not batch_thread.is_alive() and not direct_thread.is_alive()
    assert [batch.portal.metadata(j)['models'] for j in batch.enqueued] == [['pet_pair'], ['total']]


def test_direct_single_series_records_batch_fields(batch, synthetic_ct):
    source = next(synthetic_ct.iterdir())
    response = batch.client.post('/api/v1/jobs', data={'models': 'total'},
                                 files={'files': (source.name, source.read_bytes())})
    assert response.status_code == 202, response.text
    data = batch.client.get('/api/v1/jobs/'+response.json()['id']).json()
    assert data['batch_id'] and data['batch_index'] == data['batch_size'] == 1
    assert data['series']['modality'] == 'CT' and data['series']['slices'] == 1

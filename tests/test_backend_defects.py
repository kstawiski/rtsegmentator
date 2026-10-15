"""Backend regression cases using synthetic inputs and CPU-only worker doubles."""
import json
import logging
from pathlib import Path
import shlex
import threading
import time
from types import SimpleNamespace

import pydicom
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parents[1]


def expose(portal, monkeypatch, names, **overrides):
    items = [{**(portal.EXTERNAL_TASKS.get(name) or portal.INFORMATIONAL_MODELS.get(name, {})),
              'name': name, 'enabled': True, **overrides} for name in names]
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': items})
    monkeypatch.setattr(portal, 'process', lambda *args: None)
    return items


def submit(client, route, source, model, prompt=None):
    payload = source.read_bytes()
    data = {'models': model}
    if prompt is not None:
        data['prompt_json'] = json.dumps(prompt)
    if route == 'assessed':
        assessed = client.post('/api/uploads', files={'files': ('image.dcm', payload)}).json()
        data['series_key'] = assessed['series'][0]['key']
        return client.post(f"/api/uploads/{assessed['upload_id']}/start", data=data)
    if route == 'session':
        sid = client.post('/api/v1/upload-sessions').json()['session_id']
        assert client.put(f'/api/v1/upload-sessions/{sid}/files/0',
                          params={'name': 'image.dcm', 'size': len(payload), 'offset': 0},
                          content=payload).status_code == 200
        return client.post(f'/api/v1/upload-sessions/{sid}/jobs', data=data)
    return client.post(route, data=data, files={'files': ('image.dcm', payload)})


@pytest.mark.parametrize('route', ['/api/config', '/api/v1/config'])
def test_config_exact_public_contract(portal, monkeypatch, route):
    monkeypatch.setattr(portal, 'RETENTION_HOURS', 72)
    monkeypatch.setattr(portal, 'MAX_BYTES', 123456)
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {'total', 'body'})
    response = TestClient(portal.app).get(route)
    assert response.status_code == 200
    assert response.json() == dict(retention_hours=72, max_upload_bytes=123456,
        worker_mode='local', remote_gpu_tasks=['body', 'total'], version=portal.app.version,
        report_task='nv_reason_ct_report')


@pytest.mark.parametrize('route', ['/api/jobs', '/api/v1/jobs', 'session', 'assessed'])
@pytest.mark.parametrize('model', ['pam_prompted', 'medsam_vit_b_box_2d', 'voxtell_v1_1_text_prompted'])
def test_every_creation_route_requires_prompt(portal, monkeypatch, synthetic_ct, route, model):
    expose(portal, monkeypatch, [model])
    response = submit(TestClient(portal.app), route, next(synthetic_ct.iterdir()), model)
    assert response.status_code == 400, response.text
    assert 'prompt' in response.json()['detail'].lower()
    assert not list(portal.BASE.glob('*/job.json'))


@pytest.mark.parametrize('route', ['/api/v1/jobs', 'session', 'assessed'])
def test_catalog_prompt_field_is_authoritative(portal, monkeypatch, synthetic_ct, route):
    expose(portal, monkeypatch, ['new_prompt_task'], prompt='points')
    response = submit(TestClient(portal.app), route, next(synthetic_ct.iterdir()), 'new_prompt_task')
    assert response.status_code == 400
    assert 'prompt' in response.json()['detail'].lower()


@pytest.mark.parametrize('route', ['/api/jobs', '/api/v1/jobs', 'session', 'assessed'])
def test_paired_tasks_require_assessed_companion(portal, monkeypatch, synthetic_ct, route):
    model = 'isrt_pet1_early_rigid'
    expose(portal, monkeypatch, [model])
    source = next(synthetic_ct.iterdir())
    ds = pydicom.dcmread(source); ds.Modality = 'PT'; ds.save_as(source, enforce_file_format=True)
    response = submit(TestClient(portal.app), route, source, model)
    assert response.status_code == 400
    assert 'CT' in response.json()['detail']
    assert not list(portal.BASE.glob('*/job.json'))


@pytest.mark.parametrize('model,prompt', [
    ('pam_prompted', {'positive_points': [[2, 3, 0]]}),
    ('medsam_vit_b_box_2d', {'box': [[2, 3, 0], [8, 9, 0]]}),
    ('voxtell_v1_1_text_prompted', {'text': 'spleen'}),
])
def test_assessed_valid_prompt_is_preserved(portal, monkeypatch, synthetic_ct, model, prompt):
    expose(portal, monkeypatch, [model])
    response = submit(TestClient(portal.app), 'assessed', next(synthetic_ct.iterdir()), model, prompt)
    assert response.status_code == 200, response.text
    assert json.loads((portal.BASE / response.json()['id'] / 'prompt.json').read_text()) == prompt


@pytest.mark.parametrize('prompt', [{}, [], {'positive_points': [[32, 0, 0]]},
    {'positive_points': [[float('nan'), 0, 0]]}, {'box': [[0, 0, 0]]}])
def test_assessed_rejects_malformed_prompt(portal, monkeypatch, synthetic_ct, prompt):
    expose(portal, monkeypatch, ['pam_prompted'])
    assert submit(TestClient(portal.app), 'assessed', next(synthetic_ct.iterdir()),
                  'pam_prompted', prompt).status_code == 400


@pytest.mark.parametrize('model', ['sam_med3d_turbo_prompted', 'medsam2_ct_lesion_prompted', 'medical_sam2_prompted_ct'])
def test_coordinate_models_reject_text_only(portal, monkeypatch, synthetic_ct, model):
    expose(portal, monkeypatch, [model])
    response = submit(TestClient(portal.app), 'assessed', next(synthetic_ct.iterdir()), model, {'text': 'spleen'})
    assert response.status_code == 400, response.text


@pytest.mark.parametrize('route', ['/api/v1/jobs', 'session', 'assessed'])
def test_all_routes_honor_multiple_supported_modalities(portal, monkeypatch, synthetic_ct, route):
    expose(portal, monkeypatch, ['synthseg_v1_brain'])
    # Run session staging synchronously; no inference or writer threads escape the test.
    monkeypatch.setattr(portal, 'threading', SimpleNamespace(Thread=lambda target, args, **kwargs:
                        SimpleNamespace(start=lambda: target(*args))))
    response = submit(TestClient(portal.app), route, next(synthetic_ct.iterdir()), 'synthseg_v1_brain')
    assert response.status_code in {200, 202}, response.text
    assert portal.metadata(portal.BASE / response.json()['id'])['status'] == 'queued'


def catalog_with(portal, monkeypatch, details, check, informational=False):
    name = 'synthetic_model'
    monkeypatch.setattr(portal, 'EXTERNAL_TASKS', {} if informational else {name: details})
    monkeypatch.setattr(portal, 'INFORMATIONAL_MODELS', {name: details} if informational else {})
    monkeypatch.setattr(portal, 'ENABLED_EXTERNAL_MODELS', set())
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', set())
    def run(command, **kwargs):
        value = ({name: check} if check is not None else {}) if any('model_preflight.py' in str(x) for x in command) else {'tasks': ['total'], 'licensed': []}
        return SimpleNamespace(stdout=json.dumps(value))
    monkeypatch.setattr(portal.subprocess, 'run', run)
    return next(x for x in portal.build_catalog()['tasks'] if x['name'] == name)


@pytest.mark.parametrize('informational', [False, True])
@pytest.mark.parametrize('details,check,status', [
    ({'enabled': False, 'status': 'checkpoint_unavailable'}, None, 'checkpoint_unavailable'),
    ({'enabled': False, 'blocked_status': 'runtime_unsupported'}, {'ready': True, 'orientation_ready': True}, 'runtime_unsupported'),
    ({'enabled': True, 'requires_preflight': True}, None, 'preflight_unavailable'),
    ({'enabled': True}, {'ready': False, 'missing': ['runtime'], 'orientation_ready': False}, 'dependencies_missing'),
    ({'enabled': True}, {'ready': True, 'orientation_ready': False}, 'orientation_revalidation_pending'),
    ({'enabled': True}, {'ready': True, 'orientation_ready': True}, 'not_enabled_on_this_server'),
])
def test_allowlist_preserves_specific_blockers(portal, monkeypatch, informational, details, check, status):
    item = catalog_with(portal, monkeypatch, details, check, informational)
    assert not item['enabled']
    assert item['status'] == status


def test_refresh_error_preserves_catalog_and_recovers(portal, monkeypatch, caplog):
    good = {'tasks': [{'name': 'cached'}]}
    monkeypatch.setattr(portal, 'build_catalog', lambda: good)
    assert portal.tasks() == good
    built = portal.catalog_state['built']
    entered, finish = threading.Event(), threading.Event()
    def fail():
        entered.set()
        assert finish.wait(5)
        raise RuntimeError('secret /private/config/path')
    monkeypatch.setattr(portal, 'build_catalog', fail)
    monkeypatch.setattr(portal, 'gpu_memory_mb', lambda: (None, None))
    with caplog.at_level(logging.ERROR):
        try:
            assert portal.refresh_catalog_in_background()
            assert entered.wait(5)
            assert portal.health()['catalog_refreshing'] is True
            assert portal.tasks() == good
        finally:
            finish.set()
            deadline = time.monotonic() + 5
            while portal.catalog_state['refreshing'] and time.monotonic() < deadline:
                time.sleep(.01)
    health = TestClient(portal.app).get('/api/v1/health').json()
    assert health['catalog_built_at'] == built
    assert health['catalog_refreshing'] is False
    assert 'RuntimeError' in health['catalog_error']
    assert 'secret' not in health['catalog_error'] and '/private' not in health['catalog_error']
    assert health['status'] == 'degraded'
    assert any(record.exc_info for record in caplog.records)
    assert portal.tasks() == good
    monkeypatch.setattr(portal, 'build_catalog', lambda: {'tasks': []})
    portal.refresh_catalog()
    assert portal.health()['catalog_error'] is None


def test_first_catalog_failure_returns_503(portal, monkeypatch, caplog):
    def fail(): raise OSError('/private/config/path')
    monkeypatch.setattr(portal, 'build_catalog', fail)
    monkeypatch.setattr(portal, 'gpu_memory_mb', lambda: (None, None))
    client = TestClient(portal.app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        response = client.get('/api/v1/models')
    assert response.status_code == 503
    assert 'catalog' in response.json()['detail'].lower()
    health = client.get('/api/v1/health').json()
    assert health['catalog_built_at'] is None
    assert health['catalog_refreshing'] is False
    assert 'OSError' in health['catalog_error'] and '/private' not in health['catalog_error']
    monkeypatch.setattr(portal, 'build_catalog', lambda: {'tasks': []})
    assert client.get('/api/v1/models').status_code == 200


def test_whole_job_ssh_forwards_remote_environment(portal, monkeypatch):
    job = portal.BASE / 'ssh-test'; (job / 'output').mkdir(parents=True)
    (job / 'job.json').write_text(json.dumps({'models': ['total']}))
    monkeypatch.setattr(portal, 'WORKER_MODE', 'ssh')
    expected = {'RTSEG_SCRATCH_ROOT': '/scratch with spaces',
                'RTSEG_NV_REASON_CT_CACHE': '/cache with spaces', 'TOTALSEG_WEIGHTS_PATH': '/weights with spaces'}
    for key, value in zip(['REMOTE_SCRATCH_ROOT', 'REMOTE_NV_REASON_CT_CACHE', 'REMOTE_TOTALSEG_WEIGHTS_PATH'], expected.values()):
        monkeypatch.setattr(portal, key, value)
    calls = []
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs): calls.append(command)
        def poll(self): return 0
    monkeypatch.setattr(portal.subprocess, 'Popen', Process)
    for name in ['prepare_worker_job', 'retrieve_worker_results', 'remove_worker_job', 'finalize_results']:
        monkeypatch.setattr(portal, name, lambda *args: None)
    portal.process(job)
    assert len(calls) == 1
    args = shlex.split(calls[0][-1])
    assert all(f'{key}={value}' in args for key, value in expected.items())


def test_ssh_alias_replaces_unused_port_setting():
    assert 'WORKER_SSH_PORT=' not in (ROOT / '.env.example').read_text()
    deployment = (ROOT / 'docs/DEPLOYMENT.md').read_text()
    assert 'Host rtsegmentator-worker' in deployment and 'Port 2222' in deployment


def test_docker_rtstruct_runtime_libraries_and_import_check():
    dockerfile = (ROOT / 'Dockerfile').read_text()
    packages = {'libgl1', 'libglib2.0-0', 'libxcb1', 'libx11-6', 'libxext6', 'libsm6', 'libice6'}
    assert packages <= set(dockerfile.replace('\\\n', ' ').split())
    assert 'import cv2' in dockerfile and 'import rt_utils' in dockerfile


@pytest.mark.parametrize('pair_state', ['matching', 'different_frame', 'missing_identifiers'])
def test_assessed_pair_requires_present_matching_identifiers(portal, monkeypatch, synthetic_ct, pair_state):
    model = 'isrt_pet1_early_rigid'
    expose(portal, monkeypatch, [model])
    ct = pydicom.dcmread(next(synthetic_ct.iterdir()))
    pet_path = synthetic_ct.parent / 'pet.dcm'
    pet = ct.copy()
    pet.Modality = 'PT'
    pet.SeriesInstanceUID = pydicom.uid.generate_uid()
    pet.SOPInstanceUID = pydicom.uid.generate_uid()
    if pair_state == 'different_frame':
        pet.FrameOfReferenceUID = pydicom.uid.generate_uid()
    pet.save_as(pet_path, enforce_file_format=True)
    ct = pydicom.dcmread(next(synthetic_ct.iterdir()))
    ct_path = synthetic_ct.parent / 'pair-ct.dcm'
    ct.save_as(ct_path, enforce_file_format=True)
    if pair_state == 'missing_identifiers':
        for path in [ct_path, pet_path]:
            ds = pydicom.dcmread(path)
            del ds.StudyInstanceUID
            del ds.FrameOfReferenceUID
            ds.save_as(path, enforce_file_format=True)
    client = TestClient(portal.app)
    assessment = client.post('/api/uploads', files=[('files', (p.name, p.read_bytes())) for p in [ct_path, pet_path]]).json()
    selected = next(g for g in assessment['series'] if g['modality'] == 'PT')
    response = client.post(f"/api/uploads/{assessment['upload_id']}/start",
                           data={'series_key': selected['key'], 'models': model})
    assert response.status_code == (200 if pair_state == 'matching' else 400), response.text
    if pair_state == 'matching':
        job = portal.BASE / response.json()['id']
        assert len(list((job / 'input').iterdir())) == len(list((job / 'input_ct').iterdir())) == 1


def test_allowlist_preserves_report_acceptance_blocker(portal, monkeypatch):
    monkeypatch.setattr(portal, 'REPORT_TASK', 'synthetic_model')
    item = catalog_with(portal, monkeypatch, {'enabled': True},
                        {'ready': True, 'orientation_ready': True, 'acceptance_ready': False})
    assert not item['enabled'] and item['status'] == 'report_acceptance_pending'

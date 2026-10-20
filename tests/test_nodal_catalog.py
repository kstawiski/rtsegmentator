"""Catalogue-only checks with synthetic preflight responses; no model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).parents[1]


def mock_catalog(portal, monkeypatch, blocked=None):
    enabled = set(portal.EXTERNAL_TASKS) | set(portal.INFORMATIONAL_MODELS)
    monkeypatch.setattr(portal, 'enabled_external_models', lambda: enabled)
    def run(command, **kwargs):
        if any('model_preflight.py' in str(arg) for arg in command):
            checks = {name: {'ready': True, 'orientation_ready': True, 'acceptance_ready': True}
                      for name in enabled}
            if blocked:
                checks[blocked] = {'ready': False, 'orientation_ready': False,
                                   'missing': ['synthetic missing dependency']}
            return SimpleNamespace(stdout=json.dumps(checks))
        return SimpleNamespace(stdout=json.dumps({'tasks': [], 'licensed': []}))
    monkeypatch.setattr(portal.subprocess, 'run', run)
    return {t['name']: t for t in portal.build_catalog()['tasks']}


def test_nodal_catalogue_metadata_and_unchanged_gate_reasons(portal, monkeypatch):
    tasks = mock_catalog(portal, monkeypatch)
    for name, data in portal.NODAL_MODEL_DETAILS.items():
        task = tasks[name]
        assert task['code_license'] == data['code_license']
        assert task['weights_license'] == data['weights_license']
        assert task['evidence_sources']
        assert task['license'].startswith('Code: ')
        assert 'license_note' not in task
        if task['enabled']:
            assert not task.get('status', '').startswith('validated')
    for name, status in [('choi_tmli', 'ontology_review_required'),
                         ('clnet_235', 'nodal_decoders_unavailable'),
                         ('lnctvseg_swinunetr', 'inference_contract_unresolved'),
                         ('segrap2023_gtv', 'requires_paired_ct'),
                         ('stunet_hntsmrg_pre', 'packaging_blocked')]:
        assert not tasks[name]['enabled']
        assert tasks[name]['status'] == status
    assert tasks['hnlnl_2d3d']['status'] == 'technical_review_required'
    assert tasks['hnlnl_2d3d']['structure_count'] == 20
    assert tasks['lnctvseg_nnunet_v1']['structure_count'] == 6
    assert tasks['lnctvseg_nnunet_v1']['structure_names'] == json.loads(
        (ROOT / 'catalog/structures.json').read_text())['lnctvseg_nnunet_v1']['names']
    assert 'pathologic' not in tasks['lnsegfm_resenc_l']['structures']
    for name in ('lnq_mediastinal_v1', 'lnq_abdominopelvic_v1', 'lnq_inguinal_v1', 'lnq_axillary_v1'):
        assert not tasks[name].get('noncommercial', False)
        assert tasks[name]['code_license'] == 'Apache-2.0'
        assert tasks[name]['weights_license'] == 'CC BY 4.0'
        assert '19 October 2026' in tasks[name]['evidence_note']
        assert 'Attribution is required' in tasks[name]['evidence_note']
        assert any('fffa35913d9d08234638e482159268fa27322221' in url
                   for url in tasks[name]['evidence_sources'])
    assert tasks['voxtell_v1_1_text_prompted']['noncommercial']
    assert not tasks['lnq_atlas']['noncommercial'] and tasks['lnq_atlas']['code_noncommercial']
    assert not tasks['medical_sam2_prompted_ct']['noncommercial']
    assert tasks['medical_sam2_prompted_ct']['weights_license_unresolved']


def test_descriptive_metadata_cannot_override_live_preflight_failure(portal, monkeypatch):
    name = 'hnlnl_2d3d'
    before = mock_catalog(portal, monkeypatch, blocked=name)[name]
    # Inject after loading as well: even an in-memory overlay cannot change execution.
    forbidden = {'name': 'total_mr', 'enabled': True, 'status': 'validated', 'missing': [],
                 'modality': 'MR', 'modalities': ['MR'], 'prompt': None, 'prompt_contract': {},
                 'requires_ct': False, 'requires_preflight': False, 'licensed': False,
                 'output': 'report', 'execution_backend': 'ssh', 'blocked_status': None,
                 'structure_names': ['Invented_ROI'], 'structure_kind': 'fixed'}
    monkeypatch.setitem(portal.NODAL_MODEL_DETAILS, name,
                        {**portal.NODAL_MODEL_DETAILS[name], **forbidden, 'description': 'Updated description'})
    tasks = mock_catalog(portal, monkeypatch, blocked=name)
    after = tasks[name]
    assert not after['enabled']
    assert after['status'] == 'dependencies_missing'
    assert after['missing'] == ['synthetic missing dependency']
    assert after['description'] == 'Updated description'
    # Clinical display tags are derived from description text; execution fields are not.
    descriptive = {'description', 'clinical_tags'}
    assert {k: v for k, v in before.items() if k not in descriptive} == {
        k: v for k, v in after.items() if k not in descriptive}
    with pytest.raises(portal.HTTPException, match='Unavailable or invalid models'):
        portal.validate_job_selection([name], {'tasks': list(tasks.values())})
    with pytest.raises(ValueError, match='Incompatible MR tasks'):
        portal.validate_job_series([name], {'modality': 'MR'}, tasks)


@pytest.mark.parametrize('key', [
    'enabled', 'status', 'name', 'modality', 'modalities', 'prompt', 'prompt_contract',
    'requires_ct', 'requires_preflight', 'licensed', 'output', 'execution_backend',
    'blocked_status', 'missing', 'structure_names', 'descripton',
])
def test_nodal_loader_rejects_unknown_fields(portal, tmp_path, key):
    path = tmp_path / 'nodal-models.json'
    path.write_text(json.dumps({'hnlnl_2d3d': {'description': 'Allowed', key: True}}))
    with pytest.raises(ValueError, match=rf'nodal-models.json: hnlnl_2d3d has unsupported fields: {key}'):
        portal.load_nodal_model_details(path)


def test_startup_rejects_unknown_nodal_field(portal, monkeypatch):
    read_text = Path.read_text
    def injected(path, *args, **kwargs):
        if path == ROOT / 'catalog/nodal-models.json':
            return json.dumps({'hnlnl_2d3d': {'enabled': True}})
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', injected)
    spec = importlib.util.spec_from_file_location('invalid_nodal_portal', ROOT / 'app.py')
    with pytest.raises(ValueError, match='hnlnl_2d3d has unsupported fields: enabled'):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_api_exposes_nodal_facts_and_static_region_data(portal, monkeypatch):
    tasks = mock_catalog(portal, monkeypatch)
    monkeypatch.setattr(portal, 'build_catalog', lambda: {'tasks': list(tasks.values()), 'clinical_tags': []})
    with TestClient(portal.app) as client:
        response = client.get('/api/v1/models')
        assert response.status_code == 200
        returned = {t['name']: t for t in response.json()['tasks']}
        assert returned['clnet_235']['description'].startswith('CL-Net configuration has 33 nodal entries: 18 neck and 15 chest.')
        assert returned['choi_tmli']['enabled'] is False
        data = client.get('/static/ui-catalog.json').json()
        assert [r['id'] for r in data['nodal']['regions']] == ['head_neck', 'mediastinum', 'abdomen', 'pelvis']


def test_lnctvseg_pinned_manifest_not_invalidated_and_choi_unconfirmed():
    # Digest from the supplied acceptance audit; wording changes must not break the runner's guard.
    import subprocess
    original = subprocess.run(['git', 'show', 'HEAD:models/manifests/lnctvseg.json'],
                              cwd=ROOT, capture_output=True, check=True).stdout
    current = (ROOT / 'models/manifests/lnctvseg.json').read_bytes()
    assert hashlib.sha256(current).digest() == hashlib.sha256(original).digest()
    original_labels = subprocess.run(['git', 'show', 'HEAD:labels/lnctvseg.json'],
                                    cwd=ROOT, capture_output=True, check=True).stdout
    assert (ROOT / 'labels/lnctvseg.json').read_bytes() == original_labels
    mapping = json.loads((ROOT / 'labels/choi-tmli.json').read_text())
    assert mapping['confirmed'] is False
    assert all(value is None for value in mapping['labels'].values())


def test_lnctvseg_catalogue_preserves_vc_coverage_and_export_names(portal, monkeypatch):
    tasks = mock_catalog(portal, monkeypatch)
    for name in ('lnctvseg_nnunet_v1', 'lnctvseg_swinunetr', 'lnctvseg_3duxnet', 'lnctvseg_repuxnet'):
        task = tasks[name]
        assert 'left/right II+III+Va, IV+Vb+Vc and Ib' in task['description']
        assert 'IV+Vb+Vc' in task['structures']
        assert 'released export names abbreviate this pair to IV–Vb' in task['limitations']
        if name == 'lnctvseg_nnunet_v1':
            assert task['structure_count'] == 6
        else:
            assert not task['enabled']
            assert task['status'] == 'inference_contract_unresolved'

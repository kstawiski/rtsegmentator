"""Catalogue-only checks with synthetic preflight responses; no model execution."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

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
    assert tasks['lnq_mediastinal_v1']['noncommercial']
    assert tasks['lnq_abdominopelvic_v1']['noncommercial']
    assert not tasks['lnq_inguinal_v1'].get('noncommercial', False)
    assert tasks['voxtell_v1_1_text_prompted']['noncommercial']
    assert not tasks['lnq_atlas']['noncommercial'] and tasks['lnq_atlas']['code_noncommercial']
    assert not tasks['medical_sam2_prompted_ct']['noncommercial']
    assert tasks['medical_sam2_prompted_ct']['weights_license_unresolved']


def test_descriptive_metadata_cannot_override_live_preflight_failure(portal, monkeypatch):
    tasks = mock_catalog(portal, monkeypatch, blocked='hnlnl_2d3d')
    assert not tasks['hnlnl_2d3d']['enabled']
    assert tasks['hnlnl_2d3d']['status'] == 'dependencies_missing'
    assert tasks['hnlnl_2d3d']['missing'] == ['synthetic missing dependency']
    allowed = {'code_license', 'weights_license', 'weights_license_unresolved',
               'description', 'limitations', 'evidence_note', 'evidence_sources', 'noncommercial', 'code_noncommercial'}
    assert all(set(item) <= allowed for item in portal.NODAL_MODEL_DETAILS.values())


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
    mapping = json.loads((ROOT / 'labels/choi-tmli.json').read_text())
    assert mapping['confirmed'] is False
    assert all(value is None for value in mapping['labels'].values())

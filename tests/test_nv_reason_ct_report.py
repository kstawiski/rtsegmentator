from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RTStructureSetStorage, generate_uid
import pytest
from fastapi.testclient import TestClient

from nv_reason_ct_report import (TASK, REVISION, LUNGS, detect_cbct, hu_check, mask_stats, padding_values_hu,
    covered_regions, parse_sections, check_consistency, polygon_area, rtstruct_summary,
    render_html, thumbnail, write_report)
from test_api import load_app, make_slice


@pytest.mark.parametrize('name', ['Patient Verification', 'Advanced Reconstruction', 'OBI Cone-beam CT',
    'Acuity Cone-beam CT', 'Halcyon - PVA', 'Elekta XVI', 'XVI'])
def test_cbct_names(name):
    assert detect_cbct(SimpleNamespace(ManufacturerModelName=name))['is_cbct']


def test_cbct_heuristic_and_diagnostic():
    assert detect_cbct(SimpleNamespace(Manufacturer='Varian', ConvolutionKernel='Ram-Lak', KVP=125))['is_cbct']
    assert detect_cbct(SimpleNamespace(Manufacturer='Elekta', SeriesDescription='kV CBCT'))['is_cbct']
    assert not detect_cbct(SimpleNamespace(Manufacturer='Siemens', ManufacturerModelName='SOMATOM',
                                           ConvolutionKernel='B31', KVP=120))['is_cbct']
    assert not detect_cbct(SimpleNamespace())['is_cbct']


def test_hu_contract():
    assert hu_check(np.array([-1000] * 20 + [0, 500]))['looks_like_hu']
    assert not hu_check(np.arange(2000))['looks_like_hu']
    assert not hu_check(np.array([-1000, np.nan]))['looks_like_hu']


def padded_ct(pad=-8192.0, slices=8, size=64):
    """Axial slices (slice axis 2): air inside the reconstruction circle, a soft-tissue body with bone, and
    padding outside the circle."""
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    r = np.hypot(x, y)
    plane = np.full((size, size), -1000.0)
    plane[r < size * 0.35] = 40.0
    plane[r < size * 0.08] = 900.0
    plane[r > size / 2 - 0.5] = pad
    return np.repeat(plane[:, :, None], slices, axis=2)


def test_hu_check_excludes_exterior_reconstruction_circle_padding():
    for pad in (-8192, -3024, -2048, -2000):
        volume = padded_ct(pad)
        check = hu_check(volume)
        assert check['looks_like_hu'], pad
        assert check['padding_fraction'] == round(float((volume == pad).mean()), 6) and 0 < check['padding_fraction'] < 0.5
        assert check['raw_hu_min'] == pad and check['hu_min'] == -1000 and check['hu_percentiles'][0] == -1000
        assert hu_check(np.moveaxis(volume, 2, 0), slice_axis=0)['looks_like_hu']
    # The input array is not modified.
    volume = padded_ct(); copy = volume.copy(); hu_check(volume); assert np.array_equal(volume, copy)


def test_hu_check_still_refuses_unexplained_low_values_and_mostly_padding():
    # Low values inside the image are not padding (review counter-example): the original rule refuses them.
    a = np.full((32, 32, 32), -1000, dtype=np.float32)
    a[4:28, 4:28, 4:28] = 40
    a[8:12, 8:12, 8:12] = 900
    a[12:20, 12:20, 12:20] = -8192
    check = hu_check(a)
    assert not check['looks_like_hu'] and check['padding_fraction'] == 0 and check['hu_percentiles'][0] == -8192
    a[12:20, 12:20, 12:20] = np.linspace(-8192, -1501, 512).reshape(8, 8, 8)
    assert not hu_check(a)['looks_like_hu']
    # A volume that is nearly all exterior padding is refused even if the remainder looks like HU.
    b = np.full((40, 40, 4), -8192.0)
    b[18:22, 18:22, :] = -1000
    b[19:21, 19:21, :] = 500
    check = hu_check(b)
    assert not check['looks_like_hu'] and check['reason'] == 'Padding values cover most of slice 0'
    assert not hu_check(np.full((8, 8, 3), -8192.0))['looks_like_hu']
    # Exterior padding does not rescue intensities that are not HU.
    c = padded_ct()
    c[c > -8192] = np.random.default_rng(0).uniform(0, 4000, int((c > -8192).sum()))
    assert not hu_check(c)['looks_like_hu']


def test_hu_check_refuses_border_connected_interior_and_lost_slices():
    # Review round 2: a thin path to the border must not turn interior corruption into padding.
    a = np.full((32, 32, 32), -1000, dtype=np.float32)
    a[4:28, 4:28, 4:28] = 40
    a[8:12, 8:12, 8:12] = 900
    a[12:20, 12:20, 12:20] = -8192
    a[:12, 12, 12:20] = -8192
    check = hu_check(a)
    assert not check['looks_like_hu'] and check['padding_fraction'] == 0 and check['hu_percentiles'][0] == -8192
    a[12:20, 12:20, 12:20] = np.linspace(-8192, -1501, 512).reshape(8, 8, 8)
    assert not hu_check(a)['looks_like_hu']
    # Review round 2: a completely lost acquired slice is refused, not silently dropped.
    b = padded_ct()
    b[:, :, 4] = -8192
    check = hu_check(b)
    assert not check['looks_like_hu'] and check['reason'] == 'Padding values cover most of slice 4'


def test_padding_values_from_dicom_header():
    from types import SimpleNamespace as H
    # Unsigned storage with intercept -8192 (stored 0 fills the exterior), as on the production CT.
    assert padding_values_hu(H(PixelRepresentation=0, RescaleSlope=1, RescaleIntercept=-8192)) == [-8192]
    assert padding_values_hu(H(PixelRepresentation=0, RescaleSlope=1, RescaleIntercept=-1024)) == []
    assert padding_values_hu(H(PixelRepresentation=1, RescaleSlope=1, RescaleIntercept=0)) is None
    assert padding_values_hu(H(PixelRepresentation=1, PixelPaddingValue=-2000, RescaleSlope=1, RescaleIntercept=0)) == [-2000]
    assert padding_values_hu(H(PixelRepresentation=1, PixelPaddingValue=-3024, PixelPaddingRangeLimit=-2000,
                               RescaleSlope=1, RescaleIntercept=0)) == [-3024, -2000]
    # Header-identified padding is excluded; other low values are not, even in the exterior.
    assert hu_check(padded_ct(-8192), padding_values=[-8192])['looks_like_hu']
    assert not hu_check(padded_ct(-3024), padding_values=[-8192])['looks_like_hu']
    assert hu_check(padded_ct(-3024), padding_values=[-3024, -2000])['looks_like_hu']


def test_hu_check_review_round3_regressions():
    from types import SimpleNamespace as H
    # A partly lost slice in a long scan is refused before geometry can discard its padding values.
    a = padded_ct(slices=1079).astype(np.float32)
    a[24:, :, 539] = -8192
    check = hu_check(a, padding_values=[-8192])
    assert not check['looks_like_hu'] and check['reason'] == 'Padding values cover most of slice 539'
    # Header padding that is not below the cutoff authorises no exclusion (it is not "unknown").
    header = H(PixelRepresentation=1, BitsStored=16, PixelPaddingValue=0, RescaleSlope=1, RescaleIntercept=-1024)
    assert padding_values_hu(header) == []
    check = hu_check(padded_ct(-8192).astype(np.float32), padding_values=[])
    assert not check['looks_like_hu'] and check['padding_fraction'] == 0
    # A fractional rescale is compared in the image's float32 representation.
    header = H(PixelRepresentation=0, BitsStored=16, PixelPaddingValue=1, RescaleSlope=0.1, RescaleIntercept=-2048)
    values = padding_values_hu(header)
    assert values == [1 * 0.1 - 2048]
    check = hu_check(padded_ct(values[0]).astype(np.float32), padding_values=values)
    assert check['looks_like_hu'] and check['padding_fraction'] > 0.2
    assert not hu_check(padded_ct(-3024).astype(np.float32), padding_values=values)['looks_like_hu']


def field_volume(outside, size=128, slices=3):
    """Air/tissue/bone image with -8192 wherever outside(y, x) is true."""
    y, x = np.mgrid[:size, :size].astype(float)
    plane = np.full((size, size), -1000.0, dtype=np.float32)
    r = np.hypot(y - (size - 1) / 2, x - (size - 1) / 2)
    plane[r < size * 0.3] = 40
    plane[r < size * 0.06] = 900
    plane[outside(y, x)] = -8192
    return np.repeat(plane[:, :, None], slices, axis=2)


def test_hu_check_reconstruction_field_geometry():
    c = 63.5
    accepted = {
        # Production-like: a large field clipped by the image edges (corners only), centre ~2 px off.
        'clipped, off-centre': lambda y, x: np.hypot(y - (c - 1.7 * 128 / 512), x - c) > 280.4 * 128 / 512,
        'centred circle': lambda y, x: np.hypot(y - c, x - c) > 63.5,
        'shifted 4 px': lambda y, x: np.hypot(y - c - 4, x - c) > 58,
    }
    for name, outside in accepted.items():
        check = hu_check(field_volume(outside), padding_values=[-8192])
        assert check['looks_like_hu'] and check['padding_fraction'] > 0, name
    refused = {
        'ellipse': lambda y, x: np.hypot((y - c) / 60, (x - c) / 40) > 1,
        'rectangular frame': lambda y, x: (y < 8) | (y > 119) | (x < 8) | (x > 119),
        'upper rows only': lambda y, x: y < 16,
    }
    for name, outside in refused.items():
        check = hu_check(field_volume(outside), padding_values=[-8192])
        assert not check['looks_like_hu'] and check['padding_fraction'] == 0, name


def test_hu_check_without_padding_is_unchanged():
    # Air reconstructed slightly below -1000 HU stays part of the distribution; no padding is reported.
    volume = padded_ct(pad=-1024.0)
    check = hu_check(volume)
    assert check['looks_like_hu'] and check['padding_fraction'] == 0 and check['raw_hu_min'] == -1024
    assert check['hu_percentiles'] == [round(float(x), 2) for x in np.percentile(volume, [0.1, 1, 50, 99, 99.9])]


def entry(volume, first=False, last=False):
    return dict(volume_ml=volume, slice_range=[0, 2], touches_first_slice=first, touches_last_slice=last)


def test_regions_boundary_and_truncation():
    coverage = {name: entry(200) for name in LUNGS}
    coverage['heart'] = entry(100)
    assert covered_regions(coverage) == ['chest']
    coverage['heart'] = entry(99.999)
    assert covered_regions(coverage) == []
    coverage.update(liver=entry(500), kidney_left=entry(50))
    assert covered_regions(coverage) == ['abdomen']
    coverage['liver'] = entry(499.999)
    assert covered_regions(coverage) == []
    mask = np.zeros((4, 4, 5), bool); mask[1:3, 1:3, :3] = True
    stats = mask_stats(mask, 1000)
    assert stats['volume_ml'] == 12 and stats['slice_range'] == [0, 2]
    assert stats['touches_first_slice'] and not stats['touches_last_slice']
    assert mask_stats(np.zeros_like(mask), 1)['slice_range'] is None


def test_real_format_parsing_and_flags():
    raw = (Path(__file__).parent / 'fixtures/nv-reason-ct-report.txt').read_text()
    parsed = parse_sections(raw)
    assert len(parsed['findings']) == 7 and len(parsed['impression']) == 2
    assert len(parsed['labels']) == 2
    assert 'continued on' in parsed['findings'][5]['text']
    result = check_consistency(raw, {'liver': entry(600, first=True), 'heart': entry(150)})
    findings = {x['heading']: x for x in result['sections']['findings']}
    assert 'withheld' in findings['Lungs and airways']['text']
    assert findings['Heart and pericardium']['organ_status'] == 'present'
    assert findings['Liver and biliary system']['organ_status'] == 'partial'
    assert result['sections']['impression'][0].startswith('ABSENT ANATOMY FLAG')
    assert result['sections']['labels'][1].startswith('ABSENT ANATOMY FLAG')
    assert all('Synthetic lung statement' not in x['text'] for x in findings.values())


def test_markdown_headings_unmapped_and_unfinished_thinking():
    r = parse_sections('## TECHNIQUE\nSynthetic.\n## FINDINGS\n**Spleen:** Text.\n## IMPRESSION\n- Splenic finding.\n<answer>Splenic lesion</answer>')
    assert r['findings'][0]['heading'] == 'Spleen'
    assert parse_sections('<think>Unfinished private reasoning')['findings'] == []
    r = check_consistency('FINDINGS\nUnknown structure: Text.', {})
    assert any(f['status'] == 'unmapped' for f in r['flags'])


def make_rtstruct(path):
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = RTStructureSetStorage; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = RTStructureSetStorage; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = 'NEVER_COPY_THIS'; ds.PatientID = 'NEVER_COPY_ID'
    roi = Dataset(); roi.ROINumber = 1; roi.ROIName = 'Synthetic organ'; ds.StructureSetROISequence = [roi]
    contour = Dataset(); contour.ReferencedROINumber = 1; contour.ContourSequence = []
    for z in [0, 2]:
        c = Dataset(); c.ContourGeometricType = 'CLOSED_PLANAR'; c.NumberOfContourPoints = 4
        c.ContourData = [0,0,z, 10,0,z, 10,10,z, 0,10,z]; contour.ContourSequence.append(c)
    ds.ROIContourSequence = [contour]; ds.save_as(path, enforce_file_format=True)


def test_contour_volume(tmp_path):
    path = tmp_path / 'total_RTSTRUCT.dcm'; make_rtstruct(path)
    result = rtstruct_summary(path, slice_positions=[0, 2, 4])
    assert result['model'] == 'total'
    assert result['rois'][0]['approx_volume_ml'] == .4
    assert result['rois'][0]['slice_range'] == [0, 1]
    assert result['rois'][0]['slice_range_mm'] == [0, 2]
    assert 'NEVER_COPY' not in json.dumps(result)
    assert polygon_area([[0,0,0],[10,0,0],[10,0,10],[0,0,10]]) == 100


def report(status='not_generated'):
    return dict(status=status, reason='No supported region', coverage={}, regions=[], segmentation_summary=[])


def test_self_contained_html(tmp_path):
    r = report('generated'); region = check_consistency('FINDINGS\nSpleen: <script>alert(1)</script>', {})
    region.update(region='abdomen', crop_path='segmentation_guided', crop_qc={},
                  raw_text='<think>reasoning</think><script>alert(1)</script>', thumbnail_png=thumbnail(np.zeros((4,4,4))))
    r['regions'] = [region]; text = render_html(r)
    assert 'https://' not in text and 'http://' not in text
    assert '<script>' not in text and '<details>' in text
    assert 'data:image/png;base64,' in text and 'Requires review by a radiologist' in text
    assert all(x not in text for x in ['PatientName', 'PatientID', 'PatientBirthDate'])
    write_report(tmp_path, r)
    assert len(list(tmp_path.iterdir())) == 3
    assert json.loads((tmp_path / (TASK + '.json')).read_text())['regions'][0]['raw_text'] == region['raw_text']


def portal(tmp_path, monkeypatch):
    api = load_app(tmp_path); mod = sys.modules['app']
    monkeypatch.setattr(mod, 'tasks', lambda: {'tasks': [{'name': TASK, 'enabled': True}, {'name': 'total', 'enabled': True}]})
    monkeypatch.setattr(mod, 'process', lambda job, *args: None)
    return api, mod


@pytest.mark.parametrize('modality,cbct', [('MR', False), ('PT', False), ('CT', True)])
def test_api_rejects_incompatible_series(tmp_path, monkeypatch, modality, cbct):
    api, mod = portal(tmp_path, monkeypatch)
    path = tmp_path / 'slice.dcm'; make_slice(path, generate_uid(), 1)
    ds = pydicom.dcmread(path); ds.Modality = modality
    if cbct: ds.ManufacturerModelName = 'Patient Verification'
    ds.save_as(path, enforce_file_format=True)
    with TestClient(api) as client:
        response = client.post('/api/v1/jobs', data={'models': TASK}, files={'files': ('slice.dcm', path.read_bytes())})
        assert response.status_code == 400
        if cbct: assert 'CBCT' in response.text
        upload = client.post('/api/uploads', files={'files': ('slice.dcm', path.read_bytes())}).json()
        series = upload['series'][0]
        assert series['cbct'] == cbct
        response = client.post('/api/uploads/' + upload['upload_id'] + '/start',
                               data={'models': TASK, 'series_key': series['key']})
        assert response.status_code == 400


def test_catalog_report_gate(tmp_path, monkeypatch):
    load_app(tmp_path); mod = sys.modules['app']
    def run(cmd, **kwargs):
        if any('model_preflight.py' in str(x) for x in cmd):
            return SimpleNamespace(stdout=json.dumps({TASK: {'ready': True, 'acceptance_ready': True,
                                                            'orientation_ready': False}}))
        return SimpleNamespace(stdout=json.dumps({'tasks': ['total'], 'licensed': []}))
    monkeypatch.setattr(mod.subprocess, 'run', run)
    monkeypatch.setattr(mod, 'ENABLED_EXTERNAL_MODELS', {TASK})
    item = next(x for x in mod.tasks()['tasks'] if x['name'] == TASK)
    assert item['enabled'] and item['output'] == 'report' and item['modalities'] == ['CT']
    assert item['license'] == 'OpenMDW-1.1'
    monkeypatch.setattr(mod, 'ENABLED_EXTERNAL_MODELS', set())
    assert not next(x for x in mod.refresh_catalog()['tasks'] if x['name'] == TASK)['enabled']
    assert not next(x for x in mod.tasks()['tasks'] if x['name'] == 'lungtumormask_ct')['enabled']


@pytest.mark.parametrize('with_segmentation', [False, True])
def test_finalize_and_routes(tmp_path, monkeypatch, with_segmentation):
    api, mod = portal(tmp_path, monkeypatch)
    job = mod.BASE / 'test-job'; (job / 'output').mkdir(parents=True); (job / 'input').mkdir()
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'status': 'processing'}))
    write_report(job / 'output/report', report())
    if with_segmentation: make_rtstruct(job / 'output/total_RTSTRUCT.dcm')
    mod.finalize_results(job)
    assert mod.metadata(job)['status'] == 'complete'
    assert mod.metadata(job)['report_status'] == 'not_generated'
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as z:
        assert {'report/' + TASK + x for x in ('.json', '.md', '.html')} <= set(z.namelist())
        assert ('total_RTSTRUCT.dcm' in z.namelist()) == with_segmentation
    with TestClient(api) as client:
        assert client.get('/api/v1/jobs/test-job/report').json()['status'] == 'not_generated'
        response = client.get('/jobs/test-job/report'); assert response.status_code == 200
        assert "default-src 'none'" in response.headers['content-security-policy']
        assert client.get('/api/v1/jobs/test-job/result').status_code == 200


def test_ordered_execution_and_ssh_retrieval(tmp_path, monkeypatch):
    load_app(tmp_path); mod = sys.modules['app']; calls = []
    job = mod.BASE / 'ordered'; job.mkdir()
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'models': [TASK, 'total', 'body']}))
    monkeypatch.setattr(mod, 'run_local_task', lambda job, task, i, total: calls.append(task))
    monkeypatch.setattr(mod, 'finalize_results', lambda job, *args: None)
    mod.process(job)
    assert calls == ['total', 'body', TASK]
    calls.clear(); monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kw: calls.append(command))
    mod.retrieve_worker_results(job, '/jobs/example')
    assert '-r' in calls[0] and any('/output/.' in x for x in calls[0])  # includes report/ and DICOMs


def test_preflight_runtime_fallback_and_marker(tmp_path, monkeypatch):
    import model_preflight as m
    monkeypatch.setattr(m, 'MODEL_ROOT', tmp_path)
    monkeypatch.setattr(m, 'RUNTIME_ROOT', tmp_path / 'runtime')
    monkeypatch.delenv('RTSEG_NV_REASON_CT_PYTHON', raising=False)
    fallback = tmp_path / 'runtime-envs/nv-reason-ct/bin/python'; fallback.parent.mkdir(parents=True); fallback.touch()
    assert m.nv_reason_python() == fallback
    monkeypatch.setenv('RTSEG_NV_REASON_CT_PYTHON', '/explicit/python')
    assert str(m.nv_reason_python()) == '/explicit/python'
    marker = tmp_path / 'validation/nv_reason_ct_report.json'; marker.parent.mkdir()
    marker.write_text('{}'); assert not m.report_acceptance_ready()
    marker.write_text(json.dumps({'status': 'passed', 'revision': REVISION, 'checks': ['chest','abdomen','pelvis','cbct'], 'coverage_mode':'fast'}))
    assert m.report_acceptance_ready()
    assert m.audit()[TASK]['acceptance_ready']


def test_report_crash_preserves_segmentation(tmp_path, monkeypatch):
    load_app(tmp_path); mod = sys.modules['app']; job = mod.BASE / 'crash'
    (job / 'output').mkdir(parents=True)
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'models': ['total', TASK]}))
    def run(job, task, *args):
        if task == 'total': (job / 'output/total_RTSTRUCT.dcm').write_bytes(b'existing result')
        else: raise RuntimeError('model crashed')
    monkeypatch.setattr(mod, 'run_local_task', run)
    mod.process(job)
    assert mod.metadata(job)['status'] == 'failed'
    assert (job / 'output/total_RTSTRUCT.dcm').read_bytes() == b'existing result'


def test_ssh_report_crash_retrieves_existing_results(tmp_path, monkeypatch):
    load_app(tmp_path); mod = sys.modules['app']; job = mod.BASE / 'remote-crash'
    (job / 'output').mkdir(parents=True); (job / 'input').mkdir()
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'models': ['total', TASK]}))
    monkeypatch.setattr(mod, 'WORKER_MODE', 'ssh')
    commands = []
    monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kw: commands.append(command))
    monkeypatch.setattr(mod, 'transfer_to_worker', lambda *a: None)
    def run(job, remote, task, *args):
        if task == TASK: raise RuntimeError('report crashed')
    monkeypatch.setattr(mod, 'run_worker_task', run)
    mod.process(job)
    assert mod.metadata(job)['status'] == 'failed'
    assert mod.metadata(job)['completed_models'] == ['total']
    retrieval = next(i for i,c in enumerate(commands) if c[0] == 'scp')
    deletion = next(i for i,c in enumerate(commands) if c[0] == 'ssh' and 'rm -rf' in c[-1])
    assert retrieval < deletion and '/output/.' in commands[retrieval][-2]


def test_retry_report_preserves_finished_segmentation(tmp_path, monkeypatch):
    api, mod = portal(tmp_path, monkeypatch); job = mod.BASE / 'retry-report'
    (job / 'output').mkdir(parents=True); (job / 'input').mkdir()
    (job / 'input/source.dcm').write_bytes(b'source')
    (job / 'output/total_RTSTRUCT.dcm').write_bytes(b'finished')
    # A job from this version records its complete input; retry checks it before requeueing.
    (job / 'job.json').write_text(json.dumps({'id':job.name,'status':'failed','models':['total',TASK],
                                            'completed_models':['total'],
                                            'input_manifest': mod.job_input_manifest(job)}))
    with TestClient(api) as client:
        assert client.post('/api/v1/jobs/retry-report/retry').status_code == 202
    assert (job / 'output/total_RTSTRUCT.dcm').read_bytes() == b'finished'


def test_report_acceptance_does_not_bypass_segmentation_gate(tmp_path, monkeypatch):
    load_app(tmp_path); mod = sys.modules['app']
    monkeypatch.setattr(mod, 'ENABLED_EXTERNAL_MODELS', {TASK, 'lungtumormask_ct'})
    def run(cmd, **kwargs):
        if any('model_preflight.py' in str(x) for x in cmd):
            return SimpleNamespace(stdout=json.dumps({
                TASK: {'ready': True, 'acceptance_ready': False, 'orientation_ready': True},
                'lungtumormask_ct': {'ready': True, 'orientation_ready': False}}))
        return SimpleNamespace(stdout=json.dumps({'tasks':['total'],'licensed':[]}))
    monkeypatch.setattr(mod.subprocess, 'run', run)
    items = {x['name']:x for x in mod.tasks()['tasks']}
    assert items[TASK]['status'] == 'report_acceptance_pending'
    assert not items[TASK]['enabled']
    assert items['lungtumormask_ct']['status'] == 'orientation_revalidation_pending'
    assert not items['lungtumormask_ct']['enabled']


def test_gpu_wait_is_bounded_and_can_recover():
    from nv_reason_ct_report import wait_for_free_memory
    now = [0.0]; attempts = []
    def clock(): return now[0]
    def sleep(seconds): now[0] += seconds
    def unavailable(): attempts.append(now[0]); return None
    assert wait_for_free_memory(unavailable, 7, clock=clock, sleep=sleep) is None
    assert attempts == [0, 5, 7]
    now[0] = 0
    def available_later(): return (13 if now[0] < 5 else 14) * 2**30
    assert wait_for_free_memory(available_later, 7, clock=clock, sleep=sleep) == 14
    assert now[0] == 5


def test_acceptance_headings_bones_and_single_unchecked_note():
    from nv_reason_ct_report import render_markdown, organ_status, MAPPING
    raw = '''FINDINGS:
Medical devices: None present.
CHEST:
Pleura: No effusion.
Mediastinum and hila: Synthetic statement.
Great vessels: Synthetic statement.
Chest wall and breast: Synthetic statement.
Bones: Synthetic statement.
Liver: Synthetic statement.
Spleen: Synthetic statement.
Unknown heading: Unchanged text.
Unknown heading: Second text.
IMPRESSION:
Synthetic impression.'''
    coverage = {name: entry(250) for name in LUNGS}
    coverage.update(heart=entry(150), aorta=entry(30), esophagus=entry(10), trachea=entry(10),
                    pulmonary_vein=entry(10), rib_left_1=entry(5,first=True), sternum=entry(5),
                    costal_cartilages=entry(5), liver=entry(600,first=True), spleen=entry(100,last=True))
    region = check_consistency(raw, coverage, 'chest')
    findings = {f['heading']:f for f in region['sections']['findings']}
    assert findings['Bones']['organ_status'] == 'present'
    assert 'partly' not in findings['Bones']['text']
    assert findings['Liver']['text'].endswith('[partly in the scanned volume]')
    assert findings['Spleen']['organ_status'] == 'partial'
    for heading in ['Pleura','Mediastinum and hila','Great vessels','Chest wall and breast']:
        assert findings[heading]['organ_status'] != 'unmapped'
    assert findings['Medical devices']['text'] == 'None present.'
    assert findings['CHEST']['text'] == ''
    assert findings['Unknown heading']['text'] == 'Second text.'
    assert region['unchecked_headings'] == ['Medical devices','CHEST','Unknown heading']
    region.update(region='chest',crop_path='publisher_anatomy',crop_qc={},raw_text=raw)
    r=report('generated'); r['regions']=[region,region]
    for rendered in (render_html(r), render_markdown(r)):
        assert 'COVERAGE NOT CHECKED:' not in rendered
        assert 'PARTIAL COVERAGE' not in rendered
        assert rendered.count('Not checked against the segmentation: Medical devices; CHEST; Unknown heading.') == 1
    assert organ_status({'hip_left':entry(200)}, MAPPING['bones'], 'chest') == 'absent'


def test_abdominal_acceptance_headings():
    raw = '''FINDINGS:
ABDOMEN:
Bowel and appendix: Synthetic bowel statement.
Peritoneum and omentum: Synthetic peritoneal statement.
Lymph nodes: Synthetic nodal statement.
Abdominal wall and soft tissues: Synthetic soft tissue statement.
Bones: Synthetic bones statement.
IMPRESSION:
Synthetic impression.'''
    r=check_consistency(raw, {'small_bowel':entry(150),'colon':entry(150),'duodenum':entry(20),
                             'vertebrae_L1':entry(3,first=True)}, 'abdomen')
    findings={f['heading']:f for f in r['sections']['findings']}
    assert findings['Bowel and appendix']['organ_status'] == 'present'
    assert findings['Bones']['organ_status'] == 'present'
    assert r['unchecked_headings'] == ['ABDOMEN','Peritoneum and omentum','Lymph nodes','Abdominal wall and soft tissues']


@pytest.mark.parametrize('mode,fast', [('fast',True),('full',False)])
def test_coverage_mode_command_and_offline_weights(tmp_path, monkeypatch, mode, fast):
    import run_nv_reason_ct as worker
    weights=tmp_path/'weights'; weights.mkdir()
    prefixes=['Dataset297_'] if fast else ['Dataset291_','Dataset292_','Dataset293_','Dataset294_','Dataset295_']
    for prefix in prefixes: (weights/(prefix+'synthetic')).mkdir()
    monkeypatch.setenv('TOTALSEG_WEIGHTS_PATH',str(weights))
    work=tmp_path/'work'; work.mkdir(); masks=work/'masks'; masks.mkdir()
    calls=[]
    def run(command, **kwargs): calls.append((command,kwargs)); return SimpleNamespace(stdout='2.17.0')
    monkeypatch.setattr(worker.subprocess,'run',run)
    coverage,version=worker.coverage_segmentation(tmp_path/'input.nii.gz',masks,work,tmp_path/'bin/python',mode)
    assert version == '2.17.0' and coverage == {}
    command,options=calls[-1]
    assert ('--fast' in command) is fast
    assert '--roi_subset' not in command
    assert options['env']['TOTALSEG_WEIGHTS_PATH'] == str(weights)
    config=json.loads((work/'totalsegmentator/config.json').read_text())
    assert config['send_usage_stats'] is False
    for child in weights.iterdir(): child.rmdir()
    calls.clear()
    with pytest.raises(RuntimeError,match='weights are missing'):
        worker.coverage_segmentation(tmp_path/'input.nii.gz',masks,work,tmp_path/'bin/python',mode)
    assert not calls


def test_acceptance_marker_matches_coverage_mode(tmp_path, monkeypatch):
    import model_preflight as m
    monkeypatch.setattr(m,'MODEL_ROOT',tmp_path)
    marker=tmp_path/'validation/nv_reason_ct_report.json'; marker.parent.mkdir()
    marker.write_text(json.dumps({'status':'passed','revision':REVISION,'checks':['chest','abdomen','pelvis','cbct']}))
    monkeypatch.setenv('RTSEG_NV_REASON_CT_COVERAGE_MODE','fast')
    assert not m.report_acceptance_ready()  # Historical markers attest full resolution only.
    monkeypatch.setenv('RTSEG_NV_REASON_CT_COVERAGE_MODE','full')
    assert m.report_acceptance_ready()
    data=json.loads(marker.read_text()); data['coverage_mode']='fast'; marker.write_text(json.dumps(data))
    assert not m.report_acceptance_ready()
    monkeypatch.setenv('RTSEG_NV_REASON_CT_COVERAGE_MODE','fast')
    assert m.report_acceptance_ready()

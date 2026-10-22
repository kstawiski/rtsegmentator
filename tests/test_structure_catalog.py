"""Check generated contracts against runner inputs, without starting inference."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_structure_catalog as builder
from structure_labels import (INLINE_LABELS, PROMPTED_TASKS, REPO_LABEL_FILES,
                              dataset_labels, final_roi_names, kept_label,
                              repo_label_path, spec_labels, unique_roi_names)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'catalog/structures.json').read_text())
REGISTRIES = builder.registries()
SPEC_REGISTRIES = ('CERR_TASKS', 'ANTSPYNET_TASKS', 'PYCAD_TASKS', 'MRANNOTATOR_TASKS',
                   'CADS_OPEN_TASKS', 'CADS_RESEARCH_TASKS', 'NNUNET_V1_TASKS', 'MONAI_BUNDLE_TASKS')


def test_catalog_covers_all_registry_tasks_and_respects_writer_names():
    tasks = {**{name: {} for name in REGISTRIES['FALLBACK_TASKS']},
             **REGISTRIES['EXTERNAL_TASKS'], **REGISTRIES['INFORMATIONAL_MODELS']}
    assert tasks.keys() <= CATALOG.keys()
    for task, entry in CATALOG.items():
        assert set(entry) == {'kind', 'names', 'count', 'source', 'note'}, task
        assert entry['kind'] in {'fixed', 'variable', 'prompted', 'report', 'unresolved'}, task
        names = entry['names']
        assert names == unique_roi_names(names), task
        assert len({name.casefold() for name in names}) == len(names), task
        assert all(0 < len(name) <= 64 for name in names), task
        assert entry['source'] and entry['note'], task
        if entry['kind'] == 'fixed':
            assert entry['count'] == len(names) > 0, task
        else:
            assert entry['count'] is None, task
        details = tasks.get(task, {})
        if details.get('enabled', True) and not details.get('blocked_status'):
            assert entry['kind'] != 'unresolved', task
        if entry['kind'] == 'unresolved':
            assert details.get('enabled') is False or details.get('blocked_status'), task
            assert not names, task


@pytest.mark.parametrize('task', sorted(REPO_LABEL_FILES))
def test_every_repo_file_contract_uses_runtime_helper(task):
    labels = json.loads(repo_label_path(task, ROOT).read_text())
    entry, provenance = builder.resolve(task, {}, REGISTRIES, Path('unused'), {}, Path('unused'))
    assert CATALOG[task]['names'] == entry['names'] == final_roi_names(labels)
    assert 'structure_labels.repo_label_path' in provenance
    assert entry['kind'] == ('prompted' if task in PROMPTED_TASKS else 'fixed')


REPO_SPECS = [(registry, task) for registry in SPEC_REGISTRIES
              for task, spec in REGISTRIES[registry].items()
              if spec.get('labels_file') or 'labels' in spec]


@pytest.mark.parametrize('registry,task', REPO_SPECS)
def test_every_repo_registry_label_contract_uses_runtime_helper(registry, task):
    labels = spec_labels(REGISTRIES[registry][task], ROOT, Path('unused'))
    entry, provenance = builder.resolve(task, {}, REGISTRIES, Path('unused'), {}, Path('unused'))
    assert CATALOG[task]['names'] == entry['names'] == final_roi_names(labels)
    assert 'structure_labels.spec_labels' in provenance


@pytest.mark.parametrize('task', sorted(INLINE_LABELS))
def test_inline_export_contracts(task):
    assert CATALOG[task]['names'] == final_roi_names(INLINE_LABELS[task])


def test_repo_derived_special_exporters():
    from structure_labels import sct_labels
    for task, spec in REGISTRIES['SCT_TASKS'].items():
        if task in builder.SCT_SINGLE_TARGET_MODELS:
            continue
        assert CATALOG[task]['names'] == final_roi_names(sct_labels(spec, ROOT)), task
    masks = json.loads((ROOT / 'labels/multitalent-v1-masks.json').read_text())
    for task in REGISTRIES['MULTITALENT_V1_TASKS']:
        assert CATALOG[task]['names'] == unique_roi_names(masks.values())
    for task, spec in REGISTRIES['NNUNET_V2_TASKS'].items():
        if 'keep_label' in spec:
            assert CATALOG[task]['names'] == final_roi_names(kept_label(spec['keep_label'], spec.get('keep_name')))


def test_shared_functions_remain_on_runner_paths():
    calls = {node.func.id for node in ast.walk(ast.parse((ROOT / 'run_task.py').read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {'spec_labels', 'repo_label_path'} <= calls
    for filename, function in [('run_moose.py', 'dataset_labels'), ('run_nnunet_v2.py', 'dataset_labels'),
                               ('lut_to_labels.py', 'lut_labels'), ('labels_from_segmentation_lut.py', 'lut_labels'),
                               ('combine_sct_outputs.py', 'sct_output_name')]:
        calls = {node.func.id for node in ast.walk(ast.parse((ROOT / filename).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert function in calls, filename
    from rtstruct_geometry import unique_roi_names as writer_names
    assert writer_names is unique_roi_names


def test_name_normalization_and_dataset_export_semantics():
    assert final_roi_names({'2': ' Second ', '0': 'background', '1': ' First '}) == ['First', 'Second']
    assert dataset_labels({'background': 0, 'Liver': 7, 'Spleen': 2}) == {'7': 'Liver', '2': 'Spleen'}
    with pytest.raises(SystemExit, match='Overlapping region'):
        dataset_labels({'Region': [1, 2]})
    for names in ([''], ['A' * 65], ['Target', ' Target ']):
        with pytest.raises(SystemExit):
            unique_roi_names(names)


def test_named_spot_checks_and_aggregate_node_export():
    expected = {'nv_segment_ctmr_ct_body': 117, 'total': 117, 'moose_clin_ct_lungs': 5,
                'sct_spinalcord': 1, 'multitalent_v1_unet_ct': 47, 'monai_whole_brain_unest_mr': 132,
                'gouhfi_v2_brain_mr': 35, 'cervix_cbct': 4, 'hippunfold_adni_t1_subfields_mr': 16,
                'fastsurfervinn_t1_mr': 95}
    for task, count in expected.items():
        assert CATALOG[task]['count'] == count, task
    assert CATALOG['nv_segment_ctmr_ct_body']['names'] == final_roi_names(json.loads(
        (ROOT / 'labels/nv_segment_ctmr_ct_body.json').read_text()))
    assert CATALOG['cervix_cbct']['names'] == ['CTVT', 'CP', 'RECTUM', 'VESSIE']
    assert CATALOG['sct_spinalcord']['names'] == ['Spinal_Cord']
    assert CATALOG['lnq_mediastinal_v1']['kind'] == 'fixed'
    assert CATALOG['lnq_mediastinal_v1']['names'] == ['Mediastinal_LN_Candidates']
    assert 'aggregate ROI' in CATALOG['lnq_mediastinal_v1']['note']
    assert CATALOG['nninteractive_v1_prompted']['kind'] == 'prompted'
    assert CATALOG['nninteractive_v1_prompted']['names'] == ['Prompted_Target']


def test_no_structure_totals_in_presentation_overrides():
    data = json.loads((ROOT / 'static/ui-catalog.json').read_text())
    assert all(not ({'count', 'structures'} & item.keys()) for item in data['overrides'].values())


def test_unknown_worker_task_is_disabled_until_contract_is_generated(portal, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(portal, 'EXTERNAL_TASKS', {})
    monkeypatch.setattr(portal, 'INFORMATIONAL_MODELS', {})
    def metadata(command, **kwargs):
        data = {} if any('model_preflight.py' in str(arg) for arg in command) else {'tasks': ['new_worker_task'], 'licensed': []}
        return SimpleNamespace(stdout=json.dumps(data))
    monkeypatch.setattr(portal.subprocess, 'run', metadata)
    item = portal.build_catalog()['tasks'][0]
    assert item['structure_kind'] == 'unresolved'
    assert item['structure_names'] == [] and item['structure_count'] is None
    assert not item['enabled'] and item['status'] == 'structure_metadata_unavailable'


def bundle_root():
    value = os.environ.get('RTSEG_STRUCTURE_BUNDLE_ROOT')
    if not value:
        pytest.skip('Set RTSEG_STRUCTURE_BUNDLE_ROOT for the read-only metadata integration check')
    return Path(value)


def test_bundle_resolution_and_check_mode(tmp_path):
    bundle = bundle_root()
    catalog, log = builder.build(bundle)
    assert catalog == CATALOG
    assert catalog.keys() == log.keys()
    maps, tasks, _ = builder.total_maps()
    for task in tasks:
        assert catalog[task]['names'] == final_roi_names(maps[task])
    # No ML module or checkpoint should be imported by the generator.
    code = "import sys; from scripts.build_structure_catalog import build; from pathlib import Path; build(Path(sys.argv[1])); assert not ({'torch','tensorflow','monai','nnunetv2'} & sys.modules.keys())"
    result = subprocess.run([sys.executable, '-c', code, str(bundle)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    command = [sys.executable, str(ROOT / 'scripts/build_structure_catalog.py'), '--bundle-root', str(bundle), '--check']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'Checked {len(CATALOG)} tasks' in result.stdout
    altered = tmp_path / 'structures.json'
    changed = json.loads(json.dumps(catalog))
    changed['total']['names'][0] = 'wrong'
    altered.write_text(json.dumps(changed, indent=2) + '\n')
    result = subprocess.run([*command, '--output', str(altered)], capture_output=True, text=True)
    assert result.returncode == 1
    assert '-      "wrong"' in result.stdout and '+      "spleen"' in result.stdout


def test_enabled_task_metadata_failure_stops_generation(monkeypatch):
    monkeypatch.setattr(builder, 'registries', lambda root: {'FALLBACK_TASKS': [],
        'EXTERNAL_TASKS': {'required_task': {}}, 'INFORMATIONAL_MODELS': {}})
    monkeypatch.setattr(builder, 'total_maps', lambda root: ({}, set(), Path('map_to_binary.py')))
    with pytest.raises(ValueError, match='Enabled task resolution failed.*',):
        builder.build(Path('missing'))


def test_lookup_tables_and_sct_aliases_keep_exporter_semantics(tmp_path):
    from structure_labels import lut_labels, sct_labels, sct_matched_name, sct_output_name
    lut = tmp_path / 'labels.txt'
    lut.write_text('0 Background 0 0 0 0\n7 Left Cortex 1 2 3 0\n2 Right-Cortex 4 5 6 0\n# comment\n')
    assert lut_labels(lut) == {'7': 'Left_Cortex', '2': 'Right-Cortex'}
    assert lut_labels(lut, freesurfer=True) == {'7': 'Left', '2': 'Right-Cortex'}
    aliases = {'gmseg': 'Gray_Matter', 'gm_seg': 'Gray_Matter'}
    assert sct_output_name('prediction_gm_seg.nii.gz', aliases) == 'Gray_Matter'
    assert sct_output_name('prediction.nii.gz', aliases, single=True) == 'Gray_Matter'
    assert sct_output_name('prediction_other.nii.gz', aliases) == 'other'
    assert sct_matched_name('prediction_other.nii.gz', aliases) is None
    assert final_roi_names(sct_labels({'output_names': aliases}, tmp_path)) == ['Gray_Matter']
    spec = {'output_names': {'sc_seg': 'Spinal_Cord', 'lesion_seg': 'SCI_Lesion'}, 'preserve_overlaps': True}
    assert final_roi_names(sct_labels(spec, tmp_path)) == ['SCI_Lesion', 'Spinal_Cord']


def test_disabled_missing_metadata_is_explicit_and_public_safe(monkeypatch):
    monkeypatch.setattr(builder, 'registries', lambda root: {'FALLBACK_TASKS': [],
        'EXTERNAL_TASKS': {}, 'INFORMATIONAL_MODELS': {'disabled_task': {'enabled': False}}})
    monkeypatch.setattr(builder, 'total_maps', lambda root: ({}, set(), Path('map_to_binary.py')))
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, 'Missing exporter metadata', '/private-bundle/export/labels.json')
    monkeypatch.setattr(builder, 'resolve', missing)
    result, log = builder.build(Path('missing'))
    entry = result['disabled_task']
    assert entry['kind'] == 'unresolved' and entry['names'] == [] and entry['count'] is None
    assert entry['note'] == 'Required exporter metadata file is absent: labels.json.'
    assert '/private-bundle' not in json.dumps(result)
    assert '/private-bundle' in json.dumps(log)


def test_total_segmentator_site_filters_keep_generated_anatomical_terms(portal):
    task = {'name': 'total', 'description': 'TotalSegmentator CT segmentation',
            'reference': 'https://github.com/wasserth/TotalSegmentator',
            'structures': 'Task-specific TotalSegmentator structures'}
    names = CATALOG['total']['names']
    previous = portal.clinical_tags({**task, 'structures': ', '.join(names)})
    current = portal.clinical_tags({**task, 'structure_names': names})
    assert previous and current == previous


def test_sct_single_target_catalogue_matches_what_the_combiner_writes(tmp_path):
    """sct_deepseg keeps prediction.nii.gz for a one-target model; run the real combiner on it."""
    import subprocess, sys
    nib = pytest.importorskip('nibabel')
    np = pytest.importorskip('numpy')
    for task in builder.SCT_SINGLE_TARGET_MODELS:
        spec = REGISTRIES['SCT_TASKS'][task]
        work = tmp_path / task
        work.mkdir()
        data = np.zeros((4, 4, 4), dtype=np.uint8)
        data[1:3, 1:3, 1:3] = 1
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(work / 'prediction.nii.gz'))
        subprocess.run([sys.executable, str(ROOT / 'combine_sct_outputs.py'), '--glob', str(work / 'prediction*.nii.gz'),
                        '--output', str(work / 'combined.nii.gz'), '--labels', str(work / 'labels.json'),
                        '--output-names', json.dumps(spec['output_names'])], check=True, cwd=ROOT)
        written = final_roi_names(json.loads((work / 'labels.json').read_text()))
        entry = CATALOG[task]
        if entry['kind'] == 'fixed':
            assert entry['names'] == written
        else:
            assert entry['kind'] == 'unresolved' and written == ['SCT_Label_1']
            assert 'SCT_Label_1' in entry['note']


def test_lut_parsers_keep_their_runtime_edge_behaviour(tmp_path):
    from structure_labels import lut_labels
    lut = tmp_path / 'leading-zero.txt'
    lut.write_text('007 Left Cortex 1 2 3 0\n')
    assert lut_labels(lut) == {'007': 'Left_Cortex'}
    assert lut_labels(lut, freesurfer=True) == {'7': 'Left'}
    bad = tmp_path / 'invalid.txt'
    bad.write_bytes(b'7 Left\xffCortex 1 2 3 0\n')
    with pytest.raises(UnicodeDecodeError):
        lut_labels(bad)
    assert lut_labels(bad, freesurfer=True) == {'7': 'Left\ufffdCortex'}


def test_catalogue_order_is_numeric_while_runner_maps_sort_as_strings():
    labels = {str(i): f'Label_{i}' for i in range(1, 14)}
    serialised = json.loads(json.dumps(labels, sort_keys=True))
    names = final_roi_names(labels)
    assert names[:3] == ['Label_1', 'Label_2', 'Label_3']
    assert list(serialised)[:3] == ['1', '10', '11']
    assert sorted(names) == sorted(serialised.values())


def test_lut_commands_keep_their_baseline_errors_and_empty_results(tmp_path):
    import subprocess, sys
    nib = pytest.importorskip('nibabel')
    np = pytest.importorskip('numpy')
    bad = tmp_path / 'invalid.txt'
    bad.write_bytes(b'7 Left\xffCortex 1 2 3 0\n')
    run = subprocess.run([sys.executable, str(ROOT / 'lut_to_labels.py'), str(bad), str(tmp_path / 'a.json')],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode != 0 and 'UnicodeDecodeError' in run.stderr
    empty = tmp_path / 'background.txt'
    empty.write_text('0 Background 0 0 0 0\n')
    run = subprocess.run([sys.executable, str(ROOT / 'lut_to_labels.py'), str(empty), str(tmp_path / 'b.json')],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode != 0 and 'No labels parsed from' in run.stderr
    seg = tmp_path / 'seg.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)), str(seg))
    out = tmp_path / 'c.json'
    subprocess.run([sys.executable, str(ROOT / 'labels_from_segmentation_lut.py'), str(seg), str(empty), str(out)],
                   check=True, cwd=ROOT)
    assert json.loads(out.read_text()) == {}

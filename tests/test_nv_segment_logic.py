"""CPU contracts for NV-Segment; no installed bundle or ML runtime is needed."""
import hashlib
import json
from pathlib import Path

import pytest

from nv_segment_logic import MODES, check_modality, generate, roi_names, selected_indices, validate_acceptance

ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((ROOT / 'nv_segment_manifest.json').read_text())['modes']
CT_SELECTION = '$list(set([i+1 for i in range(132)]) - set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))'


@pytest.mark.parametrize('mode', MODES)
def test_correct_modality(mode):
    check_modality(mode, MODES[mode]['modality'])


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('modality', ['CT', 'MR', 'PT', 'MRI', ''])
def test_reject_wrong_modality(mode, modality):
    if modality != MODES[mode]['modality']:
        with pytest.raises(ValueError):
            check_modality(mode, modality)


def test_reject_unknown_mode():
    with pytest.raises(ValueError):
        check_modality('vista3d_prompted_ct', 'CT')


@pytest.mark.parametrize('mode,expected', list(zip(MODES, [117, 117, 50, 132])))
def test_pinned_counts_names_sparse_indices_and_hashes(mode, expected):
    path = ROOT / 'labels' / f'{mode}.json'
    labels = json.loads(path.read_text())
    indices = MANIFEST[mode]['indices']
    assert len(labels) == len(indices) == expected
    assert sorted(map(int, labels)) == sorted(indices)
    assert all(0 < int(i) <= 345 for i in labels)
    assert len(set(v.casefold() for v in labels.values())) == expected
    assert all(len(v) <= 64 and v.isascii() and '\\' not in v for v in labels.values())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST[mode]['labels_sha256']
    assert all(MANIFEST[mode][k] == v for k, v in MODES[mode].items())


def test_brain_255_is_real_foreground():
    labels = json.loads((ROOT / 'labels/nv_segment_ctmr_mri_brain.json').read_text())
    assert labels['255'] == 'Left-AnG---angular-gyrus'


@pytest.mark.parametrize('mode', ['nv_segment_ct_auto', 'nv_segment_ctmr_ct_body'])
def test_ct_automatic_excludes_unsupported_heads(mode):
    ids = selected_indices({'everything_labels': CT_SELECTION, 'ct_body': CT_SELECTION}, {}, mode)
    assert ids == MANIFEST[mode]['indices']
    assert not set(ids) & {2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132}
    assert {116, 117}.issubset(ids)


@pytest.mark.parametrize('mode,key', [('nv_segment_ctmr_mri_body', 'TotalSegmentatorMR'),
                                     ('nv_segment_ctmr_mri_brain', 'LUMIR')])
def test_mri_selection_preserves_sparse_ids(mode, key):
    ids = MANIFEST[mode]['indices']
    assert selected_indices({}, {key: list(enumerate(ids))}, mode) == ids


def test_generate_reads_metadata_and_mapping(tmp_path):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'inference.json').write_text('{}')
    (configs / 'metadata.json').write_text(json.dumps({'network_data_format': {
        'outputs': {'pred': {'channel_def': {'255': 'Left angular gyrus', '345': 'Left temporal gyrus'}}}}}))
    (configs / 'label_mappings.json').write_text('{"LUMIR": [[1, 255], [2, 345]]}')
    labels, indices = generate(tmp_path, 'nv_segment_ctmr_mri_brain')
    assert indices == [255, 345]
    assert labels == {'255': 'Left angular gyrus', '345': 'Left temporal gyrus'}


def test_candidate_names():
    names = roi_names({'1': 'liver tumor', '2': 'renal lesion', '3': 'kidney cyst',
                       '4': 'lung nodule', '5': 'liver'}, [1, 2, 3, 4, 5])
    assert all(names[str(i)].startswith('Lesion candidate: ') for i in [1, 2, 3, 4])
    assert names['5'] == 'liver'


def test_safe_unique_truncation():
    names = roi_names({'1': 'A' * 80, '2': 'A' * 80 + 'B', '3': 'é\\\x01 left kidney'}, [1, 2, 3])
    assert len(set(names.values())) == 3
    assert all(len(n) <= 64 for n in names.values())
    assert names['3'] == 'e left kidney'


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        roi_names({'1': '\x01\\'}, [1])


def test_unknown_config_expression_not_executed():
    with pytest.raises(ValueError):
        selected_indices({'everything_labels': "$__import__('os').system('false')"}, {}, 'nv_segment_ct_auto')


def test_duplicate_mapping_rejected():
    with pytest.raises(ValueError):
        selected_indices({}, {'LUMIR': [[1, 214], [2, 214]]}, 'nv_segment_ctmr_mri_brain')


@pytest.mark.parametrize('mode', MODES)
def test_acceptance_binds_task_and_pins(mode):
    spec = MODES[mode]
    marker = {'task': mode, 'result': 'passed', 'source_commit': spec['source_commit'],
              'checkpoint_revision': spec['revision'], 'checkpoint_sha256': spec['sha256']}
    validate_acceptance(marker, mode)
    for key in marker:
        with pytest.raises(ValueError):
            validate_acceptance({**marker, key: 'wrong'}, mode)
    for invalid in [None, [], 'passed', {}]:
        with pytest.raises(ValueError):
            validate_acceptance(invalid, mode)

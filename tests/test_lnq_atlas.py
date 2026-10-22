import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from run_lnq_atlas import (average_probabilities, probability_threshold, verify_bundle,
                           required_files, validate_manifest, SOURCE_REVISION)


def test_upstream_probability_threshold_and_mean_softmax():
    atlas=np.array([0.,127.5,255.])
    np.testing.assert_allclose(probability_threshold(atlas),[.3,.225,.15])
    folds=[np.array([.1,.9],dtype=np.float32) for _ in range(4)]+[np.array([.9,.1],dtype=np.float32)]
    np.testing.assert_allclose(average_probabilities(folds),[.26,.74],rtol=1e-6)
    with pytest.raises(ValueError,match='five'):average_probabilities(folds[:4])
    with pytest.raises(ValueError,match='shapes'):average_probabilities(folds[:4]+[np.ones(3)])
    with pytest.raises(ValueError,match='probability'):average_probabilities([np.array([np.nan])]*5)
    with pytest.raises(ValueError,match='Degenerate'):probability_threshold(np.ones(5))


def test_missing_or_unpinned_weights_fail_before_inference(tmp_path):
    record={'source_revision':SOURCE_REVISION,'sha256':'0'*64,'files_sha256':{}}
    manifest=tmp_path/'pin.json';manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Missing verified artifact pin'):verify_bundle(tmp_path,manifest)
    for name in required_files():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
        record['files_sha256'][name]=hashlib.sha256(name.encode()).hexdigest()
    manifest.write_text(json.dumps(record))
    assert verify_bundle(tmp_path,manifest)==record
    (tmp_path/required_files()[-1]).write_bytes(b'changed atlas')
    with pytest.raises(ValueError,match='hash mismatch'):verify_bundle(tmp_path,manifest)
    with pytest.raises(ValueError,match='revision'):validate_manifest({**record,'source_revision':'wrong'})
    with pytest.raises(ValueError,match='archive'):validate_manifest({**record,'sha256':'not-a-hash'})


def test_preflight_rejects_unpinned_bundle(monkeypatch,tmp_path):
    import model_preflight
    monkeypatch.setattr(model_preflight,'MODEL_ROOT',tmp_path)
    monkeypatch.setattr(model_preflight,'COMMON',[])
    monkeypatch.setattr(model_preflight,'REQUIREMENTS',{'lnq_atlas':[]})
    result=model_preflight.audit(['lnq_atlas'])['lnq_atlas']
    assert not result['ready'] and 'LNQ artifact provenance missing or invalid' in result['missing']


def _task256_tree(root):
    from run_lnq_atlas import TASK256_CHECKPOINT, TASK256_CHECKPOINT_PKL, TASK256_PLANS
    checkpoint = root / TASK256_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'pinned-task-256')
    (root / TASK256_CHECKPOINT_PKL).write_bytes(b'checkpoint-pickle')
    plans = root / TASK256_PLANS
    plans.parent.mkdir(parents=True, exist_ok=True)
    plans.write_bytes(b'plans-pickle')
    return root


def test_legacy_total_state_is_isolated_and_home_environment_unchanged(monkeypatch,tmp_path):
    import os,tempfile
    from pathlib import Path
    import run_total_v1
    weights = _task256_tree(tmp_path / 'task256')
    original_home=Path.home();home_environment=os.environ.get('HOME');seen=[]
    monkeypatch.setattr(run_total_v1,'version',lambda _: '1.5.7')
    monkeypatch.setattr('run_lnq_atlas.require_task256_metadata', lambda *_args, **_kwargs: {})
    monkeypatch.setenv('TOTALSEG_WEIGHTS_PATH', str(weights))
    def fake_run(input_path,output_path):
        seen.append((Path.home(),input_path,output_path))
        (Path.home()/'legacy-config.json').write_text('{}')
        with tempfile.TemporaryDirectory(prefix='nnunet_tmp_') as directory:
            assert Path(directory).parent==state/'bulk'
        with tempfile.TemporaryDirectory(prefix='pymp-',dir=tmp_path) as directory:
            assert Path(directory).parent==tmp_path
        assert os.environ['TOTALSEG_WEIGHTS_PATH']==str(weights)
    monkeypatch.setattr(run_total_v1,'run_prediction',fake_run)
    state=tmp_path/'isolated'
    arguments=['--state-dir',str(state),'--input','ct.nii.gz','--output','total.nii.gz']
    run_total_v1.main(arguments)
    assert seen==[(state,Path('ct.nii.gz'),Path('total.nii.gz'))]
    assert (state/'legacy-config.json').is_file()
    assert json.loads((state/'.totalsegmentator/config.json').read_text())['send_usage_stats'] is False
    assert Path.home()==original_home and os.environ.get('HOME')==home_environment
    assert os.environ.get('TOTALSEG_WEIGHTS_PATH')==str(weights)
    monkeypatch.setattr(run_total_v1,'version',lambda _: '2.0.0')
    with pytest.raises(ValueError,match='1.5.7'):run_total_v1.main(arguments)


def test_total_refuses_download_without_pinned_task256(monkeypatch,tmp_path):
    import run_total_v1
    from run_lnq_atlas import require_task256
    monkeypatch.setattr(run_total_v1,'version',lambda _: '1.5.7')
    monkeypatch.setattr(run_total_v1,'run_prediction',lambda *_: (_ for _ in ()).throw(AssertionError('prediction ran')))
    arguments=['--state-dir',str(tmp_path/'state'),'--input','ct.nii.gz','--output','total.nii.gz']
    monkeypatch.delenv('TOTALSEG_WEIGHTS_PATH',raising=False)
    with pytest.raises(SystemExit,match='unset'):
        run_total_v1.main(arguments)
    monkeypatch.setenv('TOTALSEG_WEIGHTS_PATH', str(tmp_path/'missing'))
    with pytest.raises(SystemExit,match='missing'):
        run_total_v1.main(arguments)
    weights=_task256_tree(tmp_path/'weights')
    extra=weights/'nnUNet/3d_fullres/Task256_TotalSegmentator_3mm_1139subj/nnUNetTrainerV2_ep8000_nomirror__other'
    extra.mkdir()
    with pytest.raises(SystemExit,match='ambiguous'):
        require_task256(weights)


def test_acceptance_rejects_waiver_and_inexact_dice(tmp_path):
    from run_lnq_atlas import validate_acceptance
    record={'task':'lnq_atlas','result':'passed','waiver':'ignored','dice':1}
    with pytest.raises(ValueError,match='waiver'):
        validate_acceptance(record,tmp_path,tmp_path/'provenance.json',tmp_path,tmp_path/'manifest.json')
    record={'task':'lnq_atlas','result':'passed','dice':True,'identical_voxels':True,'both_nonempty':True,
            'geometry_matches_input':True,'fixture':'msd-lung/dicom','source_revision':SOURCE_REVISION,
            'totalsegmentator_version':'1.5.7','validator':'convert_validate_rtstruct.py','nonempty_rois':1,
            'roi_names':['Mediastinal_Lymph_Node_Candidate'],'referenced_images_with_contours':1,
            'rtstruct_bytes':1,'checks':[]}
    with pytest.raises(ValueError,match='exactly 1'):
        validate_acceptance(record,tmp_path,tmp_path/'provenance.json',tmp_path,tmp_path/'manifest.json')


def test_absent_marker_does_not_hash_and_a_bad_marker_cannot_pass(monkeypatch,tmp_path):
    import model_preflight
    provenance={'source_revision':SOURCE_REVISION,'sha256':'a'*64,
                'files_sha256':{name:'b'*64 for name in required_files()}}
    weights=tmp_path/'weights/lnq-atlas-20261015'
    weights.mkdir(parents=True)
    (weights/'provenance.json').write_text(json.dumps(provenance))
    monkeypatch.setattr(model_preflight,'MODEL_ROOT',tmp_path)
    monkeypatch.setattr(model_preflight,'COMMON',[])
    monkeypatch.setattr(model_preflight,'REQUIREMENTS',{'lnq_atlas':['validation/lnq_atlas.json']})
    monkeypatch.setattr('run_lnq_atlas.sha256',lambda *_: (_ for _ in ()).throw(AssertionError('hashed without a marker')))
    absent=model_preflight.audit(['lnq_atlas'])['lnq_atlas']
    assert absent['acceptance_ready'] is False and not absent['ready']
    assert 'validation/lnq_atlas.json' in absent['missing']
    (tmp_path/'validation').mkdir()
    (tmp_path/'validation/lnq_atlas.json').write_text(json.dumps({'task':'lnq_atlas','result':'passed','dice':False}))
    rejected=model_preflight.audit(['lnq_atlas'])['lnq_atlas']
    assert rejected['acceptance_ready'] is False and not rejected['ready']
    assert any('inconsistent acceptance' in item for item in rejected['missing'])


def test_sm61_is_refused_and_a_supported_device_is_accepted():
    from run_lnq_atlas import device_supported, require_execution_device
    arches = ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
    assert device_supported(True, 6, 1, arches) is False
    assert device_supported(True, 9, 0, arches) is True
    assert device_supported(False, 9, 0, arches) is False
    assert device_supported(True, 9, 0, ['compute_80']) is True
    assert device_supported(True, 6, 1, ['compute_75']) is False
    refused = {'cuda_available': True, 'major': 6, 'minor': 1, 'arch_list': arches}
    with pytest.raises(SystemExit, match='sm_61'):
        require_execution_device(refused)
    require_execution_device({'cuda_available': True, 'major': 9, 'minor': 0, 'arch_list': ['sm_90']})


def test_preflight_refuses_sm61_and_accepts_sm90(monkeypatch, tmp_path):
    import model_preflight
    python = tmp_path / 'runtime-envs/lnq-atlas-ref/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('')
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', tmp_path)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', tmp_path)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {'lnq_atlas': []})
    calls = []

    def probe(path):
        calls.append(path)
        capability = probe.capability
        return {'cuda_available': True, 'major': capability[0], 'minor': capability[1],
                'arch_list': ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']}

    probe.capability = (6, 1)
    monkeypatch.setattr(model_preflight, 'probe_lnq_device', probe)
    refused = model_preflight.audit(['lnq_atlas'])['lnq_atlas']
    assert any('sm_61' in item for item in refused['missing'])
    assert refused['ready'] is False
    probe.capability = (9, 0)
    accepted = model_preflight.audit(['lnq_atlas'])['lnq_atlas']
    assert not any('cannot execute' in item for item in accepted['missing'])
    assert calls and all(Path(path) == python for path in calls)


def test_runner_refuses_unsupported_device_before_inference(monkeypatch):
    import run_lnq_atlas
    monkeypatch.setattr(run_lnq_atlas, 'require_execution_device',
                        lambda payload=None: (_ for _ in ()).throw(SystemExit(
                            'LNQ runtime cannot execute on this GPU (sm_61; compiled sm_75); '
                            'route lnq_atlas to the H100 worker')))
    monkeypatch.setattr(run_lnq_atlas, 'verify_bundle',
                        lambda *_: (_ for _ in ()).throw(AssertionError('inference started')))
    with pytest.raises(SystemExit, match='sm_61'):
        run_lnq_atlas.run(None)


def test_deterministic_contract_is_applied_after_nnunet_turns_benchmark_on():
    from run_lnq_atlas import DETERMINISTIC_CONTRACT, apply_deterministic_after_nnunet, apply_deterministic_environment
    import os
    class Cudnn:
        deterministic = False
        benchmark = True
    class Backends:
        cudnn = Cudnn()
    backends = Backends()
    apply_deterministic_after_nnunet(backends)
    assert backends.cudnn.deterministic is True and backends.cudnn.benchmark is False
    apply_deterministic_environment()
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == DETERMINISTIC_CONTRACT['cublas_workspace_config']
    assert os.environ['nnUNet_compile'] == 'false'
    assert DETERMINISTIC_CONTRACT['applied_after_nnunet_predictor_init'] is True


def test_changed_revision_rejects_unchanged_marker_and_provenance(monkeypatch, tmp_path):
    from run_lnq_atlas import (DETERMINISTIC_CONTRACT, LNQ_CHECKS, PUBLIC_FIXTURE, ROI_NAMES,
                               SOURCE_REVISION, required_files, validate_acceptance)
    monkeypatch.setattr('run_lnq_atlas.sha256', lambda *_: (_ for _ in ()).throw(AssertionError('hashed')))
    provenance = {'source_revision': SOURCE_REVISION, 'sha256': 'a' * 64,
                  'files_sha256': {name: 'b' * 64 for name in required_files()}}
    provenance_path = tmp_path / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance))
    source = tmp_path / 'LNQ2023'
    (source / '.git').mkdir(parents=True)
    (source / '.git' / 'HEAD').write_text('b' * 40 + '\n')
    record = {
        'task': 'lnq_atlas', 'result': 'passed', 'fixture': PUBLIC_FIXTURE,
        'source_revision': SOURCE_REVISION, 'dice': 1, 'identical_voxels': True,
        'both_nonempty': True, 'geometry_matches_input': True, 'totalsegmentator_version': '1.5.7',
        'validator': 'convert_validate_rtstruct.py', 'nonempty_rois': 1, 'roi_names': ROI_NAMES,
        'referenced_images_with_contours': 1, 'rtstruct_bytes': 10, 'checks': sorted(LNQ_CHECKS),
        'deterministic_contract': DETERMINISTIC_CONTRACT, 'inference_tree_sha256': 'c' * 64,
    }
    with pytest.raises(ValueError, match='revision'):
        validate_acceptance(record, tmp_path, provenance_path, tmp_path, tmp_path / 'manifest.json', source)


def test_changed_inference_blob_fails_with_the_reviewed_commit(tmp_path):
    from run_lnq_atlas import SOURCE_REVISION, require_installed_source
    source = tmp_path / 'LNQ2023'
    (source / '.git').mkdir(parents=True)
    (source / '.git' / 'HEAD').write_text(SOURCE_REVISION + '\n')
    with pytest.raises(ValueError, match='inference file missing'):
        require_installed_source(source)


def test_changed_or_removed_task256_pickle_is_refused(tmp_path, monkeypatch):
    from pathlib import Path
    from run_lnq_atlas import (TASK256_CHECKPOINT, TASK256_CHECKPOINT_PKL, TASK256_PLANS,
                               TASK256_PLANS_SHA256, TASK256_SHA256, require_task256,
                               require_task256_metadata)
    weights = _task256_tree(tmp_path / 'weights')

    def digest(path):
        path = Path(path)
        if path == weights / TASK256_CHECKPOINT:
            return TASK256_SHA256
        if path == weights / TASK256_PLANS:
            return TASK256_PLANS_SHA256
        return '0' * 64

    monkeypatch.setattr('run_lnq_atlas.sha256', digest)
    with pytest.raises(SystemExit, match='model_final_checkpoint.model.pkl'):
        require_task256_metadata(weights)
    (weights / TASK256_CHECKPOINT_PKL).unlink()
    with pytest.raises(SystemExit, match='missing'):
        require_task256(weights)


def test_empty_lnq_prediction_is_a_completed_empty_result(synthetic_ct, tmp_path):
    import app
    from test_rtstruct_geometry import ROOT, convert, inputs
    from test_viewer import make_series
    source = tmp_path / 'source'
    make_series(synthetic_ct, source)
    image_path, prediction, _mask = inputs(source, tmp_path, np.zeros((48, 32, 6), dtype=np.uint8))
    labels = json.loads((ROOT / 'labels/lnq-atlas.json').read_text())
    result, output = convert(source, tmp_path, image_path, prediction, labels=labels)
    assert result.returncode != 0
    assert 'empty prediction: no RTSTRUCT created' in result.stderr
    assert app.is_empty_prediction(RuntimeError(result.stderr))
    assert not output.exists()


def test_ordinary_cubin_covers_same_major_and_higher_minor():
    from run_lnq_atlas import device_supported
    assert device_supported(True, 6, 1, ['sm_60']) is True
    assert device_supported(True, 8, 6, ['sm_80']) is True
    assert device_supported(True, 6, 0, ['sm_61']) is False
    assert device_supported(True, 6, 1, ['sm_75']) is False
    assert device_supported(True, 9, 0, ['sm_80']) is False
    assert device_supported(True, 9, 0, ['compute_80']) is True
    # Integer capability 9.0 is the token sm_90. sm_90a is not that token.
    assert device_supported(True, 9, 0, ['sm_90a']) is False
    assert device_supported(True, 9, 0, ['sm_90']) is True


def test_preflight_accepts_sm61_when_runtime_lists_sm60(monkeypatch, tmp_path):
    import model_preflight
    python = tmp_path / 'runtime-envs/lnq-atlas-ref/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('')
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', tmp_path)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {'lnq_atlas': []})
    monkeypatch.setattr(model_preflight, 'probe_lnq_device', lambda path: {
        'cuda_available': True, 'major': 6, 'minor': 1, 'name': 'Quadro P6000',
        'arch_list': ['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']})
    result = model_preflight.audit(['lnq_atlas'])['lnq_atlas']
    assert not any('cannot execute' in item or 'sm_61' in item for item in result['missing'])


def test_total_hook_reapplies_cudnn_after_model_load(monkeypatch, tmp_path):
    import os
    import sys
    import types
    import run_total_v1
    from run_lnq_atlas import TOTAL_DETERMINISTIC_CONTRACT, apply_total_deterministic_after_nnunet
    from cads_variants import marker_has_failure
    assert marker_has_failure(TOTAL_DETERMINISTIC_CONTRACT) is False

    class Cudnn:
        deterministic = False
        benchmark = True

    class Backends:
        cudnn = Cudnn()

    backends = Backends()
    apply_total_deterministic_after_nnunet(backends)
    assert backends.cudnn.deterministic is True and backends.cudnn.benchmark is False

    events = []
    predict = types.ModuleType('nnunet.inference.predict')

    def load_model_and_checkpoint_files(*_args, **_kwargs):
        events.append('load')
        backends.cudnn.deterministic = False
        backends.cudnn.benchmark = True
        return 'model'

    predict.load_model_and_checkpoint_files = load_model_and_checkpoint_files
    saved = {name: sys.modules.get(name) for name in
             ('nnunet', 'nnunet.inference', 'nnunet.inference.predict',
              'totalsegmentator', 'totalsegmentator.config', 'totalsegmentator.python_api')}
    for name in ('nnunet', 'nnunet.inference'):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules.pop('nnunet.inference.predict', None)

    # nnunet.paths reads RESULTS_FOLDER at import, so record it when the predictor is imported.
    class PredictFinder:
        def find_spec(self, name, path=None, target=None):
            if name != 'nnunet.inference.predict':
                return None
            import importlib.util

            class Loader:
                def create_module(self, spec):
                    events.append(('import predict', os.environ.get('RESULTS_FOLDER')))
                    return predict

                def exec_module(self, module):
                    pass

            return importlib.util.spec_from_loader(name, Loader())

    finder = PredictFinder()
    sys.meta_path.insert(0, finder)
    config = types.ModuleType('totalsegmentator.config')

    def setup_nnunet():
        events.append('setup')
        os.environ['RESULTS_FOLDER'] = str(tmp_path / 'weights')

    config.setup_nnunet = setup_nnunet
    sys.modules['totalsegmentator.config'] = config
    api = types.ModuleType('totalsegmentator.python_api')

    def totalsegmentator(input_path, output_path, ml, fast, nr_thr_resamp, nr_thr_saving):
        events.append(('api', ml, fast, nr_thr_resamp, nr_thr_saving))
        sys.modules['nnunet.inference.predict'].load_model_and_checkpoint_files()
        Path(output_path).write_bytes(b'mask')

    api.totalsegmentator = totalsegmentator
    sys.modules['totalsegmentator'] = types.ModuleType('totalsegmentator')
    sys.modules['totalsegmentator.python_api'] = api
    run_total_v1._LOAD_MODEL = None
    run_total_v1._DETERMINISTIC_APPLIED = False

    def apply(backends=None):
        events.append('apply')
        target = backends.cudnn if backends is not None else backends_box
        target.deterministic = True
        target.benchmark = False

    backends_box = backends.cudnn
    monkeypatch.setattr(run_total_v1, 'apply_total_deterministic_after_nnunet', apply)
    output = tmp_path / 'total-v1.nii.gz'
    env_keys = ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'nnUNet_def_n_proc', 'RESULTS_FOLDER')
    saved_env = {key: os.environ.get(key) for key in env_keys}
    os.environ.pop('RESULTS_FOLDER', None)
    try:
        run_total_v1.run_prediction(tmp_path / 'ct.nii.gz', output)
    finally:
        sys.meta_path.remove(finder)
        run_total_v1._LOAD_MODEL = None
        run_total_v1._DETERMINISTIC_APPLIED = False
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert events == ['setup', ('import predict', str(tmp_path / 'weights')),
                      ('api', True, True, 1, 1), 'load', 'apply']
    assert backends.cudnn.deterministic is True and backends.cudnn.benchmark is False
    contract = json.loads(output.with_name('total_deterministic_contract.json').read_text())
    assert contract == TOTAL_DETERMINISTIC_CONTRACT


def test_total_prediction_refuses_an_unmarked_result(monkeypatch, tmp_path):
    import sys
    import types
    import run_total_v1
    predict = types.ModuleType('nnunet.inference.predict')
    predict.load_model_and_checkpoint_files = lambda *_args, **_kwargs: 'model'
    saved = {name: sys.modules.get(name) for name in
             ('nnunet', 'nnunet.inference', 'nnunet.inference.predict',
              'totalsegmentator', 'totalsegmentator.python_api')}
    sys.modules['nnunet'] = types.ModuleType('nnunet')
    sys.modules['nnunet.inference'] = types.ModuleType('nnunet.inference')
    sys.modules['nnunet.inference.predict'] = predict
    api = types.ModuleType('totalsegmentator.python_api')
    api.totalsegmentator = lambda *_args, **_kwargs: None
    sys.modules['totalsegmentator'] = types.ModuleType('totalsegmentator')
    sys.modules['totalsegmentator.python_api'] = api
    config = types.ModuleType('totalsegmentator.config')
    config.setup_nnunet = lambda: None
    monkeypatch.setitem(sys.modules, 'totalsegmentator.config', config)
    run_total_v1._LOAD_MODEL = None
    run_total_v1._DETERMINISTIC_APPLIED = False
    monkeypatch.setattr(run_total_v1, 'apply_total_process_environment', lambda: None)
    monkeypatch.setattr(run_total_v1, 'apply_total_deterministic_after_nnunet', lambda backends=None: None)
    try:
        with pytest.raises(RuntimeError, match='loaded no model'):
            run_total_v1.run_prediction(tmp_path / 'ct.nii.gz', tmp_path / 'total-v1.nii.gz')
        assert not (tmp_path / 'total_deterministic_contract.json').exists()
    finally:
        run_total_v1._LOAD_MODEL = None
        run_total_v1._DETERMINISTIC_APPLIED = False
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _acceptance_files(root):
    hashes = {}
    for name in required_files():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b'lnq-artifact:' + name.encode()
        path.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return hashes


def _bind_acceptance(monkeypatch, tmp_path, hashes):
    from run_lnq_atlas import (INFERENCE_TREE_SHA256, SOURCE_REVISION, TASK256_CHECKPOINT,
                               TASK256_CHECKPOINT_PKL, TASK256_PLANS, TASK256_CHECKPOINT_PKL_SHA256,
                               TASK256_PLANS_SHA256, TASK256_SHA256)
    weights = tmp_path / 'weights'
    _acceptance_files(weights)
    for name, digest in hashes.items():
        (weights / name).write_bytes(b'lnq-artifact:' + name.encode())
        assert hashlib.sha256((weights / name).read_bytes()).hexdigest() == digest
    total = _task256_tree(tmp_path / 'task256')
    provenance = {'source_revision': SOURCE_REVISION, 'sha256': 'a' * 64, 'files_sha256': dict(hashes)}
    provenance_path = tmp_path / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance))
    manifest = tmp_path / 'task256.json'
    manifest.write_text(json.dumps({
        'checkpoint_sha256': TASK256_SHA256,
        'checkpoint_pkl_sha256': TASK256_CHECKPOINT_PKL_SHA256,
        'plans_pkl_sha256': TASK256_PLANS_SHA256,
    }))
    import run_lnq_atlas
    original = run_lnq_atlas.sha256

    def digest(path):
        path = Path(path)
        pins = {
            total / TASK256_CHECKPOINT: TASK256_SHA256,
            total / TASK256_CHECKPOINT_PKL: TASK256_CHECKPOINT_PKL_SHA256,
            total / TASK256_PLANS: TASK256_PLANS_SHA256,
        }
        if path in pins:
            return pins[path]
        return original(path)

    monkeypatch.setattr(run_lnq_atlas, 'sha256', digest)
    monkeypatch.setattr(run_lnq_atlas, 'require_installed_source',
                        lambda _source: (SOURCE_REVISION, INFERENCE_TREE_SHA256))
    return weights, provenance_path, total, manifest, provenance


def _passing_record(hashes):
    from run_lnq_atlas import (DETERMINISTIC_CONTRACT, INFERENCE_TREE_SHA256, LNQ_CHECKS,
                               PUBLIC_FIXTURE, REQUIRED_DNA, REQUIRED_NNUNET, ROI_NAMES,
                               SOURCE_REVISION, TASK256_CHECKPOINT_PKL_SHA256, TASK256_PLANS_SHA256,
                               TASK256_SHA256, TOTAL_DETERMINISTIC_CONTRACT)
    checkpoints = {name: digest for name, digest in hashes.items() if name.endswith('checkpoint_best.pth')}
    return {
        'task': 'lnq_atlas', 'result': 'passed', 'fixture': PUBLIC_FIXTURE,
        'source_revision': SOURCE_REVISION, 'inference_tree_sha256': INFERENCE_TREE_SHA256,
        'dice': 1, 'identical_voxels': True, 'both_nonempty': True, 'geometry_matches_input': True,
        'totalsegmentator_version': '1.5.7', 'validator': 'convert_validate_rtstruct.py',
        'nonempty_rois': 1, 'roi_names': ROI_NAMES, 'referenced_images_with_contours': 1,
        'rtstruct_bytes': 10, 'checks': sorted(LNQ_CHECKS),
        'deterministic_contract': DETERMINISTIC_CONTRACT,
        'totalsegmentator_contract': TOTAL_DETERMINISTIC_CONTRACT,
        'files_sha256': dict(hashes), 'checkpoint_hashes': checkpoints,
        'nnunetv2': REQUIRED_NNUNET, 'dynamic_network_architectures': REQUIRED_DNA,
        'runtime': 'runtime-envs/lnq-atlas-ref',
        'execution_device_name': 'NVIDIA H100', 'execution_device_major': 9, 'execution_device_minor': 0,
        'task256_checkpoint_sha256': TASK256_SHA256,
        'task256_checkpoint_pkl_sha256': TASK256_CHECKPOINT_PKL_SHA256,
        'task256_plans_pkl_sha256': TASK256_PLANS_SHA256,
    }


def test_changed_plan_or_atlas_with_old_marker_is_rejected(monkeypatch, tmp_path):
    from run_lnq_atlas import validate_acceptance
    hashes = _acceptance_files(tmp_path / 'seed')
    weights, provenance_path, total, manifest, _provenance = _bind_acceptance(monkeypatch, tmp_path, hashes)
    device = {'name': 'NVIDIA H100', 'major': 9, 'minor': 0}
    record = _passing_record(hashes)
    for name in (next(item for item in hashes if item.endswith('plans.json')),
                 next(item for item in hashes if item.endswith('patient2-ct.nii.gz'))):
        target = weights / name
        target.write_bytes(target.read_bytes() + b'changed')
        with pytest.raises(ValueError, match='does not match provenance'):
            validate_acceptance(record, weights, provenance_path, total, manifest, tmp_path, device=device)
        target.write_bytes(b'lnq-artifact:' + name.encode())


def test_updated_provenance_with_old_marker_is_rejected(monkeypatch, tmp_path):
    from run_lnq_atlas import validate_acceptance
    hashes = _acceptance_files(tmp_path / 'seed')
    weights, provenance_path, total, manifest, provenance = _bind_acceptance(monkeypatch, tmp_path, hashes)
    plan = next(item for item in hashes if item.endswith('plans.json'))
    (weights / plan).write_bytes(b'updated-plan')
    provenance['files_sha256'][plan] = hashlib.sha256(b'updated-plan').hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    record = _passing_record(hashes)
    device = {'name': 'NVIDIA H100', 'major': 9, 'minor': 0}
    with pytest.raises(ValueError, match='hash map is missing or does not match'):
        validate_acceptance(record, weights, provenance_path, total, manifest, tmp_path, device=device)


def test_marker_without_artifact_map_is_rejected(monkeypatch, tmp_path):
    from run_lnq_atlas import validate_acceptance
    hashes = _acceptance_files(tmp_path / 'seed')
    weights, provenance_path, total, manifest, _provenance = _bind_acceptance(monkeypatch, tmp_path, hashes)
    record = _passing_record(hashes)
    record.pop('files_sha256')
    device = {'name': 'NVIDIA H100', 'major': 9, 'minor': 0}
    with pytest.raises(ValueError, match='hash map is missing or does not match'):
        validate_acceptance(record, weights, provenance_path, total, manifest, tmp_path, device=device)


def test_matching_artifact_map_is_accepted_and_h100_marker_rejects_p6000(monkeypatch, tmp_path):
    from run_lnq_atlas import validate_acceptance
    hashes = _acceptance_files(tmp_path / 'seed')
    weights, provenance_path, total, manifest, _provenance = _bind_acceptance(monkeypatch, tmp_path, hashes)
    record = _passing_record(hashes)
    device = {'name': 'NVIDIA H100', 'major': 9, 'minor': 0, 'cuda_available': True, 'arch_list': ['sm_90']}
    validate_acceptance(record, weights, provenance_path, total, manifest, tmp_path, device=device)
    with pytest.raises(ValueError, match='device'):
        validate_acceptance(record, weights, provenance_path, total, manifest, tmp_path,
                            device={'name': 'Quadro P6000', 'major': 6, 'minor': 1})
    with pytest.raises(ValueError, match='compute capability'):
        validate_acceptance(record, weights, provenance_path, total, manifest, tmp_path,
                            device={'name': 'NVIDIA H100', 'major': 6, 'minor': 1})


def test_lnq_card_names_cohort_thresholds_and_h100_routing():
    import app
    card = app.INFORMATIONAL_MODELS['lnq_atlas']['description']
    nodal = json.loads((Path(__file__).parents[1] / 'catalog/nodal-models.json').read_text())['lnq_atlas']
    blob = ' '.join([card, nodal['description'], nodal['limitations'], nodal['evidence_note']])
    assert '0.6033' in blob and '0.5690' in blob and '6.8976' in blob
    assert '100 patients' in blob and '861' in blob
    assert '0.2' in blob and '0.3' in blob and '5 mm' in blob
    assert 'not the hidden-test' not in blob.lower()
    assert 'H100' in blob and '6.1' in blob
    assert 'no RTSTRUCT' in blob

import hashlib
import json
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


def test_legacy_total_state_is_isolated_and_home_environment_unchanged(monkeypatch,tmp_path):
    import os,tempfile
    from pathlib import Path
    import run_total_v1
    original_home=Path.home();home_environment=os.environ.get('HOME');seen=[]
    monkeypatch.setattr(run_total_v1,'version',lambda _: '1.5.7')
    monkeypatch.delenv('TOTALSEG_WEIGHTS_PATH',raising=False)
    def fake_run(input_path,output_path):
        seen.append((Path.home(),input_path,output_path))
        (Path.home()/'legacy-config.json').write_text('{}')
        with tempfile.TemporaryDirectory(prefix='nnunet_tmp_') as directory:
            assert Path(directory).parent==state/'bulk'
        with tempfile.TemporaryDirectory(prefix='pymp-',dir=tmp_path) as directory:
            assert Path(directory).parent==tmp_path
        assert os.environ['TOTALSEG_WEIGHTS_PATH']==str(state/'weights')
    monkeypatch.setattr(run_total_v1,'run_prediction',fake_run)
    state=tmp_path/'isolated'
    arguments=['--state-dir',str(state),'--input','ct.nii.gz','--output','total.nii.gz']
    run_total_v1.main(arguments)
    assert seen==[(state,Path('ct.nii.gz'),Path('total.nii.gz'))]
    assert (state/'legacy-config.json').is_file()
    assert json.loads((state/'.totalsegmentator/config.json').read_text())['send_usage_stats'] is False
    assert Path.home()==original_home and os.environ.get('HOME')==home_environment
    assert 'TOTALSEG_WEIGHTS_PATH' not in os.environ
    monkeypatch.setattr(run_total_v1,'version',lambda _: '2.0.0')
    with pytest.raises(ValueError,match='1.5.7'):run_total_v1.main(arguments)

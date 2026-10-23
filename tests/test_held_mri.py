"""Synthetic, offline contracts for held bpMRI integrations; no model forwards."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from held_mri import (NCI, UMAMBA, TASKS, COMMITS, REVISIONS, UMAMBA_CHANNEL_ROLES,
                      binding, export_masks, extract_umamba, require_capability,
                      roi_name, source_files, validate_marker, weights)
from held_mri_upstream import relocated_code, bound_operator
from prostate_models import artifact_names, requirements, validate_acceptance, validate_orientation, role_directories


@pytest.fixture
def installed(tmp_path,tmp_path_factory,monkeypatch):
    cache_dir=tmp_path_factory.mktemp('held-mri-preflight-cache')
    monkeypatch.setenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE',str(cache_dir))
    for task,(model,env,runner) in TASKS.items():
        source=tmp_path/'sources'/model
        source.mkdir(parents=True)
        (source/'RTSEG_MANIFEST.json').write_text(json.dumps({'commit':COMMITS[model]}))
        for path in [*weights(task),*source_files(task),Path('runtime-envs')/env/'requirements.lock', Path('runtime-envs')/env/'bin/python',
                     Path('runtime-envs')/env/'lib/python3.11/site-packages/torch/kernel.so']:
            full=tmp_path/path
            full.parent.mkdir(parents=True,exist_ok=True)
            full.write_bytes(b'synthetic-test-only:'+str(path).encode())
    import held_mri
    from prostate_models import file_sha256
    monkeypatch.setattr(held_mri,'pinned_source_hashes',lambda task:{p.as_posix():file_sha256(tmp_path/p) for p in source_files(task)})
    import held_mri_runtime
    monkeypatch.setattr(held_mri_runtime, 'runtime_identity', lambda python: {'version':'synthetic', 'packages':[], 'stdlib':str(tmp_path/'stdlib'),
        'prefix': str(Path(python).parent.parent if Path(python).is_relative_to(tmp_path) else tmp_path/'worker')})
    monkeypatch.setattr(held_mri, 'gpu_preflight', lambda task, root: {'compute_capability':[9,0], 'name':'NVIDIA H100 synthetic-test-record'})
    return tmp_path


def marker(task,root,*,worker_python=None):
    return {'task':task,'result':'passed','acceptance_modes':['reference','dicom'],
            'reference_mode':{'passed':True,'runs':2,'result_sha256':'a'*64},
            'dicom_mode':{'passed':True,'runs':2,'result_sha256':'b'*64},**binding(task,root,worker_python=worker_python),
            'gpu':{'compute_capability':[9,0],'name':'NVIDIA H100 synthetic-test-record'},
            'checks':['gpu_inference','unmodified_reference','repeatability','exact_input_grid',
                      'rtstruct_roundtrip','orientation','both_acceptance_modes','lesion_measurements'],
            'tolerances':{'map_atol':.001 if task==UMAMBA else 0.0,'mask_voxel_differences':0,'grid_atol_mm':.0001}}


@pytest.mark.parametrize('task',[NCI,UMAMBA])
def test_bound_markers_accept_both_modes(task,installed):
    record=marker(task,installed)
    validate_acceptance(record,task,installed)
    validate_orientation(record,task,installed)


@pytest.mark.parametrize('field,value',[
    ('checkpoint_revision','other'),('checkpoint_sha256','0'*64),('source_commit','f'*40),
    ('checkpoint_hashes',{}),('source_file_hashes',{}),('environment_lock_sha256','a'*64),
    ('acceptance_modes',['reference']),('checks',['gpu_inference']),
    ('reference_mode',{'passed':True,'runs':1,'result_sha256':'a'*64}),
    ('dicom_mode',{'passed':False,'runs':2,'result_sha256':'a'*64}),
    ('deviations',['waived']),('nested',{'passed':False}),
    ('tolerances',{'map_atol':1.0,'mask_voxel_differences':0,'grid_atol_mm':.0001}),
    ('gpu',{'compute_capability':[6,1]})])
def test_marker_rejects_stale_incomplete_or_waived(installed,field,value):
    record=marker(UMAMBA,installed);record[field]=value
    with pytest.raises(ValueError): validate_marker(record,UMAMBA,installed)


def test_replaced_weights_and_locks_invalidate(installed):
    record=marker(NCI,installed)
    (installed/weights(NCI)[0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint_hashes'): validate_marker(record,NCI,installed)
    record=marker(NCI,installed)
    (installed/'runtime-envs/nci-monai/requirements.lock').write_text('new-environment')
    with pytest.raises(ValueError,match='runtime_manifest'): validate_marker(record,NCI,installed)


def test_umamba_gpu_refusal():
    with pytest.raises(ValueError,match='P6000'): require_capability((6,1))
    require_capability((7,0));require_capability((9,0))


@pytest.mark.parametrize('task',[NCI,UMAMBA])
def test_registry_roles_artifacts_and_requirements(task):
    from prostate_models import TASKS as registered
    assert task in registered
    assert registered[task]==TASKS[task]
    names=artifact_names(task)
    assert len(names)==(4 if task==NCI else 3)
    assert all(n.startswith(task+'_RTSTRUCT.dcm.') for n in names)
    assert any(p.endswith('requirements.lock') for p in requirements(task))
    assert '@convert_masks_validate_rtstruct.py' in requirements(task)
    card=json.loads((Path(__file__).parents[1]/'catalog/prostate.json').read_text())[task]
    assert [r['role'] for r in card['input_roles']]==['t2w','adc','hbv']
    assert card['reference_role']=='t2w'
    assert 'Owner accepted these terms on 2026-10-22' in card['license_note']
    assert card['requires_preflight'] is True
    if task==NCI:
        assert card['noncommercial'] is True
        assert 'excludes diagnosis and treatment' in card['description']
    else:
        assert 'Weight licence not stated' in card['license_note']
        assert 'H100' in card['description']


def test_role_reader_rejects_swapped_directory_and_partial_manifest(tmp_path):
    roles={r:{'dir':'input' if r=='t2w' else 'input_'+r} for r in ('t2w','adc','hbv')}
    manifest={}
    for entry in roles.values():
        directory=tmp_path/entry['dir'];directory.mkdir();(directory/'one.dcm').write_bytes(b'synthetic')
        manifest[entry['dir']]=[{'name':'one.dcm','size':9}]
    path=tmp_path/'job.json';path.write_text(json.dumps({'input_roles':roles,'input_manifest':manifest}))
    assert tuple(role_directories(tmp_path))==UMAMBA_CHANNEL_ROLES
    roles['adc']['dir']='input_hbv'
    path.write_text(json.dumps({'input_roles':roles,'input_manifest':manifest}))
    with pytest.raises(ValueError,match='input role adc'): role_directories(tmp_path)
    roles['adc']['dir']='input_adc';manifest['input_adc']=[]
    path.write_text(json.dumps({'input_roles':roles,'input_manifest':manifest}))
    with pytest.raises(ValueError,match='manifest'): role_directories(tmp_path)


def test_umamba_equal_score_components_keep_identity_and_volume():
    pytest.importorskip('report_guided_annotation')
    array=np.zeros((40,40,40),np.float32)
    array[2:7,2:7,2:7]=.73
    array[25:30,25:30,25:30]=.73
    detection,masks,report=extract_umamba(array,(.5,.5,3))
    assert len(masks)==2
    assert len(set(report['lesions'][i]['candidate_id'] for i in (0,1)))==2
    assert all(row['voxel_count']==125 for row in report['lesions'])
    assert all(row['volume_ml']==pytest.approx(125*.75/1000) for row in report['lesions'])
    assert report['case_likelihood']==float(np.float32(.73))
    assert report['channel_order']==['t2w','adc','hbv']
    assert report['upstream_channel_keys']==['t2w','hbv','adc']
    assert np.array_equal(detection>0,array>0)


def test_umamba_minimum_and_maximum():
    pytest.importorskip('report_guided_annotation')
    array=np.zeros((64,64,64),np.float32)
    array[1:4,1:5,1:7]=1 # exactly 72; removed by published <= rule
    detection,masks,report=extract_umamba(array,(1,1,1))
    assert not masks and not detection.any()
    for i in range(4): array[10*i+10:10*i+15,20:25,20:25]=.9-i*.1
    detection,masks,report=extract_umamba(array,(1,1,1))
    assert len(masks)==3
    assert all(row['voxel_count']==125 for row in report['lesions'])


def test_empty_lesions_are_explicit_and_gland_only_is_nonempty(tmp_path):
    pytest.importorskip('report_guided_annotation')
    reference=nib.Nifti1Image(np.zeros((6,7,8)),np.diag([.5,.5,3,1]))
    detection,masks,report=extract_umamba(np.zeros(reference.shape),( .5,.5,3))
    output=tmp_path/'result.dcm'
    export_masks(tmp_path,output,reference,masks,report)
    assert json.loads((tmp_path/'result.json').read_text())['empty'] is True
    assert json.loads(Path(str(output)+'.lesions.json').read_text())['status']=='empty'
    gland=np.zeros(reference.shape,bool);gland[1:4,1:5,2:6]=True
    export_masks(tmp_path,output,reference,{'NCI_Prostate':gland},{'status':'empty','lesions':[]})
    assert json.loads((tmp_path/'result.json').read_text())['empty'] is False
    restored=nib.load(json.loads((tmp_path/'masks.json').read_text())['NCI_Prostate'])
    assert np.array_equal(restored.get_fdata()>0,gland)
    assert np.array_equal(restored.affine,reference.affine)


def test_export_refuses_shape_mismatch(tmp_path):
    image=nib.Nifti1Image(np.zeros((5,6,7)),np.eye(4))
    with pytest.raises(ValueError,match='shape'): export_masks(tmp_path,tmp_path/'r.dcm',image,{'wrong':np.ones((5,6,8))},{})


@pytest.mark.parametrize('value',[-.1,float('nan'),1.01])
def test_extraction_refuses_invalid_scores(value):
    pytest.importorskip('report_guided_annotation')
    array=np.zeros((10,10,10));array[0,0,0]=value
    with pytest.raises(ValueError): extract_umamba(array,(1,1,1))


def test_names_and_limits():
    assert roi_name(NCI,123,5)=='NCI_L123_PIRADS5'
    assert roi_name(UMAMBA,3,.731234)=='UMMTL_L03_p0.73'
    with pytest.raises(ValueError): roi_name(UMAMBA,4,.7)
    with pytest.raises(ValueError): roi_name(NCI,1,6)


def test_io_relocation_changes_only_exact_paths(tmp_path):
    path=tmp_path/'source.py'
    path.write_text("root='/input/'\nx=0.6344772701607316\nchannels=['t2w','hbv','adc']\ndef f(v): return v*3.0\n")
    code,changes=relocated_code(path,{'/input/':'/job/input/'})
    scope={};exec(code,scope)
    assert scope['root']=='/job/input/' and scope['x']==0.6344772701607316
    assert scope['channels']==['t2w','hbv','adc'] and scope['f'](4)==12
    assert len(changes)==1


def test_plain_operator_binding_skips_scheduler():
    class Operator:
        def __init__(self): raise AssertionError('scheduler must not be constructed')
        def compute(self,x): return x+self.offset
    assert bound_operator(Operator,offset=7).compute(3)==10


def test_structure_contracts():
    data=json.loads((Path(__file__).parents[1]/'catalog/structures.json').read_text())
    assert data[NCI]['kind']=='variable' and 'NCI_Prostate' in data[NCI]['names']
    assert 'maximum 3' in data[UMAMBA]['note']
    assert 'No zone ROIs' in data[UMAMBA]['note']


def test_preflight_queries_installed_interpreter_and_missing_driver(tmp_path,monkeypatch):
    import held_mri
    calls=[]
    gpu={'name':'NVIDIA H100','compute_capability':[9,0]}
    def query(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(gpu))
    monkeypatch.setattr(held_mri.subprocess,'run',query)
    assert held_mri.gpu_preflight(NCI,tmp_path)==gpu
    assert calls[0][0]==str(tmp_path/'runtime-envs/nci-monai/bin/python')
    assert '-I' in calls[0]
    def absent(*args,**kwargs):raise FileNotFoundError('runtime absent')
    monkeypatch.setattr(held_mri.subprocess,'run',absent)
    with pytest.raises(ValueError,match='Cannot verify'): held_mri.gpu_preflight(UMAMBA,tmp_path)


def test_runner_refuses_cpu_fallback_and_old_gpu(monkeypatch):
    from held_mri import runtime_gpu
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
    monkeypatch.setitem(sys.modules,'torch',fake)
    with pytest.raises(RuntimeError,match='CPU fallback'):runtime_gpu(NCI)
    fake.cuda=SimpleNamespace(is_available=lambda:True,get_device_capability=lambda:(6,1))
    with pytest.raises(ValueError,match='P6000'):runtime_gpu(UMAMBA)


def test_json_sidecar_schema_accepts_negative_umamba(tmp_path):
    jsonschema = pytest.importorskip('jsonschema')
    pytest.importorskip('report_guided_annotation')
    _,masks,report=extract_umamba(np.zeros((4,5,6)),(1,1,1))
    report['artifacts']={}
    schema=json.loads((Path(__file__).parents[1]/'catalog/held-mri-lesions.schema.json').read_text())
    jsonschema.validate(report,schema)
    report['channel_order']=['t2w','hbv','adc']
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(report,schema)


def test_nci_classifier_grid_restoration_and_pirads_promotion(tmp_path):
    pytest.importorskip('monai')
    from monai.data import MetaTensor
    from monai.transforms import Spacing
    from run_nci_prostate import nci_measurements
    # Native T2 has a negative axis and anisotropic slices. The classifier's grid
    # differs in voxel volume; it must not be pasted under the native header.
    affine=np.array([[-.8,0,0,14],[0,.8,0,-9],[0,0,2,30],[0,0,0,1.]])
    reference=nib.Nifti1Image(np.zeros((40,42,24),np.float32),affine)
    gland=np.zeros(reference.shape,np.uint8);gland[3:37,3:39,2:22]=1
    lesion=np.zeros(reference.shape,np.uint8);lesion[7:32,14:23,7:16]=1
    for name,array in [('gland',gland),('lesion',lesion)]:nib.save(nib.Nifti1Image(array,affine),tmp_path/f'{name}.nii.gz')
    spaced=Spacing(pixdim=(.5,.5,.5),mode='nearest')(MetaTensor(lesion[None].astype(float),affine=affine))
    result={'gland_path':str(tmp_path/'gland.nii.gz'),'lesion_path':str(tmp_path/'lesion.nii.gz'),
            'classifier_data':{'nda_pred_resize':np.asarray(spaced)[0]},
            'classifier_logits':np.array([[0.,0.,4.,0.]])}
    masks,report=nci_measurements(reference,result)
    row=report['lesions'][0]
    assert row['pi_rads']==5 and row['classifier_argmax_pi_rads']==4
    assert np.array_equal(masks['NCI_Prostate'],gland>0)
    assert row['volume_ml']==pytest.approx(masks[row['roi_name']].sum()*.8*.8*2/1000)
    assert row['major_axis_length_mm']>20
    assert sum(row['classifier_probabilities'])==pytest.approx(1)
    assert report['gland']['upstream_first_region_is_background'] is True
    assert report['gland']['volume_ml']!=report['gland']['upstream_volume_mm3_assuming_0_5mm']/1000


def test_nci_no_lesion_returns_gland_without_classifier(tmp_path):
    pytest.importorskip('monai')
    from run_nci_prostate import nci_measurements
    reference=nib.Nifti1Image(np.zeros((10,11,12)),np.diag([.7,.7,3,1]))
    gland=np.zeros(reference.shape,np.uint8);gland[2:8,2:9,2:10]=1
    path=tmp_path/'gland.nii.gz';nib.save(nib.Nifti1Image(gland,reference.affine),path)
    masks,report=nci_measurements(reference,{'gland_path':path,'empty_lesions':True})
    assert list(masks)==['NCI_Prostate'] and report['status']=='empty' and report['lesions']==[]


def test_h100_marker_never_accepts_p6000(installed,monkeypatch):
    import held_mri
    record=marker(NCI,installed)
    envelope={'schema_version':2,'devices':{'H100':record}}
    monkeypatch.setattr(held_mri,'gpu_preflight',lambda *a:{'name':'Quadro P6000','compute_capability':[6,1]})
    with pytest.raises(ValueError,match='executing device P6000'):
        validate_marker(envelope,NCI,installed)
    with pytest.raises(ValueError,match='GPU/runtime'):
        validate_marker(record,NCI,installed)


@pytest.mark.parametrize('relative',['bin/python','lib/python3.11/site-packages/torch/kernel.so'])
def test_runtime_changed_with_unchanged_lock_invalidates(installed,relative):
    record=marker(NCI,installed)
    runtime=installed/'runtime-envs/nci-monai'
    lock=(runtime/'requirements.lock').read_bytes()
    (runtime/relative).write_bytes(b'changed installed implementation')
    assert (runtime/'requirements.lock').read_bytes()==lock
    with pytest.raises(ValueError,match='runtime_manifest'):
        validate_marker(record,NCI,installed)


def test_changed_conversion_file_invalidates(installed,monkeypatch,tmp_path):
    import held_mri
    # Redirect the binding's application root to an owned synthetic copy.
    app=tmp_path/'synthetic-app';app.mkdir()
    (app/'dicom_series_to_nifti.py').write_text('conversion v1')
    monkeypatch.setattr(held_mri,'__file__',str(app/'held_mri.py'))
    record=marker(NCI,installed)
    (app/'dicom_series_to_nifti.py').write_text('conversion v2')
    with pytest.raises(ValueError,match='adapter_hashes'):
        validate_marker(record,NCI,installed)


def test_device_records_preserved_and_invalidated_independently(installed):
    from held_mri import store_device_marker,invalidate_device
    h100=marker(NCI,installed)
    p6000=copy.deepcopy(h100);p6000['gpu']={'name':'Quadro P6000','compute_capability':[6,1]}
    store_device_marker(installed,NCI,h100);store_device_marker(installed,NCI,p6000)
    for folder in ('validation','orientation-validation'):
        assert set(json.loads((installed/folder/f'{NCI}.json').read_text())['devices'])=={'H100','P6000'}
    invalidate_device(installed,NCI,'H100')
    for folder in ('validation','orientation-validation'):
        assert set(json.loads((installed/folder/f'{NCI}.json').read_text())['devices'])=={'P6000'}


@pytest.mark.parametrize('task,deterministic',[(NCI,False),(UMAMBA,True)])
def test_runtime_gpu_applies_the_bound_policy(monkeypatch,task,deterministic):
    from held_mri import runtime_gpu,inference_policy
    flags=SimpleNamespace();matmul=SimpleNamespace()
    observed={}
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
        get_device_capability=lambda:(9,0),get_device_name=lambda:'NVIDIA H100',get_arch_list=lambda:['sm_90']),
        backends=SimpleNamespace(cudnn=flags,cuda=SimpleNamespace(matmul=matmul)),version=SimpleNamespace(cuda='synthetic'),
        use_deterministic_algorithms=lambda value:observed.update(deterministic_algorithms=value),
        set_float32_matmul_precision=lambda value:observed.update(float32_matmul_precision=value))
    monkeypatch.setitem(sys.modules,'torch',fake)
    gpu=runtime_gpu(task)
    assert flags.deterministic is deterministic
    assert flags.benchmark is False and flags.allow_tf32 is True and matmul.allow_tf32 is False
    assert all(gpu[k]==v for k,v in inference_policy(task).items())
    assert observed=={'deterministic_algorithms':False,'float32_matmul_precision':'highest'}


def test_runner_start_rechecks_execution_device_before_inference(installed,monkeypatch):
    import held_mri
    monkeypatch.setattr('held_mri_runtime.check_python_policy', lambda: None)
    record=marker(NCI,installed)
    held_mri.store_device_marker(installed,NCI,record)
    monkeypatch.setattr(held_mri.sys,'prefix',str(installed/'runtime-envs/nci-monai'))
    monkeypatch.setattr(held_mri.os.path,'samefile',lambda *args:True)
    monkeypatch.setattr(held_mri,'runtime_gpu',lambda task:{'name':'Quadro P6000','compute_capability':[6,1]})
    with pytest.raises(ValueError,match='executing device P6000'):
        held_mri.runtime_start(NCI,installed,worker_python=sys.executable)


def test_production_dispatch_requires_markers_before_model_launch(tmp_path):
    from prostate_models import run_task
    commands=[]
    with pytest.raises(FileNotFoundError):
        run_task(NCI,tmp_path,Path(__file__).parents[1],tmp_path/'bundle',sys.executable,commands.append)
    assert not commands


def test_stale_bytecode_cannot_change_execution_or_binding(installed):
    import os
    import py_compile
    import subprocess
    from held_mri_runtime import python_command
    module = installed/'runtime-envs/nci-monai/lib/python3.11/site-packages/torch/payload.py'
    module.write_text('VALUE = "OLD"\n')
    stamp = module.stat()
    pyc = Path(py_compile.compile(str(module), doraise=True))
    module.write_text('VALUE = "NEW"\n')
    os.utime(module, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    record = marker(NCI, installed)
    script = 'import sys;sys.path.insert(0,sys.argv[1]);import payload;print(payload.VALUE)'
    # Reproduce the reviewer's stale cache before checking the new policy.
    assert subprocess.check_output([sys.executable,'-E','-s','-B','-c',script,str(module.parent)],text=True).strip() == 'OLD'
    with python_command(sys.executable, directory=installed) as command:
        assert subprocess.check_output([*command,'-c',script,str(module.parent)],text=True).strip() == 'NEW'
        assert not list(Path(command[-1].split('=',1)[1]).iterdir())
    old = pyc.read_bytes()
    py_compile.compile(str(module), doraise=True)
    assert pyc.read_bytes() != old
    validate_marker(record, NCI, installed)
    with python_command(sys.executable, directory=installed) as command:
        assert subprocess.check_output([*command,'-c',script,str(module.parent)],text=True).strip() == 'NEW'


def test_sourceless_bytecode_is_bound(installed):
    file = installed/'runtime-envs/nci-monai/lib/python3.11/site-packages/legacy.pyc'
    file.write_bytes(b'synthetic-bytecode-v1')
    record = marker(NCI, installed)
    assert any(k.endswith('legacy.pyc') for k in record['runtime_manifest']['files'])
    file.write_bytes(b'synthetic-bytecode-v2')
    with pytest.raises(ValueError, match='runtime_manifest'):
        validate_marker(record, NCI, installed)


@pytest.mark.parametrize('relative', ['bin/python','lib/python3.11/site-packages/SimpleITK/geometry.so'])
def test_worker_dependency_change_rejects_both_tasks(installed, relative):
    worker = installed/'worker'
    for name in ['bin/python','lib/python3.11/site-packages/SimpleITK/geometry.so']:
        file = worker/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'synthetic-v1')
    python = worker/'bin/python'
    records = {task:marker(task, installed, worker_python=python) for task in TASKS}
    (worker/relative).write_bytes(b'synthetic-v2')
    for task in TASKS:
        assert binding(task, installed, worker_python=python)['runtime_manifest'] == records[task]['runtime_manifest']
        with pytest.raises(ValueError, match='worker_runtime_manifest'):
            validate_marker(records[task], task, installed, worker_python=python)


@pytest.mark.parametrize('replacement', [False, True])
def test_preserved_mtime_change_invalidates_long_lived_preflight(installed, monkeypatch, replacement):
    import os
    import held_mri
    import model_preflight
    record = marker(NCI, installed)
    held_mri.store_device_marker(installed,NCI,record)
    monkeypatch.setattr(model_preflight,'MODEL_ROOT',installed)
    monkeypatch.setattr(model_preflight,'RUNTIME_ROOT',installed)
    monkeypatch.setattr(model_preflight,'COMMON',[])
    monkeypatch.setattr(model_preflight,'REQUIREMENTS',{NCI:[]})
    before = model_preflight.audit([NCI])[NCI]
    assert before['ready'] and before['acceptance_ready'] and before['orientation_ready'], before
    file = installed/'runtime-envs/nci-monai/lib/python3.11/site-packages/torch/kernel.so'
    stamp = file.stat(); data = file.read_bytes(); changed = bytes([data[0] ^ 1]) + data[1:]
    if replacement:
        other = file.with_suffix('.replacement');other.write_bytes(changed)
        os.utime(other,ns=(stamp.st_atime_ns,stamp.st_mtime_ns));other.replace(file)
    else:
        file.write_bytes(changed);os.utime(file,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert file.stat().st_size == stamp.st_size and file.stat().st_mtime_ns == stamp.st_mtime_ns
    after = model_preflight.audit([NCI])[NCI]
    assert after['ready'] is after['acceptance_ready'] is after['orientation_ready'] is False


def test_publication_and_runner_start_do_not_trust_digest_cache(installed, monkeypatch):
    import held_mri
    import prostate_models
    record = marker(NCI, installed)
    held_mri.store_device_marker(installed, NCI, record)
    file = installed/'runtime-envs/nci-monai/lib/python3.11/site-packages/torch/kernel.so'
    old = prostate_models.file_sha256(file)
    data = file.read_bytes();file.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    st = file.stat()
    # Deliberately poison even the new metadata key: decisive checks must bypass it.
    key = (str(file.resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    prostate_models._SHA256_CACHE[key] = old
    with pytest.raises(ValueError,match='runtime_manifest'):
        held_mri.store_device_marker(installed,NCI,record)
    prostate_models._SHA256_CACHE[key] = old
    monkeypatch.setattr('held_mri_runtime.check_python_policy',lambda:None)
    monkeypatch.setattr(held_mri.sys,'prefix',str(installed/'runtime-envs/nci-monai'))
    monkeypatch.setattr(held_mri.os.path,'samefile',lambda *args:True)
    monkeypatch.setattr(held_mri,'runtime_gpu',lambda task:record['gpu'])
    with pytest.raises(ValueError,match='runtime_manifest'):
        held_mri.runtime_start(NCI,installed,worker_python=sys.executable)


def test_runner_rejects_unsafe_flags_before_package_import(tmp_path):
    import subprocess
    runner = Path(__file__).parents[1]/'run_nci_prostate.py'
    result = subprocess.run([sys.executable,'-E','-s','-B',str(runner),'--help'],text=True,capture_output=True)
    assert result.returncode and 'pycache_prefix' in result.stderr


def test_fold_outputs_are_fresh_and_missing_folds_are_named(tmp_path):
    from held_mri_upstream import prepare_nci_folds, verify_nci_folds
    paths = prepare_nci_folds(tmp_path)
    for fold in (0,4):paths[fold].write_bytes(b'synthetic')
    with pytest.raises(ValueError,match='folds 1, 2, 3'):verify_nci_folds(paths)
    with pytest.raises(ValueError,match='pre-existing.*folds 0, 4'):prepare_nci_folds(tmp_path)


@pytest.mark.parametrize('defect',['populated','world_accessible','optimization','extra_xoption'])
def test_runner_checks_private_cache_and_flag_set(tmp_path,defect):
    import subprocess
    from held_mri_runtime import python_command
    runner = Path(__file__).parents[1]/'run_nci_prostate.py'
    with python_command(sys.executable,directory=tmp_path) as command:
        cache = Path(command[-1].split('=',1)[1])
        if defect=='populated': (cache/'stray.pyc').write_bytes(b'not eligible')
        elif defect=='world_accessible': cache.chmod(0o755)
        elif defect=='optimization': command.append('-O')
        else: command.extend(['-X','frozen_modules=off'])
        result = subprocess.run([*command,str(runner),'--help'],capture_output=True,text=True)
        assert result.returncode and 'pycache_prefix' in result.stderr


def test_worker_import_environment_change_invalidates(installed,monkeypatch):
    import held_mri_runtime
    records = {task:marker(task,installed) for task in TASKS}
    original = held_mri_runtime.runtime_identity
    def changed(python):
        result = original(python)
        if str(python)==sys.executable:
            result['import_path'] = [str(installed/'added-worker-import-root')]
        return result
    monkeypatch.setattr(held_mri_runtime,'runtime_identity',changed)
    for task in TASKS:
        assert binding(task,installed)['runtime_manifest']==records[task]['runtime_manifest']
        with pytest.raises(ValueError,match='worker_runtime_manifest'):
            validate_marker(records[task],task,installed)


def test_python_policy_check_runs_on_python_310_flags(monkeypatch, tmp_path):
    # The portal image runs Python 3.10, whose sys.flags has no safe_path attribute.
    import types
    import held_mri_runtime
    cache = tmp_path / 'held-pycache'
    cache.mkdir(mode=0o700)
    flags = types.SimpleNamespace(ignore_environment=1, no_user_site=1, optimize=0, isolated=0)
    monkeypatch.setattr(sys, 'flags', flags)
    monkeypatch.setattr(sys, 'pycache_prefix', str(cache))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(sys, '_xoptions', {'pycache_prefix': str(cache)})
    held_mri_runtime.check_python_policy()
    monkeypatch.setattr(sys, 'flags', types.SimpleNamespace(ignore_environment=1, no_user_site=1, optimize=0, isolated=1))
    with pytest.raises(ValueError, match='requires -E -s -B'):
        held_mri_runtime.check_python_policy()


def test_second_preflight_process_reuses_persistent_cache_without_rehashing(installed, monkeypatch):
    import os
    import subprocess
    import held_mri
    import model_preflight
    import prostate_models
    for task in TASKS:
        held_mri.store_device_marker(installed, task, marker(task, installed))
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {NCI: [], UMAMBA: []})
    prostate_models._SHA256_CACHE.clear()
    real_hash = prostate_models._hash_file_bytes
    hashed_first = []
    monkeypatch.setattr(prostate_models, '_hash_file_bytes', lambda path, *a: (hashed_first.append(str(path)), real_hash(path, *a))[1])
    real_binding = held_mri.binding
    binding_calls = []
    monkeypatch.setattr(held_mri, 'binding', lambda *a, **kw: (binding_calls.append(a[0]), real_binding(*a, **kw))[1])
    first = model_preflight.audit([NCI, UMAMBA])
    assert first[NCI]['ready'] and first[UMAMBA]['ready']
    assert len(hashed_first) > 0
    # Acceptance and orientation validation in the same preflight share one binding evaluation per task.
    assert binding_calls == [NCI, UMAMBA]
    cache_file = Path(os.environ['RTSEG_HELD_MRI_PREFLIGHT_CACHE']) / prostate_models.PREFLIGHT_CACHE_FILENAME
    assert cache_file.is_file()
    assert (cache_file.stat().st_mode & 0o777) == 0o600
    # Simulate a second process by clearing the in-memory cache and counting hash calls.
    prostate_models._SHA256_CACHE.clear()
    hashed_second = []
    monkeypatch.setattr(prostate_models, '_hash_file_bytes', lambda path, *a: (hashed_second.append(str(path)), real_hash(path, *a))[1])
    second = model_preflight.audit([NCI, UMAMBA])
    assert second[NCI]['ready'] and second[UMAMBA]['ready']
    assert hashed_second == []
    # Also verify reuse across a genuinely separate Python subprocess.
    script = (
        "import json, pathlib, sys\n"
        "import held_mri, held_mri_runtime, model_preflight, prostate_models\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "held_mri.pinned_source_hashes = lambda task: {p.as_posix(): prostate_models.file_sha256(root / p) for p in held_mri.source_files(task)}\n"
        "held_mri_runtime.runtime_identity = lambda python: {'version': 'synthetic', 'packages': [], 'stdlib': str(root / 'stdlib'), "
        "'prefix': str(pathlib.Path(python).parent.parent if pathlib.Path(python).is_relative_to(root) else root / 'worker')}\n"
        "held_mri.gpu_preflight = lambda task, r: {'compute_capability': [9, 0], 'name': 'NVIDIA H100 synthetic-test-record'}\n"
        "model_preflight.MODEL_ROOT = root\n"
        "model_preflight.RUNTIME_ROOT = root\n"
        "model_preflight.COMMON = []\n"
        "model_preflight.REQUIREMENTS = {t: [] for t in held_mri.TASKS}\n"
        "calls = []\n"
        "real = prostate_models._hash_file_bytes\n"
        "prostate_models._hash_file_bytes = lambda p, *a: (calls.append(str(p)), real(p, *a))[1]\n"
        "res = model_preflight.audit(list(held_mri.TASKS))\n"
        "print(json.dumps({'calls': len(calls), 'ready': {k: v['ready'] for k, v in res.items()}}))\n"
    )
    out = json.loads(subprocess.check_output([sys.executable, '-c', script, str(installed)], text=True))
    assert out == {'calls': 0, 'ready': {NCI: True, UMAMBA: True}}


@pytest.mark.parametrize('change', ['size', 'mtime', 'ctime', 'inode'])
def test_preflight_cache_rehashes_when_size_mtime_ctime_or_inode_changes(installed, monkeypatch, change):
    import os
    import held_mri
    import model_preflight
    import prostate_models
    record = marker(NCI, installed)
    held_mri.store_device_marker(installed, NCI, record)
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {NCI: []})
    prostate_models._SHA256_CACHE.clear()
    assert model_preflight.audit([NCI])[NCI]['ready'] is True
    target = installed / 'runtime-envs/nci-monai/lib/python3.11/site-packages/torch/kernel.so'
    resolved_target = str(target.resolve())
    before_stat = target.stat()
    data = target.read_bytes()
    flipped = bytes([data[0] ^ 1]) + data[1:]
    if change == 'size':
        target.write_bytes(data + b'!')
        os.utime(target, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        assert target.stat().st_size != before_stat.st_size
    elif change == 'mtime':
        target.write_bytes(flipped)
        os.utime(target, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns + 1_000_000))
        assert target.stat().st_mtime_ns != before_stat.st_mtime_ns
    elif change == 'ctime':
        target.write_bytes(flipped)
        os.utime(target, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        after_stat = target.stat()
        assert after_stat.st_size == before_stat.st_size
        assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
        assert after_stat.st_ino == before_stat.st_ino
        assert after_stat.st_ctime_ns != before_stat.st_ctime_ns
    elif change == 'inode':
        replacement = target.with_suffix('.replacement')
        replacement.write_bytes(flipped)
        os.utime(replacement, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        replacement.replace(target)
        after_stat = target.stat()
        assert after_stat.st_size == before_stat.st_size
        assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
        assert after_stat.st_ino != before_stat.st_ino
    prostate_models._SHA256_CACHE.clear()
    real_hash = prostate_models._hash_file_bytes
    hashed = []
    monkeypatch.setattr(prostate_models, '_hash_file_bytes', lambda path, *a: (hashed.append(str(Path(path).resolve())), real_hash(path, *a))[1])
    after = model_preflight.audit([NCI])[NCI]
    assert after['ready'] is False
    assert hashed == [resolved_target]


def test_poisoned_persistent_cache_cannot_pass_publication_dispatch_or_runner_start(installed, monkeypatch, tmp_path):
    import os
    import held_mri
    import model_preflight
    import prostate_models
    record = marker(NCI, installed)
    held_mri.store_device_marker(installed, NCI, record)
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {NCI: []})
    prostate_models._SHA256_CACHE.clear()
    assert model_preflight.audit([NCI])[NCI]['ready'] is True
    target = installed / 'runtime-envs/nci-monai/lib/python3.11/site-packages/torch/kernel.so'
    resolved = str(target.resolve())
    old_digest = prostate_models.file_sha256(target, uncached=True)
    data = target.read_bytes()
    target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    st = target.stat()
    # Poison both the persistent cache on disk and the in-process cache with the old digest for the new stat signature.
    cache_file = Path(os.environ['RTSEG_HELD_MRI_PREFLIGHT_CACHE']) / prostate_models.PREFLIGHT_CACHE_FILENAME
    payload = json.loads(cache_file.read_text())
    payload['entries'][resolved] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, old_digest]
    cache_file.write_text(json.dumps(payload))
    cache_file.chmod(0o600)
    key = (resolved, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    prostate_models._SHA256_CACHE[key] = old_digest
    # Preflight trusts the cache and reports ready, proving the poisoned entry matches.
    assert model_preflight.audit([NCI])[NCI]['ready'] is True
    # Publication, dispatch, and runner start must all bypass the cache and reject the modified file.
    with prostate_models.preflight_digest_cache(installed):
        with pytest.raises(ValueError, match='runtime_manifest'):
            held_mri.store_device_marker(installed, NCI, record)
        commands = []
        with pytest.raises(ValueError, match='runtime_manifest'):
            prostate_models.run_task(NCI, tmp_path / 'job', Path(__file__).parents[1], installed, sys.executable, commands.append)
        assert not commands
        monkeypatch.setattr('held_mri_runtime.check_python_policy', lambda: None)
        monkeypatch.setattr(held_mri.sys, 'prefix', str(installed / 'runtime-envs/nci-monai'))
        monkeypatch.setattr(held_mri.os.path, 'samefile', lambda *args: True)
        monkeypatch.setattr(held_mri, 'runtime_gpu', lambda task: record['gpu'])
        with pytest.raises(ValueError, match='runtime_manifest'):
            held_mri.runtime_start(NCI, installed, worker_python=sys.executable)


@pytest.mark.parametrize('defect', ['other_uid_file', 'other_uid_dir', 'group_writable_file', 'world_writable_file',
                                    'group_writable_dir', 'world_writable_dir', 'symlink_file', 'symlink_dir', 'inside_bundle'])
def test_preflight_cache_ignores_unsafe_owner_permissions_symlinks_or_bundle_paths(installed, monkeypatch, tmp_path, defect):
    import os
    import held_mri
    import model_preflight
    import prostate_models
    record = marker(NCI, installed)
    held_mri.store_device_marker(installed, NCI, record)
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {NCI: []})
    prostate_models._SHA256_CACHE.clear()
    assert model_preflight.audit([NCI])[NCI]['ready'] is True
    cache_dir = Path(os.environ['RTSEG_HELD_MRI_PREFLIGHT_CACHE'])
    cache_file = cache_dir / prostate_models.PREFLIGHT_CACHE_FILENAME
    # Poison the cache entry for kernel.so after modifying kernel.so; if the unsafe cache is ignored, preflight will re-hash and fail.
    target = installed / 'runtime-envs/nci-monai/lib/python3.11/site-packages/torch/kernel.so'
    resolved = str(target.resolve())
    old_digest = prostate_models.file_sha256(target, uncached=True)
    data = target.read_bytes()
    target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    st = target.stat()
    payload = json.loads(cache_file.read_text())
    payload['entries'][resolved] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, old_digest]
    cache_file.write_text(json.dumps(payload))
    cache_file.chmod(0o600)
    real_uid = os.getuid()
    if defect == 'other_uid_file':
        orig_load = prostate_models._load_preflight_cache_file
        def fake_uid_load(path):
            monkeypatch.setattr(prostate_models.os, 'getuid', lambda: real_uid + 1)
            try:
                return orig_load(path)
            finally:
                monkeypatch.setattr(prostate_models.os, 'getuid', lambda: real_uid)
        monkeypatch.setattr(prostate_models, '_load_preflight_cache_file', fake_uid_load)
    elif defect == 'other_uid_dir':
        monkeypatch.setattr(prostate_models.os, 'getuid', lambda: real_uid + 1)
    elif defect == 'group_writable_file':
        cache_file.chmod(0o620)
    elif defect == 'world_writable_file':
        cache_file.chmod(0o602)
    elif defect == 'group_writable_dir':
        cache_dir.chmod(0o770)
    elif defect == 'world_writable_dir':
        cache_dir.chmod(0o777)
    elif defect == 'symlink_file':
        external = tmp_path / 'external-cache.json'
        external.write_text(cache_file.read_text())
        external.chmod(0o600)
        cache_file.unlink()
        cache_file.symlink_to(external)
    elif defect == 'symlink_dir':
        link_dir = tmp_path / 'symlink-cache-dir'
        link_dir.symlink_to(cache_dir)
        monkeypatch.setenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', str(link_dir))
    elif defect == 'inside_bundle':
        inside = installed / 'cache-inside-bundle'
        inside.mkdir(mode=0o700)
        (inside / prostate_models.PREFLIGHT_CACHE_FILENAME).write_text(cache_file.read_text())
        (inside / prostate_models.PREFLIGHT_CACHE_FILENAME).chmod(0o600)
        monkeypatch.setenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', str(inside))
    prostate_models._SHA256_CACHE.clear()
    real_hash = prostate_models._hash_file_bytes
    hashed = []
    monkeypatch.setattr(prostate_models, '_hash_file_bytes', lambda path, *a: (hashed.append(str(path)), real_hash(path, *a))[1])
    after = model_preflight.audit([NCI])[NCI]
    assert after['ready'] is False
    assert len(hashed) > 0


@pytest.mark.parametrize('corrupt_content', [
    '{not-valid-json',
    '["not", "an", "object"]',
    '{"schema_version": 999, "entries": {}}',
    '{"schema_version": 1, "entries": "not-a-dict"}',
    '{"schema_version": 1, "entries": {"/some/path": ["bad", "tuple"]}}',
    '\x00\xff\xfe',
    pytest.param('[' * 1100 + '0' + ']' * 1100, id='deeply-nested-json'),  # json.loads raises RecursionError
])
def test_corrupt_preflight_cache_file_is_ignored_and_repaired(installed, monkeypatch, corrupt_content):
    import os
    import held_mri
    import model_preflight
    import prostate_models
    record = marker(NCI, installed)
    held_mri.store_device_marker(installed, NCI, record)
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', installed)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {NCI: []})
    cache_dir = Path(os.environ['RTSEG_HELD_MRI_PREFLIGHT_CACHE'])
    cache_file = cache_dir / prostate_models.PREFLIGHT_CACHE_FILENAME
    cache_file.write_bytes(corrupt_content.encode('latin1'))
    cache_file.chmod(0o600)
    prostate_models._SHA256_CACHE.clear()
    real_hash = prostate_models._hash_file_bytes
    hashed = []
    monkeypatch.setattr(prostate_models, '_hash_file_bytes', lambda path, *a: (hashed.append(str(path)), real_hash(path, *a))[1])
    first = model_preflight.audit([NCI])[NCI]
    assert first['ready'] is True
    assert len(hashed) > 0
    repaired = json.loads(cache_file.read_text())
    assert repaired['schema_version'] == prostate_models.PREFLIGHT_CACHE_SCHEMA
    assert isinstance(repaired['entries'], dict) and len(repaired['entries']) > 0
    assert (cache_file.stat().st_mode & 0o777) == 0o600
    prostate_models._SHA256_CACHE.clear()
    hashed_second = []
    monkeypatch.setattr(prostate_models, '_hash_file_bytes', lambda path, *a: (hashed_second.append(str(path)), real_hash(path, *a))[1])
    assert model_preflight.audit([NCI])[NCI]['ready'] is True
    assert hashed_second == []


def test_concurrent_preflight_writers_leave_valid_cache_file(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import prostate_models
    scratch_root = tmp_path / 'scratch-root'
    unwritable_home = tmp_path / 'unwritable-home'
    unwritable_home.write_text('not-a-directory')
    monkeypatch.delenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', raising=False)
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
    monkeypatch.setenv('HOME', str(unwritable_home))
    monkeypatch.setenv('RTSEG_SCRATCH_ROOT', str(scratch_root))
    cache_dir = prostate_models.preflight_cache_directory()
    assert cache_dir == scratch_root / 'held-mri-preflight-cache'
    files_root = tmp_path / 'files'
    files_root.mkdir()
    files = []
    for idx in range(24):
        f = files_root / f'file_{idx}.bin'
        f.write_bytes(f'payload-{idx}'.encode())
        files.append(f)

    def worker(worker_id: int) -> list[str]:
        subset = files[worker_id * 4:(worker_id + 1) * 4] + files[:2]
        with prostate_models.preflight_digest_cache(tmp_path / 'forbidden'):
            return [prostate_models.file_sha256(f) for f in subset]

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(worker, range(6)))
    assert len(results) == 6
    cache_file = cache_dir / prostate_models.PREFLIGHT_CACHE_FILENAME
    assert cache_file.is_file()
    assert (cache_file.stat().st_mode & 0o777) == 0o600
    assert not list(cache_dir.glob('*.tmp'))
    payload = json.loads(cache_file.read_text())
    assert payload['schema_version'] == prostate_models.PREFLIGHT_CACHE_SCHEMA
    for f in files:
        resolved = str(f.resolve())
        assert resolved in payload['entries']
        assert payload['entries'][resolved][5] == prostate_models.file_sha256(f, uncached=True)


def test_preflight_cache_default_location_resolution_order(tmp_path, monkeypatch):
    import prostate_models
    bundle = tmp_path / 'bundle'
    bundle.mkdir(mode=0o700)
    sample = tmp_path / 'sample.bin'
    sample.write_bytes(b'location-order-payload')
    env_cache = tmp_path / 'explicit-env-cache'
    xdg_root = tmp_path / 'xdg-cache-home'
    home_root = tmp_path / 'user-home'
    scratch_root = tmp_path / 'scratch-root'
    monkeypatch.setenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', str(env_cache))
    monkeypatch.setenv('XDG_CACHE_HOME', str(xdg_root))
    monkeypatch.setenv('HOME', str(home_root))
    monkeypatch.setenv('RTSEG_SCRATCH_ROOT', str(scratch_root))

    # 1. Explicit RTSEG_HELD_MRI_PREFLIGHT_CACHE wins over XDG, HOME, and scratch.
    prostate_models._SHA256_CACHE.clear()
    assert prostate_models.preflight_cache_directory(protected_roots=(bundle,)) == env_cache
    with prostate_models.preflight_digest_cache(bundle) as session:
        assert session['cache_dir'] == env_cache.resolve()
        prostate_models.file_sha256(sample)
    assert (env_cache / prostate_models.PREFLIGHT_CACHE_FILENAME).is_file()
    assert not xdg_root.exists() and not home_root.exists() and not scratch_root.exists()

    # 2. When RTSEG_HELD_MRI_PREFLIGHT_CACHE is unset, $XDG_CACHE_HOME/rtsegmentator/held-mri-preflight-cache wins over HOME and scratch.
    monkeypatch.delenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', raising=False)
    expected_xdg = xdg_root / 'rtsegmentator' / 'held-mri-preflight-cache'
    prostate_models._SHA256_CACHE.clear()
    assert prostate_models.preflight_cache_directory(protected_roots=(bundle,)) == expected_xdg
    with prostate_models.preflight_digest_cache(bundle) as session:
        assert session['cache_dir'] == expected_xdg.resolve()
        prostate_models.file_sha256(sample)
    assert (expected_xdg / prostate_models.PREFLIGHT_CACHE_FILENAME).is_file()
    assert not home_root.exists() and not scratch_root.exists()

    # 3. When XDG_CACHE_HOME is also unset, ~/.cache/rtsegmentator/held-mri-preflight-cache wins over scratch.
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
    expected_home = home_root / '.cache' / 'rtsegmentator' / 'held-mri-preflight-cache'
    prostate_models._SHA256_CACHE.clear()
    assert prostate_models.preflight_cache_directory(protected_roots=(bundle,)) == expected_home
    with prostate_models.preflight_digest_cache(bundle) as session:
        assert session['cache_dir'] == expected_home.resolve()
        prostate_models.file_sha256(sample)
    assert (expected_home / prostate_models.PREFLIGHT_CACHE_FILENAME).is_file()
    assert not scratch_root.exists()

    # 4. Fallback to <RTSEG_SCRATCH_ROOT>/held-mri-preflight-cache only when the home cache directory cannot be created or validated.
    for reason in ('file_home', 'readonly_home', 'world_writable_home', 'symlink_home', 'bundle_home', 'invalid_xdg'):
        case_dir = tmp_path / f'fallback-{reason}'
        case_dir.mkdir(mode=0o700)
        case_scratch = case_dir / 'scratch'
        monkeypatch.setenv('RTSEG_SCRATCH_ROOT', str(case_scratch))
        monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
        if reason == 'file_home':
            bad_home = case_dir / 'home-is-file'
            bad_home.write_text('not a directory')
            monkeypatch.setenv('HOME', str(bad_home))
        elif reason == 'readonly_home':
            ro_home = case_dir / 'ro-home'
            ro_home.mkdir(mode=0o500)
            monkeypatch.setenv('HOME', str(ro_home))
        elif reason == 'world_writable_home':
            ww_home = case_dir / 'ww-home'
            ww_home.mkdir(mode=0o700)
            ww_home.chmod(0o777)
            monkeypatch.setenv('HOME', str(ww_home))
        elif reason == 'symlink_home':
            real_home = case_dir / 'real-home'
            real_home.mkdir(mode=0o700)
            link_home = case_dir / 'link-home'
            link_home.symlink_to(real_home)
            monkeypatch.setenv('HOME', str(link_home))
        elif reason == 'bundle_home':
            monkeypatch.setenv('HOME', str(bundle / 'home-inside-bundle'))
        elif reason == 'invalid_xdg':
            bad_xdg = case_dir / 'xdg-is-file'
            bad_xdg.write_text('not a directory')
            monkeypatch.setenv('XDG_CACHE_HOME', str(bad_xdg))
            monkeypatch.setenv('HOME', str(home_root))
        expected_scratch = case_scratch / 'held-mri-preflight-cache'
        prostate_models._SHA256_CACHE.clear()
        assert prostate_models.preflight_cache_directory(protected_roots=(bundle,)) == expected_scratch, reason
        with prostate_models.preflight_digest_cache(bundle) as session:
            assert session['cache_dir'] == expected_scratch.resolve(), reason
            prostate_models.file_sha256(sample)
        assert (expected_scratch / prostate_models.PREFLIGHT_CACHE_FILENAME).is_file(), reason
        if reason == 'readonly_home':
            (case_dir / 'ro-home').chmod(0o700)


def test_preflight_cache_periodic_flush_writes_before_session_end(tmp_path, monkeypatch):
    import prostate_models
    cache_dir = tmp_path / 'periodic-cache'
    bundle = tmp_path / 'bundle'
    bundle.mkdir(mode=0o700)
    monkeypatch.setenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', str(cache_dir))
    files_dir = tmp_path / 'files'
    files_dir.mkdir(mode=0o700)
    files = []
    for idx in range(6):
        f = files_dir / f'item_{idx}.bin'
        f.write_bytes(f'periodic-content-{idx}'.encode())
        files.append(f)
    clock = [1000.0]
    monkeypatch.setattr(prostate_models.time, 'monotonic', lambda: clock[0])
    prostate_models._SHA256_CACHE.clear()
    cache_file = cache_dir / prostate_models.PREFLIGHT_CACHE_FILENAME

    with prostate_models.preflight_digest_cache(bundle):
        # At t=1000.0 (0 s elapsed) and t=1029.9 (29.9 s elapsed), no periodic flush has occurred yet.
        prostate_models.file_sha256(files[0])
        assert not cache_file.exists()
        clock[0] = 1029.9
        prostate_models.file_sha256(files[1])
        assert not cache_file.exists()

        # At t=1030.0 (exactly 30.0 s elapsed), hashing flushes files[0..2] before session end.
        clock[0] = 1030.0
        prostate_models.file_sha256(files[2])
        assert cache_file.is_file()
        assert (cache_file.stat().st_mode & 0o777) == 0o600
        mid_payload_1 = json.loads(cache_file.read_text())
        assert set(mid_payload_1['entries']) == {str(f.resolve()) for f in files[:3]}

        # At t=1045.0 (15.0 s since last flush at 1030.0), files[3] is kept in memory until next interval.
        clock[0] = 1045.0
        prostate_models.file_sha256(files[3])
        assert set(json.loads(cache_file.read_text())['entries']) == {str(f.resolve()) for f in files[:3]}

        # At t=1060.5 (30.5 s since last flush at 1030.0), hashing flushes files[0..4] before session end.
        clock[0] = 1060.5
        prostate_models.file_sha256(files[4])
        mid_payload_2 = json.loads(cache_file.read_text())
        assert set(mid_payload_2['entries']) == {str(f.resolve()) for f in files[:5]}

        # At t=1070.0 (<30 s since 1060.5), files[5] is hashed and flushed when the session ends.
        clock[0] = 1070.0
        prostate_models.file_sha256(files[5])
        assert set(json.loads(cache_file.read_text())['entries']) == {str(f.resolve()) for f in files[:5]}

    final_payload = json.loads(cache_file.read_text())
    assert set(final_payload['entries']) == {str(f.resolve()) for f in files}


@pytest.mark.parametrize('sig_name', ['SIGTERM', 'SIGHUP'])
def test_interrupted_preflight_session_flushes_on_sigterm_and_sighup(tmp_path, monkeypatch, sig_name):
    import os
    import signal
    import subprocess
    import prostate_models
    sig = getattr(signal, sig_name)
    cache_dir = tmp_path / f'cache-{sig_name}'
    bundle = tmp_path / 'bundle'
    bundle.mkdir(mode=0o700)
    files_dir = tmp_path / f'files-{sig_name}'
    files_dir.mkdir(mode=0o700)
    files = []
    for idx in range(6):
        f = files_dir / f'part_{idx}.bin'
        f.write_bytes(f'signal-payload-{sig_name}-{idx}'.encode())
        files.append(f)

    # Verify in-process that previous signal handlers are restored after preflight_digest_cache exits.
    monkeypatch.setenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', str(cache_dir))
    prev_term = signal.getsignal(signal.SIGTERM)
    prev_hup = signal.getsignal(signal.SIGHUP)
    sentinel_term = lambda s, f: None
    sentinel_hup = lambda s, f: None
    signal.signal(signal.SIGTERM, sentinel_term)
    signal.signal(signal.SIGHUP, sentinel_hup)
    try:
        with prostate_models.preflight_digest_cache(bundle):
            assert signal.getsignal(signal.SIGTERM) is not sentinel_term
            assert signal.getsignal(signal.SIGHUP) is not sentinel_hup
        assert signal.getsignal(signal.SIGTERM) is sentinel_term
        assert signal.getsignal(signal.SIGHUP) is sentinel_hup
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGHUP, prev_hup)

    # Simulate SIGTERM/SIGHUP in a child process mid-hashing (on the 4th file, before the 30 s periodic flush).
    child_script = (
        "import json, pathlib, sys, time\n"
        "import prostate_models\n"
        "bundle = pathlib.Path(sys.argv[1])\n"
        "files = [pathlib.Path(p) for p in json.loads(sys.argv[2])]\n"
        "real_hash = prostate_models._hash_file_bytes\n"
        "count = 0\n"
        "def slow_hash(path, *a):\n"
        "    global count\n"
        "    count += 1\n"
        "    if count == 4:\n"
        "        sys.stdout.write('MID_HASHING\\n')\n"
        "        sys.stdout.flush()\n"
        "        time.sleep(30)\n"
        "    return real_hash(path, *a)\n"
        "prostate_models._hash_file_bytes = slow_hash\n"
        "with prostate_models.preflight_digest_cache(bundle):\n"
        "    for f in files:\n"
        "        prostate_models.file_sha256(f)\n"
    )
    env = {**os.environ, 'RTSEG_HELD_MRI_PREFLIGHT_CACHE': str(cache_dir),
           'PYTHONPATH': os.pathsep.join([str(Path(__file__).parents[1]), os.environ.get('PYTHONPATH', '')])}
    cache_file = cache_dir / prostate_models.PREFLIGHT_CACHE_FILENAME
    assert not cache_file.exists()
    proc = subprocess.Popen(
        [sys.executable, '-c', child_script, str(bundle), json.dumps([str(f) for f in files])],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    line = proc.stdout.readline().strip()
    assert line == 'MID_HASHING'
    # Before sending the signal, less than 30 s has elapsed so digests.json has not been written yet.
    assert not cache_file.exists()
    proc.send_signal(sig)
    _, stderr = proc.communicate(timeout=15)
    assert proc.returncode == 128 + int(sig), stderr
    assert cache_file.is_file()
    assert (cache_file.stat().st_mode & 0o777) == 0o600
    assert not list(cache_dir.glob('*.tmp'))
    payload = json.loads(cache_file.read_text())
    assert payload['schema_version'] == prostate_models.PREFLIGHT_CACHE_SCHEMA
    expected_before_signal = {str(f.resolve()) for f in files[:3]}
    assert set(payload['entries']) == expected_before_signal
    for f in files[:3]:
        assert payload['entries'][str(f.resolve())][5] == prostate_models.file_sha256(f, uncached=True)

    # A subsequent session reuses the 3 flushed entries and hashes only the remaining 3 files.
    prostate_models._SHA256_CACHE.clear()
    real_hash = prostate_models._hash_file_bytes
    hashed_after = []
    monkeypatch.setattr(prostate_models, '_hash_file_bytes',
                        lambda path, *a: (hashed_after.append(str(Path(path).resolve())), real_hash(path, *a))[1])
    with prostate_models.preflight_digest_cache(bundle):
        for f in files:
            prostate_models.file_sha256(f)
    assert hashed_after == [str(f.resolve()) for f in files[3:]]
    assert set(json.loads(cache_file.read_text())['entries']) == {str(f.resolve()) for f in files}


def test_existing_read_only_home_cache_falls_back_to_scratch(tmp_path, monkeypatch):
    # Release review of the cache (round 1): an existing, safe but unwritable default cache must not be selected.
    import os
    import prostate_models
    bundle = tmp_path / 'bundle'
    bundle.mkdir(mode=0o700)
    sample = tmp_path / 'sample.bin'
    sample.write_bytes(b'read-only-home-payload')
    home_root = tmp_path / 'home'
    scratch_root = tmp_path / 'scratch'
    scratch_root.mkdir(mode=0o700)
    home_cache = home_root / '.cache' / 'rtsegmentator' / 'held-mri-preflight-cache'
    home_cache.mkdir(parents=True, mode=0o700)
    for d in (home_root, home_root / '.cache', home_root / '.cache' / 'rtsegmentator'):
        d.chmod(0o700)
    home_cache.chmod(0o500)
    monkeypatch.delenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', raising=False)
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
    monkeypatch.setenv('HOME', str(home_root))
    monkeypatch.setenv('RTSEG_SCRATCH_ROOT', str(scratch_root))
    try:
        if os.access(home_cache, os.W_OK):
            pytest.skip('running with privileges that bypass directory permissions')
        prostate_models._SHA256_CACHE.clear()
        with prostate_models.preflight_digest_cache(bundle) as session:
            assert session['cache_dir'] == (scratch_root / 'held-mri-preflight-cache').resolve()
            prostate_models.file_sha256(sample)
        assert (scratch_root / 'held-mri-preflight-cache' / prostate_models.PREFLIGHT_CACHE_FILENAME).is_file()
        assert not (home_cache / prostate_models.PREFLIGHT_CACHE_FILENAME).exists()
    finally:
        home_cache.chmod(0o700)


def test_flush_restores_callers_signal_mask(tmp_path):
    # Release review of the cache (round 1): a signal the caller blocked must stay blocked, and a pending
    # blocked signal must not be delivered by the cache when the session ends.
    import os
    import subprocess
    script = (
        "import os, pathlib, signal, sys\n"
        "import prostate_models\n"
        "work = pathlib.Path(sys.argv[1])\n"
        "bundle = work / 'bundle'; bundle.mkdir(mode=0o700)\n"
        "sample = work / 'sample.bin'; sample.write_bytes(b'mask-payload')\n"
        "os.environ['RTSEG_HELD_MRI_PREFLIGHT_CACHE'] = str(work / 'cache')\n"
        "signal.signal(signal.SIGHUP, lambda s, f: print('UNEXPECTED_EARLY_HUP', flush=True))\n"
        "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGHUP})\n"
        "with prostate_models.preflight_digest_cache(bundle):\n"
        "    prostate_models.file_sha256(sample)\n"
        "    os.kill(os.getpid(), signal.SIGHUP)\n"
        "    print('HUP_QUEUED_WHILE_BLOCKED', flush=True)\n"
        "still_blocked = signal.SIGHUP in signal.pthread_sigmask(signal.SIG_BLOCK, set())\n"
        "print('STILL_BLOCKED' if still_blocked else 'UNBLOCKED', flush=True)\n"
        "print('AFTER_CONTEXT', flush=True)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != 'RTSEG_HELD_MRI_PREFLIGHT_CACHE'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True,
                            env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['HUP_QUEUED_WHILE_BLOCKED', 'STILL_BLOCKED', 'AFTER_CONTEXT']

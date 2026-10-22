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
def installed(tmp_path,monkeypatch):
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

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from choi_ontology import REGIONS, confirmed_labels
from choi_anatomy import analyze, load_image, side_evidence, propose
from run_choi_tmli import prepare_input, prepare_model, prediction_command


def save(path, data, affine=None):
    img=nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    img.header.set_xyzt_units('mm'); nib.save(img,path)
    return path


def test_confirmation_rejects_proposal_missing_duplicate_and_unattributed(tmp_path):
    path=tmp_path/'labels.json'
    record={'schema_version':1,'confirmed':True,'reviewer':'Test reviewer','reviewed_utc':'2026-10-15',
            'evidence':['synthetic review'],'labels':dict(zip(map(str,range(1,19)),REGIONS))}
    path.write_text(json.dumps(record));assert confirmed_labels(path)==record['labels']
    for change in ({'confirmed':False},{'reviewer':''},{'evidence':[]},
                   {'labels':{'1':'Cervical_Left'}},
                   {'labels':{str(i):'Cervical_Left' for i in range(1,19)}}):
        path.write_text(json.dumps({**record,**change}))
        with pytest.raises(ValueError):confirmed_labels(path)
    with pytest.raises(ValueError):confirmed_labels(Path(__file__).parents[1]/'labels/choi-tmli.json')


def test_left_right_uses_patient_ras_and_local_midline():
    mid=np.array([[20,0,0],[30,0,100]])
    left=side_evidence(np.array([[10,0,10],[20,0,90]]),mid)
    right=side_evidence(np.array([[40,0,10],[40,0,90]]),mid)
    assert left['side']=='Left' and right['side']=='Right'
    assert left['left_fraction']==1 and not left['midline_extrapolated']
    assert side_evidence(np.array([[0,0,150]]),mid)['midline_extrapolated']
    assert side_evidence(np.array([[0,0,0]]),np.empty((0,3)))['side']=='unresolved'


def test_model_mirror_is_private_for_metadata_and_command_disables_mirroring(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'plans.pkl').write_bytes(b'plans')
    fold=source/'fold_0';fold.mkdir()
    for name in ('model_final_checkpoint.model','model_final_checkpoint.model.pkl'):(fold/name).write_bytes(b'test')
    mirror=prepare_model(source,tmp_path/'results',[0])
    assert (mirror/'fold_0/model_final_checkpoint.model').is_symlink()
    assert not (mirror/'plans.pkl').is_symlink()
    (mirror/'plans.pkl').write_bytes(b'private');assert (source/'plans.pkl').read_bytes()==b'plans'
    command=prediction_command('predict','in','out',[0,1,2,3,4])
    assert command[command.index('-f')+1:command.index('--disable_tta')]==['0','1','2','3','4']
    with pytest.raises(FileExistsError):prepare_model(source,tmp_path/'results',[0])


def test_geometry_validation_rejects_misalignment_noninteger_and_unknown_units(tmp_path):
    ref=save(tmp_path/'ref.nii.gz',np.zeros((4,5,6),dtype=np.uint8))
    image,_=load_image(ref)
    aff=np.eye(4);aff[0,3]=10
    mismatch=save(tmp_path/'wrong.nii.gz',np.zeros((4,5,6),dtype=np.uint8),aff)
    with pytest.raises(ValueError,match='Grid mismatch'):load_image(mismatch,image,labels=True)
    bad=save(tmp_path/'bad.nii.gz',np.ones((4,5,6),dtype=np.float32)*.5)
    with pytest.raises(ValueError,match='integer'):load_image(bad,image,labels=True)
    img=nib.load(ref);img.header.set_xyzt_units('unknown');nib.save(img,bad)
    with pytest.raises(ValueError,match='millimetre'):load_image(bad)
    # A binary mask can inherit mm only from a matching, explicitly mm CT.
    load_image(bad,image,labels=True)


def test_report_volume_laterality_distances_reorientation_and_missing_labels(tmp_path):
    shape=(26,24,32);aff=np.diag([2.,3.,4.,1.]);aff[:3,3]=[-26,-36,-64]
    ct=save(tmp_path/'ct.nii.gz',np.zeros(shape,dtype=np.int16),aff)
    pred=np.zeros(shape,dtype=np.uint8);pred[4:7,10:13,21:24]=1;pred[19:22,10:13,21:24]=2
    prediction=save(tmp_path/'pred.nii.gz',pred,aff)
    masks=tmp_path/'total';masks.mkdir()
    for name,z in [('vertebrae_C7',22),('vertebrae_T1',18),('vertebrae_T6',12),('vertebrae_T12',8),('vertebrae_L5',4)]:
        mask=np.zeros(shape,dtype=np.uint8);mask[12:15,8:11,z:z+2]=1;save(masks/(name+'.nii.gz'),mask,aff)
    report=analyze(prediction,ct,masks,tmp_path/'review')
    first,second=report['labels'][:2]
    assert first['voxels']==27 and first['volume_ml']==pytest.approx(.648)
    assert first['side']['side']=='Left' and second['side']['side']=='Right'
    assert first['proposal']=='Cervical_Left' and second['proposal']=='Cervical_Right'
    assert first['centroid_lps_mm'][0]>0
    expected=np.linalg.norm(np.array([12,10,22])-np.array([6,10,22]))*2
    cervical=next(n for n in first['neighbours'] if n['structure']=='vertebrae_C7')
    assert cervical['minimum_voxel_centre_distance_mm']==pytest.approx(expected)
    assert len(list((tmp_path/'review').glob('label_*.png')))==18
    sheet=(tmp_path/'review/review.html').read_text()
    assert sheet.count('src="data:image/png;base64,')==18 and 'ABSENT' not in sheet  # absent text is rendered in the PNG
    assert report['labels'][17]['proposal'] is None and report['confirmed'] is False
    # A lossless flip is still the same physical geometry, not an opposite laterality.
    flipped=nib.load(prediction).as_reoriented(np.array([[0,-1],[1,1],[2,1]]));nib.save(flipped,tmp_path/'flip.nii.gz')
    _,reloaded=load_image(tmp_path/'flip.nii.gz',nib.as_closest_canonical(nib.load(ct)),labels=True)
    np.testing.assert_array_equal(reloaded,pred)


def test_oblique_physical_centroid_and_empty_prediction(tmp_path):
    shape=(7,8,9);aff=np.array([[2,-.3,0,-30],[.3,2,0,10],[0,0,4,-10],[0,0,0,1.]])
    ct=save(tmp_path/'ct.nii.gz',np.zeros(shape,dtype=np.float32),aff)
    pred=np.zeros(shape,dtype=np.uint8);pred[2,3,4]=1
    masks=tmp_path/'total';masks.mkdir()
    path=save(tmp_path/'pred.nii.gz',pred,aff)
    report=analyze(path,ct,masks,tmp_path/'review')
    np.testing.assert_allclose(report['labels'][0]['centroid_ras_mm'],nib.affines.apply_affine(aff,[2,3,4]))
    assert report['labels'][0]['proposal'] is None
    save(path,np.zeros(shape,dtype=np.uint8),aff)
    empty=analyze(path,ct,masks,tmp_path/'empty')
    assert all(r['volume_ml']==0 and r['confidence']==0 for r in empty['labels'])


def test_mediastinal_rule_retains_ambiguity():
    landmarks={f'vertebrae_{k}':{'centroid_ras_mm':[0,0,z]} for k,z in [('T1',100),('T4',70),('T5',60),('T12',0)]}
    row={'voxels':100,'centroid_ras_mm':[0,20,75],'side':{'side':'midline_or_mixed','mean_offset_ras_x_mm':0,'midline_extrapolated':False},
         'neighbours':[{'structure':'trachea','centroid_to_mask_mm':10}]}
    candidates,flags=propose(row,landmarks)
    assert candidates[0]['region']=='Superior_Mediastinal'
    assert {c['region'] for c in candidates}=={'Superior_Mediastinal','Inferior_Mediastinal'}
    assert candidates[0]['score']<=.4 and any('clinician review' in f for f in flags)


def test_disconnected_components_keep_full_extent_and_review_largest(tmp_path):
    shape=(32,10,12);aff=np.diag([5.,5.,5.,1.])
    ct=save(tmp_path/'ct.nii.gz',np.zeros(shape,dtype=np.int16),aff)
    pred=np.zeros(shape,dtype=np.uint8);pred[2:6,3:6,4:7]=1;pred[25:27,3:6,4:7]=1
    path=save(tmp_path/'pred.nii.gz',pred,aff);total=tmp_path/'total';total.mkdir()
    row=analyze(path,ct,total,tmp_path/'review')['labels'][0]
    assert row['component_count']==2 and row['largest_component_fraction']==pytest.approx(2/3)
    assert row['bounds_ras_mm'][1][0]==130
    assert row['largest_components'][0]['bounds_ras_mm'][1][0]==25
    assert row['review_plane_point_ras_mm'][0]<=25
    assert any('disconnected components' in flag for flag in row['flags'])


def test_preflight_blocks_even_when_all_files_exist(monkeypatch, tmp_path):
    import model_preflight
    monkeypatch.setattr(model_preflight, 'APP_ROOT', tmp_path)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {'choi_tmli': []})
    (tmp_path/'labels').mkdir()
    path=tmp_path/'labels/choi-tmli.json'
    path.write_text(json.dumps({'confirmed':False}))
    check=model_preflight.audit(['choi_tmli'])['choi_tmli']
    assert check['ready'] is False and any('clinician' in x for x in check['missing'])
    path.write_text(json.dumps({'schema_version':1,'confirmed':True,'reviewer':'Test',
        'reviewed_utc':'2026-10-15','evidence':['synthetic'],
        'labels':dict(zip(map(str,range(1,19)),REGIONS))}))
    assert model_preflight.audit(['choi_tmli'])['choi_tmli']['ready'] is True


def test_catalog_stays_disabled_even_with_passing_preflight(portal):
    for name in ('choi_tmli','clnet_235','lnq_atlas'):
        item={'name':name,**portal.INFORMATIONAL_MODELS[name]}
        portal.apply_model_gates(item,{'ready':True,'orientation_ready':True,'missing':[]},{name})
        assert item['enabled'] is False


def test_dispatch_rejects_unconfirmed_choi_before_spawning(monkeypatch,tmp_path):
    import os,runpy,signal,subprocess,sys,tempfile
    root=Path(__file__).parents[1]
    job=tmp_path/'job';(job/'output').mkdir(parents=True)
    monkeypatch.setenv('RTSEG_APP_ROOT',str(root));monkeypatch.setenv('RTSEG_MODEL_ROOT',str(tmp_path/'models'))
    monkeypatch.setattr(sys,'argv',['run_task.py',str(job),'choi_tmli'])
    monkeypatch.setattr(signal,'signal',lambda *args:None)
    def fail(*args,**kwargs):raise AssertionError('An unconfirmed mapping started inference')
    monkeypatch.setattr(subprocess,'Popen',fail)
    fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile,'tempdir',f'/proc/self/fd/{fd}')
        with pytest.raises(ValueError,match='clinician confirmation'):
            runpy.run_path(str(root/'run_task.py'),run_name='__main__')
    finally:os.close(fd)


def test_review_summary_component_spans_csv_and_fold_agreement(tmp_path):
    import csv
    from html.parser import HTMLParser
    shape=(12,12,30);aff=np.diag([5.,5.,5.,1.])
    ct=save(tmp_path/'ct.nii.gz',np.zeros(shape,dtype=np.int16),aff)
    pred=np.zeros(shape,dtype=np.uint8)
    pred[1:4,3:6,10:12]=1  # 2.25 mL, largest, near T6
    pred[7:9,3:5,20:22]=1  # 1.00 mL, second, near T1
    pred[2,2,29]=1         # tiny distant fragment near C1
    prediction=save(tmp_path/'pred.nii.gz',pred,aff)
    masks=tmp_path/'total';masks.mkdir()
    for name,z in [('T6',10),('T1',20),('C1',29)]:
        mask=np.zeros(shape,dtype=np.uint8);mask[5:7,5:7,z:z+1]=1
        save(masks/f'vertebrae_{name}.nii.gz',mask,aff)
    report=analyze(prediction,ct,masks,tmp_path/'review',{'1':{'mean_dice':.75}})
    row=report['labels'][0]
    assert row['component_count']==3
    assert row['largest_component_vertebral_extent']['cranial_nearest_centroid']=='vertebrae_T6'
    assert row['components_ge_1ml_vertebral_extent']['cranial_nearest_centroid']=='vertebrae_T1'
    assert row['full_label_vertebral_extent']['cranial_nearest_centroid']=='vertebrae_C1'
    assert row['components_ge_1ml']['component_count']==2
    records=list(csv.DictReader((tmp_path/'review/choi-review-template.csv').open()))
    assert len(records)==18 and list(records[0])==['label','proposed region','reviewer region','confidence','comment']
    assert all(record['reviewer region']==record['confidence']==record['comment']=='' for record in records)
    sheet=(tmp_path/'review/review.html').read_text()
    assert sheet.index('id="summary"') < sheet.index('id="label-1"')
    assert '0.750 (mean Dice)' in sheet and 'Components ≥1 mL:' in sheet
    class DetailsCheck(HTMLParser):
        def __init__(self):super().__init__();self.depth=0;self.raw_blocks=0
        def handle_starttag(self,tag,attrs):
            if tag=='details':self.depth+=1
            if tag=='pre':
                assert self.depth>0
                self.raw_blocks+=1
        def handle_endtag(self,tag):
            if tag=='details':self.depth-=1
    parser=DetailsCheck();parser.feed(sheet)
    assert parser.raw_blocks>=36

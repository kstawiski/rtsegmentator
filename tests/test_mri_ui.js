'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {roleCandidates,reviewedRole,roleOptionLabel,seriesPlan,roleChoicesForSubmit}=require('../static/app.js');
const roles=[{role:'t2w',modality:'MR',label:'T2W',required:true},{role:'adc',modality:'MR',label:'ADC map',required:true},{role:'hbv',modality:'MR',label:'High b-value DWI',required:true}];
const primary={key:'t2',modality:'MR',study_uid:'study',frame_uid:'frame',description:'T2 axial',rows:8,columns:8,slices:3};
const adc={...primary,key:'adc',description:'ADC map',is_adc:true};
const dwi={...primary,key:'dwi',description:'DWI',hbv_options:[{b_value:800,b_value_source:'(0018,9087)',dimensions:[8,8,3]},{b_value:1400,b_value_source:'(0018,9087)',dimensions:[8,8,3]}]};
const multi={name:'synthetic_bpmri',modality:'MR',input_roles:roles,reference_role:'t2w'};
const single={name:'total_mr',modality:'MR'};

test('supervised PI-CAI card gives the numeric reason for its explicit block',()=>{
  const {blockedReason}=require('../static/app.js');
  const task=require('../catalog/prostate.json').picai_nnunet_supervised_bpmri;
  const reason=blockedReason({...task,name:'picai_nnunet_supervised_bpmri'},primary,[primary,adc,dwi]);
  assert.equal(task.enabled,false);
  assert.equal(task.blocked_status,task.status);
  assert.match(reason,/390 vs 1,132 voxels/);
  assert.match(reason,/outside the authors' reproduction tolerance/);
});

test('ADC evidence ranks above description; candidates stay within the study and modality',()=>{
  const rows=[primary,adc,dwi,{...adc,key:'named',is_adc:false},{...adc,key:'foreign',study_uid:'other'},{...adc,key:'ct',modality:'CT'}];
  const options=roleCandidates(primary,rows,roles[1]);
  assert.equal(options[0].series.key,'adc');
  assert.deepEqual(options.map(o=>o.series.key),['adc','named','dwi']);
  assert.deepEqual(reviewedRole(primary,rows,roles[1]).choice,{series_key:'adc'});
});

test('HBV suggestions choose the highest documented b-value, while every volume stays editable',()=>{
  const rows=[primary,adc,dwi,{...dwi,key:'unknown',hbv_options:[]}];
  const review=reviewedRole(primary,rows,roles[2]);
  assert.equal(review.choice.b_value,1400);
  assert.deepEqual(review.options.map(o=>o.choice.b_value),[1400,800]);
  const chosen=reviewedRole(primary,rows,roles[2],{hbv:{series_key:'dwi',b_value:800}});
  assert.equal(chosen.choice.b_value,800);
  assert.match(roleOptionLabel(chosen.selected,primary),/b=800 s\/mm² · \(0018,9087\) · 8 × 8 × 3/);
});

test('equal ranked series require a choice; explicit choices and explicit missing values are retained',()=>{
  const rows=[primary,adc,dwi,{...adc,key:'adc-repeat'}];
  assert.match(reviewedRole(primary,rows,roles[1]).reason,/Ambiguous ADC map/);
  assert.equal(reviewedRole(primary,rows,roles[1],{adc:'adc-repeat'}).choice.series_key,'adc-repeat');
  assert.match(reviewedRole(primary,rows,roles[1],{adc:null}).reason,/Missing ADC map/);
  assert.match(reviewedRole(primary,[primary,adc,dwi,{...dwi,key:'dwi-repeat'}],roles[2]).reason,/Ambiguous/);
});

test('different Frame of Reference remains selectable with a visible warning',()=>{
  const rows=[primary,{...adc,frame_uid:'different'}];
  const review=reviewedRole(primary,rows,roles[1]);
  assert.equal(review.reason,'');
  assert.match(roleOptionLabel(review.selected,primary),/Frame of Reference differs/);
});

test('missing required roles block only the multi-sequence model',()=>{
  const selected=new Set([single.name,multi.name]);
  const [plan]=seriesPlan([primary],selected,[single,multi],[primary,adc]);
  assert.deepEqual(plan.models.map(t=>t.name),['total_mr']);
  assert.equal(plan.reason,'');
  assert.match(plan.skipped[0].reason,/Missing High/);
  const [blocked]=seriesPlan([primary],new Set([multi.name]),[multi],[primary,adc]);
  assert.equal(blocked.models.length,0);
  assert.match(blocked.reason,/Missing High/);
});

test('each primary retains independent role selection',()=>{
  const second={...primary,key:'t2-repeat'};
  const plans=seriesPlan([primary,second],new Set([single.name,multi.name]),[single,multi],
    [primary,second,adc,dwi],'nv_reason_ct_report',{'t2-repeat':{hbv:null}});
  assert.equal(plans[0].models.length,2);
  assert.equal(plans[1].models.length,1);
});

test('single-series planning remains independent of companion metadata',()=>{
  const [plan]=seriesPlan([primary],new Set([single.name]),[single],[primary]);
  assert.deepEqual(plan.models,[single]);
  assert.equal(plan.reason,'');
  assert.deepEqual(plan.skipped,[]);
});

for (const description of ['DWI for ADC calculation','ADC map']) {
  test(`description-only ${description} requires a user choice`,()=>{
    const named={...adc,is_adc:false,description};
    const rows=[primary,named,dwi];
    const review=reviewedRole(primary,rows,roles[1]);
    assert.equal(review.choice,null);
    assert.match(review.reason,/ImageType/);
    assert.equal(review.options[0].series.key,'adc');
    const explicit=reviewedRole(primary,rows,roles[1],{adc:'adc'});
    assert.deepEqual(explicit.choice,{series_key:'adc'});
    assert.equal(explicit.reason,'');
    const [plan]=seriesPlan([primary],new Set([multi.name]),[multi],rows);
    assert.equal(plan.models.length,0);
  });
}

test('assessment failure blocks reference and companion selection with a reason',()=>{
  const bad={...adc,unavailable_reason:'Series assessment rejected one instance'};
  assert.equal(reviewedRole(primary,[primary,bad,dwi],roles[1],{adc:'adc'}).choice,null);
  assert.match(reviewedRole(primary,[primary,bad,dwi],roles[1],{adc:'adc'}).reason,/assessment/);
  const [plan]=seriesPlan([{...primary,unavailable_reason:bad.unavailable_reason}],
    new Set([multi.name]),[multi],[primary,adc,dwi]);
  assert.equal(plan.models.length,0);
  assert.match(plan.reason,/assessment/);
});


test('submitting suggested roles preserves automatic versus user selection',()=>{
  assert.deepEqual(roleChoicesForSubmit(roles,{}),{});
  assert.deepEqual(roleChoicesForSubmit(roles,{adc:{series_key:'adc'}}),{adc:{series_key:'adc'}});
  assert.deepEqual(roleChoicesForSubmit(roles,{adc:null}),{adc:null});
});

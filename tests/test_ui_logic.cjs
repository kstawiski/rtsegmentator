'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const ui = require('../static/app.js');
const metadata = require('../static/ui-catalog.json');
const ct = {modality:'CT',study_uid:'study',frame_uid:'frame',columns:32,rows:24,slices:8};
const pet = {...ct,modality:'PT'};
const mr = {...ct,modality:'MR'};
const task = (name,extra={})=>({name,modality:'CT',enabled:true,...extra});

test('clinical grouping uses explicit ambiguous-task overrides and an unknown-task fallback',()=>{
  assert.equal(ui.presentation(task('siam_v3_head_mr_ct'),metadata).group,'brain');
  assert.equal(ui.presentation(task('cervix_cbct'),metadata).group,'ctv');
  assert.equal(ui.presentation(task('hnlnl_2d3d'),metadata).group,'ctv');
  assert.equal(ui.presentation(task('cerr_ct_lung_smit'),metadata).group,'tumour');
  assert.equal(ui.presentation(task('abdominal_muscles'),metadata).group,'oar');
  assert.equal(ui.presentation(task('future_worker_xyz'),metadata).group,'other');
  assert.equal(ui.presentation(task('head_tumor_prompt',{prompt:'box'}),metadata).group,'interactive');
  assert.equal(ui.presentation(task('future_report',{output:'report'}),metadata).group,'report');
});

test('modality filtering retains multi-modality SIAM and PET-only models',()=>{
  const siam=task('siam_v3_head_mr_ct',{modality:'MR',modalities:['MR','CT']});
  assert(ui.supports(siam,ct)); assert(ui.supports(siam,mr)); assert(!ui.supports(siam,pet));
  assert(!ui.supports(task('total_mr',{modality:'MR'}),ct));
  assert(ui.supports(task('pet_model',{modality:'PT'}),pet));
});

test('availability retains exact catalog and missing dependency reasons',()=>{
  const restricted=task('external',{enabled:false,status:'orientation_revalidation_pending',missing:['checkpoint','orientation marker']});
  assert.equal(ui.blockedReason(restricted,ct,[ct]),'orientation_revalidation_pending · Missing: checkpoint; orientation marker');
  assert.equal(ui.blockedReason(task('licensed',{licensed:true}),ct,[ct]),'');
});

test('reports are refused for CBCT and non-CT series',()=>{
  const report=task('nv_reason_ct_report',{output:'report'});
  assert.match(ui.blockedReason(report,{...ct,cbct:true,cbct_reason:'cone-beam acquisition'},[ct]),/cone-beam acquisition/);
  assert.match(ui.blockedReason(report,mr,[mr]),/Requires CT/);
  assert.equal(ui.blockedReason(report,ct,[ct]),'');
});

test('PET pairing uses both Study and Frame of Reference',()=>{
  const model=task('isrt_pet1_early_rigid',{modality:'PT',requires_ct:true});
  assert.match(ui.blockedReason(model,pet,[pet]),/same Study and Frame/);
  assert.match(ui.blockedReason(model,pet,[pet,{...ct,frame_uid:'different'}]),/same Study and Frame/);
  assert.equal(ui.blockedReason(model,pet,[pet,ct]),'');
  assert.equal(ui.blockedReason(task('dotatate_pet_lesions',{modality:'PT'}),pet,[pet]),'');
});

test('SBRT post-processing is only exposed for local CT total inference',()=>{
  const selected=new Set(['total']),catalog=[task('total')];
  assert(ui.cropSupported(ct,selected,catalog,{worker_mode:'local'},null));
  assert(ui.cropSupported(ct,selected,catalog,null,{worker_mode:'local'}));
  assert(!ui.cropSupported(ct,selected,catalog,{worker_mode:'ssh'},null));
  assert(!ui.cropSupported(mr,selected,catalog,{worker_mode:'local'},null));
  assert(!ui.cropSupported(ct,new Set(['total_mr']),catalog,{worker_mode:'local'},null));
  assert(!ui.cropSupported(ct,selected,[task('total',{execution_backend:'ssh'})],{worker_mode:'local'},null));
  assert(!ui.cropSupported(ct,selected,catalog,{worker_mode:'local',remote_gpu_tasks:['total']},null));
  assert(!ui.cropSupported(ct,selected,catalog,null,null));
});

test('point, box and text validation follows each model input contract',()=>{
  const validate=(value,names)=>ui.validatePrompt(value,names.map(name=>task(name)),ct);
  assert(validate({},['pam_prompted']));
  assert.equal(validate({positive_points:[[3,5,2]]},['pam_prompted','sat3d_prompted']),'');
  assert.match(validate({negative_points:[[3,5,2]]},['sat3d_prompted']),/positive/);
  assert.match(validate({positive_points:[[32,5,2]]},['pam_prompted']),/outside/);
  assert.match(validate({positive_points:[[3,5,Infinity]]},['pam_prompted']),/finite/);
  assert.match(validate({positive_points:[['3',5,2]]},['pam_prompted']),/finite/);
  assert.match(validate({box:[[1,2,3],[5,6,4]]},['medsam_vit_b_box_2d']),/same slice/);
  assert.equal(validate({box:[[1,2,3],[5,6,4]]},['segvol_prompted_ct']),'');
  assert.match(validate({box:[[1,2,3],[5,6,3]]},['medical_sam3_3d_prompted']),/medical concept/);
  assert.equal(validate({box:[[1,2,3],[5,6,3]],text:'spleen'},['medical_sam3_3d_prompted']),'');
  assert.equal(validate({text:'spleen'},['voxtell_v1_1_text_prompted']),'');
  assert.match(validate({text:'x'.repeat(513)},['voxtell_v1_1_text_prompted']),/512/);
  assert.match(validate({text:'spleen\n'},['voxtell_v1_1_text_prompted']),/printable/);
  assert.match(validate({text:'spleen'},['voxtell_v1_1_text_prompted','pam_prompted']),/positive/);
  assert.match(validate([],['pam_prompted']),/object/);
  assert.match(validate({positive_points:null,text:'spleen'},['segvol_prompted_ct']),/lists/);
});

test('counts use label maps and runtime lists without counting commas in prose',()=>{
  const nv=task('nv_segment_ct_auto',{structures:'117 foreground labels; organs and anatomical structures'});
  assert.match(ui.structureInfo(nv,ui.presentation(nv,metadata)).label,/^117 structures/);
  const total=task('total',{description:'TotalSegmentator CT segmentation',structures:'liver, spleen, kidney_left'});
  assert.equal(ui.structureInfo(total,{}).label,'3 structures · details');
  assert.match(ui.structureInfo(task('unknown',{structures:'Paired organs, vessels and tissue groups'}),{}).label,/count not supplied/);
  assert.match(ui.structureInfo(task('report',{output:'report'}),{}).label,/Report formats/);
});

test('references permit only explicit web links, never executable URLs',()=>{
  assert.equal(ui.safeReference('javascript:alert(1)'),'');
  assert.equal(ui.safeReference('data:text/html,example'),'');
  assert.equal(ui.safeReference('/api/v1/jobs'),'');
});

test('remote execution badges cover both routed tasks and whole SSH workers',()=>{
  assert(ui.executesRemotely(task('total'),{worker_mode:'ssh'},null));
  assert(ui.executesRemotely(task('total'),null,{worker_mode:'ssh'}));
  assert(ui.executesRemotely(task('total'),{worker_mode:'local',remote_gpu_tasks:['total']},null));
  assert(ui.executesRemotely(task('total',{execution_backend:'ssh'}),null,null));
  assert(!ui.executesRemotely(task('total'),{worker_mode:'local'},null));
});

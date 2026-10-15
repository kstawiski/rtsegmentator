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

const viewer = require('../static/viewer.js');
const near = (a,b) => assert.ok(Math.abs(a-b) < 1e-9, `${a} != ${b}`);

test('viewer fits the stored pixel grid, with contours at pixel centres and no implicit flips',()=>{
  const t = viewer.imageTransform(100, 50, 400, 300);
  assert.deepEqual(t,{scale:4,x:2,y:52});
  assert.deepEqual(viewer.pixelToScreen([0,0],t),[2,52]);
  assert.deepEqual(viewer.pixelToScreen([99,49],t),[398,248]);
  // Image edges are half a pixel from the centres and share the contour transform.
  assert.deepEqual(viewer.pixelToScreen([-.5,-.5],t),[0,50]);
  assert.deepEqual(viewer.pixelToScreen([99.5,49.5],t),[400,250]);
  const panned = viewer.imageTransform(100,50,400,300,{zoom:2,x:35,y:-20});
  assert.deepEqual(viewer.pixelToScreen([0,0],panned),[-161,-66]);
  assert.deepEqual(viewer.pixelToScreen([10,12],panned),[-81,30]);
});

test('viewer inverse transform and zoom preserve the image point under the chosen anchor',()=>{
  for(const [columns,rows,width,height] of [[512,512,600,400],[48,32,537,333],[32,48,1024,300]]){
    const g = {columns,rows,width,height}, view={zoom:1.7,x:-21,y:38},anchor=[123.25,222.5];
    const before = viewer.imageTransform(columns,rows,width,height,view);
    const pixel = viewer.screenToPixel(anchor,before);
    const afterView = viewer.zoomAt(view,1.25,anchor,g);
    const after = viewer.imageTransform(columns,rows,width,height,afterView);
    const projected = viewer.pixelToScreen(pixel,after);
    near(projected[0],anchor[0]);near(projected[1],anchor[1]);
    const roundTrip = viewer.screenToPixel(viewer.pixelToScreen([27.2,13.6],after),after);
    near(roundTrip[0],27.2);near(roundTrip[1],13.6);
    const restored = viewer.zoomAt(afterView,.8,anchor,g);
    near(restored.zoom,view.zoom);near(restored.x,view.x);near(restored.y,view.y);
    assert.deepEqual(view,{zoom:1.7,x:-21,y:38});
    assert.equal(viewer.zoomAt(view,100,anchor,g).zoom,8);
    assert.equal(viewer.zoomAt(view,.001,anchor,g).zoom,.5);
  }
});

test('ROI visibility uses structure-set keys, survives filtering, and jumps to an existing plane',()=>{
  const keys=['modelA:1','modelB:1','modelA:2'];
  const all=viewer.visibility(keys);
  const hidden=viewer.setVisible(all,['modelA:1'],false);
  assert.deepEqual([...hidden],['modelB:1','modelA:2']);
  assert.equal(all.size,3);
  assert.deepEqual([...viewer.visibility(keys,hidden)],['modelB:1','modelA:2']);
  assert.equal(viewer.setVisible(hidden,keys,false).size,0);
  assert.equal(viewer.setVisible(hidden,keys,true).size,3);
  assert.equal(viewer.middleSlice({slices:[1,2,9]}),2);
  assert.equal(viewer.middleSlice({slices:[]}),null);
});

test('primary actions require explicit choice, support either or both, and use modality defaults',()=>{
  const catalog=[task('total'),task('total_mr',{modality:'MR'}),task('other'),task('nv_reason_ct_report',{output:'report'})];
  const selected=new Set();
  let next=ui.primarySelection(selected,'segment',true,catalog,ct);
  assert.deepEqual([...next],['total']);assert.equal(selected.size,0);
  next=ui.primarySelection(next,'report',true,catalog,ct);
  assert.deepEqual([...next],['total','nv_reason_ct_report']);
  next=ui.primarySelection(next,'segment',false,catalog,ct);
  assert.deepEqual([...next],['nv_reason_ct_report']);
  next=ui.primarySelection(next,'report',false,catalog,ct);assert.equal(next.size,0);
  assert.deepEqual([...ui.primarySelection(new Set(),'segment',true,catalog,mr)],['total_mr']);
  assert.deepEqual([...ui.primarySelection(new Set(['other']),'segment',true,catalog,ct)],['other']);
  for(const series of [mr,pet,{...ct,cbct:true,cbct_reason:'Detected cone-beam CT'}]){
    assert.equal(ui.primarySelection(new Set(),'report',true,catalog,series).size,0);
  }
  const blocked=[task('total',{enabled:false,status:'unavailable'}),task('nv_reason_ct_report',{output:'report',enabled:false,status:'acceptance_missing'})];
  assert.equal(ui.primarySelection(new Set(),'segment',true,blocked,ct).size,0);
  assert.equal(ui.primarySelection(new Set(),'report',true,blocked,ct).size,0);
});

test('job deletion remains disabled for every nonterminal and unknown state',()=>{
  for(const status of ['queued','processing','transferring','retrieving','unknown',undefined])assert.equal(ui.canDeleteJob({status}),false);
  for(const status of ['complete','failed','cancelled'])assert.equal(ui.canDeleteJob({status}),true);
  assert.equal(ui.canDeleteJob(null),false);
});

test('report rendering retains refusal, partial and withheld markers, flags and numeric measurements',()=>{
  const report={status:'generated',regions:[{region:'chest',sections:{findings:[
    {heading:'Lungs',text:'Not in scanned volume — statement withheld',organ_status:'absent'},
    {heading:'Heart',text:'Synthetic 12.5 mm finding [partly in the scanned volume]',organ_status:'partial'},
    {heading:'<script>',text:'<img src=x onerror=alert(1)>',organ_status:'unmapped'}],technique:['Synthetic'],impression:['Research draft'],labels:['Test']},
    flags:[{status:'generation_truncated',reason:'Reached token limit 1024'}],unchecked_headings:['Unknown']}],coverage:{heart:{volume_ml:12.345,slice_range:[1,4],touches_first_slice:true}},segmentation_summary:[]};
  const html=ui.reportMarkup(report),md=ui.reportMarkdown(report);
  for(const text of ['12.5','12.345','1024','withheld','Partial coverage','requires radiologist review'])assert.ok(html.includes(text),text);
  assert.ok(!html.includes('<script>'));assert.ok(!html.includes('<img src=x'));assert.ok(html.includes('&lt;script&gt;'));
  for(const text of ['absent','partial','generation_truncated','12.345','1024'])assert.ok(md.includes(text),text);
  const refused=ui.reportMarkup({status:'not_generated',reason:'HU check failed',regions:[]});
  assert.ok(refused.includes('Not generated'));assert.ok(refused.includes('HU check failed'));assert.ok(refused.includes('No region report generated'));
});

test('report coverage remains distinct from generated regions and the viewer slice grid',()=>{
  const report={status:'not_generated',reason:'Insufficient resources',covered_regions:['chest'],regions:[],
    coverage_slice_convention:'Zero based, inferior to superior in closest-canonical mask grid',
    flags:[{section:'impression',index:0,absent_organs:['liver']}],model:'example-model',revision:'123abc'};
  const html=ui.reportMarkup(report),md=ui.reportMarkdown(report);
  for(const text of ['Chest','closest-canonical','Absent anatomy flag','liver','index 0','123abc'])assert.ok(html.includes(text),text);
  assert.ok(md.includes('Regions covered: chest'));
  assert.ok(md.includes('closest-canonical'));
});

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
  assert.deepEqual(t,{scale:4,a:4,b:0,c:0,d:4,x:2,y:52});
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

test('counts distinguish zero, one and multiple models and structures',()=>{
  for(const noun of ['model','structure','active job','file','image','slice']){
    assert.equal(ui.counted(0,noun),`0 ${noun}s`);
    assert.equal(ui.counted(1,noun),`1 ${noun}`);
    assert.equal(ui.counted(2,noun),`2 ${noun}s`);
  }
  assert.equal(ui.structureInfo(task('one',{structures:['Heart']}),{}).label,'1 structure · details');
  assert.equal(ui.structureInfo(task('one',{description:'TotalSegmentator task',structures:'Heart'}),{}).label,'1 structure · details');
});

test('licence badges use explicit catalog flags independently of descriptions',()=>{
  assert.deepEqual(ui.licenceBadges({licensed:true}),['Licence key required']);
  assert.deepEqual(ui.licenceBadges({licensed:false,noncommercial:true}),['Non-commercial weights']);
  assert.deepEqual(ui.licenceBadges({licensed:true,noncommercial:true}),['Licence key required','Non-commercial weights']);
  assert.deepEqual(ui.licenceBadges({licensed:false,license_note:'NVIDIA Open Model License.'}),[]);
  assert.deepEqual(ui.licenceBadges({description:'Research-only UI; requires review.',license:'OpenMDW-1.1'}),[]);
});

test('empty anatomical report labels are headings and real unmapped statements remain findings',()=>{
  const findings=[
    {heading:'CHEST',text:'',organ_status:'unmapped'},
    {heading:'ABDOMEN',text:'  ',organ_status:'unmapped'},
    {heading:'Medical devices',text:'Synthetic 12.5 mm device [1].',organ_status:'unmapped'},
    {heading:'CHEST',text:'Synthetic observation [2].',organ_status:'unmapped'},
    {heading:'UNKNOWN',text:'',organ_status:'unmapped'},
    {heading:'Liver',text:'Statement withheld',organ_status:'absent'}
  ];
  const report={status:'generated',regions:[{region:'chest',sections:{findings},unchecked_headings:['CHEST','ABDOMEN','Medical devices','UNKNOWN']}]};
  const html=ui.reportMarkup(report),md=ui.reportMarkdown(report);
  assert.equal((html.match(/class="report-section-heading"/g)||[]).length,2);
  assert.match(html,/<h5[^>]*>CHEST<\/h5>/);
  assert.match(html,/<h5[^>]*>ABDOMEN<\/h5>/);
  assert.equal((html.match(/class="finding"/g)||[]).length,4);
  assert.ok(html.includes('not checked against segmentation'));
  assert.ok(!html.includes('>unmapped<'));
  assert.ok(html.includes('12.5 mm device [1]'));
  assert.ok(html.includes('Synthetic observation [2]'));
  assert.ok(html.includes('Withheld · absent anatomy'));
  assert.ok(html.includes('Not checked against segmentation: Medical devices; UNKNOWN.'));
  assert.match(md,/#### CHEST\n#### ABDOMEN/);
  assert.ok(md.includes('Medical devices [not checked against segmentation]: Synthetic 12.5 mm device [1].'));
  assert.equal(report.regions[0].sections.findings[0].organ_status,'unmapped');
});

test('report status distinguishes earlier failures, report failures and uncertain legacy jobs',()=>{
  const report='nv_reason_ct_report';
  const base={status:'failed',models:['total','lung',report],report_status:'failed',report_reason:'Task failed; see job error'};
  const notRun='AI report: not run (an earlier step failed)';
  const failed='AI report: Failed · Task failed; see job error';
  assert.equal(ui.reportStatusText({...base,model_index:1,completed_models:[]}),notRun);
  assert.equal(ui.reportStatusText({...base,model_index:2,completed_models:['total']}),notRun);
  assert.equal(ui.reportStatusText({...base,completed_models:[]}),notRun);
  assert.equal(ui.reportStatusText({...base,model_index:0}),notRun);
  assert.equal(ui.reportStatusText({...base,model_index:2}),notRun);
  assert.equal(ui.reportStatusText({...base,model_index:3}),failed);
  assert.equal(ui.reportStatusText({...base,model_index:2,remote_task_active:report}),failed);
  // The report can fail before its first heartbeat, leaving the segmentation index.
  assert.equal(ui.reportStatusText({...base,model_index:2,completed_models:['total','lung']}),failed);
  assert.equal(ui.reportStatusText({...base,model_index:2,completed_models:['total','lung',report]}),failed);
  assert.equal(ui.reportStatusText(base),failed);
  assert.equal(ui.reportStatusText({...base,models:[report],model_index:0}),failed);
  assert.equal(ui.reportStatusText({...base,status:'processing',report_status:'running',report_reason:'',model_index:2}),'AI report: Running');
  assert.equal(ui.reportStatusText({...base,status:'queued',report_status:'pending',report_reason:''}),'AI report: Pending');
  assert.equal(ui.reportStatusText({...base,status:'complete',report_status:'not_generated',report_reason:'Coverage insufficient'}),'AI report: Not generated · Coverage insufficient');
  assert.equal(ui.reportStatusText({...base,status:'complete',report_status:'generated',report_reason:''}),'AI report: Generated');
  assert.equal(ui.reportStatusText({status:'failed',models:['total']}),'');
  assert.equal(ui.reportStatusText({...base,models:['total','custom_report'],model_index:1},'custom_report'),notRun);
});


test('section labels do not appear as failed coverage checks in report flags',()=>{
  const sectionFlag={region:'chest',section:'findings',heading:'CHEST',status:'unmapped'};
  const actualFlag={region:'chest',section:'findings',heading:'Devices',status:'unmapped'};
  const report={status:'generated',regions:[{region:'chest',sections:{findings:[
    {heading:'CHEST',text:'',organ_status:'unmapped'},{heading:'Devices',text:'12 mm [1].',organ_status:'unmapped'}]},
    flags:[sectionFlag,actualFlag]}],flags:[sectionFlag,actualFlag]};
  const html=ui.reportMarkup(report);
  assert.ok(html.includes('<h5 class="report-section-heading">CHEST</h5>'));
  assert.ok(!html.includes(': CHEST'));
  assert.ok(html.includes(': Devices'));
  assert.equal(report.flags.length,2,'Original JSON evidence remains unchanged');
});

// Independently enumerate the eight possible stored in-plane axis orders/signs
// for each patient plane, including the reported MSD vertical-flip case.
for (const [plane,right,down,expected] of [
  ['axial',[1,0,0],[0,1,0],{left:'R',right:'L',top:'A',bottom:'P'}],
  ['coronal',[1,0,0],[0,0,-1],{left:'R',right:'L',top:'H',bottom:'F'}],
  ['sagittal',[0,1,0],[0,0,-1],{left:'A',right:'P',top:'H',bottom:'F'}],
]) {
  for (const swap of [false,true]) for (const sx of [-1,1]) for (const sy of [-1,1]) {
    test(`${plane} standard display: swap=${swap}, column sign=${sx}, row sign=${sy}`,()=>{
      const series={column_direction:(swap?down:right).map(v=>v*sx),row_direction:(swap?right:down).map(v=>v*sy)};
      const display=viewer.displayOrientation(series);
      assert.deepEqual(viewer.orientationMarkers(series,display),expected);
      assert.deepEqual(viewer.displayOrientation(series,false),{a:1,b:0,c:0,d:1});
      const columns=7,rows=5,width=420,height=300,view={zoom:1.4,x:21,y:-17};
      const t=viewer.imageTransform(columns,rows,width,height,view,display);
      const displayColumns=swap?rows:columns,displayRows=swap?columns:rows;
      const scale=Math.min(width/displayColumns,height/displayRows)*view.zoom;
      const origin=[(width-displayColumns*scale)/2+view.x,(height-displayRows*scale)/2+view.y];
      // Independent expected raster location for every asymmetric source pixel.
      for(let x=0;x<columns;x++) for(let y=0;y<rows;y++) {
        const u=swap?(sy>0?y:rows-1-y):(sx>0?x:columns-1-x);
        const v=swap?(sx>0?x:columns-1-x):(sy>0?y:rows-1-y);
        const screen=viewer.pixelToScreen([x,y],t);
        near(screen[0],origin[0]+(u+.5)*scale);near(screen[1],origin[1]+(v+.5)*scale);
        const roundTrip=viewer.screenToPixel(screen,t);near(roundTrip[0],x);near(roundTrip[1],y);
        // Canvas drawImage starts at (-.5,-.5). Its source pixel centre (x+.5,y+.5)
        // and the contour vertex (x,y) must coincide under the SAME canvas matrix.
        const imagePoint=[t.a*((x+.5)-.5)+t.c*((y+.5)-.5)+t.x,t.b*((x+.5)-.5)+t.d*((y+.5)-.5)+t.y];
        near(imagePoint[0],screen[0]);near(imagePoint[1],screen[1]);
      }
      const anchor=[151.2,92.8],g={columns,rows,width,height,display};
      const pixel=viewer.screenToPixel(anchor,t),next=viewer.zoomAt(view,1.25,anchor,g);
      const projected=viewer.pixelToScreen(pixel,viewer.imageTransform(columns,rows,width,height,next,display));
      near(projected[0],anchor[0]);near(projected[1],anchor[1]);
    });
  }
}

test('edge letters follow stored orientation when standard display is disabled',()=>{
  const series={column_direction:[1,0,0],row_direction:[0,-1,0]};
  assert.deepEqual(viewer.orientationMarkers(series,viewer.displayOrientation(series,false)),{left:'R',right:'L',top:'P',bottom:'A'});
  assert.deepEqual(viewer.orientationMarkers(series,viewer.displayOrientation(series,true)),{left:'R',right:'L',top:'A',bottom:'P'});
});

test('oblique edges stay as stored and use at most two letters, strongest first',()=>{
  for(const series of [
    {column_direction:[.8,0,.6],row_direction:[0,1,0]},
    {column_direction:[Math.SQRT1_2,Math.SQRT1_2,0],row_direction:[-Math.SQRT1_2,Math.SQRT1_2,0]},
  ]) assert.deepEqual(viewer.displayOrientation(series),{a:1,b:0,c:0,d:1});
  assert.deepEqual(viewer.orientationMarkers({column_direction:[.8,0,.6],row_direction:[0,1,0]}),{left:'RF',right:'LH',top:'A',bottom:'P'});
  assert.equal(viewer.directionLabel([Math.sqrt(1-.3**2),-.3,0]),'LA');
  assert.equal(viewer.directionLabel([Math.sqrt(1-.299**2),-.299,0]),'L');
  assert.equal(viewer.directionLabel([-.4,.5,.7681145748]),'HP');
  assert.equal(viewer.directionLabel([0,0,-1]),'F');
  assert.equal(viewer.directionLabel([NaN,0,0]),'?');
});

test('canvas image and contours use one affine for every signed permutation at device pixel ratio 2',()=>{
  const priorWindow=global.window;global.window={devicePixelRatio:2};
  try {
    for(const swap of [false,true]) for(const sx of [-1,1]) for(const sy of [-1,1]) {
      const series={columns:7,rows:5,slices:3,modality:'CT',orientation:'axial',
        column_direction:swap?[0,sx,0]:[sx,0,0],row_direction:swap?[sy,0,0]:[0,sy,0],
        window_presets:[{id:'test',label:'Test',center:40,width:400}]};
      const display=viewer.displayOrientation(series),g={...series,width:420,height:300,display};
      const elements=Object.fromEntries(['canvas','top','bottom','left','right','readout','opacity','mode'].map(k=>[k,{style:{},setAttribute(){}}]));
      elements.opacity.value='80';elements.mode.value='outline';
      let matrix,imagePixel,contourPixel,drawCount=0;
      const map=([x,y])=>[matrix[0]*x+matrix[2]*y+matrix[4],matrix[1]*x+matrix[3]*y+matrix[5]];
      const ctx={setTransform(...m){matrix=m;drawCount++;},drawImage(img,x,y,w,h){
        assert.equal(w,7);assert.equal(h,5);imagePixel=map([x+2.5,y+1.5]);
      },beginPath(){},rect(){},clip(){},moveTo(x,y){contourPixel=map([x,y]);},lineTo(){},closePath(){},stroke(){}};
      viewer.Viewer.prototype.draw.call({el:k=>elements[k],geometry:()=>g,info:{series},window:'test',index:1,
        view:{zoom:1.7,x:19,y:-31},ctx,image:{},paths:[{key:'roi',paths:[[[2,1],[4,1],[4,3]]]}],
        visible:new Set(['roi']),rois:[{key:'roi',color:'#123456'}]});
      assert.equal(drawCount,1);assert.deepEqual(contourPixel,imagePixel);
      assert.deepEqual(['top','bottom','left','right'].map(k=>elements[k].textContent),['A','P','R','L']);
    }
  } finally {global.window=priorWindow;}
});

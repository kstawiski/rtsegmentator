'use strict';
// Synthetic browser fixture. Loaded only by the standalone UI test harness.
window.uiFixtureErrors=[];
window.addEventListener('error',e=>window.uiFixtureErrors.push(e.message));
window.addEventListener('unhandledrejection',e=>window.uiFixtureErrors.push(String(e.reason)));
const reportName='nv_reason_ct_report';
const model=(name,extra={})=>({name,modality:'CT',enabled:true,status:'available',structures:['Synthetic structure'],...extra});
const fixtureModels=[model('total'),model('nv_segment_ct_auto',{licensed:false,license_note:'NVIDIA Open Model License.'}),
  model('nv_segment_ctmr_ct_body',{licensed:false,noncommercial:true,license_note:'NVIDIA OneWay Noncommercial weight licence.'}),
  model('total_mr',{modality:'MR'}),model('nv_segment_ctmr_mri_body',{modality:'MR',noncommercial:true}),
  model('synthetic_prompt',{prompt:'points'}),model('synthetic_unavailable',{enabled:false,status:'checkpoint_unavailable'}),model('heart_key_task',{licensed:true}),model(reportName,{output:'report',license:'OpenMDW-1.1'}),
  ...Array.from({length:14},(_,i)=>model('synthetic_anatomy_model_with_long_name_'+i))];
const fixtureReport={status:'generated',regions:[{region:'chest',sections:{technique:['Synthetic UI fixture. No patient images.'],findings:[
  {heading:'CHEST',text:'',organ_status:'unmapped'},
  {heading:'Medical devices',text:'Synthetic measurement 12.5 mm [1].',organ_status:'unmapped'},
  {heading:'Lungs',text:'Synthetic finding retained.',organ_status:'present'},
  {heading:'ABDOMEN',text:'',organ_status:'unmapped'},
  {heading:'Liver',text:'Not in scanned volume — statement withheld',organ_status:'absent'}],impression:['Synthetic text for UI tests.'],labels:[]},unchecked_headings:['CHEST','Medical devices','ABDOMEN']}],coverage:{heart:{volume_ml:12.345,slice_range:[1,4]}}};
const failed={status:'failed',created:1700000000,updated:1700000060,models:['total',reportName],report_status:'failed',report_reason:'Task failed; see job error',error:'Synthetic test failure',detail:'Processing failed',model_total:2};
const fixtureJobs=[{id:'qa-active',status:'processing',created:Date.now()/1000-300,updated:Date.now()/1000,models:['total','nv_segment_ct_auto',reportName],current_task:'nv_segment_ct_auto',stage:'Validating RTSTRUCT geometry',stage_key:'rtstruct_validation',stage_done:43,stage_total:117,stage_started:Date.now()/1000-60,last_activity:Date.now()/1000-5,last_cpu_activity:Date.now()/1000-2,stall_threshold:1200,completed_models:['total'],task_timings:{total:{status:'done',duration:120},nv_segment_ct_auto:{status:'running',started:Date.now()/1000-180}}},{...failed,id:'qa-earlier',model_index:1,completed_models:[]},
  {...failed,id:'qa-report',model_index:2,completed_models:['total']},
  {...failed,id:'qa-complete',status:'complete',report_status:'generated',report_reason:'',error:null,model_index:2,completed_models:['total',reportName]}];
const fixtureSeries=[
  {key:'synthetic-ct',series_uid:'synthetic-ct',modality:'CT',description:'Synthetic CT',slices:24,rows:32,columns:32,study_date:'20000101'},
  {key:'synthetic-mr',series_uid:'synthetic-mr',modality:'MR',description:'Synthetic MR',slices:30,rows:32,columns:32,study_date:'20000101'},
  {key:'synthetic-localiser',series_uid:'synthetic-localiser',modality:'CT',description:'Synthetic localiser',slices:3,rows:32,columns:32,study_date:'20000101'}];
const archivedBatchJobs=fixtureSeries.slice(0,2).map((series,i)=>({id:'qa-archived-'+i,status:'queued',created:1+i,
  models:[i?'total_mr':'total'],batch_id:'qa-archived-batch',batch_index:i+1,batch_size:2,
  series:{...series,description:'Archived '+series.description}}));
window.uiBatchRequests=[];
const originalFetch=window.fetch.bind(window);
window.fetch=async (input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href),path=url.pathname;
  if(!path.startsWith('/api/'))return originalFetch(input,options);
  let body={},status=200;
  if(path==='/api/v1/models')body={tasks:fixtureModels,clinical_tags:[]};
  else if(path==='/api/v1/config')body={worker_mode:'local',report_task:reportName};
  else if(path.includes('health'))body={worker_mode:'local',worker_reachable:false};
  else if(path==='/api/v1/jobs')body={jobs:url.searchParams.get('batch_id')==='qa-archived-batch'?archivedBatchJobs:fixtureJobs};
  else if(/^\/api\/v1\/jobs\/[^/]+$/.test(path))body=[...fixtureJobs,...archivedBatchJobs].find(j=>j.id===path.split('/').at(-1))||{};
  else if(path.endsWith('/log'))body={task:url.searchParams.get('task'),lines:['RTSEG_PROGRESS {\"stage\":\"rtstruct_validation\",\"done\":43,\"total\":117}','Synthetic task log: <no markup>']};
  else if(path.endsWith('/report'))body=fixtureReport;
  else if(path.endsWith('/viewer')){body={detail:'Synthetic UI fixture: no image pixels supplied.'};status=404;}
  else if(path==='/api/v1/upload-sessions')body={session_id:'ui-fixture',max_request_bytes:1048576};
  else if(path.includes('/files/'))body={received:options.body.size};
  else if(path.endsWith('/assess'))body={};
  else if(path==='/api/v1/upload-sessions/ui-fixture')body={status:'ready',result:{upload_id:'ui-fixture',series:fixtureSeries}};
  else if(path==='/api/v1/uploads/ui-fixture/jobs'){
    const keys=options.body.getAll('series_keys'),names=options.body.getAll('models');
    window.uiBatchRequests.push({keys,models:names});
    const now=Date.now()/1000,batch_id='synthetic-batch';
    const jobs=keys.map((key,i)=>({id:'qa-batch-'+i,series_key:key,models:names.filter(name=>fixtureModels.find(t=>t.name===name).modality===fixtureSeries.find(s=>s.key===key).modality)}));
    fixtureJobs.push(...jobs.map((job,i)=>({...job,status:i?'queued':'processing',created:now+i*.2,
      detail:i?'Queued behind 1 earlier job':'total (1/2): validating RTSTRUCT geometry · 43/117 structures',
      ...(i?{}:{current_task:'total',model_index:1,stage:'Validating RTSTRUCT geometry',stage_key:'rtstruct_validation',
        stage_done:43,stage_total:117,stage_started:now-60,last_activity:now-5,last_cpu_activity:now-2,
        stall_threshold:1200,task_timings:{total:{status:'running',started:now-180}}}),
      batch_id,batch_index:i+1,batch_size:jobs.length,series:fixtureSeries.find(s=>s.key===job.series_key)})));
    fixtureJobs.push({id:'qa-unrelated',status:'queued',created:now+.1,models:['total']});
    body={batch_id,ids:jobs.map(j=>j.id),jobs};status=202;
  }
  else {body={detail:'Unexpected fixture request: '+path};status=404;window.uiFixtureErrors.push(body.detail);}
  return new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
};

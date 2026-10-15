'use strict';
// Synthetic browser fixture. Loaded only by the standalone UI test harness.
window.uiFixtureErrors=[];
window.addEventListener('error',e=>window.uiFixtureErrors.push(e.message));
window.addEventListener('unhandledrejection',e=>window.uiFixtureErrors.push(String(e.reason)));
const reportName='nv_reason_ct_report';
const model=(name,extra={})=>({name,modality:'CT',enabled:true,status:'available',structures:['Synthetic structure'],...extra});
const fixtureModels=[model('total'),model('nv_segment_ct_auto',{licensed:false,license_note:'NVIDIA Open Model License.'}),
  model('nv_segment_ctmr_ct_body',{licensed:false,noncommercial:true,license_note:'NVIDIA OneWay Noncommercial weight licence.'}),
  model('heart_key_task',{licensed:true}),model(reportName,{output:'report',license:'OpenMDW-1.1'}),
  ...Array.from({length:14},(_,i)=>model('synthetic_anatomy_model_with_long_name_'+i))];
const fixtureReport={status:'generated',regions:[{region:'chest',sections:{technique:['Synthetic UI fixture. No patient images.'],findings:[
  {heading:'CHEST',text:'',organ_status:'unmapped'},
  {heading:'Medical devices',text:'Synthetic measurement 12.5 mm [1].',organ_status:'unmapped'},
  {heading:'Lungs',text:'Synthetic finding retained.',organ_status:'present'},
  {heading:'ABDOMEN',text:'',organ_status:'unmapped'},
  {heading:'Liver',text:'Not in scanned volume — statement withheld',organ_status:'absent'}],impression:['Synthetic text for UI tests.'],labels:[]},unchecked_headings:['CHEST','Medical devices','ABDOMEN']}],coverage:{heart:{volume_ml:12.345,slice_range:[1,4]}}};
const failed={status:'failed',created:1700000000,updated:1700000060,models:['total',reportName],report_status:'failed',report_reason:'Task failed; see job error',error:'Synthetic test failure',detail:'Processing failed',model_total:2};
const fixtureJobs=[{...failed,id:'qa-earlier',model_index:1,completed_models:[]},
  {...failed,id:'qa-report',model_index:2,completed_models:['total']},
  {...failed,id:'qa-complete',status:'complete',report_status:'generated',report_reason:'',error:null,model_index:2,completed_models:['total',reportName]}];
const originalFetch=window.fetch.bind(window);
window.fetch=async (input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href),path=url.pathname;
  if(!path.startsWith('/api/'))return originalFetch(input,options);
  let body={},status=200;
  if(path==='/api/v1/models')body={tasks:fixtureModels,clinical_tags:[]};
  else if(path==='/api/v1/config')body={worker_mode:'local',report_task:reportName};
  else if(path.includes('health'))body={worker_mode:'local',worker_reachable:false};
  else if(path==='/api/v1/jobs')body={jobs:fixtureJobs};
  else if(/^\/api\/v1\/jobs\/[^/]+$/.test(path))body=fixtureJobs.find(j=>j.id===path.split('/').at(-1))||{};
  else if(path.endsWith('/report'))body=fixtureReport;
  else if(path.endsWith('/viewer')){body={detail:'Synthetic UI fixture: no image pixels supplied.'};status=404;}
  else if(path==='/api/v1/upload-sessions')body={session_id:'ui-fixture',max_request_bytes:1048576};
  else if(path.includes('/files/'))body={received:options.body.size};
  else if(path.endsWith('/assess'))body={};
  else if(path==='/api/v1/upload-sessions/ui-fixture')body={status:'ready',result:{upload_id:'ui-fixture',series:[{key:'synthetic',modality:'CT',description:'Synthetic browser fixture',slices:12,rows:32,columns:32,study_date:'20000101'}]}};
  else {body={detail:'Unexpected fixture request: '+path};status=404;window.uiFixtureErrors.push(body.detail);}
  return new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
};

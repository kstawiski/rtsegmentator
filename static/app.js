'use strict';

// Presentation never grants availability. The live catalog and selected series do.
const GROUPS = [
  ['whole', 'Whole-body organs'], ['oar', 'Organs at risk by site'],
  ['tumour', 'Tumours and lymph nodes'], ['ctv', 'Target volumes (CTV)'],
  ['brain', 'Brain and head'], ['spine', 'Spine'],
  ['interactive', 'Interactive (needs your clicks)'], ['report', 'AI draft reports'],
  ['other', 'Other anatomy and research models']
];
const TERMINAL = new Set(['complete', 'failed', 'cancelled']);
const PROMPT_HELP = {
  pam_prompted: 'PAM: place a positive point or a tight box on one slice. The seed propagates through the volume.',
  sat3d_prompted: 'SAT3D: at least one positive point is required. Negative points exclude adjacent structures within the prompt-centred 128-voxel region.',
  sam_med2d_prompted: 'SAM-Med2D: prompt every intended slice. It does not propagate to unprompted slices.',
  medsam_vit_b_box_2d: 'MedSAM ViT-B: draw a tight box on one slice. Only that slice is segmented.',
  medical_sam3_3d_prompted: 'Medical-SAM3: a box and a medical concept in Target description are required. Output covers the publisher’s eight-slice window.',
  sam_med3d_turbo_prompted: 'SAM-Med3D Turbo: place a positive point inside the 3D object. A box is interpreted as its centre point.',
  medsam2_ct_lesion_prompted: 'MedSAM2: place points or a tight box on one axial lesion slice. The CT-lesion model propagates in both directions.',
  medical_sam2_prompted_ct: 'Medical-SAM2 (ImprintLab): place points or a box on one axial slice. This is a distinct model from MedSAM2.',
  nninteractive_v1_prompted: 'nnInteractive: place a positive point or an axial box. Negative points can exclude adjacent structures.',
  vista3d_prompted_ct: 'VISTA3D: at least one positive point is required. Negative points can exclude adjacent structures.',
  segvol_prompted_ct: 'SegVol: use text, points or a 3D box. Text and spatial prompts can be combined; edit a 3D box in the JSON panel.',
  voxtell_v1_1_text_prompted: 'VoxTell: enter one structure, tumour, lesion or clinical description (up to 512 printable characters). Points and boxes are ignored.'
};
function humanize(name) {
  const words = {ct:'CT',mr:'MR',mri:'MRI',pt:'PET',pet:'PET',petct:'PET/CT',ctv:'CTV',gtv:'GTV',oar:'OAR',oars:'OARs',cbct:'CBCT',fdg:'FDG',psma:'PSMA',t1:'T1',t2:'T2',t1c:'T1 contrast',t1ce:'T1 contrast',dkt:'DKT',sct:'SCT',cads:'CADS',moose:'MOOSE',monai:'MONAI',bamf:'BAMF',pycad:'PYCAD',nv:'NV',ctmr:'CT/MR',nnunetv1:'nnU-Net v1',resenc:'ResEnc',swinunetr:'Swin UNETR',unet:'U-Net',isrt:'ISRT',lnsegfm:'LN-Seg-FM',hn:'head and neck',han:'head and neck',hnlnl:'head and neck nodal levels',lnq:'LNQ',siam:'SIAM',rano2:'RANO2',uwlair:'UWLAIR',hntsmrg:'HNTS-MRG',gbm:'glioblastoma',fgt:'fibroglandular tissue',pdac:'pancreatic cancer',coala:'COALA',crlm:'colorectal liver metastases',2:'2',auto:'automatic'};
  const text = String(name || 'Unnamed model').split('_').map(w => words[w.toLowerCase()] || w).join(' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}
function presentation(task, metadata = {overrides:{}}) {
  const override = metadata.overrides?.[task.name] || {};
  let group = override.group;
  // Prompt and report contracts take precedence over the naming fallback.
  if (task.output === 'report') group = 'report';
  else if (task.prompt) group = 'interactive';
  if (!group) {
    const name = task.name.toLowerCase();
    if (/^isrt_|\bctv\b/.test(name.replaceAll('_', ' ')) || name.startsWith('isrt_')) group = 'ctv';
    else if (/^(sct_|totalspineseg)|spine|spinal|vertebr/.test(name)) group = 'spine';
    else if (/tumou?r|lesion|lymph|gtv|^lnsegfm|^lnq_|^rano|^uwlair|^stunet|^hecktor|^hd_glio|^hd_bm|brats|^pam_|^sat3d|^coala|^compai|^dbdmp|^wcode|hnlnl|^choi_tmli/.test(name)) group = 'tumour';
    else if (/brain|head|hipp|cereb|skull|dental|^siam|^gouhfi|^fastsurfer|^antspynet|^synthseg|^hd_(ct)?bet|face|oculomotor/.test(name)) group = 'brain';
    else if (/^(total(_v\d+|_mr)?|mrsegmentator|mrisegmentator|vibesegmentator)$|whole.?body|^nv_segment.*body|^nv_segment_ct_auto|^universal_model|^multitalent/.test(name)) group = 'whole';
    else if (/organ|vessel|muscle|bone|lung|liver|kidney|prostate|pelvi|pancrea|heart|cardiac|aort|breast|tissue|body|spleen|adrenal|abdomen|rib|bowel|airway|oar|segthor|btcv|gland|cavit|thigh|shoulder|appendicular/.test(name)) group = 'oar';
    else group = 'other'; // Explicit fallback for future worker-discovered tasks.
  }
  const purpose = override.purpose || task.description || 'Task-specific anatomy; inspect the output details.';
  return { ...override, group, title: override.title || humanize(task.name), purpose };
}
function supports(task, series) { return !!series && (task.modalities || [task.modality]).includes(series.modality); }
function pairedCT(series, allSeries) {
  return allSeries.find(s => s.modality === 'CT' && s.study_uid === series?.study_uid && s.frame_uid === series?.frame_uid);
}
function blockedReason(task, series, allSeries, reportTask = 'nv_reason_ct_report') {
  const reasons = [];
  if (task.enabled === false) {
    reasons.push(task.status || 'unavailable');
    if (task.missing?.length) reasons.push(`Missing: ${Array.isArray(task.missing) ? task.missing.join('; ') : task.missing}`);
    if (task.remote_preflight_error) reasons.push(task.remote_preflight_error);
  }
  if (!supports(task, series)) reasons.push(`Requires ${(task.modalities || [task.modality]).map(modalityName).join(' / ')} images`);
  if ((task.name === reportTask || task.output === 'report') && series?.cbct) reasons.push(`CBCT is not supported for AI reports: ${series.cbct_reason || 'CBCT detected'}`);
  if (task.requires_ct && series && !pairedCT(series, allSeries)) reasons.push('Requires a CT series in the same Study and Frame of Reference');
  return reasons.join(' · ');
}
function primarySelection(selected, action, checked, catalog, series, allSeries = [], reportName = 'nv_reason_ct_report') {
  const next = new Set(selected), isReport = t => t.output === 'report' || t.name === reportName;
  if (!checked) {
    catalog.filter(t => action === 'report' ? t.name === reportName : !isReport(t)).forEach(t => next.delete(t.name));
    return next;
  }
  if (action === 'segment' && catalog.some(t => !isReport(t) && next.has(t.name))) return next;
  const name = action === 'report' ? reportName : series?.modality === 'CT' ? 'total' : series?.modality === 'MR' ? 'total_mr' : null;
  const task = catalog.find(t => t.name === name);
  if (task && !blockedReason(task, series, allSeries, reportName)) next.add(task.name);
  return next;
}
function canDeleteJob(job) { return !!job && TERMINAL.has(job.status); }
function reportMarkdown(report) {
  const lines = ['# NV-Reason-CT draft report', '', '**Research draft. Requires radiologist review.**', '',
    `Status: ${report.status || 'Not supplied'}`, report.reason || '', '', 'Segmentation can be wrong. Coverage checks do not verify diagnostic findings or image quality.'];
  lines.push('', 'Regions covered: ' + (report.covered_regions || (report.regions || []).map(r => r.region)).join(', '));
  for (const region of report.regions || []) {
    lines.push('', `## ${region.region}`, '', '### Technique', ...(region.sections?.technique || []), '', '### Findings');
    for (const f of region.sections?.findings || []) lines.push(`${f.heading} [${f.organ_status || 'not checked'}]: ${f.text}`);
    lines.push('', '### Impression', ...(region.sections?.impression || []), '', '### Abnormality labels', ...(region.sections?.labels || []));
    if (region.flags?.length) lines.push('', '### Flags', ...region.flags.map(f => JSON.stringify(f)));
    if (region.unchecked_headings?.length) lines.push('', 'Not checked against segmentation: ' + region.unchecked_headings.join('; '));
  }
  for (const [title, key] of [['Anatomical coverage','coverage'], ['Segmentation summary','segmentation_summary']]) {
    lines.push('', '## ' + title, '', '```json', JSON.stringify(report[key] || {}, null, 2), '```');
  }
  if (report.coverage_slice_convention) lines.push('', 'Coverage slice convention: ' + report.coverage_slice_convention);
  if (report.flags?.length) lines.push('', '## All report flags', ...report.flags.map(f => JSON.stringify(f)));
  if (report.model || report.revision) lines.push('', 'Model: ' + (report.model || 'Not supplied'), 'Revision: ' + (report.revision || 'Not supplied'));
  // Preserve every additional field, numeric value and citation in the JSON download.
  lines.push('', 'Generated from the report JSON by the portal. The JSON and original files in the result ZIP retain all provenance and raw output.');
  return lines.join('\n') + '\n';
}
function reportMarkup(report) {
  const paragraphs = values => (values || []).map(v => `<p>${esc(v)}</p>`).join('');
  const flags = values => `<ul class="report-flags">${values.map(f => `<li><strong>${esc(humanize(f.status || (f.absent_organs ? 'Absent anatomy flag' : 'Report flag')))}</strong>${f.region ? ' · '+esc(f.region) : ''}${f.heading ? ': '+esc(f.heading) : ''}${f.section ? ' · '+esc(f.section) : ''}${Number.isInteger(f.index) ? ' (index '+esc(f.index)+')' : ''}${f.absent_organs?.length ? ': '+esc(f.absent_organs.join(', ')) : ''}${f.reason ? ' · '+esc(f.reason) : ''}</li>`).join('')}</ul>`;
  return `<div class="notice report-banner"><strong>Research AI draft · requires radiologist review</strong><p>Coverage checks do not verify diagnostic findings or image quality.</p></div>
    <p><strong>Status: ${esc(humanize(report.status || 'Not supplied'))}</strong>${report.reason ? `<br>${esc(report.reason)}` : ''}</p>
    <p><strong>Regions covered:</strong> ${(report.covered_regions || (report.regions || []).map(r => r.region)).map(r => esc(humanize(r))).join(', ') || 'No region report generated'}</p>
    ${(report.regions || []).map(r => `<section class="region-report"><h3>${esc(humanize(r.region))}</h3><h4>Technique</h4>${paragraphs(r.sections?.technique)}<h4>Findings</h4>
      ${(r.sections?.findings || []).map(f => `<div class="finding"><strong>${esc(f.heading)}</strong> <span class="badge ${f.organ_status === 'present' ? '' : 'license'}">${esc(f.organ_status === 'absent' ? 'Withheld · absent anatomy' : f.organ_status === 'partial' ? 'Partial coverage' : f.organ_status || 'Not checked')}</span><p>${esc(f.text)}</p></div>`).join('')}
      <h4>Impression</h4>${paragraphs(r.sections?.impression)}<h4>Abnormality labels</h4>${paragraphs(r.sections?.labels)}
      ${r.flags?.length ? `<h4>Flags</h4>${flags(r.flags)}` : ''}
      ${r.unchecked_headings?.length ? `<p class="notice">Not checked against segmentation: ${esc(r.unchecked_headings.join('; '))}.</p>` : ''}</section>`).join('')}
    ${report.flags?.length ? `<details><summary>All report flags</summary>${flags(report.flags)}</details>` : ''}
    <details class="coverage-details"><summary>Anatomical coverage measurements</summary><p class="small muted">${esc(report.coverage_slice_convention || 'Zero-based indices in the report coverage grid; these can differ from the viewer series.')}</p><div class="table-scroll"><table><thead><tr><th>Organ</th><th>mL</th><th>Slices (index)</th><th>Truncated</th></tr></thead><tbody>${Object.entries(report.coverage || {}).map(([name,r]) => `<tr><td>${esc(name)}</td><td>${esc(r.volume_ml)}</td><td>${esc(r.slice_range?.join('–') || '—')}</td><td>${r.touches_first_slice || r.touches_last_slice ? 'Yes' : 'No'}</td></tr>`).join('')}</tbody></table></div></details>
    ${report.model || report.revision ? `<p class="small muted">Model: ${esc(report.model || 'Not supplied')} · Revision: ${esc(report.revision || 'Not supplied')}</p>` : ''}`;
}
function executesRemotely(task, config, health) {
  return task.execution_backend === 'ssh' || (config?.worker_mode || health?.worker_mode) === 'ssh' ||
    [...(config?.remote_gpu_tasks || []), ...(health?.remote_gpu_tasks || [])].includes(task.name);
}
function cropSupported(series, selected, catalog, config, health) {
  const mode = config?.worker_mode || health?.worker_mode;
  const remote = [...(config?.remote_gpu_tasks || []), ...(health?.remote_gpu_tasks || [])];
  return series?.modality === 'CT' && selected.has('total') && mode === 'local' &&
    catalog.find(t => t.name === 'total')?.execution_backend !== 'ssh' && !remote.includes('total');
}
function validatePrompt(value, tasks, series) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return 'Prompt JSON must be an object.';
  const positives = value.positive_points === undefined ? [] : value.positive_points;
  const negatives = value.negative_points === undefined ? [] : value.negative_points, box = value.box;
  if (!Array.isArray(positives) || !Array.isArray(negatives)) return 'Positive and negative points must be lists.';
  if (box != null && (!Array.isArray(box) || box.length !== 2)) return 'A box needs exactly two [column, row, slice] corners.';
  const coords = [...positives, ...negatives, ...(box || [])], names = tasks.map(t => t.name);
  for (const p of coords) {
    if (!Array.isArray(p) || p.length !== 3 || p.some(v => typeof v !== 'number' || !Number.isFinite(v))) return 'Every coordinate must contain three finite numbers: [column, row, slice].';
    if (p.some((v, i) => v < 0 || v >= [series.columns, series.rows, series.slices][i])) return `Coordinate [${p.join(', ')}] is outside this series.`;
  }
  const has = (...n) => n.some(x => names.includes(x)), text = typeof value.text === 'string' ? value.text.trim() : '';
  if (!coords.length && !text) return 'Add a point, box or target description for the selected model.';
  if (has('pam_prompted', 'nninteractive_v1_prompted', 'sam_med3d_turbo_prompted') && !positives.length && !box) return 'This model requires a positive point or a box.';
  if (has('sat3d_prompted', 'vista3d_prompted_ct') && !positives.length) return 'SAT3D and VISTA3D require at least one positive point.';
  if (has('sam_med2d_prompted', 'medsam2_ct_lesion_prompted', 'medical_sam2_prompted_ct') && !coords.length) return 'The selected model requires points or a box.';
  if (has('medsam_vit_b_box_2d', 'medical_sam3_3d_prompted') && !box) return 'MedSAM ViT-B and Medical-SAM3 require a two-corner box.';
  if (has('medical_sam3_3d_prompted', 'voxtell_v1_1_text_prompted') && !text) return 'Enter the required medical concept in Target description.';
  if (has('voxtell_v1_1_text_prompted') && (value.text.length > 512 || /[\x00-\x1f\x7f]/.test(value.text))) return 'VoxTell text must be at most 512 printable characters.';
  if (box && box[0][2] !== box[1][2] && tasks.some(t => t.name !== 'segvol_prompted_ct')) return 'Both box corners must be on the same slice for these models.';
  return '';
}
function structureInfo(task, meta) {
  if (task.output === 'report') return {label:'Report formats and model details', text:task.structures || 'JSON, Markdown and HTML; no RTSTRUCT'};
  const structures = task.structures || 'Structure list not supplied by the catalog.';
  if (Array.isArray(structures)) return {label:`${structures.length} structures · details`, text:structures.join(', ')};
  // Prefer the runtime label list to a packaged list when TotalSegmentator supplies one.
  if (String(task.description).startsWith('TotalSegmentator ') && structures !== 'Task-specific TotalSegmentator structures') {
    const count = structures.split(',').filter(x => x.trim()).length;
    return {label:`${count} structures · details`, text:structures};
  }
  if (Number.isInteger(meta.count)) return {label:`${meta.count} ${meta.count === 1 ? 'structure' : 'structures'} · details`, text:meta.structures?.join(', ') || structures};
  const count = structures.match(/^(\d+) (?:foreground labels|organs|major|whole-body|bilateral|head-and-neck|anatomy classes)/);
  return {label:count ? `${count[1]} structures · details` : 'Structures · count not supplied', text:structures};
}
function modalityName(m) { return m === 'PT' ? 'PET' : m; }
function esc(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function formatBytes(n) { if (!Number.isFinite(n)) return '—'; const units = ['B','KiB','MiB','GiB']; let i=0; while(n>=1024 && i<3){n/=1024;i++;} return `${n.toFixed(i?1:0)} ${units[i]}`; }
function duration(seconds) { const n=Math.max(0,Math.floor(seconds)); return n<60?`${n}s`:n<3600?`${Math.floor(n/60)}m ${n%60}s`:`${Math.floor(n/3600)}h ${Math.floor(n%3600/60)}m`; }
function studyDate(value) { return /^\d{8}$/.test(value || '') ? `${value.slice(0,4)}-${value.slice(4,6)}-${value.slice(6,8)}` : value || 'Date not supplied'; }
function safeReference(value) { try { const u=new URL(value); return ['https:','http:'].includes(u.protocol) ? u.href : ''; } catch (_) { return ''; } }
const sleep = ms => new Promise(resolve => setTimeout(resolve,ms));
async function request(url, options={}) {
  const response=await fetch(url,options), text=await response.text(); let body;
  try { body=JSON.parse(text); } catch (_) { body={detail:text || `HTTP ${response.status}`}; }
  if(!response.ok){const error=new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail || body));error.status=response.status;throw error;}
  return body;
}

if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const state = {step:0,upload:null,series:null,catalog:[],metadata:{overrides:{}},selected:new Set(),config:null,health:null,catalogReady:false,starting:false};
  let seriesViewer=null, jobViewer=null, detailId=null, detailToken=0, detailLoaded=false, detailInfo=null, detailReport=null, detailLoading=false;
  const stepPanels=[$('uploadStep'),$('seriesStep'),$('modelsStep'),$('reviewStep')];
  const stepTitles=[$('uploadTitle'),$('seriesTitle'),$('modelsTitle'),$('reviewTitle')];
  const uploads={
    seg:{files:[],session:null,busy:false,selection:$('selection'),drop:$('dropZone'),filesInput:$('files'),folderInput:$('folder'),buttons:[$('chooseFiles'),$('chooseFolder')],progress:$('uploadProgress'),bar:$('uploadBar'),text:$('uploadText'),error:$('uploadError')},
    crop:{files:[],session:null,busy:false,selection:$('cropSelection'),drop:$('cropDropZone'),filesInput:$('cropFiles'),folderInput:$('cropFolder'),buttons:[$('chooseCropFiles'),$('chooseCropFolder')],progress:$('cropProgress'),bar:$('cropBar'),text:$('cropText'),error:$('cropError')}
  };
  let prompt={positive_points:[],negative_points:[],box:null,text:''}, boxCorner=null, previewToken=0;
  const reportTask=()=>state.config?.report_task || 'nv_reason_ct_report';
  const chosen=()=>state.catalog.filter(t=>state.selected.has(t.name));
  const prompted=()=>chosen().filter(t=>t.prompt);
  const meta=t=>presentation(t,state.metadata);
  const isRemote=t=>executesRemotely(t,state.config,state.health);
  const reason=t=>blockedReason(t,state.series,state.upload?.series || [],reportTask());
  function showError(el,message){el.textContent=message;el.hidden=!message;}
  function setArea(area, focus=false){
    const match = /^jobs\/([A-Za-z0-9-]+)$/.exec(area);
    const id = match?.[1] || null;
    if(match) area='detail';
    if(!['new','jobs','tools','detail'].includes(area)) area='new';
    if(detailId !== id){
      detailId=id; ++detailToken; detailLoaded=false; detailLoading=false; detailInfo=null; detailReport=null;
      jobViewer?.destroy();jobViewer=null;$('jobResults').hidden=true;showError($('jobDetailError'),'');
      $('deleteJob').disabled=true;$('jobDetailTitle').textContent='Job details';$('jobDetailMeta').textContent=id||'';
      $('jobDetailStatus').textContent='Loading…';$('jobDetailProgress').textContent='';$('jobDetailReportStatus').textContent='';$('jobDetailErrors').hidden=true;$('retryDetail').hidden=true;$('jobDetailBar').removeAttribute('value');
      selectResultTab('report');
    }
    $('newPage').hidden=area!=='new';$('jobsPage').hidden=area!=='jobs';$('toolsPage').hidden=area!=='tools';$('jobDetailPage').hidden=area!=='detail';
    $('moreTools').open=false;
    document.querySelectorAll('[data-area]').forEach(a=>{if(a.dataset.area===(area==='detail'?'jobs':area))a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
    if(area==='detail')refreshDetail();
    if(focus) (area==='new'?$ ('newTitle'):area==='jobs'?$('jobsTitle'):area==='detail'?$('jobDetailTitle'):$('toolsTitle')).focus();
  }
  window.addEventListener('hashchange',()=>setArea(location.hash.slice(1),true));
  function steps(){
    $('uploadSummary').textContent=state.upload?`${state.upload.series.length} series · ${state.upload.series.reduce((n,s)=>n+s.slices,0)} images`:'Files, folder or ZIP';
    $('seriesSummary').textContent=state.series?`${modalityName(state.series.modality)} · ${state.series.description || state.series.slices+' slices'}`:'Choose the image series';
    $('modelsSummary').textContent=state.selected.size?`${state.selected.size} model${state.selected.size===1?'':'s'} selected`:'Choose the outputs';
    document.querySelectorAll('[data-step]').forEach(b=>{const n=Number(b.dataset.step);b.disabled=state.starting || (n===1&&!state.upload)||(n===2&&!state.series)||(n===3&&(!state.series||!state.selected.size));b.classList.toggle('done',n<state.step);if(n===state.step)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  }
  function goStep(n,focus=true){
    if((n>0&&!state.upload)||(n>1&&!state.series)||(n>2&&!state.selected.size)||state.starting)return;
    state.step=n;stepPanels.forEach((p,i)=>p.hidden=i!==n);steps();
    if(n===1&&state.series)loadSeriesViewer();if(n===2)renderCatalog();if(n===3)renderReview();
    if(focus){stepTitles[n].focus({preventScroll:true});window.scrollTo({top:0,behavior:'instant'});}
  }
  document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>goStep(Number(b.dataset.step))));
  document.querySelectorAll('[data-back]').forEach(b=>b.addEventListener('click',()=>goStep(Number(b.dataset.back))));
  $('toSeries').onclick=()=>goStep(1);
  $('toModels').onclick=()=>{if(!state.series){showError($('seriesError'),'Select one image series.');return;}goStep(2);};
  $('toReview').onclick=()=>{if(!state.selected.size){showError($('modelError'),'Choose segmentation, an AI draft report, or both.');return;}goStep(3);};
  function uploadStatus(c,text){c.progress.hidden=false;c.text.textContent=text;}
  function uploadBusy(c,busy){c.busy=busy;c.buttons.forEach(b=>b.disabled=busy);c.filesInput.disabled=busy;c.folderInput.disabled=busy;c.drop.setAttribute('aria-busy',String(busy));}
  async function entryFiles(entry){
    if(!entry)return[];
    if(entry.isFile)return new Promise((resolve,reject)=>entry.file(f=>resolve([f]),reject));
    if(entry.isDirectory){const reader=entry.createReader(),files=[];for(;;){const batch=await new Promise((resolve,reject)=>reader.readEntries(resolve,reject));if(!batch.length)break;for(const item of batch)files.push(...await entryFiles(item));}return files;}
    return[];
  }
  async function setFiles(c,files){
    if(c.busy||!files.length)return;c.files=files;c.session=null;
    c.selection.textContent=`${files.length} files · ${formatBytes(files.reduce((n,f)=>n+f.size,0))}`;
    showError(c.error,'');c.progress.hidden=true;
    if(c===uploads.seg){state.upload=null;state.series=null;state.selected.clear();seriesViewer?.destroy();seriesViewer=null;$('assessment').textContent='';$('toSeries').hidden=true;resetPrompt();goStep(0,false);await inspect();}
  }
  Object.values(uploads).forEach(c=>{
    c.buttons[0].onclick=()=>c.filesInput.click();c.buttons[1].onclick=()=>c.folderInput.click();
    [c.filesInput,c.folderInput].forEach(input=>input.addEventListener('change',()=>{const files=[...input.files];input.value='';setFiles(c,files);}));
    c.drop.addEventListener('dragover',e=>{e.preventDefault();if(!c.busy)c.drop.classList.add('dragging');});
    c.drop.addEventListener('dragleave',()=>c.drop.classList.remove('dragging'));
    c.drop.addEventListener('drop',async e=>{e.preventDefault();c.drop.classList.remove('dragging');if(c.busy)return;const items=[...(e.dataTransfer.items||[])],plain=[...e.dataTransfer.files],entries=items.map(i=>i.webkitGetAsEntry?.()).filter(Boolean);try{let files=[];for(const entry of entries)files.push(...await entryFiles(entry));await setFiles(c,files.length?files:plain);}catch(error){showError(c.error,`Could not read the dropped folder: ${error.message}`);}});
  });
  ['dragover','drop'].forEach(event=>window.addEventListener(event,e=>e.preventDefault()));
  async function uploadRequest(url,options={},keepGoing=()=>true){
    for(let attempt=0;;attempt++){
      if(!keepGoing())throw Error('Upload stopped');let res,body;
      try{res=await fetch(url,options);const text=await res.text();try{body=JSON.parse(text);}catch(_){body={detail:text||`HTTP ${res.status}`};}}
      catch(error){if(attempt>=5)throw Error('Network error during upload. Check the connection, then retry to resume this session.');await sleep(1000*2**attempt);continue;}
      if(res.status===409&&Number.isInteger(body.expected_offset))return{status:409,body};
      if(res.ok)return{status:res.status,body};
      if([408,429,500,501,502,503,504,520,521,522,523,524].includes(res.status)&&attempt<5){await sleep(1000*2**attempt);continue;}
      const error=new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body));error.status=res.status;throw error;
    }
  }
  async function uploadSession(c){
    const files=c.files,total=files.reduce((n,f)=>n+f.size,0),start=Date.now();
    if(state.config?.max_upload_bytes && total>state.config.max_upload_bytes)throw Error(`Selected files exceed the ${formatBytes(state.config.max_upload_bytes)} upload limit.`);
    if(!c.session){uploadStatus(c,'Starting upload…');const created=(await uploadRequest('/api/v1/upload-sessions',{method:'POST'})).body;
      if(!Number.isSafeInteger(created.max_request_bytes)||created.max_request_bytes<=0)throw Error('Invalid server upload chunk limit');
      c.session={base:'/api/v1/upload-sessions/'+encodeURIComponent(created.session_id),limit:created.max_request_bytes,max:created.max_total_bytes,offsets:files.map(()=>0),uploaded:false};}
    const session=c.session;
    if(session.max&&total>session.max)throw Error(`Selected files exceed the ${formatBytes(session.max)} upload limit.`);
    if(session.uploaded)return session.base;
    let next=0,stopped=false;const initial=session.offsets.reduce((a,b)=>a+b,0);
    function progress(){const loaded=session.offsets.reduce((a,b)=>a+b,0),pct=total?Math.floor(100*loaded/total):100,rate=(loaded-initial)/Math.max((Date.now()-start)/1000,.001);c.bar.value=pct;uploadStatus(c,`Uploading ${formatBytes(loaded)} of ${formatBytes(total)} (${pct}%)${rate>0?' · '+formatBytes(rate)+'/s':''}`);}
    async function worker(){try{while(!stopped&&next<files.length){const i=next++,f=files[i];let offset=session.offsets[i],resumes=0;if(offset===f.size&&f.size>0)continue;do{
      const end=Math.min(f.size,offset+session.limit),r=await uploadRequest(`${session.base}/files/${i}?name=${encodeURIComponent(f.name)}&offset=${offset}&size=${f.size}`,{method:'PUT',body:f.slice(offset,end)},()=>!stopped),n=r.status===409?r.body.expected_offset:r.body.received;
      if(!Number.isSafeInteger(n)||n<0||n>f.size)throw Error('Invalid server upload offset');
      if(r.status===409){if(++resumes>5)throw Error('Upload offset could not be reconciled');}else{if(n!==end)throw Error('Incomplete chunk acknowledgement');resumes=0;}
      offset=n;session.offsets[i]=n;if(!stopped)progress();
    }while(!stopped&&offset<f.size);}}catch(error){stopped=true;throw error;}}
    // Wait for all in-flight requests to settle before allowing a retry.
    const results=await Promise.allSettled(Array.from({length:Math.min(4,files.length)},()=>worker()));
    const failure=results.find(r=>r.status==='rejected');if(failure)throw failure.reason;
    session.uploaded=true;c.bar.value=100;return session.base;
  }
  async function inspect(){
    const c=uploads.seg;if(c.busy)return;uploadBusy(c,true);$('resumeUpload').hidden=true;showError(c.error,'');
    try{
      const base=await uploadSession(c);uploadStatus(c,'Upload complete. Inspecting DICOM series…');
      await uploadRequest(base+'/assess',{method:'POST'});
      let result;for(;;){const data=(await uploadRequest(base)).body;if(data.status==='ready'){result=data.result;break;}if(data.status==='failed'){c.session=null;throw Error(data.error||'Inspection failed');}await sleep(1000);}
      if(!result?.series?.length)throw Error('No valid CT, MR or PET image series found.');
      state.upload=result;state.series=null;state.selected.clear();
      const valid=result.series.reduce((n,s)=>n+s.slices,0);
      $('assessment').textContent=`${valid} valid image files in ${result.series.length} series · ${result.ignored || 0} ignored files`;
      uploadStatus(c,'Upload and inspection complete.');$('toSeries').hidden=false;
      if(result.series.length===1)selectSeries(result.series[0]);
      renderSeries();steps();if(location.hash===''||location.hash==='#new')goStep(1);
    }catch(error){if(error.status===404)c.session=null;showError(c.error,error.message);$('resumeUpload').hidden=false;}
    finally{uploadBusy(c,false);}
  }
  $('resumeUpload').onclick=inspect;
  function previewURL(s,index){return `/api/uploads/${encodeURIComponent(state.upload.upload_id)}/preview?series_key=${encodeURIComponent(s.key)}&index=${index}`;}
  function renderSeries(){
    $('series').innerHTML=state.upload.series.map((s,i)=>`<label class="series-card"><input type="radio" name="series_key" value="${esc(s.key)}" data-series-index="${i}" ${state.series?.key===s.key?'checked':''}><span class="series-info"><span class="pill">${esc(modalityName(s.modality))}</span>${s.cbct?' <span class="badge license">CBCT detected</span>':''}<strong>${esc(s.description||'No series description')}</strong><span>${s.slices} slices · ${s.rows} × ${s.columns}</span><br><small class="muted">Study date ${esc(studyDate(s.study_date))}</small></span></label>`).join('');
    $('series').querySelectorAll('input').forEach(input=>input.onchange=()=>selectSeries(state.upload.series[Number(input.dataset.seriesIndex)]));
    if(!seriesViewer)seriesViewer=new DicomUI.Viewer($('seriesViewer'));
    loadSeriesViewer();
  }
  function loadSeriesViewer(){
    if(!state.series||!state.upload)return;
    if(!seriesViewer)seriesViewer=new DicomUI.Viewer($('seriesViewer'));
    const base=`/api/v1/uploads/${encodeURIComponent(state.upload.upload_id)}/series/${encodeURIComponent(state.series.key)}/viewer`;
    if(seriesViewer.base!==base)seriesViewer.load(base);
  }
  function selectSeries(s){
    if(state.series?.key!==s.key){state.series=s;state.selected.clear();$('prostatePostprocess').checked=false;resetPrompt();$('promptSlice').max=s.slices-1;$('promptSlice').value=Math.floor(s.slices/2);$('promptColumn').max=s.columns-1;$('promptRow').max=s.rows-1;}
    const warnings=[];if(s.cbct)warnings.push(`AI draft report unavailable for CBCT: ${s.cbct_reason || 'CBCT detected'}.`);
    if(s.modality==='PT'){const ct=pairedCT(s,state.upload.series);warnings.push(ct?`Paired CT found: ${ct.description||'CT series'} (${ct.slices} slices). PET1 ISRT models use a CT in the same Study and Frame of Reference. This check does not establish registration.`:'PET1 ISRT models require a CT in the same Study and Frame of Reference. No matching CT was found in this upload. PET-only models remain selectable when available.');}
    $('seriesWarnings').textContent=warnings.join(' ');$('seriesWarnings').hidden=!warnings.length;showError($('seriesError'),'');steps();renderSelection();loadSeriesViewer();
  }
  function badges(t){
    const result=[];if(t.licensed)result.push(['Licence-restricted','license']);if(t.prompt)result.push(['Interactive prompt needed','prompt']);if(t.output==='report')result.push(['Research AI draft','ai']);if(isRemote(t))result.push(['Runs on remote GPU','remote']);
    if(/non[ -]?commercial|by-nc|research.or.education.use.only/i.test([t.license,t.license_note,t.description,t.limitations].join(' ')))result.push(['Non-commercial / restricted weights','license']);
    if(t.requires_ct)result.push(['Paired CT required','']);
    return result.map(([text,cls])=>`<span class="badge ${cls}">${text}</span>`).join('');
  }
  function availabilityNote(t, blocked){
    if(!blocked)return '';
    if(t.missing?.length || t.remote_preflight_error)return `<details class="reason-details"><summary>Unavailable: ${esc(t.status || 'unavailable')} · details</summary><p class="reason">${esc(blocked)}</p></details>`;
    return `<p class="reason">Unavailable: ${esc(blocked)}</p>`;
  }
  function modelRow(t){
    const m=meta(t),blocked=reason(t),structures=structureInfo(t,m),reference=safeReference(t.reference);
    return `<article class="model-row ${blocked?'is-unavailable':''}"><input type="checkbox" name="models" value="${esc(t.name)}" aria-label="Select ${esc(m.title)}" ${state.selected.has(t.name)?'checked':''} ${blocked?'disabled':''}><div><label class="model-title" data-toggle-model="${esc(t.name)}">${esc(m.title)}</label><p class="model-purpose">${esc(m.purpose)}</p><div class="badges">${badges(t)}</div>${availabilityNote(t,blocked)}<details class="model-details"><summary>${esc(structures.label)}</summary><div class="detail-body"><p><strong>Output:</strong> <span class="structures">${esc(structures.text)}</span></p><p>${esc(t.description || '')}</p>${t.limitations?`<p><strong>Limitations:</strong> ${esc(t.limitations)}</p>`:''}${t.license||t.license_note?`<p><strong>Licence:</strong> ${esc([t.license,t.license_note].filter(Boolean).join(' · '))}</p>`:''}<p><strong>Task:</strong> <code>${esc(t.name)}</code> · <strong>Catalog status:</strong> ${esc(t.status||'available')}</p>${t.clinical_tags?.length?`<p><strong>Clinical tags:</strong> ${esc(t.clinical_tags.map(x=>x.label).join(', '))}</p>`:''}${reference?`<a href="${esc(reference)}" target="_blank" rel="noopener noreferrer">Model reference (external)</a>`:''}</div></details></div></article>`;
  }
  function renderCatalog(){
    if(!state.series)return;
    $('modalityLabel').textContent=modalityName(state.series.modality)+' series';
    $('modelNote').textContent='Choose either action or both. Nothing runs until you review and start.';
    const site=$('clinicalFilter').value,q=$('modelSearch').value.trim().toLowerCase(),show=$('showUnavailable').checked;
    const compatible=state.catalog.filter(t=>supports(t,state.series)),available=compatible.filter(t=>!reason(t));
    const visible=compatible.filter(t=>(show||!reason(t))&&(!site||(t.clinical_tags||[]).some(x=>x.id===site))&&(!q||[t.name,meta(t).title,meta(t).purpose,t.description,t.structures,...(t.clinical_tags||[]).map(x=>x.label)].join(' ').toLowerCase().includes(q)));
    const opened=new Set([...$('modelGroups').querySelectorAll('details[data-group][open]')].map(el=>el.dataset.group));
    const groups=GROUPS.map(([id,title])=>({id,title,tasks:visible.filter(t=>meta(t).group===id)})).filter(g=>g.tasks.length);
    $('modelResults').textContent=state.catalogReady?`${visible.length} models shown · ${available.length} available for this ${modalityName(state.series.modality)} series · ${compatible.length-available.length} unavailable`:'';
    $('modelGroups').innerHTML=groups.map((g,i)=>`<details class="model-group" data-group="${g.id}" ${q||site||opened.has(g.id)||(!opened.size&&i===0)?'open':''}><summary>${g.title}<span class="group-count">${g.tasks.length} models</span></summary>${g.tasks.map(modelRow).join('')}</details>`).join('')||(state.catalogReady?'<p class="empty">No models match these filters. Change the search or show unavailable models.</p>':'');
    renderSelection();
  }
  function toggleModel(name){const task=state.catalog.find(t=>t.name===name);if(!task||reason(task))return;if(state.selected.has(name))state.selected.delete(name);else state.selected.add(name);showError($('modelError'),'');syncModelInputs();renderSelection();}
  function syncModelInputs(){document.querySelectorAll('[name=models]').forEach(i=>i.checked=state.selected.has(i.value));}
  $('modelGroups').addEventListener('change',e=>{if(e.target.name==='models')toggleModel(e.target.value);});
  $('modelGroups').addEventListener('click',e=>{const label=e.target.closest('[data-toggle-model]');if(label)toggleModel(label.dataset.toggleModel);});
  $('selectedChips').addEventListener('click',e=>{const b=e.target.closest('[data-remove]');if(b){state.selected.delete(b.dataset.remove);syncModelInputs();renderSelection();}});
  $('clearModels').onclick=()=>{state.selected.clear();syncModelInputs();renderSelection();};
  $('modelSearch').oninput=renderCatalog;$('clinicalFilter').onchange=renderCatalog;$('showUnavailable').onchange=renderCatalog;
  function renderActions(){
    const report=state.catalog.find(t=>t.name===reportTask());
    const segments=chosen().filter(t=>t.output!=='report'&&t.name!==reportTask());
    const defaultName=state.series?.modality==='CT'?'total':state.series?.modality==='MR'?'total_mr':null;
    const defaultTask=state.catalog.find(t=>t.name===defaultName);
    const segmentReason=segments.length?'':defaultTask?reason(defaultTask):state.catalogReady?(defaultName?'Default model is not supplied by the catalog. Choose another available model.':'Choose a PET model from the model catalog.'):'Loading model catalog…';
    $('segmentAction').checked=segments.length>0;$('segmentAction').disabled=!!segmentReason;
    $('segmentCard').classList.toggle('selected',segments.length>0);
    $('segmentDefault').textContent=defaultName?`Default: TotalSegmentator ${defaultName}`:'PET segmentation';
    $('segmentBadges').innerHTML=defaultTask?badges(defaultTask):'';
    $('segmentSelection').textContent=segments.length?segments.map(t=>meta(t).title).join(' · '):defaultName?'Select this action to add the default model.':'Choose one or more models below.';
    showError($('segmentReason'),segmentReason);
    $('reportAction').checked=state.selected.has(reportTask());
    $('reportCard').classList.toggle('selected',state.selected.has(reportTask()));
    const blocked=report?reason(report):state.catalogReady?'NV-Reason-CT is not supplied by the model catalog.':'Loading model catalog…';
    $('reportAction').disabled=!!blocked;showError($('reportReason'),blocked);
    $('reportBadges').innerHTML=report?badges(report):'';
    const reference=report?safeReference(report.reference):'';
    $('reportDetails').hidden=!report;
    $('reportModelDetails').innerHTML=report?`<p>${esc(report.description||'')}</p><p>${esc(report.limitations||'')}</p><p>${esc([report.license,report.license_note].filter(Boolean).join(' · '))}</p>${reference?`<a href="${esc(reference)}" target="_blank" rel="noopener noreferrer">Model reference (external)</a>`:''}`:'';
  }
  for(const [id,action] of [['segmentAction','segment'],['reportAction','report']])$(id).onchange=()=>{
    state.selected=primarySelection(state.selected,action,$(id).checked,state.catalog,state.series,state.upload?.series||[],reportTask());
    showError($('modelError'),'');syncModelInputs();renderSelection();
  };
  $('chooseModels').onclick=()=>{const open=$('catalogPicker').hidden;$('catalogPicker').hidden=!open;$('chooseModels').setAttribute('aria-expanded',String(open));if(open){renderCatalog();$('modelSearch').focus();}};
  function renderSelection(){
    renderActions();
    $('selectionCount').textContent=`${state.selected.size} model${state.selected.size===1?'':'s'} selected`;
    $('selectedChips').innerHTML=chosen().map(t=>`<button type="button" class="chip" data-remove="${esc(t.name)}" aria-label="Remove ${esc(meta(t).title)}">${esc(meta(t).title)} <span aria-hidden="true">×</span></button>`).join('')||'<span class="small muted">Select models to build this job.</span>';
    steps();
  }
  async function loadCatalog(){
    $('catalogStatus').textContent='Loading model catalog. Worker checks can take a few minutes…';$('reloadCatalog').hidden=true;
    try{const data=await request('/api/v1/models');state.catalog=data.tasks;state.catalogReady=true;
      const previous=$('clinicalFilter').value;$('clinicalFilter').innerHTML='<option value="">All tumour sites</option>'+(data.clinical_tags||[]).map(t=>`<option value="${esc(t.id)}">${esc(t.label)}</option>`).join('');$('clinicalFilter').value=previous;
      for(const name of state.selected){const t=state.catalog.find(t=>t.name===name);if(!t||reason(t))state.selected.delete(name);}
      $('catalogStatus').textContent='';renderCatalog();if(state.step===3)renderReview();
    }catch(error){$('catalogStatus').textContent='Model catalog could not be loaded: '+error.message;$('reloadCatalog').hidden=false;}
  }
  $('reloadCatalog').onclick=loadCatalog;
  function renderReview(){
    const s=state.series,tasks=chosen().sort((a,b)=>Number(a.output==='report')-Number(b.output==='report'));
    $('reviewContent').innerHTML=`<div class="review-card"><h3>Selected series</h3><span class="pill">${esc(modalityName(s.modality))}</span><p><strong>${esc(s.description||'No series description')}</strong><br>${s.slices} slices · ${s.rows} × ${s.columns}<br>Study date ${esc(studyDate(s.study_date))}</p><button type="button" data-edit-series>Edit series</button></div><div class="review-card"><h3>${tasks.length} model${tasks.length===1?'':'s'} to run</h3><ul>${tasks.map(t=>`<li><strong>${esc(meta(t).title)}</strong><span class="small muted">${t.output==='report'?'AI draft report · JSON, Markdown and HTML':'RTSTRUCT contours'}${isRemote(t)?' · remote GPU':''}${t.requires_ct?' · paired CT included':''}</span><div class="badges">${badges(t)}</div></li>`).join('')}</ul></div>`;
    $('reviewContent').querySelector('[data-edit-series]').onclick=()=>goStep(1);
    const crop=cropSupported(s,state.selected,state.catalog,state.config,state.health);$('postprocessPanel').hidden=!crop;if(!crop)$('prostatePostprocess').checked=false;
    $('promptPanel').hidden=!prompted().length;
    if(prompted().length){$('promptGuidance').innerHTML=prompted().map(t=>`<p>${esc(PROMPT_HELP[t.name]||`${meta(t).title}: ${t.prompt}. ${t.limitations||t.description||''}`)}</p>`).join('');loadPromptSlice();}
    const warnings=[];if(tasks.some(t=>t.output==='report'))warnings.push('AI output is a research draft requiring radiologist review. Diagnostic chest/abdomen CT only; coverage, HU or resource checks can refuse generation. Reports run after segmentation.');
    if(tasks.some(t=>t.licensed))warnings.push('Selected models include licence-restricted weights. Their licence terms remain available in model details.');
    if(tasks.some(t=>t.requires_ct))warnings.push('The matching CT is paired by Study and Frame of Reference. This does not establish registration.');
    $('reviewWarnings').textContent=warnings.join(' ');$('reviewWarnings').hidden=!warnings.length;
    $('startJob').textContent=`Start job · ${tasks.length} model${tasks.length===1?'':'s'}`;
  }
  function resetPrompt(){prompt={positive_points:[],negative_points:[],box:null,text:''};boxCorner=null;$('promptText').value='';writePrompt();showError($('promptError'),'');}
  function writePrompt(){ $('promptJson').value=JSON.stringify(prompt,null,2);$('promptText').value=prompt.text||'';drawPrompt(); }
  function readPrompt(){try{const p=JSON.parse($('promptJson').value);if(!p||typeof p!=='object'||Array.isArray(p))throw Error();return p;}catch(_){throw Error('Prompt JSON must be a valid JSON object.');}}
  function loadPromptSlice(){
    if(!state.series||!state.upload)return;const index=Number($('promptSlice').value),token=++previewToken;
    $('promptSliceLabel').textContent=`${index+1} of ${state.series.slices} (index ${index})`;$('previewError').hidden=true;
    $('promptImage').onload=()=>{if(token===previewToken)drawPrompt();};$('promptImage').onerror=()=>{$('previewError').hidden=false;};
    $('promptImage').src=previewURL(state.series,index);drawPrompt();
  }
  function drawPrompt(){
    const s=state.series;if(!s)return;const index=Number($('promptSlice').value);let html='';
    for(const [key,cls] of [['positive_points','positive'],['negative_points','negative']]){
      if(!Array.isArray(prompt[key]))continue;for(const p of prompt[key]){if(!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isFinite(v))||p[2]!==index)continue;html+=`<span class="prompt-marker ${cls}" style="left:${(p[0]+.5)/s.columns*100}%;top:${(p[1]+.5)/s.rows*100}%">${cls==='positive'?'+':'−'}</span>`;}
    }
    const box=prompt.box;if(Array.isArray(box)&&box.length===2&&box.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite))&&index>=Math.min(box[0][2],box[1][2])&&index<=Math.max(box[0][2],box[1][2]))html+=`<span class="prompt-box" style="left:${Math.min(box[0][0],box[1][0])/s.columns*100}%;top:${Math.min(box[0][1],box[1][1])/s.rows*100}%;width:${Math.abs(box[1][0]-box[0][0])/s.columns*100}%;height:${Math.abs(box[1][1]-box[0][1])/s.rows*100}%"></span>`;
    if(boxCorner&&boxCorner[2]===index)html+=`<span class="prompt-marker" style="left:${(boxCorner[0]+.5)/s.columns*100}%;top:${(boxCorner[1]+.5)/s.rows*100}%">1</span>`;
    $('promptOverlay').innerHTML=html;$('promptStatus').textContent=`${Array.isArray(prompt.positive_points)?prompt.positive_points.length:0} positive · ${Array.isArray(prompt.negative_points)?prompt.negative_points.length:0} negative · ${prompt.box?'1 box':'no box'}${boxCorner?' · Choose the second corner on this slice.':''}`;
  }
  function addPoint(point){
    try{prompt=readPrompt();const s=state.series;if(point.some((v,i)=>!Number.isFinite(v)||v<0||v>=[s.columns,s.rows,s.slices][i]))throw Error('The point must be within this series.');
      const mode=$('promptMode').value;if(mode==='box'){if(!boxCorner)boxCorner=point;else{if(boxCorner[2]!==point[2])throw Error('Both box corners must be on the same slice. Clear the prompt to restart the box.');prompt.box=[boxCorner,point];boxCorner=null;}}
      else{const key=mode+'_points';if(!Array.isArray(prompt[key]))prompt[key]=[];prompt[key].push(point);}
      showError($('promptError'),'');writePrompt();
    }catch(error){showError($('promptError'),error.message);}
  }
  $('promptImage').onclick=e=>{if(!$('previewError').hidden||!e.target.naturalWidth)return;const s=state.series,r=e.target.getBoundingClientRect();addPoint([Math.max(0,Math.min(s.columns-1,Math.floor((e.clientX-r.left)/r.width*s.columns))),Math.max(0,Math.min(s.rows-1,Math.floor((e.clientY-r.top)/r.height*s.rows))),Number($('promptSlice').value)]);};
  $('addPoint').onclick=()=>addPoint([$('promptColumn').valueAsNumber,$('promptRow').valueAsNumber,Number($('promptSlice').value)]);
  $('promptSlice').oninput=loadPromptSlice;$('clearPrompt').onclick=resetPrompt;
  $('promptText').oninput=()=>{try{prompt=readPrompt();prompt.text=$('promptText').value;$('promptJson').value=JSON.stringify(prompt,null,2);showError($('promptError'),'');}catch(error){showError($('promptError'),error.message);}};
  $('promptJson').oninput=()=>{try{prompt=readPrompt();$('promptText').value=typeof prompt.text==='string'?prompt.text:'';boxCorner=null;drawPrompt();showError($('promptError'),'');}catch(error){showError($('promptError'),error.message);}};
  function routeStartError(message){
    if(/postprocess|SBRT|crop/i.test(message)&&!$('postprocessPanel').hidden){showError($('postprocessError'),message);$('postprocessError').scrollIntoView({block:'center'});}
    else if(/prompt|point|corner|coordinate|VoxTell|MedSAM|PAM|SAT3D/i.test(message)&&prompted().length){showError($('promptError'),message);$('promptError').scrollIntoView({block:'center'});}
    else if(/series|paired|same Study|CBCT/i.test(message)){goStep(1);showError($('seriesError'),message);}
    else if(/model|task|licen/i.test(message)){goStep(2);showError($('modelError'),message);}
    else showError($('startError'),message);
  }
  $('startJob').onclick=async()=>{
    if(state.starting)return;showError($('startError'),'');showError($('promptError'),'');showError($('postprocessError'),'');
    for(const t of chosen()){const error=reason(t);if(error){goStep(2);showError($('modelError'),`${meta(t).title}: ${error}`);return;}}
    if(!state.selected.size){goStep(2);showError($('modelError'),'Choose segmentation, an AI draft report, or both.');return;}
    let promptValue='';if(prompted().length){try{prompt=readPrompt();const error=boxCorner?'Complete the second box corner or clear the prompt.':validatePrompt(prompt,prompted(),state.series);if(error)throw Error(error);promptValue=JSON.stringify(prompt);}catch(error){showError($('promptError'),error.message);$('promptError').scrollIntoView({block:'center'});return;}}
    state.starting=true;steps();$('startJob').disabled=true;$('startJob').textContent='Starting…';
    let failure=null;
    try{const form=new FormData();form.append('series_key',state.series.key);chosen().forEach(t=>form.append('models',t.name));if(promptValue)form.append('prompt_json',promptValue);
      if($('prostatePostprocess').checked&&cropSupported(state.series,state.selected,state.catalog,state.config,state.health))form.append('postprocess','prostate_sbrt_crop');
      const result=await request(`/api/uploads/${encodeURIComponent(state.upload.upload_id)}/start`,{method:'POST',body:form});
      rememberJob(result.id);$('jobNotice').textContent='Job queued. Follow progress and download results here.';$('jobNotice').hidden=false;
      state.upload=null;state.series=null;state.selected.clear();seriesViewer?.destroy();seriesViewer=null;uploads.seg.files=[];uploads.seg.session=null;uploads.seg.selection.textContent='No files selected';uploads.seg.progress.hidden=true;$('assessment').textContent='';$('toSeries').hidden=true;resetPrompt();
      location.hash='jobs/'+encodeURIComponent(result.id);pollJobs();
    }catch(error){failure=error.message;}
    finally{state.starting=false;$('startJob').disabled=false;steps();if(!state.upload)goStep(0,false);else renderReview();if(failure)routeStartError(failure);}
  };
  $('cropForm').onsubmit=async e=>{
    e.preventDefault();const c=uploads.crop;if(c.busy)return;showError(c.error,'');
    if(!c.files.length){showError(c.error,'Choose CT and matching RTSTRUCT files first.');return;}
    const margin=$('cropMargin').valueAsNumber;if(!Number.isFinite(margin)||margin<0){showError(c.error,'Margin must be finite and non-negative.');return;}
    uploadBusy(c,true);$('cropSubmit').disabled=true;$('cropMargin').disabled=true;$('cropKeepUids').disabled=true;
    try{const base=await uploadSession(c),form=new FormData();form.append('margin_mm',String(margin*10));form.append('keep_uids',String($('cropKeepUids').checked));uploadStatus(c,'Upload complete. Queueing CPU crop…');
      const result=(await uploadRequest(base+'/prostate-sbrt-crop',{method:'POST',body:form})).body;rememberJob(result.id);uploadStatus(c,'CPU crop queued. Follow progress in Jobs.');c.session=null;c.files=[];c.selection.textContent='No files selected';$('jobNotice').textContent='CPU crop queued.';$('jobNotice').hidden=false;location.hash='jobs';pollJobs();
    }catch(error){if(error.status===404)c.session=null;showError(c.error,error.message);}
    finally{uploadBusy(c,false);$('cropSubmit').disabled=false;$('cropMargin').disabled=false;$('cropKeepUids').disabled=false;}
  };
  let remembered=[];try{const saved=JSON.parse(localStorage.getItem('dicomJobs')||'[]');if(Array.isArray(saved))remembered=saved.filter(x=>typeof x==='string'&&/^[A-Za-z0-9-]+$/.test(x));}catch(_){}
  function rememberJob(id){remembered=[id,...remembered.filter(x=>x!==id)].slice(0,100);try{localStorage.setItem('dicomJobs',JSON.stringify(remembered));}catch(_){} }
  const jobRows=new Map(),jobData=new Map();let polling=false;
  function updateText(el,text){if(el.textContent!==text)el.textContent=text;}
  function updateJob(j){
    let row=jobRows.get(j.id);if(!row){row=$('jobTemplate').content.firstElementChild.cloneNode(true);row.dataset.job=j.id;jobRows.set(j.id,row);row.querySelector('[data-field=retry]').onclick=()=>retryJob(j.id,row);}
    const field=name=>row.querySelector(`[data-field="${name}"]`),busy=!TERMINAL.has(j.status),models=j.models||[],total=j.model_total||models.length,pct=Number.isFinite(j.progress)?Math.max(0,Math.min(100,j.progress)):null;
    updateText(field('title'),j.tool?'Prostate SBRT crop':models.length===1&&models[0]===reportTask()?'AI draft report':models.includes(reportTask())?'Segmentation and AI draft report':`${models.length} model${models.length===1?'':'s'} · segmentation job`);
    field('title').href=field('open').href='#jobs/'+encodeURIComponent(j.id);
    const elapsed=(busy?Date.now()/1000:(j.finished||j.completed||j.updated||j.created))-j.created;
    updateText(field('meta'),`${new Date(j.created*1000).toLocaleString()} · ${busy?'Elapsed':'Duration'} ${duration(elapsed)} · ${j.id}`);
    updateText(field('status'),humanize(j.status));field('status').className='pill '+(TERMINAL.has(j.status)||['processing','transferring','retrieving','queued'].includes(j.status)?j.status:'');
    updateText(field('models'),models.map(name=>meta(state.catalog.find(t=>t.name===name)||{name}).title).join(' · '));updateText(field('detail'),j.detail||'');
    field('progressWrap').hidden=!busy&&pct===null;if(pct===null)field('progress').removeAttribute('value');else field('progress').value=pct;
    const progress=[];if(j.model_index&&total)progress.push(`Model ${j.model_index} of ${total}`);else if(busy&&total)progress.push(`${total} model${total===1?'':'s'} queued`);
    if(j.total_bytes&&(j.status==='transferring'||j.status==='retrieving'))progress.push(`${formatBytes(j.transferred_bytes||0)} / ${formatBytes(j.total_bytes)}`);
    if(pct!==null)progress.push(`${pct}%`);updateText(field('progressText'),progress.join(' · '));
    field('report').hidden=!j.report_status;updateText(field('report'),`AI report: ${humanize(j.report_status||'')} ${j.report_reason||''}`.trim());
    field('errors').hidden=!j.error;updateText(field('error'),j.error||'');
    field('download').hidden=j.status!=='complete';field('download').href=`/api/v1/jobs/${encodeURIComponent(j.id)}/result`;
    const hasReport=j.status==='complete'&&['generated','not_generated'].includes(j.report_status);
    field('viewReport').hidden=!hasReport;field('viewReport').href=`/jobs/${encodeURIComponent(j.id)}/report`;
    field('reportJson').hidden=!hasReport;field('reportJson').href=`/api/v1/jobs/${encodeURIComponent(j.id)}/report`;
    field('retry').hidden=j.status!=='failed';return row;
  }
  async function pollJobs(){
    if(polling)return;polling=true;let summaries=[],listSucceeded=false;
    try{try{const result=await request('/api/v1/jobs');summaries=result.jobs;listSucceeded=true;showError($('jobsError'),'');}catch(error){showError($('jobsError'),'Could not refresh jobs: '+error.message);summaries=remembered.map(id=>({id}));}
      let next=0,failed=0;const jobs=[];
      await Promise.all(Array.from({length:Math.min(4,summaries.length)},async()=>{while(next<summaries.length){const summary=summaries[next++];try{const j=await request('/api/v1/jobs/'+encodeURIComponent(summary.id));jobData.set(j.id,j);jobs.push(j);}catch(_){failed++;if(jobData.has(summary.id))jobs.push(jobData.get(summary.id));}}}));
      jobs.sort((a,b)=>b.created-a.created);const present=new Set(jobs.map(j=>j.id));
      if(listSucceeded){for(const [id,row] of jobRows){if(!present.has(id)){row.remove();jobRows.delete(id);jobData.delete(id);}}}
      if(failed)showError($('jobsError'),`${failed} job detail request${failed===1?'':'s'} failed. Retained rows may be out of date; the next refresh will retry.`);
      if(jobs.length){$('jobs').querySelector('.empty')?.remove();jobs.forEach((j,i)=>{const row=updateJob(j),current=$('jobs').children[i];if(current!==row)$('jobs').insertBefore(row,current||null);});}
      else if(listSucceeded&&!failed&&!jobRows.size)$('jobs').innerHTML='<p class="empty panel">No jobs yet. Start with a study in New job.</p>';
      const count=(listSucceeded?summaries:jobs).filter(j=>j.status&&!TERMINAL.has(j.status)).length;updateText($('activeCount'),String(count));$('activeCount').setAttribute('aria-label',`${count} active jobs`);
      if(detailId)refreshDetail();
    }finally{polling=false;}
  }
  async function retryJob(id,row){const button=row.querySelector('[data-field=retry]'),error=row.querySelector('[data-field=retryError]');button.disabled=true;showError(error,'');try{await request(`/api/v1/jobs/${encodeURIComponent(id)}/retry`,{method:'POST'});await pollJobs();}catch(e){showError(error,e.message);}finally{button.disabled=false;}}
  $('refreshJobs').onclick=pollJobs;
  function selectResultTab(name, focus=false){
    document.querySelectorAll('[data-result-tab]').forEach(button=>{
      const selected=button.dataset.resultTab===name;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;
      $(button.getAttribute('aria-controls')).hidden=!selected;if(selected&&focus)button.focus();
    });
  }
  document.querySelectorAll('[data-result-tab]').forEach(button=>{
    button.onclick=()=>selectResultTab(button.dataset.resultTab);
    button.onkeydown=e=>{
      const names=['report','structures','downloads'],i=names.indexOf(button.dataset.resultTab);
      let next;if(e.key==='ArrowRight')next=(i+1)%3;else if(e.key==='ArrowLeft')next=(i+2)%3;else if(e.key==='Home')next=0;else if(e.key==='End')next=2;else return;
      e.preventDefault();selectResultTab(names[next],true);
    };
  });
  let detailRequest=0, deleting=false;
  async function refreshDetail(){
    const id=detailId,token=detailToken,serial=++detailRequest;if(!id)return;
    try{
      const job=await request(`/api/v1/jobs/${encodeURIComponent(id)}`);
      if(token!==detailToken||serial!==detailRequest)return;
      jobData.set(id,job);showError($('jobDetailError'),'');
      const models=job.models||[],reportOnly=models.length===1&&models[0]===reportTask();
      $('jobDetailTitle').textContent=job.tool?'Prostate SBRT crop':reportOnly?'AI draft report':models.includes(reportTask())?'Segmentation and AI draft report':'Segmentation results';
      $('jobDetailMeta').textContent=`${id} · ${new Date(job.created*1000).toLocaleString()} · ${models.map(name=>meta(state.catalog.find(t=>t.name===name)||{name}).title).join(' · ')}`;
      $('jobDetailStatus').textContent=humanize(job.status);$('jobDetailStatus').className='pill '+(TERMINAL.has(job.status)||['queued','processing','retrieving','transferring'].includes(job.status)?job.status:'');
      $('jobDetailProgress').textContent=[job.detail,job.model_index?`Model ${job.model_index} of ${job.model_total||models.length}`:'',Number.isFinite(job.progress)?`${job.progress}%`:'',job.total_bytes?`${formatBytes(job.transferred_bytes||0)} / ${formatBytes(job.total_bytes)}`:''].filter(Boolean).join(' · ');
      if(Number.isFinite(job.progress))$('jobDetailBar').value=Math.max(0,Math.min(100,job.progress));else $('jobDetailBar').removeAttribute('value');
      $('jobDetailBar').hidden=TERMINAL.has(job.status);
      $('jobDetailReportStatus').textContent=job.report_status?`AI report: ${humanize(job.report_status)}${job.report_reason?' · '+job.report_reason:''}`:'';
      $('jobDetailErrors').hidden=!job.error;$('jobDetailFailure').textContent=job.error||'';
      $('retryDetail').hidden=job.status!=='failed';$('deleteJob').disabled=deleting||!canDeleteJob(job);
      $('deleteJob').title=canDeleteJob(job)?'Delete job and its files':'Deletion is available after the job stops running';
      if($('deleteDialog').open)$('confirmDelete').disabled=deleting||!canDeleteJob(job);
      $('jobResults').hidden=job.status!=='complete';
      if(job.status==='complete'&&!detailLoaded&&!detailLoading)loadResults(job,token);
      if(job.status!=='complete'&&detailLoaded){detailLoaded=false;jobViewer?.destroy();jobViewer=null;}
    }catch(error){if(token===detailToken&&serial===detailRequest){showError($('jobDetailError'),`Could not load this job: ${error.message}`);$('deleteJob').disabled=true;}}
  }
  async function loadResults(job,token){
    detailLoading=true;detailReport=null;detailInfo=null;
    const base=`/api/v1/jobs/${encodeURIComponent(job.id)}`;
    $('reportContent').textContent='Loading report…';$('structuresContent').textContent='Loading structures…';$('reportDownloads').hidden=true;showError($('downloadError'),'');
    $('resultDownload').href=base+'/result';$('jsonDownload').href=base+'/report';$('htmlDownload').href=base+'/report/html';
    jobViewer?.destroy();jobViewer=new DicomUI.Viewer($('jobViewer'),{overlays:true});
    const [metadata,report]=await Promise.allSettled([jobViewer.load(base+'/viewer'),request(base+'/report')]);
    if(token!==detailToken)return;
    detailInfo=metadata.status==='fulfilled'?metadata.value:null;
    if(report.status==='fulfilled'){
      detailReport=report.value;$('reportDownloads').hidden=false;
      $('reportContent').innerHTML=`<p class="report-link"><a href="/jobs/${encodeURIComponent(job.id)}/report" target="_blank" rel="noopener">Open full report ↗</a></p>`+reportMarkup(detailReport);
    }else{
      const expected=detailInfo?.report||job.report_status||(job.models||[]).includes(reportTask());
      $('reportContent').textContent=expected?`Report unavailable: ${report.reason.message}. Use Refresh to retry.`:'No AI report was requested for this job.';
    }
    renderStructures();detailLoaded=true;detailLoading=false;
  }
  function renderStructures(){
    if(!detailInfo){$('structuresContent').textContent='Structure metadata is unavailable. Use Refresh to retry. Downloads remain available.';return;}
    const rows=detailInfo.structure_sets.flatMap(set=>set.rois.map(roi=>({model:set.model,roi})));
    if(!rows.length){$('structuresContent').textContent='No RTSTRUCT structures are available for this job.';return;}
    $('structuresContent').innerHTML=`<div class="table-scroll"><table><thead><tr><th>Model</th><th>ROI</th><th>Slice range (index)</th><th>Approx. mL</th></tr></thead><tbody>${rows.map(({model,roi})=>{
      const candidates=(detailReport?.segmentation_summary||[]).filter(s=>s.model===model).flatMap(s=>s.rois||[]).filter(r=>r.name===roi.name);
      const volume=candidates.length===1?candidates[0].approx_volume_ml:null;
      return `<tr><td>${esc(model)}</td><td>${esc(roi.name)}</td><td>${roi.slices.length?`${roi.slices[0]}–${roi.slices.at(-1)}`:'No matching slices'}</td><td>${Number.isFinite(volume)?esc(volume):'—'}</td></tr>`;
    }).join('')}</tbody></table></div>`;
  }
  $('refreshDetail').onclick=()=>{detailLoaded=false;refreshDetail();};
  $('retryDetail').onclick=async()=>{
    const id=detailId,token=detailToken;$('retryDetail').disabled=true;
    try{await request(`/api/v1/jobs/${encodeURIComponent(id)}/retry`,{method:'POST'});if(token===detailToken){detailLoaded=false;refreshDetail();}pollJobs();}
    catch(error){if(token===detailToken)showError($('jobDetailError'),error.message);}
    finally{$('retryDetail').disabled=false;}
  };
  $('markdownDownload').onclick=()=>{
    if(!detailReport)return;
    const url=URL.createObjectURL(new Blob([reportMarkdown(detailReport)],{type:'text/markdown;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download=`${detailId}-report.md`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  $('deleteJob').onclick=()=>{
    if(!canDeleteJob(jobData.get(detailId)))return;
    $('deleteIdentity').textContent=detailId;$('deleteDialog').dataset.job=detailId;showError($('deleteError'),'');$('confirmDelete').disabled=false;$('deleteDialog').showModal();
  };
  $('deleteDialog').addEventListener('cancel',e=>{if(deleting)e.preventDefault();});
  $('confirmDelete').onclick=async()=>{
    const id=$('deleteDialog').dataset.job;if(deleting||!canDeleteJob(jobData.get(id)))return;
    deleting=true;$('confirmDelete').disabled=true;$('cancelDelete').disabled=true;$('deleteJob').disabled=true;
    try{
      await request(`/api/v1/jobs/${encodeURIComponent(id)}`,{method:'DELETE'});
      jobRows.get(id)?.remove();jobRows.delete(id);jobData.delete(id);remembered=remembered.filter(value=>value!==id);
      try{localStorage.setItem('dicomJobs',JSON.stringify(remembered));}catch(_){}
      $('deleteDialog').close();$('jobNotice').textContent=`Deleted job ${id} and its files.`;$('jobNotice').hidden=false;location.hash='jobs';pollJobs();
    }catch(error){showError($('deleteError'),error.message);}
    finally{deleting=false;$('cancelDelete').disabled=false;$('confirmDelete').disabled=false;if(detailId)refreshDetail();}
  };
  async function refreshHealth(){
    try{const h=await request('/api/v1/health');state.health=h;const pieces=[];
      if(Number.isFinite(h.gpu_total_mb)&&h.gpu_total_mb>0){pieces.push(`GPU available · ${(h.gpu_free_mb/1024).toFixed(1)} / ${(h.gpu_total_mb/1024).toFixed(1)} GiB free`);}else pieces.push(h.worker_mode==='local'?'GPU unavailable / memory not reported':h.worker_reachable?'Worker reachable · GPU memory not reported':'Worker unreachable');
      const remote=[...(h.remote_gpu_tasks||[]),...(state.config?.remote_gpu_tasks||[])].length>0;if(remote)pieces.push(h.remote_worker_reachable===true?'Remote GPU worker reachable':h.remote_worker_reachable===false?'Remote GPU worker unreachable':'Remote GPU worker status unknown');
      $('health').textContent=pieces.join(' · ');$('health').className='health pill '+(h.status==='ok'&&h.gpu_total_mb!==0&&h.gpu_total_mb!=null?'good':'warning');$('version').textContent=h.version||state.config?.version||'';
      if(state.step===3){const crop=cropSupported(state.series,state.selected,state.catalog,state.config,h);$('postprocessPanel').hidden=!crop;if(!crop)$('prostatePostprocess').checked=false;}
    }catch(error){$('health').textContent='Worker status unavailable';$('health').className='health pill warning';}
  }
  async function loadConfig(){
    try{state.config=await request('/api/v1/config');const c=state.config;if(Number.isFinite(c.retention_hours))$('retention').textContent=`Uploads and results are deleted automatically after ${c.retention_hours} hours.`;
      if(Number.isFinite(c.max_upload_bytes))$('uploadLimit').textContent=`Maximum upload: ${formatBytes(c.max_upload_bytes)}. Transfers resume after interrupted chunks.`;
      if(c.version)$('version').textContent=c.version;
    }catch(_){/* Older servers keep the explicit configured-retention fallback. */}
  }
  async function initialize(){
    setArea(location.hash.slice(1));steps();resetPrompt();pollJobs();loadConfig().then(refreshHealth);
    try{state.metadata=await request('/static/ui-catalog.json');}catch(_){/* Explicit classification and title fallbacks remain available offline. */}
    loadCatalog();
  }
  initialize();setInterval(pollJobs,4000);setInterval(refreshHealth,30000);
}
if(typeof module!=='undefined'&&module.exports)module.exports={presentation,executesRemotely,supports,pairedCT,blockedReason,cropSupported,validatePrompt,structureInfo,humanize,safeReference,primarySelection,canDeleteJob,reportMarkdown,reportMarkup,GROUPS};

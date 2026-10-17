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
  return allSeries.find(s => s.modality === 'CT' && series?.study_uid && series?.frame_uid && s.study_uid === series.study_uid && s.frame_uid === series.frame_uid);
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
const DEFAULT_MODELS = {CT:['total','nv_segment_ctmr_ct_body'], MR:['total_mr','nv_segment_ctmr_mri_body']};
function preselectSeries(series) {
  if(series.length===1)return [series[0].key];
  return series.filter(s=>['CT','MR'].includes(s.modality)&&s.slices>=20&&
    !/locali[sz]er|scout|surview|topogram/i.test([s.description,...(s.image_type||[])].join(' '))).map(s=>s.key);
}
function defaultSelection(catalog, series, selected = new Set(), edited = false) {
  if(edited)return new Set(selected);
  const names=new Set(series.flatMap(s=>DEFAULT_MODELS[s.modality]||[]));
  return new Set(catalog.filter(t=>names.has(t.name)&&t.enabled!==false&&series.some(s=>supports(t,s))).map(t=>t.name));
}
function batchBlockedReason(task, series, allSeries, reportName = 'nv_reason_ct_report') {
  if(task.prompt&&series.length!==1)return 'Select a single series to use interactive models';
  if(series.some(s=>!blockedReason(task,s,allSeries,reportName)))return '';
  return [...new Set(series.map(s=>blockedReason(task,s,allSeries,reportName)))].join(' · ')||'Select an image series';
}
function seriesPlan(series, selected, catalog, allSeries = series, reportName = 'nv_reason_ct_report') {
  const tasks=catalog.filter(t=>selected.has(t.name)).sort((a,b)=>Number(a.output==='report'||a.name===reportName)-Number(b.output==='report'||b.name===reportName));
  return series.map(s=>{
    const compatible=tasks.filter(t=>supports(t,s)&&!((t.output==='report'||t.name===reportName)&&s.cbct));
    const models=compatible.filter(t=>!batchBlockedReason(t,series,allSeries,reportName)&&!blockedReason(t,s,allSeries,reportName));
    const errors=compatible.map(t=>batchBlockedReason(t,series,allSeries,reportName)||blockedReason(t,s,allSeries,reportName)).filter(Boolean);
    return {series:s,models,reason:errors.length?[...new Set(errors)].join(' · '):models.length?'':
      s.cbct&&tasks.some(t=>t.name===reportName)?'No compatible segmentation model selected; AI reports exclude CBCT':'No compatible model selected. Deselect this series or add a compatible model.'};
  });
}
function primarySelection(selected, action, checked, catalog, series, allSeries = [], reportName = 'nv_reason_ct_report') {
  const next=new Set(selected), rows=Array.isArray(series)?series:[series].filter(Boolean);
  const isReport=t=>t.output==='report'||t.name===reportName;
  if(!checked){
    for(const name of next){
      const task=catalog.find(t=>t.name===name);
      if(action==='report'?name===reportName:name!==reportName&&(!task||!isReport(task)))next.delete(name);
    }
    return next;
  }
  if(action==='segment'){
    if(catalog.some(t=>!isReport(t)&&next.has(t.name)))return next;
    defaultSelection(catalog,rows).forEach(name=>next.add(name));
  }else{
    const task=catalog.find(t=>t.name===reportName);
    if(task&&!batchBlockedReason(task,rows,allSeries,reportName))next.add(reportName);
  }
  return next;
}
function jobTitle(job, reportName = 'nv_reason_ct_report', detail = false) {
  if(job.series?.modality)return `${job.series.description||'Image series'} · ${modalityName(job.series.modality)}`;
  const models=job.models||[];
  return job.tool?'Prostate SBRT crop':models.length===1&&models[0]===reportName?'AI draft report':
    models.includes(reportName)?'Segmentation and AI draft report':detail?'Segmentation results':`${counted(models.length,'model')} · segmentation job`;
}
function groupBatchJobs(jobs) {
  const groups=new Map();
  for(const job of jobs){const key=job.batch_id||job.id;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(job);}
  return [...groups.values()].sort((a,b)=>Math.max(...b.map(j=>j.created||0))-Math.max(...a.map(j=>j.created||0)))
    .flatMap(group=>group.sort((a,b)=>(a.batch_index||0)-(b.batch_index||0)));
}
function canDeleteJob(job) { return !!job && TERMINAL.has(job.status); }
// Catalog flags describe separate licence requirements; descriptive text is not a flag.
function licenceBadges(task) {
  return [task.licensed ? 'Licence key required' : '', task.noncommercial ? 'Non-commercial weights' : ''].filter(Boolean);
}
function counted(count, noun, plural = noun + 's') { return `${count} ${Number(count) === 1 ? noun : plural}`; }
function reportStatusText(job, reportName = 'nv_reason_ct_report') {
  if (!job.report_status) return '';
  const models = job.models || [], reportIndex = models.indexOf(reportName);
  if (job.status === 'failed' && ['failed', 'pending'].includes(job.report_status) && reportIndex > 0) {
    const completed = job.completed_models;
    // Report execution follows every segmentation task. A report may fail before its
    // first progress heartbeat, so completed_models takes precedence over an old index.
    const reachedReport = completed?.includes(reportName) || job.remote_task_active === reportName || job.model_index >= reportIndex + 1;
    const earlierIncomplete = Array.isArray(completed)
      ? models.slice(0, reportIndex).some(name => !completed.includes(name))
      : Number.isInteger(job.model_index) && job.model_index >= 0 && job.model_index < reportIndex + 1;
    if (!reachedReport && earlierIncomplete) return 'AI report: not run (an earlier step failed)';
  }
  return `AI report: ${humanize(job.report_status)}${job.report_reason ? ' · ' + job.report_reason : ''}`;
}
function isReportSection(finding) {
  // Empty anatomical labels emitted by the report parser are navigation, not findings.
  const name = String(finding.heading || '').trim().replace(/:$/, '').replace(/\s+/g, ' ');
  return !String(finding.text || '').trim() && (!finding.organ_status || finding.organ_status === 'unmapped') &&
    /^(?:chest|thorax|abdomen|upper abdomen|pelvis|abdomen(?: and |[ /-])pelvis|chest(?:, | and |[ /-])abdomen(?: and |[ /-])pelvis)$/i.test(name);
}
function findingStatus(status) { return status === 'unmapped' ? 'not checked against segmentation' : status || 'Not checked'; }
function reportSectionNames(region) {
  return new Set((region.sections?.findings || []).filter(isReportSection).map(f => f.heading));
}
function reportMarkdown(report) {
  const lines = ['# NV-Reason-CT draft report', '', '**Research draft. Requires radiologist review.**', '',
    `Status: ${report.status || 'Not supplied'}`, report.reason || '', '', 'Segmentation can be wrong. Coverage checks do not verify diagnostic findings or image quality.'];
  lines.push('', 'Regions covered: ' + (report.covered_regions || (report.regions || []).map(r => r.region)).join(', '));
  for (const region of report.regions || []) {
    lines.push('', `## ${region.region}`, '', '### Technique', ...(region.sections?.technique || []), '', '### Findings');
    for (const f of region.sections?.findings || []) lines.push(isReportSection(f) ? `#### ${f.heading}` : `${f.heading} [${findingStatus(f.organ_status)}]: ${f.text}`);
    lines.push('', '### Impression', ...(region.sections?.impression || []), '', '### Abnormality labels', ...(region.sections?.labels || []));
    if (region.flags?.length) lines.push('', '### Flags', ...region.flags.map(f => JSON.stringify(f)));
    const unchecked = (region.unchecked_headings || []).filter(name => !reportSectionNames(region).has(name));
    if (unchecked.length) lines.push('', 'Not checked against segmentation: ' + unchecked.join('; '));
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
  const isSectionFlag = (flag, region) => flag.section === 'findings' && flag.status === 'unmapped' && reportSectionNames(region).has(flag.heading);
  const regionFlags = region => (region.flags || []).filter(flag => !isSectionFlag(flag, region));
  const allFlags = (report.flags || []).filter(flag => !(report.regions || []).some(region => flag.region === region.region && isSectionFlag(flag, region)));
  const flags = values => `<ul class="report-flags">${values.map(f => `<li><strong>${esc(f.status === 'unmapped' ? findingStatus(f.status) : humanize(f.status || (f.absent_organs ? 'Absent anatomy flag' : 'Report flag')))}</strong>${f.region ? ' · '+esc(f.region) : ''}${f.heading ? ': '+esc(f.heading) : ''}${f.section ? ' · '+esc(f.section) : ''}${Number.isInteger(f.index) ? ' (index '+esc(f.index)+')' : ''}${f.absent_organs?.length ? ': '+esc(f.absent_organs.join(', ')) : ''}${f.reason ? ' · '+esc(f.reason) : ''}</li>`).join('')}</ul>`;
  return `<div class="notice report-banner"><strong>Research AI draft · requires radiologist review</strong><p>Coverage checks do not verify diagnostic findings or image quality.</p></div>
    <p><strong>Status: ${esc(humanize(report.status || 'Not supplied'))}</strong>${report.reason ? `<br>${esc(report.reason)}` : ''}</p>
    <p><strong>Regions covered:</strong> ${(report.covered_regions || (report.regions || []).map(r => r.region)).map(r => esc(humanize(r))).join(', ') || 'No region report generated'}</p>
    ${(report.regions || []).map(r => `<section class="region-report"><h3>${esc(humanize(r.region))}</h3><h4>Technique</h4>${paragraphs(r.sections?.technique)}<h4>Findings</h4>
      ${(r.sections?.findings || []).map(f => isReportSection(f) ? `<h5 class="report-section-heading">${esc(f.heading)}</h5>` : `<div class="finding"><strong>${esc(f.heading)}</strong> <span class="badge ${f.organ_status === 'present' ? '' : 'license'}">${esc(f.organ_status === 'absent' ? 'Withheld · absent anatomy' : f.organ_status === 'partial' ? 'Partial coverage' : findingStatus(f.organ_status))}</span><p>${esc(f.text)}</p></div>`).join('')}
      <h4>Impression</h4>${paragraphs(r.sections?.impression)}<h4>Abnormality labels</h4>${paragraphs(r.sections?.labels)}
      ${regionFlags(r).length ? `<h4>Flags</h4>${flags(regionFlags(r))}` : ''}
      ${r.unchecked_headings?.some(name => !reportSectionNames(r).has(name)) ? `<p class="notice">Not checked against segmentation: ${esc(r.unchecked_headings.filter(name => !reportSectionNames(r).has(name)).join('; '))}.</p>` : ''}</section>`).join('')}
    ${allFlags.length ? `<details><summary>All report flags</summary>${flags(allFlags)}</details>` : ''}
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
  if (Array.isArray(structures)) return {label:`${counted(structures.length,'structure')} · details`, text:structures.join(', ')};
  // Prefer the runtime label list to a packaged list when TotalSegmentator supplies one.
  if (String(task.description).startsWith('TotalSegmentator ') && structures !== 'Task-specific TotalSegmentator structures') {
    const count = structures.split(',').filter(x => x.trim()).length;
    return {label:`${counted(count,'structure')} · details`, text:structures};
  }
  if (Number.isInteger(meta.count)) return {label:`${counted(meta.count,'structure')} · details`, text:meta.structures?.join(', ') || structures};
  const count = structures.match(/^(\d+) (?:foreground labels|organs|major|whole-body|bilateral|head-and-neck|anatomy classes)/);
  return {label:count ? `${counted(count[1],'structure')} · details` : 'Structures · count not supplied', text:structures};
}
function modalityName(m) { return m === 'PT' ? 'PET' : m; }
function esc(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function formatBytes(n) { if (!Number.isFinite(n)) return '—'; const units = ['B','KiB','MiB','GiB']; let i=0; while(n>=1024 && i<3){n/=1024;i++;} return `${n.toFixed(i?1:0)} ${units[i]}`; }
function jobProgress(job, now = Date.now()/1000) {
  const active=!TERMINAL.has(job.status),end=active?now:(job.finished||job.completed||job.updated||job.created||now);
  const task=job.current_task||'',stage=job.stage||'';
  const timing=(job.task_timings||{})[task];
  const runningStage=active&&job.status==='processing'&&Boolean(task)&&(!timing||timing.status==='running');
  const known=runningStage&&Number.isFinite(job.stage_done)&&Number.isFinite(job.stage_total)&&job.stage_total>0;
  const counts=known?` · ${job.stage_done}/${job.stage_total}${['rtstruct_contours','rtstruct_validation'].includes(job.stage_key)?' structures':''}`:'';
  const activity=job.task_remote?job.last_activity:Math.max(job.last_activity||0,job.last_cpu_activity||0);
  const age=activity?Math.max(0,Math.floor(end-activity)):null;
  const text=runningStage?[task,stage].filter(Boolean).join(': ')+counts:(job.detail||stage);
  const elapsed=`${active?'Elapsed':'Duration'} ${duration(end-(job.created||end))}`;
  const stageTime=runningStage&&job.stage_started?`stage ${duration(end-job.stage_started)}`:'';
  const last=runningStage&&age!==null?`last activity ${age} s ago`:'';
  const warning=runningStage&&(job.stall_warning||(age!==null&&age>=(job.stall_threshold||1200)))?
    `No ${job.task_remote?'worker log':'log or CPU'} activity for ${duration(age||0)}. Automatic recovery is pending.`:'';
  return {text,elapsed,stageTime,last,warning,activeStage:runningStage,
    percent:known?Math.max(0,Math.min(100,100*job.stage_done/job.stage_total)):
      (Number.isFinite(job.progress)?Math.max(0,Math.min(100,job.progress)):null)};
}
function modelChecklist(job, now = Date.now()/1000) {
  return (job.models||[]).map(name=>{
    const timing=(job.task_timings||{})[name]||{},done=(job.completed_models||[]).includes(name)||(job.no_structures_found||[]).includes(name)||timing.status==='done';
    const running=job.status==='processing'&&job.current_task===name;
    const failed=timing.status==='failed'||(job.status==='failed'&&job.current_task===name);
    const status=done?'done':running?'running':failed?'failed':'waiting';
    const seconds=done?timing.duration:running&&timing.started?now-timing.started:null;
    return {name,status,text:`${done?'✓ ':''}${name} · ${status}${Number.isFinite(seconds)?' · '+duration(seconds):''}`};
  });
}

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
  const state = {step:0,upload:null,series:null,seriesKeys:[],modelsEdited:false,catalog:[],metadata:{overrides:{}},selected:new Set(),config:null,health:null,catalogReady:false,starting:false};
  let seriesViewer=null, jobViewer=null, detailId=null, detailToken=0, detailLoaded=false, detailInfo=null, detailReport=null, detailLoading=false;
  const stepPanels=[$('uploadStep'),$('seriesStep'),$('modelsStep'),$('reviewStep')];
  const stepTitles=[$('uploadTitle'),$('seriesTitle'),$('modelsTitle'),$('reviewTitle')];
  const selectionTray = document.querySelector('.selection-tray');
  function reserveSelectionSpace() {
    const height = selectionTray.getBoundingClientRect().height;
    document.documentElement.style.setProperty('--selection-tray-space', height ? `${Math.ceil(height) + 28}px` : '0px');
  }
  new ResizeObserver(reserveSelectionSpace).observe(selectionTray);
  window.addEventListener('resize', reserveSelectionSpace);

  const uploads={
    seg:{files:[],session:null,busy:false,selection:$('selection'),drop:$('dropZone'),filesInput:$('files'),folderInput:$('folder'),buttons:[$('chooseFiles'),$('chooseFolder')],progress:$('uploadProgress'),bar:$('uploadBar'),text:$('uploadText'),error:$('uploadError')},
    crop:{files:[],session:null,busy:false,selection:$('cropSelection'),drop:$('cropDropZone'),filesInput:$('cropFiles'),folderInput:$('cropFolder'),buttons:[$('chooseCropFiles'),$('chooseCropFolder')],progress:$('cropProgress'),bar:$('cropBar'),text:$('cropText'),error:$('cropError')}
  };
  let prompt={positive_points:[],negative_points:[],box:null,text:''}, boxCorner=null, previewToken=0;
  let promptSeries=null, promptSeriesKey=null, promptTransform=null;
  const reportTask=()=>state.config?.report_task || 'nv_reason_ct_report';
  const chosen=()=>[...state.selected].map(name=>state.catalog.find(t=>t.name===name)||{name,enabled:false,modalities:[],status:'Model no longer available in catalogue'});
  const prompted=()=>chosen().filter(t=>t.prompt);
  const meta=t=>presentation(t,state.metadata);
  const isRemote=t=>executesRemotely(t,state.config,state.health);
  const selectedSeries=()=>state.upload?.series.filter(s=>state.seriesKeys.includes(s.key))||[];
  const reason=t=>batchBlockedReason(t,selectedSeries(),state.upload?.series||[],reportTask());
  const plan=()=>seriesPlan(selectedSeries(),state.selected,state.catalog,state.upload?.series||[],reportTask());
  const cropAvailable=()=>selectedSeries().some(s=>cropSupported(s,state.selected,state.catalog,state.config,state.health));
  function applyDefaults(){state.selected=defaultSelection(state.catalog,selectedSeries(),state.selected,state.modelsEdited);}
  function showError(el,message){el.textContent=message;el.hidden=!message;}
  function setArea(area, focus=false){
    const match = /^jobs\/([A-Za-z0-9-]+)$/.exec(area);
    const id = match?.[1] || null;
    if(match) area='detail';
    if(!['new','jobs','tools','detail'].includes(area)) area='new';
    if(detailId !== id){
      detailId=id; ++detailToken; detailLoaded=false; detailLoading=false; detailInfo=null; detailReport=null;
      jobViewer?.destroy();jobViewer=null;$('jobResults').hidden=true;showError($('jobDetailError'),'');
      $('batchSiblings').replaceChildren();$('batchSiblings').hidden=true;
      $('deleteJob').disabled=true;$('jobDetailTitle').textContent='Job details';$('jobDetailMeta').textContent=id||'';
      $('jobDetailModels').replaceChildren();$('jobDetailLog').textContent='';$('jobDetailStall').hidden=true;$('jobDetailLiveLog').hidden=true;$('jobDetailStatus').textContent='Loading…';$('jobDetailProgress').textContent='';$('jobDetailReportStatus').textContent='';$('jobDetailErrors').hidden=true;$('retryDetail').hidden=true;$('jobDetailBar').removeAttribute('value');
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
    $('uploadSummary').textContent=state.upload?`${state.upload.series.length} series · ${counted(state.upload.series.reduce((n,s)=>n+s.slices,0),'image')}`:'Files, folder or ZIP';
    $('seriesSummary').textContent=state.seriesKeys.length?counted(state.seriesKeys.length,'series','series'):'Choose image series';
    $('reviewSummary').textContent=state.seriesKeys.length?counted(state.seriesKeys.length,'job')+' · '+counted(state.selected.size,'model'):'Confirm the plan';
    $('modelsSummary').textContent=state.selected.size?`${counted(state.selected.size,'model')} selected`:'Choose the outputs';
    document.querySelectorAll('[data-step]').forEach(b=>{const n=Number(b.dataset.step);b.disabled=state.starting || (n===1&&!state.upload)||(n===2&&!state.seriesKeys.length)||(n===3&&(!state.seriesKeys.length||!state.selected.size));b.classList.toggle('done',n<state.step);if(n===state.step)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  }
  function goStep(n,focus=true){
    if((n>0&&!state.upload)||(n>1&&!state.seriesKeys.length)||(n>2&&!state.selected.size)||state.starting)return;
    state.step=n;stepPanels.forEach((p,i)=>p.hidden=i!==n);steps();
    if(n===1&&state.series)loadSeriesViewer();if(n===2)renderCatalog();if(n===3)renderReview();
    if(focus){stepTitles[n].focus({preventScroll:true});window.scrollTo({top:0,behavior:'instant'});}
  }
  document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>goStep(Number(b.dataset.step))));
  document.querySelectorAll('[data-back]').forEach(b=>b.addEventListener('click',()=>goStep(Number(b.dataset.back))));
  $('toSeries').onclick=()=>goStep(1);
  $('toModels').onclick=()=>{if(!state.seriesKeys.length){showError($('seriesError'),'Select at least one image series.');return;}goStep(2);};
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
    c.selection.textContent=`${counted(files.length,'file')} · ${formatBytes(files.reduce((n,f)=>n+f.size,0))}`;
    showError(c.error,'');c.progress.hidden=true;
    if(c===uploads.seg){state.upload=null;state.series=null;state.seriesKeys=[];state.modelsEdited=false;state.selected.clear();seriesViewer?.destroy();seriesViewer=null;$('assessment').textContent='';$('toSeries').hidden=true;resetPrompt();goStep(0,false);await inspect();}
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
      state.upload=result;state.series=null;state.seriesKeys=preselectSeries(result.series);state.modelsEdited=false;state.selected.clear();applyDefaults();
      const valid=result.series.reduce((n,s)=>n+s.slices,0);
      $('assessment').textContent=`${counted(valid,'valid image file')} in ${result.series.length} series · ${counted(result.ignored || 0,'ignored file')}`;
      uploadStatus(c,'Upload and inspection complete.');$('toSeries').hidden=false;
      selectSeries(result.series.find(s=>state.seriesKeys.includes(s.key))||result.series[0]);
      renderSeries();steps();if(location.hash===''||location.hash==='#new')goStep(1);
    }catch(error){if(error.status===404)c.session=null;showError(c.error,error.message);$('resumeUpload').hidden=false;}
    finally{uploadBusy(c,false);}
  }
  $('resumeUpload').onclick=inspect;
  function previewURL(s,index){return `/api/uploads/${encodeURIComponent(state.upload.upload_id)}/preview?series_key=${encodeURIComponent(s.key)}&index=${index}`;}
  function seriesWarning(s){
    if(s.cbct)return `AI draft report unavailable for CBCT: ${s.cbct_reason||'CBCT detected'}.`;
    if(s.modality!=='PT')return '';
    const ct=pairedCT(s,state.upload.series);
    return ct?`Paired CT found: ${ct.description||'CT series'} (${counted(ct.slices,'image')}). Matching Study and Frame of Reference does not establish registration.`:
      'No matching CT in this upload. Paired PET models require a CT in the same Study and Frame of Reference; PET-only models remain available.';
  }
  function renderSeries(){
    const focused=document.activeElement,focusKey=focused?.name==='series_keys'?focused.value:null,focusPreview=focused?.dataset.preview;
    $('seriesCount').textContent=`${state.seriesKeys.length} of ${state.upload.series.length} series selected`;
    $('series').innerHTML=state.upload.series.map((s,i)=>`<article class="series-card ${state.series?.key===s.key?'is-previewed':''}" data-series-index="${i}"><input type="checkbox" name="series_keys" value="${esc(s.key)}" aria-label="Select ${esc(s.description||s.modality+' series')}" ${state.seriesKeys.includes(s.key)?'checked':''}><div class="series-info"><span class="pill">${esc(modalityName(s.modality))}</span>${s.cbct?' <span class="badge license">CBCT detected</span>':''}<strong>${esc(s.description||'No series description')}</strong><span>${counted(s.slices,'image')} · ${s.rows} × ${s.columns}</span><br><small class="muted">Study date ${esc(studyDate(s.study_date))}</small>${seriesWarning(s)?`<p class="small series-warning">${esc(seriesWarning(s))}</p>`:''}<button type="button" data-preview="${i}" aria-pressed="${state.series?.key===s.key}">${state.series?.key===s.key?'Previewing':'Preview'}</button></div></article>`).join('');
    $('series').querySelectorAll('input').forEach(input=>input.onchange=()=>{
      state.seriesKeys=state.upload.series.filter(s=>s.key===input.value?input.checked:state.seriesKeys.includes(s.key)).map(s=>s.key);
      selectionChanged();
    });
    $('series').querySelectorAll('.series-card').forEach(card=>card.onclick=e=>{
      if(e.target.closest('input'))return;
      selectSeries(state.upload.series[Number(card.dataset.seriesIndex)]);renderSeries();
    });
    if(focusKey)[...$('series').querySelectorAll('input')].find(input=>input.value===focusKey)?.focus({preventScroll:true});
    else if(focusPreview!==undefined)$('series').querySelector(`[data-preview="${focusPreview}"]`)?.focus({preventScroll:true});
    loadSeriesViewer();
  }
  function selectionChanged(){
    if(state.seriesKeys.length===1)selectSeries(selectedSeries()[0]);
    applyDefaults();$('prostatePostprocess').checked=false;resetPrompt();
    showError($('seriesError'),'');renderSeries();renderSelection();
  }
  $('selectAllSeries').onclick=()=>{state.seriesKeys=state.upload.series.filter(s=>['CT','MR'].includes(s.modality)).map(s=>s.key);selectionChanged();};
  $('clearSeries').onclick=()=>{state.seriesKeys=[];selectionChanged();};
  function loadSeriesViewer(){
    if(!state.series||!state.upload)return;
    if(!seriesViewer)seriesViewer=new DicomUI.Viewer($('seriesViewer'));
    const base=`/api/v1/uploads/${encodeURIComponent(state.upload.upload_id)}/series/${encodeURIComponent(state.series.key)}/viewer`;
    if(seriesViewer.base!==base)seriesViewer.load(base);
  }
  function selectSeries(s){
    if(state.series?.key!==s.key){state.series=s;resetPrompt();$('promptSlice').max=s.slices-1;$('promptSlice').value=Math.floor(s.slices/2);$('promptColumn').max=s.columns-1;$('promptRow').max=s.rows-1;}
    loadSeriesViewer();
  }
  function badges(t){
    const result=licenceBadges(t).map(text=>[text,'license']);if(t.prompt)result.push(['Interactive prompt needed','prompt']);if(t.output==='report')result.push(['Research AI draft','ai']);if(isRemote(t))result.push(['Runs on remote GPU','remote']);
    if(t.requires_ct)result.push(['Paired CT required','']);
    if(Object.values(DEFAULT_MODELS).flat().includes(t.name))result.push(['Default','default']);
    return result.map(([text,cls])=>`<span class="badge ${cls}">${text}</span>`).join('');
  }
  function availabilityNote(t, blocked){
    if(!blocked)return '';
    if(t.missing?.length || t.remote_preflight_error)return `<details class="reason-details"><summary>Unavailable: ${esc(t.status || 'unavailable')} · details</summary><p class="reason">${esc(blocked)}</p></details>`;
    return `<p class="reason">Unavailable: ${esc(blocked)}</p>`;
  }
  function modelRow(t){
    const m=meta(t),blocked=reason(t),structures=structureInfo(t,m),reference=safeReference(t.reference);
    const applies=selectedSeries().filter(s=>supports(t,s)&&!((t.output==='report'||t.name===reportTask())&&s.cbct));
    const scope=new Set(selectedSeries().map(s=>s.modality)).size>1?[...new Set(applies.map(s=>s.modality))].map(mod=>`${applies.filter(s=>s.modality===mod).length} ${modalityName(mod)} series`).join(' · '):'';
    return `<article class="model-row ${blocked?'is-unavailable':''}"><input type="checkbox" name="models" value="${esc(t.name)}" aria-label="Select ${esc(m.title)}" ${state.selected.has(t.name)?'checked':''} ${blocked?'disabled':''}><div><label class="model-title" data-toggle-model="${esc(t.name)}">${esc(m.title)}</label><p class="model-purpose">${esc(m.purpose)}</p><div class="badges">${badges(t)}${scope?`<span class="badge">${esc(scope)}</span>`:''}</div>${availabilityNote(t,blocked)}<details class="model-details"><summary>${esc(structures.label)}</summary><div class="detail-body"><p><strong>Output:</strong> <span class="structures">${esc(structures.text)}</span></p><p>${esc(t.description || '')}</p>${t.limitations?`<p><strong>Limitations:</strong> ${esc(t.limitations)}</p>`:''}${t.license||t.license_note?`<p><strong>Licence:</strong> ${esc([t.license,t.license_note].filter(Boolean).join(' · '))}</p>`:''}<p><strong>Task:</strong> <code>${esc(t.name)}</code> · <strong>Catalog status:</strong> ${esc(t.status||'available')}</p>${t.clinical_tags?.length?`<p><strong>Clinical tags:</strong> ${esc(t.clinical_tags.map(x=>x.label).join(', '))}</p>`:''}${reference?`<a href="${esc(reference)}" target="_blank" rel="noopener noreferrer">Model reference (external)</a>`:''}</div></details></div></article>`;
  }
  function renderCatalog(){
    if(!state.seriesKeys.length)return;
    $('modalityLabel').textContent=counted(state.seriesKeys.length,'series','series');
    $('modelNote').textContent='Defaults are selected for CT and MR. Review the models for each series before starting.';
    const site=$('clinicalFilter').value,q=$('modelSearch').value.trim().toLowerCase(),show=$('showUnavailable').checked;
    const compatible=state.catalog.filter(t=>selectedSeries().some(s=>supports(t,s))),available=compatible.filter(t=>!reason(t));
    const visible=compatible.filter(t=>(show||t.enabled!==false)&&(!site||(t.clinical_tags||[]).some(x=>x.id===site))&&(!q||[t.name,meta(t).title,meta(t).purpose,t.description,t.structures,...(t.clinical_tags||[]).map(x=>x.label)].join(' ').toLowerCase().includes(q)));
    const closed=new Set([...$('modelGroups').querySelectorAll('details[data-group]:not([open])')].map(el=>el.dataset.group));
    const groups=GROUPS.map(([id,title])=>({id,title,tasks:visible.filter(t=>meta(t).group===id)})).filter(g=>g.tasks.length);
    $('modelResults').textContent=state.catalogReady?`${counted(visible.length,'model')} shown · ${available.length} available for selected series · ${compatible.length-available.length} unavailable`:'';
    $('modelGroups').innerHTML=groups.map((g,i)=>`<details class="model-group" data-group="${g.id}" ${q||site||!closed.has(g.id)?'open':''}><summary>${g.title}<span class="group-count">${counted(g.tasks.length,'model')}</span></summary>${g.tasks.map(modelRow).join('')}</details>`).join('')||(state.catalogReady?'<p class="empty">No models match these filters. Change the search or show unavailable models.</p>':'');
    renderSelection();
  }
  function toggleModel(name){const task=state.catalog.find(t=>t.name===name);if(!task||reason(task))return;state.modelsEdited=true;if(state.selected.has(name))state.selected.delete(name);else state.selected.add(name);showError($('modelError'),'');syncModelInputs();renderSelection();}
  function syncModelInputs(){document.querySelectorAll('[name=models]').forEach(i=>i.checked=state.selected.has(i.value));}
  $('modelGroups').addEventListener('change',e=>{if(e.target.name==='models')toggleModel(e.target.value);});
  $('modelGroups').addEventListener('click',e=>{const label=e.target.closest('[data-toggle-model]');if(label)toggleModel(label.dataset.toggleModel);});
  $('selectedChips').addEventListener('click',e=>{const b=e.target.closest('[data-remove]');if(b){state.modelsEdited=true;state.selected.delete(b.dataset.remove);syncModelInputs();renderSelection();}});
  $('clearModels').onclick=()=>{state.modelsEdited=true;state.selected.clear();syncModelInputs();renderSelection();};
  $('modelSearch').oninput=renderCatalog;$('clinicalFilter').onchange=renderCatalog;$('showUnavailable').onchange=renderCatalog;
  function renderActions(){
    const report=state.catalog.find(t=>t.name===reportTask());
    const segments=chosen().filter(t=>t.output!=='report'&&t.name!==reportTask());
    const defaults=[...defaultSelection(state.catalog,selectedSeries())];
    const segmentReason=segments.length||defaults.length?'':state.catalogReady?'Choose an available segmentation model from the catalogue.':'Loading model catalog…';
    $('segmentAction').checked=segments.length>0;$('segmentAction').disabled=!!segmentReason;
    $('segmentCard').classList.toggle('selected',segments.length>0);
    const defaultLabels={total:'total',total_mr:'total_mr',nv_segment_ctmr_ct_body:'NV-Segment-CTMR CT body',nv_segment_ctmr_mri_body:'NV-Segment-CTMR MRI body'};
    $('segmentDefault').textContent=[...new Set(selectedSeries().map(s=>s.modality))].map(mod=>{
      const names=(DEFAULT_MODELS[mod]||[]).filter(name=>defaults.includes(name));
      return names.length?`${modalityName(mod)}: ${names.map(name=>defaultLabels[name]).join(' + ')}`:
        mod==='PT'?'PET: choose models below':`${modalityName(mod)}: no default models available`;
    }).join(' · ');
    $('segmentBadges').innerHTML='';
    $('segmentSelection').textContent=segments.length?segments.map(t=>meta(t).title).join(' · '):'Choose models below or restore defaults.';
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
    state.modelsEdited=true;state.selected=primarySelection(state.selected,action,$(id).checked,state.catalog,selectedSeries(),state.upload?.series||[],reportTask());
    showError($('modelError'),'');syncModelInputs();renderSelection();
  };
  $('restoreDefaults').onclick=()=>{state.modelsEdited=false;applyDefaults();showError($('modelError'),'');syncModelInputs();renderSelection();};
  function renderSelection(){
    renderActions();
    $('selectionCount').textContent=`${counted(state.selected.size,'model')} selected`;
    $('selectedChips').innerHTML=chosen().map(t=>`<button type="button" class="chip" data-remove="${esc(t.name)}" aria-label="Remove ${esc(meta(t).title)}">${esc(meta(t).title)} <span aria-hidden="true">×</span></button>`).join('')||'<span class="small muted">Select models to build the plan.</span>';
    steps();
  }
  async function loadCatalog(){
    $('catalogStatus').textContent='Loading model catalog. Worker checks can take a few minutes…';$('reloadCatalog').hidden=true;
    try{const data=await request('/api/v1/models');state.catalog=data.tasks;state.catalogReady=true;
      const previous=$('clinicalFilter').value;$('clinicalFilter').innerHTML='<option value="">All tumour sites</option>'+(data.clinical_tags||[]).map(t=>`<option value="${esc(t.id)}">${esc(t.label)}</option>`).join('');$('clinicalFilter').value=previous;
      applyDefaults();
      $('catalogStatus').textContent='';renderCatalog();if(state.step===3)renderReview();
    }catch(error){$('catalogStatus').textContent='Model catalog could not be loaded: '+error.message;$('reloadCatalog').hidden=false;}
  }
  $('reloadCatalog').onclick=loadCatalog;
  function renderReview(){
    if(state.seriesKeys.length===1)selectSeries(selectedSeries()[0]);
    const tasks=chosen().sort((a,b)=>Number(a.output==='report')-Number(b.output==='report')),plans=plan();
    const invalid=tasks.filter(t=>reason(t));
    $('reviewContent').innerHTML=plans.map(p=>`<article class="review-card" data-plan-series="${esc(p.series.key)}"><h3>${esc(p.series.description||'No series description')}</h3><p><span class="pill">${esc(modalityName(p.series.modality))}</span> · ${counted(p.series.slices,'image')}</p><ul>${p.models.map(t=>`<li><strong>${esc(meta(t).title)}</strong><span class="small muted">${t.output==='report'?'AI draft report · JSON, Markdown and HTML':'RTSTRUCT contours'}${isRemote(t)?' · remote GPU':''}${t.requires_ct?' · paired CT included':''}</span><div class="badges">${badges(t)}</div></li>`).join('')}</ul>${p.reason?`<p class="error">${esc(p.reason)}</p>`:''}<button type="button" data-edit-series>Edit series selection</button></article>`).join('');
    $('reviewContent').querySelectorAll('[data-edit-series]').forEach(b=>b.onclick=()=>goStep(1));
    showError($('startError'),invalid.map(t=>`${meta(t).title}: ${reason(t)}`).join(' · '));
    const crop=cropAvailable();$('postprocessPanel').hidden=!crop;if(!crop)$('prostatePostprocess').checked=false;
    $('promptPanel').hidden=!prompted().length;
    if(prompted().length){$('promptGuidance').innerHTML=prompted().map(t=>`<p>${esc(PROMPT_HELP[t.name]||`${meta(t).title}: ${t.prompt}. ${t.limitations||t.description||''}`)}</p>`).join('');loadPromptSlice();}
    const warnings=[];if(tasks.some(t=>t.output==='report'))warnings.push('AI output is a research draft requiring radiologist review. Diagnostic chest/abdomen CT only; coverage, HU or resource checks can refuse generation. Reports run after segmentation.');
    if(tasks.some(t=>t.licensed))warnings.push('Selected models require a licence key. See the licence terms in model details.');
    if(tasks.some(t=>t.noncommercial))warnings.push('Selected models have non-commercial weights. See the licence terms in model details.');
    if(tasks.some(t=>t.requires_ct))warnings.push('The matching CT is paired by Study and Frame of Reference. This does not establish registration.');
    $('reviewWarnings').textContent=warnings.join(' ');$('reviewWarnings').hidden=!warnings.length;
    const runs=plans.reduce((n,p)=>n+p.models.length,0);
    $('startJob').textContent=`Start ${counted(plans.length,'job')} · ${counted(runs,plans.length===1?'model':'model run')}`;
    $('startJob').disabled=state.starting||!plans.length||plans.some(p=>p.reason||!p.models.length)||invalid.length>0;
  }
  function resetPrompt(){prompt={positive_points:[],negative_points:[],box:null,text:''};boxCorner=null;$('promptText').value='';writePrompt();showError($('promptError'),'');}
  function writePrompt(){ $('promptJson').value=JSON.stringify(prompt,null,2);$('promptText').value=prompt.text||'';drawPrompt(); }
  function readPrompt(){try{const p=JSON.parse($('promptJson').value);if(!p||typeof p!=='object'||Array.isArray(p))throw Error();return p;}catch(_){throw Error('Prompt JSON must be a valid JSON object.');}}
  async function loadPromptSlice(){
    if(!state.series||!state.upload)return;const index=Number($('promptSlice').value),token=++previewToken;
    $('promptSliceLabel').textContent=`${index+1} of ${state.series.slices} (index ${index})`;$('previewError').hidden=true;
    $('promptImage').style.visibility='hidden';promptTransform=null;$('promptOverlay').replaceChildren();
    $('promptCanvas').querySelectorAll('[data-p-edge]').forEach(el=>{el.hidden=true;});
    const key=`${state.upload.upload_id}/${state.series.key}`;
    try {
      if(promptSeriesKey!==key){
        const info=await request(`/api/v1/uploads/${encodeURIComponent(state.upload.upload_id)}/series/${encodeURIComponent(state.series.key)}/viewer`);
        if(token!==previewToken)return;promptSeries=info.series;promptSeriesKey=key;
      }
    }catch(_){if(token===previewToken)$('previewError').hidden=false;return;}
    $('promptImage').onload=()=>{if(token===previewToken)drawPrompt();};$('promptImage').onerror=()=>{if(token===previewToken)$('previewError').hidden=false;};
    $('promptImage').src=previewURL(state.series,index);drawPrompt();
  }
  function drawPrompt(){
    const s=state.series;if(!s)return;const index=Number($('promptSlice').value);let html='';
    $('promptStatus').textContent=`${Array.isArray(prompt.positive_points)?prompt.positive_points.length:0} positive · ${Array.isArray(prompt.negative_points)?prompt.negative_points.length:0} negative · ${prompt.box?'1 box':'no box'}${boxCorner?' · Choose the second corner on this slice.':''}`;
    if(promptSeriesKey!==`${state.upload?.upload_id}/${s.key}`)return;
    const display=DicomUI.displayOrientation(promptSeries,$('promptStandard').checked);
    const width=$('promptCanvas').getBoundingClientRect().width;if(!width)return;
    const height=width*(Math.abs(display.b)*s.columns+Math.abs(display.d)*s.rows)/(Math.abs(display.a)*s.columns+Math.abs(display.c)*s.rows);
    $('promptCanvas').style.height=`${height}px`;
    const t=promptTransform=DicomUI.imageTransform(s.columns,s.rows,width,height,{zoom:1,x:0,y:0},display);
    Object.assign($('promptImage').style,{position:'absolute',left:'0',top:'0',width:`${s.columns}px`,height:`${s.rows}px`,transformOrigin:'0 0',
      transform:`matrix(${t.a},${t.b},${t.c},${t.d},${t.x-.5*(t.a+t.c)},${t.y-.5*(t.b+t.d)})`});
    if($('promptImage').complete&&$('promptImage').naturalWidth)$('promptImage').style.visibility='visible';
    const position=p=>{const [x,y]=DicomUI.pixelToScreen(p,t);return `left:${x}px;top:${y}px`;};
    for(const [key,cls] of [['positive_points','positive'],['negative_points','negative']]){
      if(!Array.isArray(prompt[key]))continue;for(const p of prompt[key]){if(!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isFinite(v))||p[2]!==index)continue;html+=`<span class="prompt-marker ${cls}" style="${position(p)}">${cls==='positive'?'+':'−'}</span>`;}
    }
    const box=prompt.box;if(Array.isArray(box)&&box.length===2&&box.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite))&&index>=Math.min(box[0][2],box[1][2])&&index<=Math.max(box[0][2],box[1][2])){
      const [a,b]=box.map(p=>DicomUI.pixelToScreen(p,t));
      html+=`<span class="prompt-box" style="left:${Math.min(a[0],b[0])}px;top:${Math.min(a[1],b[1])}px;width:${Math.abs(b[0]-a[0])}px;height:${Math.abs(b[1]-a[1])}px"></span>`;
    }
    if(boxCorner&&boxCorner[2]===index)html+=`<span class="prompt-marker" style="${position(boxCorner)}">1</span>`;
    $('promptOverlay').innerHTML=html;
    const markers=DicomUI.orientationMarkers(promptSeries,display),positions={top:[width/2,14],bottom:[width/2,height-14],left:[18,height/2],right:[width-18,height/2]};
    $('promptCanvas').querySelectorAll('[data-p-edge]').forEach(el=>{const edge=el.dataset.pEdge;el.hidden=false;el.textContent=markers[edge];el.style.left=`${positions[edge][0]}px`;el.style.top=`${positions[edge][1]}px`;});
  }
  function addPoint(point){
    try{prompt=readPrompt();const s=state.series;if(point.some((v,i)=>!Number.isFinite(v)||v<0||v>=[s.columns,s.rows,s.slices][i]))throw Error('The point must be within this series.');
      const mode=$('promptMode').value;if(mode==='box'){if(!boxCorner)boxCorner=point;else{if(boxCorner[2]!==point[2])throw Error('Both box corners must be on the same slice. Clear the prompt to restart the box.');prompt.box=[boxCorner,point];boxCorner=null;}}
      else{const key=mode+'_points';if(!Array.isArray(prompt[key]))prompt[key]=[];prompt[key].push(point);}
      showError($('promptError'),'');writePrompt();
    }catch(error){showError($('promptError'),error.message);}
  }
  $('promptImage').onclick=e=>{if(!$('previewError').hidden||!e.target.naturalWidth||!promptTransform)return;const s=state.series,r=$('promptCanvas').getBoundingClientRect(),p=DicomUI.screenToPixel([e.clientX-r.left,e.clientY-r.top],promptTransform);addPoint([Math.max(0,Math.min(s.columns-1,Math.floor(p[0]+.5))),Math.max(0,Math.min(s.rows-1,Math.floor(p[1]+.5))),Number($('promptSlice').value)]);};
  $('promptStandard').onchange=drawPrompt;
  new ResizeObserver(()=>drawPrompt()).observe($('promptCanvas'));
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
    if(plan().some(p=>p.reason||!p.models.length)){renderReview();return;}
    if(!state.selected.size){goStep(2);showError($('modelError'),'Choose segmentation, an AI draft report, or both.');return;}
    let promptValue='';if(prompted().length){try{prompt=readPrompt();const error=boxCorner?'Complete the second box corner or clear the prompt.':validatePrompt(prompt,prompted(),state.series);if(error)throw Error(error);promptValue=JSON.stringify(prompt);}catch(error){showError($('promptError'),error.message);$('promptError').scrollIntoView({block:'center'});return;}}
    state.starting=true;steps();$('startJob').disabled=true;$('startJob').textContent='Starting…';
    let failure=null;
    try{const form=new FormData();state.seriesKeys.forEach(key=>form.append('series_keys',key));chosen().forEach(t=>form.append('models',t.name));if(promptValue)form.append('prompt_json',promptValue);
      if($('prostatePostprocess').checked&&cropAvailable())form.append('postprocess','prostate_sbrt_crop');
      const result=await request(`/api/v1/uploads/${encodeURIComponent(state.upload.upload_id)}/jobs`,{method:'POST',body:form});
      result.ids.forEach(id=>{rememberJob(id);newJobs.add(id);});$('jobNotice').textContent=result.ids.length===1?'Job queued. Follow progress and download results here.':`${result.ids.length} jobs queued from one upload; they run one after another`;$('jobNotice').hidden=false;
      state.upload=null;state.series=null;state.seriesKeys=[];state.modelsEdited=false;state.selected.clear();seriesViewer?.destroy();seriesViewer=null;uploads.seg.files=[];uploads.seg.session=null;uploads.seg.selection.textContent='No files selected';uploads.seg.progress.hidden=true;$('assessment').textContent='';$('toSeries').hidden=true;resetPrompt();
      location.hash=result.ids.length===1?'jobs/'+encodeURIComponent(result.ids[0]):'jobs';pollJobs();
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
  const jobRows=new Map(),jobData=new Map(),newJobs=new Set();let polling=false;
  function updateText(el,text){if(el.textContent!==text)el.textContent=text;}
  function updateJob(j){
    let row=jobRows.get(j.id);if(!row){row=$('jobTemplate').content.firstElementChild.cloneNode(true);row.dataset.job=j.id;jobRows.set(j.id,row);row.querySelector('[data-field=retry]').onclick=()=>retryJob(j.id,row);}
    const field=name=>row.querySelector(`[data-field="${name}"]`),busy=!TERMINAL.has(j.status),models=j.models||[],total=j.model_total||models.length,pct=jobProgress(j).percent;
    updateText(field('title'),jobTitle(j,reportTask()));row.classList.toggle('is-new',newJobs.has(j.id));row.dataset.batch=j.batch_id||j.id;
    field('title').href=field('open').href='#jobs/'+encodeURIComponent(j.id);
    const elapsed=(busy?Date.now()/1000:(j.finished||j.completed||j.updated||j.created))-j.created;
    updateText(field('meta'),`${new Date(j.created*1000).toLocaleString()} · ${busy?'Elapsed':'Duration'} ${duration(elapsed)} · ${j.batch_size>1?`Series ${j.batch_index} of ${j.batch_size} · `:''}${j.id}`);
    updateText(field('status'),humanize(j.status));field('status').className='pill '+(TERMINAL.has(j.status)||['processing','transferring','retrieving','queued'].includes(j.status)?j.status:'');
    const stage=jobProgress(j),hasStage=stage.activeStage;updateText(field('stage'),hasStage?[stage.text,stage.stageTime,stage.warning||stage.last].filter(Boolean).join(' · '):'');field('stage').classList.toggle('stall-warning',Boolean(stage.warning));field('detail').hidden=hasStage;
    updateText(field('models'),models.map(name=>meta(state.catalog.find(t=>t.name===name)||{name}).title).join(' · '));updateText(field('detail'),j.detail||'');
    field('progressWrap').hidden=!busy&&pct===null;if(pct===null)field('progress').removeAttribute('value');else field('progress').value=pct;
    const progress=[];if(j.model_index&&total)progress.push(`Model ${j.model_index} of ${total}`);else if(busy&&total)progress.push(`${counted(total,'model')} queued`);
    if(j.total_bytes&&(j.status==='transferring'||j.status==='retrieving'))progress.push(`${formatBytes(j.transferred_bytes||0)} / ${formatBytes(j.total_bytes)}`);
    if(pct!==null)progress.push(`${Math.round(pct)}%`);updateText(field('progressText'),progress.join(' · '));
    field('report').hidden=!j.report_status;updateText(field('report'),reportStatusText(j,reportTask()));
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
      const grouped=groupBatchJobs(jobs);jobs.splice(0,jobs.length,...grouped);const present=new Set(jobs.map(j=>j.id));
      if(listSucceeded){for(const [id,row] of jobRows){if(!present.has(id)){row.remove();jobRows.delete(id);jobData.delete(id);}}}
      if(failed)showError($('jobsError'),`${failed} job detail request${failed===1?'':'s'} failed. Retained rows may be out of date; the next refresh will retry.`);
      if(jobs.length){$('jobs').querySelector('.empty')?.remove();jobs.forEach((j,i)=>{const row=updateJob(j),current=$('jobs').children[i];if(current!==row)$('jobs').insertBefore(row,current||null);});}
      else if(listSucceeded&&!failed&&!jobRows.size)$('jobs').innerHTML='<p class="empty panel">No jobs yet. Start with a study in New job.</p>';
      const count=(listSucceeded?summaries:jobs).filter(j=>j.status&&!TERMINAL.has(j.status)).length;updateText($('activeCount'),String(count));$('activeCount').setAttribute('aria-label',`${counted(count,'active job')}`);
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
      const models=job.models||[];
      $('jobDetailTitle').textContent=jobTitle(job,reportTask(),true);
      $('jobDetailMeta').textContent=`${id} · ${job.batch_size>1?`Series ${job.batch_index} of ${job.batch_size} · `:''}${new Date(job.created*1000).toLocaleString()} · ${models.map(name=>meta(state.catalog.find(t=>t.name===name)||{name}).title).join(' · ')}`;
      let siblings=job.batch_size>1?groupBatchJobs([...jobData.values()].filter(j=>j.batch_id===job.batch_id)):[];
      if(job.batch_size>1&&siblings.length<job.batch_size){
        try{
          const batch=await request(`/api/v1/jobs?batch_id=${encodeURIComponent(job.batch_id)}`);
          siblings=groupBatchJobs(batch.jobs.filter(j=>j.batch_id===job.batch_id));
        }catch(_){/* Keep links already available; the next refresh retries. */}
        if(token!==detailToken||serial!==detailRequest)return;
      }
      $('batchSiblings').hidden=!siblings.length;
      $('batchSiblings').innerHTML=siblings.map(j=>`<a class="chip" href="#jobs/${encodeURIComponent(j.id)}" ${j.id===id?'aria-current="page"':''}>${esc(jobTitle(j,reportTask()))} · ${esc(humanize(j.status))}</a>`).join('');
      $('jobDetailStatus').textContent=humanize(job.status);$('jobDetailStatus').className='pill '+(TERMINAL.has(job.status)||['queued','processing','retrieving','transferring'].includes(job.status)?job.status:'');
      const progress=jobProgress(job);
      $('jobDetailProgress').textContent=[progress.text,progress.elapsed,progress.stageTime,progress.last,
        job.total_bytes?`${formatBytes(job.transferred_bytes||0)} / ${formatBytes(job.total_bytes)}`:''].filter(Boolean).join(' · ');
      if(progress.percent!==null)$('jobDetailBar').value=progress.percent;else $('jobDetailBar').removeAttribute('value');
      $('jobDetailBar').hidden=TERMINAL.has(job.status);
      $('jobDetailStall').hidden=!progress.warning;$('jobDetailStall').textContent=progress.warning;
      $('jobDetailModels').replaceChildren(...modelChecklist(job).map(item=>{
        const li=document.createElement('li');li.textContent=item.text;li.dataset.status=item.status;return li;
      }));
      $('jobDetailLiveLog').hidden=!job.current_task;
      if($('jobDetailLiveLog').open)refreshLiveLog(job,token);
      $('jobDetailReportStatus').textContent=reportStatusText(job,reportTask());
      $('jobDetailErrors').hidden=!job.error;$('jobDetailFailure').textContent=job.error||'';
      $('retryDetail').hidden=job.status!=='failed';$('deleteJob').disabled=deleting||!canDeleteJob(job);
      $('deleteJob').title=canDeleteJob(job)?'Delete job and its files':'Deletion is available after the job stops running';
      if($('deleteDialog').open)$('confirmDelete').disabled=deleting||!canDeleteJob(job);
      $('jobResults').hidden=job.status!=='complete';
      if(job.status==='complete'&&!detailLoaded&&!detailLoading)loadResults(job,token);
      if(job.status!=='complete'&&detailLoaded){detailLoaded=false;jobViewer?.destroy();jobViewer=null;}
    }catch(error){if(token===detailToken&&serial===detailRequest){showError($('jobDetailError'),`Could not load this job: ${error.message}`);$('deleteJob').disabled=true;}}
  }
  let logRequest=0;
  async function refreshLiveLog(job,token){
    const serial=++logRequest;
    if(!job.current_task)return;
    try{
      const result=await request(`/api/v1/jobs/${encodeURIComponent(job.id)}/log?task=${encodeURIComponent(job.current_task)}&lines=40`);
      if(token!==detailToken||serial!==logRequest)return;
      $('jobDetailLog').textContent=result.lines.join('\n')||'Waiting for task log…';
    }catch(error){if(token===detailToken&&serial===logRequest)$('jobDetailLog').textContent='Log unavailable: '+error.message;}
  }
  $('jobDetailLiveLog').ontoggle=()=>{const job=jobData.get(detailId);if($('jobDetailLiveLog').open&&job)refreshLiveLog(job,detailToken);};
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
      if(state.step===3){const crop=cropAvailable();$('postprocessPanel').hidden=!crop;if(!crop)$('prostatePostprocess').checked=false;}
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
if(typeof module!=='undefined'&&module.exports)module.exports={jobProgress,modelChecklist,preselectSeries,defaultSelection,seriesPlan,batchBlockedReason,jobTitle,groupBatchJobs,DEFAULT_MODELS,presentation,executesRemotely,supports,pairedCT,blockedReason,cropSupported,validatePrompt,structureInfo,humanize,safeReference,primarySelection,canDeleteJob,reportMarkdown,reportMarkup,licenceBadges,counted,reportStatusText,isReportSection,GROUPS};

'use strict';
// Run runUiBrowserChecks(iframe) in a same-origin harness that serves the real UI
// with ui_fixture.js loaded before app.js. No inference or patient data is used.
async function runUiBrowserChecks(frame) {
  const results=[],win=frame.contentWindow,doc=win.document,$=id=>doc.getElementById(id);
  const assert=(value,message)=>{if(!value)throw Error(message);results.push(message);};
  const settle=()=>new Promise(r=>setTimeout(r,100));
  async function waitFor(test){for(let i=0;i<100;i++){if(test())return;await settle();}throw Error('Timed out waiting for UI');}
  await waitFor(()=>$('catalogStatus').textContent==='');
  const transfer=new win.DataTransfer();transfer.items.add(new win.File(['synthetic'], 'synthetic.dcm'));
  $('files').files=transfer.files;$('files').dispatchEvent(new win.Event('change'));
  await waitFor(()=>!$('seriesStep').hidden);
  assert($('selection').textContent.startsWith('1 file ·'),'Upload count uses singular file');
  $('toModels').click();await settle();
  $('segmentAction').click();await settle();
  assert($('selectionCount').textContent==='1 model selected','Tray uses singular model');
  assert($('modelsSummary').textContent==='1 model selected','Step summary uses singular model');
  const tray=()=>doc.querySelector('.selection-tray').getBoundingClientRect();
  const visible=el=>{const r=el.getBoundingClientRect();return r.top>=0&&r.bottom<=tray().top&&r.right<=win.innerWidth;};
  for(const width of [1280,1024,1440]){
    frame.style.width=width+'px';await settle();
    $('chooseModels').scrollIntoView({block:'end'});await settle();
    assert(visible($('chooseModels')),`Choose other models is reachable above tray at ${width}x800`);
    $('clearModels').scrollIntoView({block:'end'});await settle();
    assert(visible($('clearModels')),`Final action is reachable above tray at ${width}x800`);
  }
  frame.style.width='1280px';$('chooseModels').click();await settle();
  $('modelSearch').value='heart_key_task';$('modelSearch').dispatchEvent(new win.Event('input'));await settle();
  assert($('modelResults').textContent.startsWith('1 model shown'),'Filtered result count uses singular model');
  assert(doc.querySelector('.group-count').textContent==='1 model','Group count uses singular model');
  assert($('modelGroups').textContent.includes('Licence key required'),'Commercial TotalSegmentator flag shows licence key badge');
  $('modelSearch').value='nv_segment';$('modelSearch').dispatchEvent(new win.Event('input'));await settle();
  const row=name=>doc.querySelector(`input[value="${name}"]`).closest('.model-row');
  assert(!row('nv_segment_ct_auto').querySelector('.badge.license'),'NV-Segment-CT has no restriction badge');
  assert(row('nv_segment_ct_auto').textContent.includes('NVIDIA Open Model License.'),'Permissive licence note remains in details');
  assert(row('nv_segment_ctmr_ct_body').querySelector('.badge.license').textContent==='Non-commercial weights','CTMR has non-commercial badge');
  $('modelSearch').value='';$('modelSearch').dispatchEvent(new win.Event('input'));await settle();
  const initialHeight=tray().height;
  for(const input of doc.querySelectorAll('input[name=models]'))if(!input.checked)input.click();
  await settle();assert(tray().height>initialHeight,'Multiple selected names expand tray');
  for(const width of [1024,1280,1440]){
    frame.style.width=width+'px';await settle();
    $('clearModels').scrollIntoView({block:'end'});await settle();
    assert(visible($('clearModels')),`Wrapped selection reserves enough room at ${width}x800`);
    const reserved=parseFloat(win.getComputedStyle($('modelsStep')).paddingBottom);
    assert(reserved>=tray().height+14,`Reserved space tracks measured tray at ${width}px`);
  }
  $('clearModels').focus();$('modelSearch').focus();await settle();assert(visible($('modelSearch')),'Keyboard focus can scroll above tray');
  $('toReview').click();await settle();assert(!$('reviewStep').hidden,'Review remains accessible');
  assert(parseFloat(win.getComputedStyle(doc.documentElement).scrollPaddingBottom)===0,'Tray reservation clears when leaving selection');
  assert($('reviewWarnings').textContent.includes('require a licence key'),'Review explains licence key requirement');
  assert($('reviewWarnings').textContent.includes('non-commercial weights'),'Review explains non-commercial weights');
  win.location.hash='jobs';await waitFor(()=>!$('jobsPage').hidden);await settle();
  const earlier=doc.querySelector('[data-job="qa-earlier"] [data-field=report]');
  assert(earlier.textContent==='AI report: not run (an earlier step failed)','Jobs list distinguishes report not run');
  assert(doc.querySelector('[data-job="qa-report"] [data-field=report]').textContent.startsWith('AI report: Failed'),'Jobs list preserves actual report failure');
  assert(doc.querySelector('[data-job="qa-active"] [data-field=stage]').textContent.includes('43/117 structures'),'Jobs list shows current task stage and counts');
  win.location.hash='jobs/qa-active';await waitFor(()=>$('jobDetailProgress').textContent.includes('43/117 structures'));
  assert($('jobDetailProgress').textContent.includes('last activity'),'Status panel shows activity age');
  assert(Math.abs($('jobDetailBar').value-100*43/117)<.01,'Stage progress bar is determinate');
  assert($('jobDetailModels').textContent.includes('✓ total · done · 2m 0s'),'Completed model shows duration');
  assert($('jobDetailModels').textContent.includes('nv_segment_ct_auto · running'),'Checklist identifies running model');
  assert($('jobDetailModels').textContent.includes('nv_reason_ct_report · waiting'),'Checklist identifies waiting model');
  $('jobDetailLiveLog').open=true;await waitFor(()=>$('jobDetailLog').textContent.includes('Synthetic task log'));
  assert($('jobDetailLog').textContent.includes('<no markup>')&&!$('jobDetailLog').children.length,'Live log is shown as plain text');
  win.location.hash='jobs/qa-earlier';await waitFor(()=>$('jobDetailReportStatus').textContent.includes('not run'));
  assert($('jobDetailReportStatus').textContent===earlier.textContent,'Job detail and list use the same not-run wording');
  win.location.hash='jobs/qa-complete';await waitFor(()=>$('reportContent').querySelector('.report-section-heading'));
  const headings=[...$('reportContent').querySelectorAll('.report-section-heading')].map(el=>el.textContent);
  assert(headings.join(',')==='CHEST,ABDOMEN','Report renders anatomical labels as section headings');
  assert(!$('reportContent').querySelector('.report-section-heading .badge'),'Section headings have no finding badges');
  assert($('reportContent').textContent.includes('not checked against segmentation'),'Unmapped finding badge uses plain wording');
  assert($('reportContent').textContent.includes('12.5 mm [1]'),'Numeric result and citation survive rendering');
  assert(win.uiFixtureErrors.length===0,'No browser script errors or unexpected API requests');
  return {passed:results.length,results,errors:win.uiFixtureErrors};
}

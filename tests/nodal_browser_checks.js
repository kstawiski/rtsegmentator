'use strict';
// Synthetic catalogue and upload responses only; no image files or model calls.
async function runNodalBrowserChecks(frame) {
  const win=frame.contentWindow,doc=win.document,$=id=>doc.getElementById(id),results=[];
  const assert=(value,message)=>{if(!value)throw Error(message);results.push(message);};
  const settle=()=>new Promise(r=>setTimeout(r,80));
  async function waitFor(fn){for(let i=0;i<100;i++){if(fn())return;await settle();}throw Error('UI timeout');}
  await waitFor(()=>$('catalogStatus').textContent==='');
  const data=new win.DataTransfer();data.items.add(new win.File(['synthetic'], 'synthetic.dcm'));
  $('files').files=data.files;$('files').dispatchEvent(new win.Event('change'));
  await waitFor(()=>!$('seriesStep').hidden);
  doc.querySelector('input[value="synthetic-mr"]').click();$('toModels').click();await settle();
  const group=()=>doc.querySelector('[data-group="nodal"]');
  const row=name=>group().querySelector(`input[value="${name}"]`)?.closest('.model-row');
  const selected=()=>[...doc.querySelectorAll('#selectedChips [data-remove]')].map(b=>b.dataset.remove).sort();
  assert(group()&&group().open,'Lymph-node group is initially open');
  assert(group().querySelectorAll('[data-nodal-region]').length===4,'Four region views are available');
  assert(row('choi_tmli').querySelector('input').disabled&&row('choi_tmli').textContent.includes('ontology_review_required'),'Choi remains visible and disabled without Show unavailable');
  assert(row('clnet_235').textContent.includes('nodal_decoders_unavailable'),'CL-Net exact blocker is shown');
  assert(row('uwlair_hntsmrg_pre_t2').querySelector('input').disabled&&row('uwlair_hntsmrg_pre_t2').textContent.includes('Requires MR'),'MR GTV model stays visible with modality reason on CT');
  assert(group().textContent.includes('20 exported elective structures'),'HNLNL combined and unsided label summary is visible');
  assert(row('lnsegfm_resenc_l').textContent.includes('Code licence:')&&row('lnsegfm_resenc_l').textContent.includes('Checkpoint licence:'),'Code and checkpoint licences are separate');
  assert(row('lnsegfm_resenc_l').textContent.includes('Checkpoint licence unresolved'),'Unresolved checkpoint terms are visible');
  for(const [region,names] of [
    ['head_neck',['hnlnl_2d3d','lnsegfm_resenc_l']],['mediastinum',['compai_lnq','lnq_mediastinal_v1']],
    ['abdomen',['lnq_abdominopelvic_v1']],['pelvis',['lnq_abdominopelvic_v1','lnq_inguinal_v1']]]){
    group().querySelector(`[data-nodal-region="${region}"]`).click();await settle();
    group().querySelector('[data-nodal-preset]').click();await settle();
    assert(JSON.stringify(selected())===JSON.stringify(names.sort()),region+' preset selects exactly its compatible defaults');
    assert(group().textContent.includes('Selected from published results; not validated on your data.'),region+' preset basis is shown');
    assert(row('choi_tmli').querySelector('input').disabled,region+' view cannot enable Choi');
    if(region==='mediastinum')assert(group().querySelector('[data-nodal-section="isrt"]')&&group().textContent.includes('decoders were not released'),'Mediastinal ISRT section and station gap are visible');
    if(['abdomen','pelvis'].includes(region))assert(group().textContent.includes('No verified '+(region==='abdomen'?'abdominal':'pelvic')+' nodal-region text strings'),region+' text vocabulary gap is shown');
  }
  assert(row('lnq_abdominopelvic_v1').textContent.includes('Non-commercial weights'),'Preset preserves LNQ non-commercial badge');
  const first=group().querySelector('input[value="lnq_abdominopelvic_v1"]');first.click();await settle();
  assert([...doc.querySelectorAll('input[value="lnq_abdominopelvic_v1"]')].every(i=>!i.checked),'Selection synchronizes across duplicate regional and original cards');
  $('clearModels').click();row('voxtell_v1_1_text_prompted').querySelector('input').click();$('toReview').click();await settle();
  assert(!$('promptPanel').hidden,'Selecting VoxTell opens the prompt panel');
  assert(!$('promptSuggestions').children.length,'Pelvis has no unsupported chips');
  $('promptRegion').value='';$('promptRegion').dispatchEvent(new win.Event('change'));
  assert(!$('promptSuggestions').children.length&&$('promptRegionNote').textContent.includes('optional guidance'),'General-target workflow can omit nodal guidance');
  $('promptRegion').value='head_neck';$('promptRegion').dispatchEvent(new win.Event('change'));await settle();
  assert($('promptSuggestions').querySelectorAll('button').length===16,'Head and neck offers exact audited chips');
  assert($('promptSuggestions').textContent.includes('experimental: no published accuracy for this term'),'Specific terms display the experimental warning');
  const prompt={text:'old target',positive_points:[[2,3,4]],negative_points:[],box:null};
  $('promptJson').value=JSON.stringify(prompt);
  $('promptSuggestions').querySelector('button').click();
  assert($('promptText').value==='cervical lymph node','Chip inserts exact text into Target description');
  assert(JSON.parse($('promptJson').value).positive_points[0].join(',')==='2,3,4','Chip preserves spatial coordinates in shared prompt JSON');
  $('promptJson').value='{invalid';$('promptSuggestions').querySelector('button').click();
  assert(!$('promptError').hidden&&$('promptJson').value==='{invalid','Invalid JSON is reported without destroying the draft');
  $('promptJson').value=JSON.stringify(prompt);
  $('promptRegion').value='mediastinum';$('promptRegion').dispatchEvent(new win.Event('change'));await settle();
  assert($('promptSuggestions').querySelectorAll('button').length===13,'Mediastinum offers exact audited strings');
  assert($('promptGuidance').textContent.includes('Points and boxes are ignored'),'VoxTell prompt consumption is explicit');
  $('promptRegion').value='abdomen';$('promptRegion').dispatchEvent(new win.Event('change'));
  assert(!$('promptSuggestions').children.length&&$('promptSuggestionsNote').textContent.includes('No verified text chips'),'Abdomen does not invent text chips');
  assert(win.uiBatchRequests.length===0,'Checks never submit a model job');
  const errors=win.uiFixtureErrors.filter(e=>!e.includes('ResizeObserver'));
  assert(errors.length===0,'No JavaScript or unexpected API errors');
  // Finish on the picker for screenshots and manual inspection.
  doc.querySelector('[data-back="2"]').click();await settle();
  return {passed:results.length,checks:results,errors};
}

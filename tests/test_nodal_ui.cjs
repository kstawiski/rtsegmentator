'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const ui=require('../static/app.js'),metadata=require('../static/ui-catalog.json');
const vocabulary=require('../catalog/voxtell-nodal-vocabulary.json'),details=require('../catalog/nodal-models.json');
const ct={modality:'CT'},mr={modality:'MR'},pet={modality:'PT'};
const model=(name,extra={})=>({name,modality:'CT',enabled:true,...extra});
const regionIds=['head_neck','mediastinum','abdomen','pelvis'];

test('four regional views preserve original groups and allow cross-region membership',()=>{
  assert.deepEqual(metadata.nodal.regions.map(r=>r.id),regionIds);
  const choi=model('choi_tmli',{enabled:false,status:'ontology_review_required'});
  assert.deepEqual(ui.presentation(choi,metadata).nodalRegions,regionIds);
  assert.equal(ui.presentation(choi,metadata).group,'ctv');
  assert.equal(ui.presentation(model('hnlnl_2d3d'),metadata).group,'ctv');
  for(const id of regionIds){
    const sections=ui.nodalSections([choi],id,metadata);
    assert(sections.some(s=>s.id==='elective'&&s.tasks[0]===choi));
    assert.deepEqual(sections.filter(s=>s.id!=='isrt').map(s=>s.id),['elective','candidates','gtv','interactive']);
  }
  assert.equal(ui.nodalSections([], 'unknown', metadata).length,0);
  assert.equal(ui.presentation(model('future_unknown'),metadata).group,'other');
});

test('regional sections keep candidates, elective regions, GTV and ISRT distinct',()=>{
  const catalog=Object.keys(details).map(name=>model(name));
  const h=ui.nodalSections(catalog,'head_neck',metadata),m=ui.nodalSections(catalog,'mediastinum',metadata);
  assert(h.find(s=>s.id==='elective').tasks.some(t=>t.name==='lnctvseg_nnunet_v1'));
  assert(h.find(s=>s.id==='candidates').tasks.every(t=>t.name.startsWith('lnsegfm_')));
  assert(h.find(s=>s.id==='gtv').tasks.some(t=>t.name==='uwlair_hntsmrg_pre_t2'));
  assert(m.find(s=>s.id==='isrt').tasks.every(t=>t.name.startsWith('isrt_')));
  assert(m.find(s=>s.id==='elective').note.includes('decoders were not released'));
  for(const id of ['abdomen','pelvis'])assert(ui.nodalSections(catalog,id,metadata).find(s=>s.id==='elective').note.includes('unresolved'));
});

test('regional presets use exactly the requested defaults and replace segmentation only',()=>{
  const expected=[['hnlnl_2d3d','lnsegfm_resenc_l'],['compai_lnq','lnq_mediastinal_v1'],['lnq_abdominopelvic_v1'],['lnq_abdominopelvic_v1','lnq_inguinal_v1']];
  const catalog=[...expected.flat().map(name=>model(name)),model('total'),model('report',{output:'report'})];
  regionIds.forEach((id,i)=>{
    const result=ui.regionPreset(catalog,id,metadata,[ct],new Set(['total','report']));
    assert.deepEqual(result.added,expected[i]);
    assert.deepEqual([...result.selected],['report',...expected[i]]);
    assert.equal(result.skipped.length,0);
    assert(![...result.selected].some(name=>name.startsWith('isrt_')));
  });
});

test('presets retain exact blockers, ignore missing tasks, respect modality and preserve licence flags',()=>{
  const disabled=model('hnlnl_2d3d',{enabled:false,status:'orientation_revalidation_pending',missing:['orientation marker']});
  const candidate=model('lnsegfm_resenc_l',{noncommercial:true,licensed:true});
  const before=JSON.stringify(candidate);
  const r=ui.regionPreset([disabled,candidate],'head_neck',metadata,[ct]);
  assert.deepEqual(r.added,['lnsegfm_resenc_l']);
  assert.match(r.skipped[0].reason,/orientation_revalidation_pending.*orientation marker/);
  assert.equal(JSON.stringify(candidate),before);
  assert.deepEqual(ui.licenceBadges(candidate),['Licence key required','Non-commercial weights']);
  assert.deepEqual(ui.licenceBadges(details.lnq_atlas),['Checkpoint licence unresolved','Non-commercial code']);
  assert.deepEqual(ui.licenceBadges(details.medical_sam2_prompted_ct),['Checkpoint licence unresolved']);
  for(const series of [[mr],[pet],[]]){
    const unavailable=ui.regionPreset([disabled,candidate],'head_neck',metadata,series,new Set(['old']));
    assert.deepEqual([...unavailable.selected],['old']);assert.equal(unavailable.added.length,0);
  }
  const absent=ui.regionPreset([candidate],'head_neck',metadata,[ct,mr]);
  assert.deepEqual(absent.added,['lnsegfm_resenc_l']);assert.equal(absent.skipped[0].reason,'Not in the live catalogue');
});

test('every chip is an exact released string with source; level/station terms are marked experimental',()=>{
  const terms=new Set(vocabulary.terms);
  for(const region of metadata.nodal.regions){
    const chips=metadata.nodal.chips[region.id];
    assert.equal(new Set(chips.map(c=>c.text)).size,chips.length);
    for(const chip of chips){
      assert(terms.has(chip.text),chip.text);
      assert.equal(chip.source,'catalog/voxtell-nodal-vocabulary.json');
      assert.equal(chip.model,'voxtell_v1_1_text_prompted');
      if(chip.text.startsWith('right level ')||(region.id==='mediastinum'&&chip.text!=='mediastinal lymph nodes'))assert.equal(chip.experimental,true);
    }
  }
  assert.equal(metadata.nodal.experimental_label,'experimental: no published accuracy for this term');
  assert.deepEqual(metadata.nodal.chips.abdomen,[]);assert.deepEqual(metadata.nodal.chips.pelvis,[]);
});

test('chips require the compatible selected enabled model and replace only prompt text',()=>{
  const vox=model('voxtell_v1_1_text_prompted',{prompt:'text'}),selection=new Set([vox.name]);
  const suggestions=(catalog,selected=selection,series=[ct])=>ui.promptSuggestions(catalog,selected,'head_neck',metadata,series);
  assert.equal(suggestions([vox]).length,16);
  assert.deepEqual(suggestions([{...vox,enabled:false,status:'weights_unavailable'}]),[]);
  assert.deepEqual(suggestions([vox],new Set()),[]);
  assert.deepEqual(suggestions([vox],selection,[mr]),[]);
  assert.deepEqual(suggestions([vox],selection,[ct,ct]),[]);
  const prompt={text:'old',positive_points:[[1,2,3]],negative_points:[[4,5,6]],box:[[0,0,3],[3,3,3]]};
  const next=ui.insertPromptSuggestion(prompt,suggestions([vox])[0]);
  assert.equal(next.text,'cervical lymph node');assert.equal(prompt.text,'old');
  for(const k of ['positive_points','negative_points','box'])assert.deepEqual(next[k],prompt[k]);
});

test('every regional model has separate licences, evidence and accurate interaction limits',()=>{
  const allowed=new Set(['code_license','weights_license','weights_license_unresolved','description','limitations','evidence_note','evidence_sources','noncommercial','code_noncommercial']);
  for(const r of metadata.nodal.regions)for(const s of r.sections)for(const name of s.models){
    assert(details[name],name);
    for(const key of Object.keys(details[name]))assert(allowed.has(key),name+': '+key);
    assert(details[name].code_license&&details[name].weights_license&&details[name].evidence_sources.length);
  }
  assert.match(metadata.nodal.guidance.medical_sam3_3d_prompted,/eight adjacent axial slices/);
  assert.match(metadata.nodal.guidance.pam_prompted,/negative points are ignored/);
  assert.match(metadata.nodal.guidance.pam_prompted,/Box2Mask workflow are not implemented/);
  assert.match(details.hnlnl_2d3d.evidence_note,/mirror-TTA change requires technical rechecking/);
  assert.match(details.clnet_235.description,/18 neck and 15 chest/);
  assert.match(details.lnctvseg_nnunet_v1.limitations,/omits 7 and logs its volume/);
});

test('LNQ author correction removes only the stale licence badges',()=>{
  for(const name of ['lnq_mediastinal_v1','lnq_abdominopelvic_v1','lnq_inguinal_v1','lnq_axillary_v1']){
    assert.equal(details[name].code_license,'Apache-2.0');
    assert.equal(details[name].weights_license,'CC BY 4.0');
    assert.deepEqual(ui.licenceBadges(details[name]),[]);
    assert.match(details[name].evidence_note,/19 October 2026/);
    assert(details[name].evidence_sources.some(url=>url.endsWith('fffa35913d9d08234638e482159268fa27322221/MODELS.md')));
  }
  assert(ui.licenceBadges(details.voxtell_v1_1_text_prompted).includes('Non-commercial weights'));
  assert(ui.licenceBadges(details.lnq_atlas).includes('Non-commercial code'));
});

test('LNCTVSeg summaries include Vc and distinguish the shorter export names',()=>{
  for(const name of ['lnctvseg_nnunet_v1','lnctvseg_swinunetr','lnctvseg_3duxnet','lnctvseg_repuxnet']){
    assert.match(metadata.overrides[name].purpose,/left\/right II\+III\+Va, IV\+Vb\+Vc and Ib/);
    assert.match(metadata.overrides[name].purpose,/export names abbreviate the lower-neck pair to IV–Vb/);
    assert.match(details[name].description,/IV\+Vb\+Vc/);
    assert.match(details[name].limitations,/export names abbreviate this pair to IV–Vb/);
  }
});

test('SAM-Med3D guidance explains point precedence and the limited box role',()=>{
  const appSource=require('node:fs').readFileSync(require.resolve('../static/app.js'),'utf8');
  const help=appSource.match(/sam_med3d_turbo_prompted: '([^']+)'/)[1];
  for(const text of [help,details.sam_med3d_turbo_prompted.description,metadata.nodal.guidance.sam_med3d_turbo_prompted]){
    assert.match(text,/Points take precedence over a box/);
    assert.match(text,/box centre is used only when no positive or negative points are given/);
    assert.match(text,/box is not a spatial boundary/);
    assert.match(text,/128³ at 1\.5 mm/);
  }
});

test('release absence notes are dated and bounded while integration blockers stay explicit',()=>{
  const regions=Object.fromEntries(metadata.nodal.regions.map(r=>[r.id,r]));
  const note=(region,id)=>regions[region].sections.find(s=>s.id===id).note;
  assert(note('mediastinum','elective').startsWith('No released model with verified IASLC station outputs was found in the sources reviewed on 20 October 2026.'));
  for(const [region,area] of [['abdomen','abdominal'],['pelvis','pelvic']]){
    assert.match(note(region,'gtv'),/was found in the sources reviewed on 20 October 2026/);
    assert.match(regions[region].prompt_note,/were found in the sources reviewed on 20 October 2026/);
    assert(note(region,'elective').startsWith('No enabled '+area+' nodal-region model.'));
  }
});

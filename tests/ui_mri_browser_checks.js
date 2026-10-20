'use strict';
async function runMriBrowserChecks(frame){
  const results=[],win=frame.contentWindow,doc=win.document,$=id=>doc.getElementById(id);
  const assert=(value,message)=>{if(!value)throw Error(message);results.push(message);};
  async function waitFor(test){for(let i=0;i<100;i++){if(test())return;await new Promise(r=>setTimeout(r,100));}throw Error('Timed out waiting for MRI UI');}
  await waitFor(()=>$('catalogStatus').textContent==='');
  const transfer=new win.DataTransfer();transfer.items.add(new win.File(['synthetic'], 'synthetic.dcm'));
  $('files').files=transfer.files;$('files').dispatchEvent(new win.Event('change'));
  await waitFor(()=>!$('seriesStep').hidden);
  $('clearSeries').click();doc.querySelector('input[value="synthetic-t2"]').click();$('toModels').click();
  doc.querySelector('[name=models][value="synthetic_bpmri"]').click();
  doc.querySelector('[data-step="3"]').click();
  const role=name=>doc.querySelector(`[data-role="${name}"]`);
  assert(doc.querySelectorAll('[data-role]').length===2,'One companion dropdown per role');
  assert(role('adc').value===''&&role('adc').textContent.includes('Ambiguous'),'Equal ADC candidates require a choice');
  assert($('reviewContent').textContent.includes('will not run'),'Missing roles show a model-specific blocking reason');
  assert(!$('startJob').disabled,'T2-only models can run with an unresolved ADC role');
  assert(role('hbv').selectedOptions[0].textContent.includes('b=1400'),'Highest documented b-value is suggested');
  assert(role('hbv').textContent.includes('(0018,9087)'),'Diffusion source tag is shown');
  assert(role('hbv').textContent.includes('160 × 160 × 24'),'Chosen-volume dimensions are shown');
  assert($('reviewContent').textContent.includes('Frame of Reference differs'),'Frame difference warning is visible');
  role('adc').value='0';role('adc').dispatchEvent(new win.Event('change'));
  assert(!$('reviewContent').textContent.includes('will not run'),'Explicit ADC selection unblocks the multi-sequence model');
  role('hbv').value='1';role('hbv').dispatchEvent(new win.Event('change'));
  assert(role('hbv').selectedOptions[0].textContent.includes('b=800'),'User can choose a lower valid b-value');
  assert(doc.activeElement===role('hbv'),'Dropdown keeps keyboard focus after editing');
  assert(!win.uiFixtureErrors.length,'No browser errors');
  return {passed:results.length,checks:results,errors:win.uiFixtureErrors};
}

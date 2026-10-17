'use strict';
// Invoke in the existing same-origin browser harness; all pixels are synthetic.
async function runViewerSpeedBrowserChecks(win) {
  const {Viewer}=win.eval('DicomUI'), doc=win.document, host=doc.createElement('div');
  host.style.cssText='width:800px;height:800px';doc.body.append(host);
  const originalFetch=win.fetch, requests=[], errors=[];
  let current=0, maximum=0, aborts=0, building=0, metadataCalls=0;
  const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const assert=(ok,message)=>{if(!ok)throw Error(message);};
  const wait=async predicate=>{for(let i=0;i<200;i++){if(predicate())return;await sleep(10);}throw Error('Viewer browser wait timed out');};
  const metadata=()=>({series:{modality:'CT',rows:128,columns:128,slices:100,orientation:'axial',
    column_direction:[1,0,0],row_direction:[0,1,0],window_presets:[
      {id:'soft-tissue',label:'Soft tissue',center:40,width:400},{id:'lung',label:'Lung',center:-600,width:1500}],default_window:'soft-tissue'},
    contours_ready:metadataCalls>1,contours_state:metadataCalls>1?'ready':'building',
    structure_sets:metadataCalls>1?[{model:'Synthetic',label:'Test',rois:[{key:'roi',name:'Synthetic ROI',color:'#112233',slices:[50,51,85]}]}]:[]});
  const roi=index=>[{key:'roi',testIndex:index,paths:[[[1,1],[10,1],[10,10]]]}];
  win.fetch=async (input,options={})=>{
    const url=String(input).replace('/api/v1/jobs/qa-complete/viewer','/viewer-speed-qa');
    if(!url.startsWith('/viewer-speed-qa'))return originalFetch(input,options);
    requests.push(url);current++;maximum=Math.max(maximum,current);
    try {
      return await new Promise((resolve,reject)=>{
        let done=false;
        const timer=setTimeout(async()=>{
          if(done)return;done=true;
          if(url==='/viewer-speed-qa'){metadataCalls++;resolve(new win.Response(JSON.stringify(metadata())));return;}
          const match=url.match(/slices\/(\d+)\.png/);
          if(match){
            const canvas=doc.createElement('canvas');canvas.width=canvas.height=128;
            const ctx=canvas.getContext('2d'),value=(Number(match[1])*3)%240;
            ctx.fillStyle=`rgb(${value},${value},${value})`;ctx.fillRect(0,0,128,128);
            resolve(new win.Response(await new Promise(r=>canvas.toBlob(r,'image/png'))));return;
          }
          if(building<2){building++;resolve(new win.Response('{"contours_ready":false}',{status:202,headers:{'Retry-After':'.25'}}));return;}
          const single=url.match(/contours\/(\d+)$/);
          if(single){const index=Number(single[1]);resolve(new win.Response(JSON.stringify({index,rois:roi(index)})));return;}
          const query=new URL(url,win.location.href).searchParams,start=Number(query.get('start')),count=Number(query.get('count'));
          resolve(new win.Response(JSON.stringify({start,slices:Object.fromEntries(Array.from({length:count},(_,i)=>[start+i,roi(start+i)]))})));
        },url.includes('slices/')?120:url.includes('contours')?250:20);
        options.signal?.addEventListener('abort',()=>{if(done)return;done=true;clearTimeout(timer);aborts++;reject(new win.DOMException('Aborted','AbortError'));},{once:true});
      });
    } finally {current--;}
  };
  const viewer=new Viewer(host,{overlays:true});
  let samples=0;
  function consistent(){
    const index=viewer.displayed.index,canvas=viewer.el('canvas');
    assert(viewer.image && viewer.el('message').hidden,'Displayed image remains visible');
    assert(canvas.dataset.sliceIndex===String(index),'Canvas index matches displayed image');
    assert(viewer.el('index').textContent.startsWith(`${index+1} / 100`),'Label describes displayed slice');
    assert(canvas.getAttribute('aria-label').includes(`slice ${index+1} of 100`),'Accessible label describes displayed slice');
    assert(viewer.paths.every(r=>r.testIndex===index),'Every contour belongs to displayed image');
    const pixel=canvas.getContext('2d').getImageData(Math.floor(canvas.width/2),Math.floor(canvas.height/2),1,1).data;
    assert(pixel[3]===255 && pixel[0]===(index*3)%240,'Visible canvas pixels match displayed index');samples++;
  }
  try {
    await viewer.load('/viewer-speed-qa');await wait(()=>viewer.image);
    consistent();assert(viewer.contourState==='loading','Image appears while contours are still pending');
    const beforeScroll=requests.length;
    for(let index=51;index<=85;index++){viewer.setSlice(index);consistent();await sleep(5);}
    assert(requests.length===beforeScroll,'Rapid changes defer new requests until scrolling settles');
    await wait(()=>viewer.displayed.index===85);consistent();
    await wait(()=>viewer.rois.length && viewer.paths.length);consistent();
    viewer.setSlice(50);assert(viewer.displayed.index===50,'Cached slice displays synchronously');consistent();
    viewer.window='lung';viewer.schedule();consistent();
    assert(viewer.displayed.window==='soft-tissue','Displayed window remains explicit during replacement');
    await wait(()=>viewer.displayed.window==='lung');consistent();
    viewer.el('standard').click();consistent();
    assert(requests.some(url=>url.includes('/contours?start=')),'Contour prefetch uses range endpoint');
    assert(maximum<=4,'At most four concurrent viewer requests');
    assert(building===2,'Building contour responses were retried');assert(aborts>0,'Distant prefetch requests were cancelled');
    viewer.destroy();await sleep(20);
    win.location.hash='jobs';await sleep(50);metadataCalls=0;
    win.location.hash='jobs/qa-complete';
    await wait(()=>doc.getElementById('structuresContent').textContent==='Contours loading…');
    await wait(()=>doc.getElementById('structuresContent').textContent.includes('Synthetic ROI'));
    assert(doc.getElementById('structuresContent').querySelector('table'),'Structures table updates after background metadata arrives');
    win.location.hash='jobs';await sleep(50);
    return {passed:true,consistency_samples:samples,max_concurrent:maximum,requests:requests.length,
      range_requests:requests.filter(url=>url.includes('/contours?')).length,building_responses:building,aborted_requests:aborts,
      structures_table_updated:true,errors};
  } finally {viewer.destroy();win.location.hash='jobs';await sleep(50);win.fetch=originalFetch;host.remove();}
}
if(typeof module!=='undefined')module.exports={runViewerSpeedBrowserChecks};

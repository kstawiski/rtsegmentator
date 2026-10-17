'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {Viewer, LRU, prefetchIndices} = require('../static/viewer.js');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate) {
  for (let i=0; i<200; i++) { if (predicate()) return; await sleep(5); }
  assert.fail('Timed out waiting for viewer');
}
const info = {series:{modality:'CT',slices:100,rows:32,columns:32,orientation:'axial',
  column_direction:[1,0,0],row_direction:[0,1,0],window_presets:[
    {id:'soft-tissue',label:'Soft tissue',center:40,width:400},{id:'lung',label:'Lung',center:-600,width:1500}],default_window:'soft-tissue'},
  contours_ready:true,structure_sets:[{model:'test',label:'Synthetic',rois:[{key:'roi',name:'ROI',color:'#112233',slices:[50]}]}]};

function fixture(t, metadata = info) {
  const elements = new Map(), draws = [], urls = new Map();
  const ctx = new Proxy({drawImage:img=>draws.push(img.index)}, {get:(o,k)=>o[k] || (()=>{})});
  const element = name => {
    if (!elements.has(name)) elements.set(name, {value:name==='opacity'?'80':name==='mode'?'outline':'',hidden:false,style:{},dataset:{},
      getContext:()=>ctx,getBoundingClientRect:()=>({width:640,height:480}),addEventListener(){},setAttribute(k,v){this[k]=v;if(k==='data-slice-index')this.dataset.sliceIndex=v;},
      querySelectorAll:()=>[],focus(){},setPointerCapture(){}});
    return elements.get(name);
  };
  const host = {querySelector:s=>element(s.match(/"([^"]+)"/)[1]),replaceChildren(){this.destroyed=true;}};
  let serial = 0, current = 0, maximum = 0;
  const requests = [];
  t.mock.method(URL, 'createObjectURL', blob=>{const url=`blob:test${serial++}`;urls.set(url,blob);return url;});
  t.mock.method(URL, 'revokeObjectURL', url=>urls.delete(url));
  t.mock.method(global, 'fetch', (url, options={}) => {
    current++; maximum=Math.max(maximum,current);
    return new Promise((resolve,reject)=>{
      const request={url,signal:options.signal,done:false,respond(body,status=200){
        if(this.done)return;this.done=true;current--;
        resolve(new Response(typeof body==='string'?body:JSON.stringify(body),{status,headers:{'Retry-After':'.01'}}));
      }};
      options.signal?.addEventListener('abort',()=>{if(!request.done){request.done=true;current--;reject(new DOMException('Aborted','AbortError'));}},{once:true});
      requests.push(request);
      if(url.split('?')[0]==='/series' || url==='/other')request.respond(metadata);
    });
  });
  for(const key of ['window','ResizeObserver','Image']) { const value=global[key];t.after(()=>{if(value===undefined)delete global[key];else global[key]=value;}); }
  global.window={devicePixelRatio:1};
  global.ResizeObserver=class {observe(){} disconnect(){}};
  global.Image=class {
    naturalWidth=32; naturalHeight=32;
    async decode(){this.index=JSON.parse(await urls.get(this.src).text()).index;}
  };
  const viewer = new Viewer(host,{overlays:true});
  t.after(()=>viewer.destroy());
  const pending = pattern => requests.find(r=>!r.done && pattern.test(r.url));
  const png = index => {const request=pending(new RegExp(`/slices/${index}\\.png`));assert.ok(request,`PNG ${index} requested`);request.respond({index});};
  const contours = index => {const request=pending(new RegExp(`/contours/${index}$`));assert.ok(request);request.respond({index,rois:[{key:'roi',paths:[[[index,0],[index,1],[index+1,1]]]}]});};
  return {viewer,element,requests,pending,png,contours,draws,maximum:()=>maximum,urls};
}

test('LRU bounds bytes and entry count, refreshes recency, and clears',()=>{
  const cache=new LRU(10,2);
  cache.set('a',1,4);cache.set('b',2,4);assert.equal(cache.get('a'),1);
  cache.set('c',3,4);assert.equal(cache.get('b'),undefined);assert.equal(cache.bytes,8);
  cache.set('a',4,9);assert.equal(cache.bytes,9);assert.equal(cache.get('c'),undefined);
  cache.set('huge',5,11);assert.equal(cache.get('huge'),undefined);
  cache.clear();assert.equal(cache.bytes,0);assert.equal(cache.items.size,0);
});

test('prefetch biases 16 forward and 6 back in either direction and clips ends',()=>{
  for (const direction of [-1,1]) {
    const indices=prefetchIndices(50,100,direction);
    assert.equal(indices.length,23);assert.equal(indices[0],50);
    assert.equal(indices.filter(i=>(i-50)*direction>0).length,16);
    assert.equal(indices.filter(i=>(i-50)*direction<0).length,6);
  }
  assert.deepEqual(prefetchIndices(0,3,-1),[0,1,2]);
});

test('images display before contours; scrolling preserves a consistent displayed slice and cached hits are synchronous',async t=>{
  const f=fixture(t),v=f.viewer;await v.load('/series');
  await until(()=>f.pending(/slices\/50\.png/));f.png(50);
  await until(()=>v.image);assert.equal(v.displayed.index,50);assert.deepEqual(v.paths,[]);
  assert.match(f.element('loading').textContent,/Contours loading/);assert.equal(f.element('message').hidden,true);
  f.contours(50);await until(()=>v.paths.length);const oldPaths=v.paths;
  v.setSlice(51);assert.equal(v.image.index,50);assert.equal(v.paths,oldPaths);
  assert.match(f.element('index').textContent,/51 \/ 100.*loading 52/);
  assert.match(f.element('canvas')['aria-label'],/slice 51 of 100/);
  await until(()=>f.pending(/slices\/51\.png/));f.png(51);
  await until(()=>v.displayed.index===51);assert.deepEqual(v.paths,[]);
  assert.equal(f.element('canvas').dataset.sliceIndex,'51');
  const calls=f.requests.length;v.setSlice(50);
  assert.equal(v.displayed.index,50);assert.equal(v.paths,oldPaths);assert.equal(f.requests.length,calls);
  assert.ok(f.maximum()<=4);assert.equal(f.urls.size,0);
});

test('fast scrolling debounces fetches, cancels distant requests, and prefetches contour ranges',async t=>{
  const f=fixture(t),v=f.viewer;await v.load('/series');
  await until(()=>f.pending(/slices\/50\.png/));f.png(50);f.contours(50);
  await until(()=>v.image && v.paths.length);
  assert.ok(f.requests.some(r=>r.url.includes('/contours?start=')));
  const count=f.requests.length;
  for(let index=70;index<90;index++){v.setSlice(index);assert.equal(v.image.index,50);assert.match(f.element('index').textContent,new RegExp(`loading ${index+1}`));}
  assert.equal(f.requests.length,count);
  assert.ok(f.requests.filter(r=>/slices\/(?:49|51)\.png/.test(r.url)).every(r=>r.signal.aborted));
  await until(()=>f.pending(/slices\/89\.png/));
  assert.ok(!f.requests.slice(count).some(r=>/slices\/(?:7\d|8[0-8])\.png/.test(r.url)));
  f.png(89);await until(()=>v.displayed.index===89);assert.ok(f.maximum()<=4);
});

test('202 retries current and prefetched contours; metadata polling discovers ROI controls',async t=>{
  const building={...info,structure_sets:[],contours_ready:false,contours_state:'building'};
  const f=fixture(t,building),v=f.viewer,updates=[];v.onMetadata=data=>updates.push(data.contours_ready);await v.load('/series');
  await until(()=>f.pending(/contours\/50$/));
  f.pending(/contours\/50$/).respond({contours_ready:false},202);
  const range=f.requests.find(r=>r.url.includes('/contours?'));assert.ok(range);range.respond({contours_ready:false},202);
  f.png(50);await until(()=>v.image);assert.match(f.element('loading').textContent,/Contours loading/);
  // Free the metadata slot and allow the retry timer to run while PNG prefetch is active.
  for (const r of f.requests.filter(r=>!r.done && /slices\//.test(r.url))) r.respond({index:Number(r.url.match(/slices\/(\d+)/)[1])});
  await until(()=>f.requests.filter(r=>/contours\/50$/.test(r.url)).length===2);
  f.contours(50);await until(()=>v.paths.length);
  assert.ok(f.maximum()<=4);
  v.updateMetadata({...info,contours_state:'ready'});v.showContours();
  assert.equal(v.rois.length,1);assert.ok(v.visible.has('roi'));assert.equal(v.contourState,'ready');
  assert.equal(updates[0],false);assert.equal(updates.at(-1),true);
});

test('window changes preserve the displayed window until pixels arrive; load and destroy clear caches and cancel work',async t=>{
  const f=fixture(t),v=f.viewer;await v.load('/series');
  await until(()=>f.pending(/slices\/50\.png/));f.png(50);await until(()=>v.image);
  v.window='lung';v.schedule();assert.equal(v.displayed.window,'soft-tissue');assert.ok(v.image);
  assert.match(f.element('readout').textContent,/Soft tissue/);
  await until(()=>f.pending(/slices\/50\.png\?window=lung/));f.png(50);
  await until(()=>v.displayed.window==='lung');assert.match(f.element('readout').textContent,/Lung/);
  await v.load('/other');assert.equal(v.images.items.size,0);assert.equal(v.contours.items.size,0);assert.equal(v.image,null);
  v.destroy();assert.equal(v.active.size,0);assert.equal(v.info,null);assert.equal(v.images.bytes,0);
  assert.ok(f.requests.filter(r=>!r.done).every(r=>r.signal.aborted));
});

test('oversized payloads display without refetch loops when memory budgets cannot hold the prefetch window',async t=>{
  const f=fixture(t),v=f.viewer;v.images.budget=1;v.contours.budget=1;
  await v.load('/series');await until(()=>f.pending(/slices\/50\.png/));
  f.png(50);await until(()=>v.image);f.contours(50);await until(()=>v.paths.length);
  assert.equal(v.images.bytes,0);assert.equal(v.contours.bytes,0);assert.equal(v.contourState,'ready');
  assert.equal(f.requests.filter(r=>/slices\/50\.png/.test(r.url)).length,1);
  assert.equal(f.requests.filter(r=>/contours\/50$/.test(r.url)).length,1);
  v.schedule();await sleep(80);
  assert.equal(f.requests.filter(r=>/slices\/50\.png/.test(r.url)).length,1);
  assert.equal(f.requests.filter(r=>/contours\/50$/.test(r.url)).length,1);
});

test('contour failure before pixels remains failed after the image and metadata arrive',async t=>{
  const f=fixture(t),v=f.viewer;await v.load('/series');
  await until(()=>f.pending(/contours\/50$/));
  f.pending(/contours\/50$/).respond({detail:'Synthetic contour failure'},500);
  await until(()=>v.retryAt.get(v.contourKey(50))===Infinity);
  assert.equal(v.image,null);
  f.png(50);await until(()=>v.image);
  assert.equal(v.contourState,'failed');
  assert.match(f.element('loading').textContent,/Contours unavailable/);
  assert.doesNotMatch(f.element('loading').textContent,/Contours loading/);
  assert.match(f.element('error').textContent,/Synthetic contour failure/);
  v.updateMetadata(info);v.showContours();
  assert.equal(v.contourState,'failed');
  assert.equal(f.requests.filter(r=>/contours\/50$/.test(r.url)).length,1);
});

for(const initial of ['failed','building'])test(`${initial} contour index stops polling at failure; explicit retry restarts`,async t=>{
  const metadata={...info,structure_sets:[],contours_ready:false,contours_state:initial,contours_error:'Synthetic invalid RTSTRUCT'};
  const f=fixture(t,metadata),v=f.viewer;await v.load('/series');
  await until(()=>f.pending(/slices\/50\.png/));
  f.png(50);await until(()=>v.image);
  if(initial==='building'){
    metadata.contours_state='failed';
    // The initial immediate poll can still contain building metadata; force
    // the next due poll after a slot becomes available.
    await sleep(10);v.metadataRetry=0;
    for(const r of f.requests.filter(r=>!r.done)){
      if(/contours/.test(r.url))r.respond({contours_ready:false},202);
      else if(/slices\//.test(r.url))r.respond({index:Number(r.url.match(/slices\/(\d+)/)[1])});
    }
    v.fetchSlice();await until(()=>v.info.contours_state==='failed');
  }
  const polls=f.requests.filter(r=>r.url==='/series').length;
  await sleep(1100);
  assert.equal(f.requests.filter(r=>r.url==='/series').length,polls);
  assert.equal(v.contourState,'failed');
  assert.match(f.element('empty').textContent,/Contours unavailable: Synthetic invalid RTSTRUCT/);
  assert.doesNotMatch(f.element('empty').textContent,/loading/);
  assert.equal(f.element('retry').hidden,false);
  metadata.contours_state='building';metadata.contours_error=null;
  await f.element('retry').onclick();
  assert.ok(f.requests.some(r=>r.url==='/series?retry_contours=true'));
  await until(()=>f.pending(/contours\/50$/));
  assert.match(f.element('empty').textContent,/Contours loading/);
  assert.equal(v.info.contours_state,'building');
});

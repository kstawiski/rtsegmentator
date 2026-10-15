'use strict';

// Pixel coordinates refer to pixel centres, including the top-left pixel (0, 0).
// The same affine transform is used for the PNG and every contour path.
const DicomUI = (() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const identity = () => ({a:1, b:0, c:0, d:1});
  function displayOrientation(series, standard = true) {
    if (!standard) return identity();
    const column = series?.column_direction, row = series?.row_direction;
    // Only signed patient-axis permutations qualify. An axial plane can still
    // have oblique in-plane edges; do not snap those to a different orientation.
    const axis = v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)
      && v.every(n => Math.abs(n - Math.round(n)) <= 1e-4)
      && v.reduce((sum,n) => sum + Math.abs(Math.round(n)), 0) === 1;
    if (!axis(column) || !axis(row)) return identity();
    const c = column.map(Math.round), r = row.map(Math.round);
    const dot = (u,v) => u.reduce((sum,n,i) => sum + n*v[i], 0);
    if (dot(c,r) !== 0) return identity();
    const normalAxis = [0,1,2].find(i => !c[i] && !r[i]);
    // DICOM LPS: screen right/down = L/P (axial), L/F (coronal), P/F (sagittal).
    const right = normalAxis === 0 ? [0,1,0] : [1,0,0];
    const down = normalAxis === 2 ? [0,1,0] : [0,0,-1];
    return {a:dot(c,right), b:dot(c,down), c:dot(r,right), d:dot(r,down)};
  }
  function directionLabel(vector) {
    if (!Array.isArray(vector) || vector.length !== 3 || !vector.every(Number.isFinite)) return '?';
    const length = Math.hypot(...vector); if (!length) return '?';
    return vector.map((v,i) => ({magnitude:Math.abs(v)/length, letter:(v < 0 ? ['R','A','F'] : ['L','P','H'])[i]}))
      .sort((a,b) => b.magnitude-a.magnitude).filter((v,i) => i === 0 || v.magnitude >= .3 - 1e-12)
      .slice(0,2).map(v => v.letter).join('');
  }
  function orientationMarkers(series, display = identity()) {
    const column = series?.column_direction, row = series?.row_direction;
    if (!Array.isArray(column) || !Array.isArray(row)) return {left:'?',right:'?',top:'?',bottom:'?'};
    const right = column.map((v,i) => v*display.a + row[i]*display.c);
    const down = column.map((v,i) => v*display.b + row[i]*display.d);
    return {left:directionLabel(right.map(v => -v)), right:directionLabel(right),
      top:directionLabel(down.map(v => -v)), bottom:directionLabel(down)};
  }
  function imageTransform(columns, rows, width, height, view = {zoom:1, x:0, y:0}, display = identity()) {
    const {a,b,c,d} = display;
    const displayColumns = Math.abs(a)*columns + Math.abs(c)*rows;
    const displayRows = Math.abs(b)*columns + Math.abs(d)*rows;
    const scale = Math.min(width / displayColumns, height / displayRows) * view.zoom;
    const minX = Math.min(0,a*(columns-1)) + Math.min(0,c*(rows-1));
    const minY = Math.min(0,b*(columns-1)) + Math.min(0,d*(rows-1));
    return {scale, a:a*scale, b:b*scale, c:c*scale, d:d*scale,
      x:(width - displayColumns*scale)/2 + view.x + scale/2 - minX*scale,
      y:(height - displayRows*scale)/2 + view.y + scale/2 - minY*scale};
  }
  function pixelToScreen(point, transform) {
    const {a,b,c,d,x,y} = transform;
    return [point[0]*a + point[1]*c + x, point[0]*b + point[1]*d + y];
  }
  function screenToPixel(point, transform) {
    const {a,b,c,d,x,y} = transform, determinant = a*d-b*c;
    return [(d*(point[0]-x)-c*(point[1]-y))/determinant, (-b*(point[0]-x)+a*(point[1]-y))/determinant];
  }
  function zoomAt(view, factor, anchor, geometry) {
    const {columns, rows, width, height, display} = geometry;
    const pixel = screenToPixel(anchor, imageTransform(columns, rows, width, height, view, display));
    const next = {...view, zoom:Math.max(.5, Math.min(8, view.zoom * factor))};
    const moved = pixelToScreen(pixel, imageTransform(columns, rows, width, height, next, display));
    next.x += anchor[0] - moved[0]; next.y += anchor[1] - moved[1];
    return next;
  }
  function visibility(keys, previous = null) {
    return new Set(keys.filter(key => !previous || previous.has(key)));
  }
  function setVisible(previous, keys, show) {
    const next = new Set(previous);
    keys.forEach(key => show ? next.add(key) : next.delete(key));
    return next;
  }
  const middleSlice = roi => roi.slices?.length ? roi.slices[Math.floor(roi.slices.length / 2)] : null;
  const color = value => /^#[0-9a-f]{6}$/i.test(value) ? value : '#ffd166';
  async function json(url, signal) {
    const response = await fetch(url, {signal});
    const data = await response.json();
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
    return data;
  }
  class Viewer {
    constructor(host, {overlays = false} = {}) {
      this.host = host; this.overlays = overlays; this.view = {zoom:1, x:0, y:0};
      this.standard = true;
      this.visible = new Set(); this.image = null; this.paths = []; this.token = 0;
      host.innerHTML = `<div class="dicom-viewer">
        <div class="viewer-toolbar"><label>Window<select data-v="window" disabled></select></label>
        <label class="viewer-orientation-option"><input type="checkbox" data-v="standard" checked> Standard orientation</label>
        <div class="button-row"><button type="button" data-v="out" aria-label="Zoom out">−</button><button type="button" data-v="in" aria-label="Zoom in">+</button><button type="button" data-v="reset">Reset view</button></div></div>
        <div class="viewer-stage" data-v="stage" tabindex="0" role="group" aria-label="DICOM image. Arrow keys or wheel: slices. Shift and arrow keys or drag: pan. Plus or minus: zoom. Zero: reset. One: soft tissue, two: lung, three: bone, four: brain.">
          <canvas data-v="canvas" role="img" aria-label="DICOM slice"></canvas><span class="viewer-message" data-v="message" role="status">Choose a series to preview.</span>
          <span class="orientation-marker" data-v="top" hidden></span><span class="orientation-marker" data-v="bottom" hidden></span>
          <span class="orientation-marker" data-v="left" hidden></span><span class="orientation-marker" data-v="right" hidden></span>
        </div>
        <div class="slice-controls"><button type="button" data-v="previous" aria-label="Previous slice" disabled>←</button><label>Slice <span data-v="index">—</span><input data-v="slice" type="range" min="0" max="0" value="0" disabled></label><button type="button" data-v="next" aria-label="Next slice" disabled>→</button></div>
        <p class="viewer-readout" data-v="readout"></p><p class="viewer-help">Scroll or ↑ ↓: slices · Drag or Shift + arrows: pan · + / −: zoom · 0: reset · Windows: 1 soft tissue, 2 lung, 3 bone, 4 brain. Focus the image to use shortcuts.</p>
        <p class="error" data-v="error" role="alert" hidden></p><button type="button" data-v="retry" hidden>Retry viewer</button>
        <section class="overlay-controls" data-v="overlays" aria-label="Contour overlays" hidden>
          <div class="section-heading"><h3>Contour overlays <span class="small muted" data-v="count"></span></h3><div class="button-row"><button type="button" data-v="show">Show all</button><button type="button" data-v="hide">Hide all</button></div></div>
          <div class="overlay-options"><label>Find a structure<input type="search" data-v="search" placeholder="ROI or model"></label><label>Opacity <output data-v="opacityValue">80%</output><input data-v="opacity" type="range" min="0" max="100" value="80"></label><label>Display<select data-v="mode"><option value="outline">Outline</option><option value="filled">Filled</option></select></label></div>
          <p class="small muted" data-v="empty"></p><div class="roi-groups" data-v="groups"></div>
        </section></div>`;
      this.el = name => host.querySelector(`[data-v="${name}"]`);
      this.ctx = this.el('canvas').getContext('2d');
      this.el('slice').oninput = () => this.setSlice(Number(this.el('slice').value));
      this.el('previous').onclick = () => this.setSlice(this.index - 1);
      this.el('next').onclick = () => this.setSlice(this.index + 1);
      this.el('window').onchange = () => { this.window = this.el('window').value; this.schedule(); };
      this.el('standard').onchange = () => {
        const g = this.geometry(), anchor = [g.width/2,g.height/2];
        const pixel = this.info ? screenToPixel(anchor, imageTransform(g.columns,g.rows,g.width,g.height,this.view,g.display)) : null;
        this.standard = this.el('standard').checked;
        if (pixel) {
          const moved = pixelToScreen(pixel, imageTransform(g.columns,g.rows,g.width,g.height,this.view,displayOrientation(this.info.series,this.standard)));
          this.view.x += anchor[0]-moved[0]; this.view.y += anchor[1]-moved[1];
        }
        this.draw();
      };
      this.el('reset').onclick = () => { this.view = {zoom:1, x:0, y:0}; this.draw(); };
      this.el('in').onclick = () => this.zoom(1.25); this.el('out').onclick = () => this.zoom(.8);
      this.el('retry').onclick = () => this.load(this.base);
      const stage = this.el('stage');
      stage.addEventListener('wheel', e => { if (!this.info || !e.deltaY) return; e.preventDefault(); this.setSlice(this.index + Math.sign(e.deltaY)); }, {passive:false});
      stage.addEventListener('keydown', e => {
        if (!this.info || e.altKey || e.ctrlKey || e.metaKey) return;
        if (e.shiftKey && ['ArrowUp','ArrowLeft','ArrowDown','ArrowRight'].includes(e.key)) {
          this.view.x += e.key === 'ArrowLeft' ? -20 : e.key === 'ArrowRight' ? 20 : 0;
          this.view.y += e.key === 'ArrowUp' ? -20 : e.key === 'ArrowDown' ? 20 : 0; this.draw();
        }
        else if (['ArrowUp','ArrowLeft','ArrowDown','ArrowRight'].includes(e.key)) this.setSlice(this.index + (['ArrowUp','ArrowLeft'].includes(e.key) ? -1 : 1));
        else if (['+','=','-'].includes(e.key)) this.zoom(e.key === '-' ? .8 : 1.25);
        else if (e.key === '0') this.el('reset').click();
        else if (['1','2','3','4'].includes(e.key)) {
          const preset = ['soft-tissue','lung','bone','brain'][Number(e.key)-1];
          if (this.info.series.window_presets.some(p => p.id === preset)) { this.window = preset; this.el('window').value = preset; this.schedule(); }
        } else return;
        e.preventDefault();
      });
      stage.addEventListener('pointerdown', e => { if (e.button !== 0 || !this.info) return; stage.focus(); stage.setPointerCapture(e.pointerId); this.drag = {x:e.clientX, y:e.clientY, view:{...this.view}}; });
      stage.addEventListener('pointermove', e => { if (!this.drag) return; this.view.x = this.drag.view.x + e.clientX - this.drag.x; this.view.y = this.drag.view.y + e.clientY - this.drag.y; this.draw(); });
      stage.addEventListener('lostpointercapture', () => { this.drag = null; });
      this.el('search').oninput = () => this.filterROIs();
      for (const [name, show] of [['show',true],['hide',false]]) this.el(name).onclick = () => {
        this.visible = setVisible(this.visible, this.rois.map(r => r.key), show); this.syncROIs(); this.draw();
      };
      this.el('opacity').oninput = () => { this.el('opacityValue').textContent = this.el('opacity').value + '%'; this.draw(); };
      this.el('mode').onchange = () => this.draw();
      this.el('groups').addEventListener('change', e => {
        if (!e.target.matches('[data-roi]')) return;
        this.visible = setVisible(this.visible, [e.target.dataset.roi], e.target.checked); this.syncROIs(); this.draw();
      });
      this.el('groups').addEventListener('click', e => {
        const key = e.target.closest('[data-jump]')?.dataset.jump, roi = this.rois?.find(r => r.key === key);
        if (!roi || middleSlice(roi) === null) return;
        this.visible = setVisible(this.visible, [roi.key], true); this.syncROIs(); this.setSlice(middleSlice(roi)); this.el('stage').focus();
      });
      this.resize = new ResizeObserver(() => this.draw()); this.resize.observe(stage);
    }
    error(message) { this.el('error').textContent = message; this.el('error').hidden = !message; this.el('retry').hidden = !message; }
    async load(base) {
      this.controller?.abort(); clearTimeout(this.timer); const token = ++this.token;
      this.controller = new AbortController(); this.base = base; this.info = null; this.image = null; this.paths = [];
      this.view = {zoom:1,x:0,y:0}; this.error(''); this.el('message').hidden = false; this.el('message').textContent = 'Loading series…';
      this.el('overlays').hidden = true; this.el('readout').textContent = ''; this.el('index').textContent = '—';
      ['slice','window','previous','next'].forEach(key => { this.el(key).disabled = true; }); this.draw();
      try {
        const info = await json(base, this.controller.signal); if (token !== this.token) return;
        this.info = info; this.index = Math.floor(info.series.slices / 2); this.window = info.series.default_window;
        this.el('window').innerHTML = info.series.window_presets.map(p => `<option value="${escape(p.id)}">${escape(p.label)}</option>`).join('');
        this.el('window').value = this.window; this.el('window').disabled = false;
        this.el('slice').max = info.series.slices - 1; this.el('slice').disabled = false;
        this.rois = info.structure_sets.flatMap(s => s.rois.map(r => ({...r, model:s.model})));
        this.visible = visibility(this.rois.map(r => r.key)); this.renderROIs();
        this.setSlice(this.index);
        return info;
      } catch (e) { if (token === this.token && e.name !== 'AbortError') { this.el('message').textContent = 'Viewer unavailable'; this.error(e.message); } }
    }
    setSlice(index) {
      if (!this.info) return;
      this.index = Math.max(0, Math.min(this.info.series.slices - 1, index));
      this.el('slice').value = this.index; this.el('index').textContent = `${this.index + 1} / ${this.info.series.slices} (index ${this.index})`;
      this.el('previous').disabled = this.index === 0; this.el('next').disabled = this.index === this.info.series.slices - 1;
      this.schedule();
    }
    schedule() {
      clearTimeout(this.timer); this.controller?.abort(); ++this.token;
      // Never display old contours over a new slice or leave a stale image labelled as current.
      this.image = null; this.paths = []; this.draw(); this.error('');
      this.el('message').textContent = 'Loading slice…'; this.el('message').hidden = false;
      this.timer = setTimeout(() => this.fetchSlice(), 35);
    }
    async fetchSlice() {
      const token = this.token, index = this.index, controller = this.controller = new AbortController();
      const base = this.base, window = this.window;
      const imageTask = (async () => {
        const response = await fetch(`${base}/slices/${index}.png?window=${encodeURIComponent(window)}`, {signal:controller.signal});
        if (!response.ok) { const data = await response.json(); throw Error(data.detail || `HTTP ${response.status}`); }
        const url = URL.createObjectURL(await response.blob());
        try { const img = new Image(); img.src = url; await img.decode(); return img; } finally { URL.revokeObjectURL(url); }
      })();
      const contourTask = this.overlays && this.rois.length ? json(`${base}/contours/${index}`, controller.signal) : Promise.resolve({index,rois:[]});
      const [pixels, contours] = await Promise.allSettled([imageTask, contourTask]);
      if (token !== this.token) return;
      const errors = [];
      if (pixels.status === 'fulfilled') { this.image = pixels.value; this.el('message').hidden = true; }
      else { errors.push(`Image: ${pixels.reason.message}`); this.el('message').textContent = 'Image unavailable'; }
      if (contours.status === 'fulfilled' && contours.value.index === index) this.paths = contours.value.rois;
      else errors.push(`Contours unavailable: ${contours.reason?.message || 'slice mismatch'}`);
      this.error(errors.join(' · ')); this.draw();
    }
    geometry() { const r = this.el('stage').getBoundingClientRect(); return {...this.info?.series, width:r.width, height:r.height,
      display:displayOrientation(this.info?.series,this.standard)}; }
    zoom(factor) {
      if (!this.info) return; const g = this.geometry(); if (!g.width || !g.height) return;
      this.view = zoomAt(this.view, factor, [g.width / 2, g.height / 2], g); this.draw();
    }
    draw() {
      const canvas = this.el('canvas'), g = this.geometry(); if (!g.width || !g.height) return;
      const ratio = window.devicePixelRatio || 1; canvas.width = Math.round(g.width * ratio); canvas.height = Math.round(g.height * ratio);
      for (const edge of ['top','bottom','left','right']) this.el(edge).hidden = !this.info;
      if (!this.info) return;
      const p = this.info.series.window_presets.find(p => p.id === this.window);
      this.el('readout').textContent = `${this.info.series.modality} · ${this.info.series.orientation} · ${g.columns} × ${g.rows} · ${p?.label || this.window} · C ${p?.center ?? '—'} / W ${p?.width ?? '—'} · Zoom ${Math.round(this.view.zoom * 100)}%`;
      const t = imageTransform(g.columns, g.rows, g.width, g.height, this.view, g.display), ctx = this.ctx;
      const markers = orientationMarkers(this.info.series,g.display);
      const corners = [[-.5,-.5],[g.columns-.5,-.5],[-.5,g.rows-.5],[g.columns-.5,g.rows-.5]].map(p => pixelToScreen(p,t));
      const xs = corners.map(p => p[0]), ys = corners.map(p => p[1]);
      const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
      const positions = {top:[(left+right)/2,top+14],bottom:[(left+right)/2,bottom-14],left:[left+14,(top+bottom)/2],right:[right-14,(top+bottom)/2]};
      for (const [edge,label] of Object.entries(markers)) {
        const el = this.el(edge); el.textContent = label; el.setAttribute('aria-label', `${edge}: ${label}`);
        el.style.left = `${Math.max(18,Math.min(g.width-18,positions[edge][0]))}px`;
        el.style.top = `${Math.max(14,Math.min(g.height-14,positions[edge][1]))}px`;
      }
      if (!this.image) return;
      ctx.setTransform(t.a * ratio, t.b * ratio, t.c * ratio, t.d * ratio, t.x * ratio, t.y * ratio);
      ctx.drawImage(this.image, -.5, -.5, g.columns, g.rows);
      // Image and contour pixel centres share this signed affine, including clipping.
      ctx.beginPath(); ctx.rect(-.5, -.5, g.columns, g.rows); ctx.clip();
      ctx.lineWidth = 1.6 / t.scale; ctx.lineJoin = 'round';
      ctx.globalAlpha = Number(this.el('opacity').value) / 100;
      for (const roi of this.paths) {
        if (!this.visible.has(roi.key)) continue;
        const metadata = this.rois.find(r => r.key === roi.key); ctx.strokeStyle = ctx.fillStyle = color(metadata?.color);
        ctx.beginPath();
        for (const path of roi.paths) { path.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.closePath(); }
        if (this.el('mode').value === 'filled') ctx.fill('evenodd');
        ctx.stroke();
      }
      canvas.setAttribute('aria-label', `DICOM ${this.info.series.modality}, slice ${this.index + 1} of ${g.slices}, ${p?.label || this.window} window`);
    }
    renderROIs() {
      this.el('overlays').hidden = !this.overlays;
      this.el('empty').textContent = this.rois.length ? '' : 'No RTSTRUCT structures are available for this job.';
      this.el('groups').innerHTML = this.info.structure_sets.map(s => `<details open><summary>${escape(s.model)} <span class="muted">· ${escape(s.label)} · ${s.rois.length} ROI${s.rois.length === 1 ? '' : 's'}</span></summary>${s.rois.map(r => `<div class="roi-row" data-search="${escape((s.model+' '+r.name).toLowerCase())}"><label><input type="checkbox" data-roi="${escape(r.key)}" checked><span class="swatch" style="background:${color(r.color)}" aria-hidden="true"></span><span>${escape(r.name)}</span></label><button type="button" data-jump="${escape(r.key)}" aria-label="Jump to middle slice of ${escape(r.name)}" ${middleSlice(r) === null ? 'disabled title="No matching source slices"' : ''}>Jump to</button></div>`).join('')}</details>`).join('');
      this.syncROIs(); this.filterROIs();
    }
    syncROIs() {
      this.el('groups').querySelectorAll('[data-roi]').forEach(el => { el.checked = this.visible.has(el.dataset.roi); });
      this.el('count').textContent = `${this.visible.size} / ${this.rois.length} shown`;
    }
    filterROIs() {
      const query = this.el('search').value.trim().toLowerCase();
      this.el('groups').querySelectorAll('details').forEach(group => {
        const rows = [...group.querySelectorAll('[data-search]')]; rows.forEach(row => { row.hidden = !row.dataset.search.includes(query); });
        group.hidden = rows.every(row => row.hidden); if (query) group.open = true;
      });
    }
    destroy() { ++this.token; clearTimeout(this.timer); this.controller?.abort(); this.resize.disconnect(); this.host.replaceChildren(); }
  }
  return {Viewer, displayOrientation, directionLabel, orientationMarkers, imageTransform, pixelToScreen, screenToPixel, zoomAt, visibility, setVisible, middleSlice};
})();
if (typeof module !== 'undefined' && module.exports) module.exports = DicomUI;

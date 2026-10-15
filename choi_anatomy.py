"""Anatomical evidence for Choi numeric labels; proposals require clinician review.

All measurements use NIfTI RAS millimetres: +x right, +y anterior, +z superior.
DICOM LPS is obtained by negating x and y; patient left is +x in LPS, -x in RAS.
No registration or interpolation of the measured masks is performed.
"""
import argparse
import base64
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_erosion, map_coordinates, label as connected_components
from scipy.spatial import cKDTree

from choi_ontology import REGIONS

NEIGHBOURS = ('aorta', 'inferior_vena_cava', 'iliac_artery_left', 'iliac_artery_right',
              'iliac_vena_left', 'iliac_vena_right', 'femur_left', 'femur_right',
              'hip_left', 'hip_right', 'sacrum', 'humerus_left', 'humerus_right',
              'scapula_left', 'scapula_right', 'trachea', 'heart', 'spinal_cord')
CONVENTION = ('NIfTI affine maps voxels to RAS mm (+x right, +y anterior, +z superior). '
              'DICOM LPS = (-RAS x, -RAS y, RAS z). Left is negative RAS x relative to the local spinal midline. '
              'Review panels are radiological: patient right at image left; axial top anterior, coronal top superior.')
RULES = '''Rules use physical centroids and centroid-to-mask distances, not numeric label order.
Cervical: above T1, within 80 mm of a cervical vertebra. Axillary: lateral displacement >60 mm,
within 60 mm of the ipsilateral humerus/scapula, above T6. Mediastinal: T1–T12, displacement <70 mm,
within 80 mm of trachea/heart/aorta; T4/T5 midpoint is a provisional superior/inferior separator.
The inferior tracheal end is reported as a carina proxy, not a detected bifurcation.
Para-aortic: T12–L5, within 60 mm of aorta/IVC. Common iliac: L4 to 20 mm below L5,
within 50 mm of iliac vessels. Presacral: within the sacral z extent, <30 mm from midline,
0–100 mm anterior to sacral centroid, within 60 mm of sacrum. Below L5 and within 70 mm of
ipsilateral hip/iliac vessels: external iliac if anterior to hip centroid; internal iliac
if posterior; obturator if medial to hip centroid and at/below proximal femur +20 mm.
Inguinal: below proximal femur top, anterior to proximal femur centroid, within 80 mm of
femur/iliac vessels. Total masks do not separate internal/external iliac vessels or delineate
the obturator fossa, femoral head or carina. Bone/vessel proxies cannot resolve those boundaries.
Scores (0–1) are heuristic evidence strength, not calibrated probabilities. Missing landmarks,
midline extrapolation, competing rules, duplicate proposed regions, widely separated components and sparse predictions lower
confidence. Missing labels are unresolved; a one-to-one assignment is never forced.'''


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_image(path, reference=None, *, labels=False):
    img = nib.load(path)
    if len(img.shape) != 3 or not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-8:
        raise ValueError(f'Invalid 3D geometry: {Path(path).name}')
    units=img.header.get_xyzt_units()[0]
    if units != 'mm' and not (labels and reference is not None and units == 'unknown'):
        raise ValueError(f'Explicit millimetre NIfTI units required: {Path(path).name}')
    # Lossless permutations/reversals are allowed. Other grid differences fail closed.
    img = nib.as_closest_canonical(img)
    if reference is not None and (img.shape != reference.shape or not np.allclose(img.affine, reference.affine, atol=1e-3, rtol=0)):
        raise ValueError(f'Grid mismatch; use masks of the same CT: {Path(path).name}')
    arr = np.asanyarray(img.dataobj)
    if not np.isfinite(arr).all():
        raise ValueError(f'Non-finite values: {Path(path).name}')
    if labels and (np.any(arr < 0) or np.any(arr != np.rint(arr))):
        raise ValueError(f'Expected nonnegative integer labels: {Path(path).name}')
    return img, arr


def describe(points, voxel_volume):
    centre = points.mean(axis=0)
    return {'voxels': len(points), 'volume_ml': float(len(points) * voxel_volume / 1000),
            'centroid_ras_mm': centre.tolist(), 'centroid_lps_mm': (centre * [-1, -1, 1]).tolist(),
            'bounds_ras_mm': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
            'z_extent_mm': [float(points[:, 2].min()), float(points[:, 2].max())]}


def side_evidence(points, midline):
    if not len(midline):
        return {'side': 'unresolved', 'reason': 'No spinal cord or vertebral midline', 'midline_extrapolated': True}
    z, x = midline[:, 2], midline[:, 0]
    dx = points[:, 0] - np.interp(points[:, 2], z, x)
    left, right = float(np.mean(dx < -3)), float(np.mean(dx > 3))
    return {'side': 'Left' if left >= .8 else 'Right' if right >= .8 else 'midline_or_mixed',
            'left_fraction': left, 'right_fraction': right, 'midline_band_fraction': 1-left-right,
            'mean_offset_ras_x_mm': float(dx.mean()),
            'midline_extrapolated': bool(points[:, 2].min() < z.min() or points[:, 2].max() > z.max())}


def propose(row, landmarks):
    if not row['voxels']:
        return [], ['Label absent; review another prediction or case']
    centre = np.array(row['centroid_ras_mm']); x, y, z = centre
    side = row['side']['side']; candidates = []; flags = []
    dist = {n['structure']: n['centroid_to_mask_mm'] for n in row['neighbours']}
    def near(*names): return min((dist.get(n, float('inf')) for n in names), default=float('inf'))
    def level(name):
        item = landmarks.get('vertebrae_' + name)
        return item['centroid_ras_mm'][2] if item else None
    def between(low, high): return low is not None and high is not None and low <= z <= high
    def add(region, score, rule, bilateral=False):
        choices = [side] if side in ('Left', 'Right') else ['Left', 'Right']
        for s in choices if bilateral else ['']:
            candidates.append({'region': region + ('_' + s if s else ''),
                               'score': score if not bilateral or len(choices)==1 else min(score,.2), 'rule': rule})
    offset = abs(row['side'].get('mean_offset_ras_x_mm', float('inf')))
    t1,t6,t12,l4,l5 = [level(n) for n in ('T1','T6','T12','L4','L5')]
    if t1 is not None and z > t1 and near(*[n for n in landmarks if n.startswith('vertebrae_C')]) < 80:
        add('Cervical', .75, 'Above T1 and near cervical vertebrae', True)
    for s in ([side] if side in ('Left','Right') else []):
        if t6 is not None and z > t6 and offset > 60 and near('humerus_'+s.lower(),'scapula_'+s.lower()) < 60:
            add('Axillary', .7, 'Lateral shoulder region above T6', True)
    if between(t12,t1) and offset < 70 and near('trachea','heart','aorta') < 80:
        t4,t5 = level('T4'),level('T5')
        split = (t4+t5)/2 if t4 is not None and t5 is not None else None
        add('Superior_Mediastinal', .4 if split is not None and z >= split else .25, 'Thoracic midline; T4/T5 separator is provisional')
        add('Inferior_Mediastinal', .4 if split is not None and z < split else .25, 'Thoracic midline; T4/T5 separator is provisional')
        flags.append('Superior/inferior mediastinal boundary requires clinician review; carina is only a proxy')
        extent=row.get('z_extent_mm',[])
        if split is not None and len(extent)==2 and extent[0]<=split<=extent[1]:
            flags.append('Label crosses the provisional T4/T5 separator')
    if between(l5,t12) and near('aorta','inferior_vena_cava') < 60:
        add('Para_Aortic', .65, 'T12–L5 beside aorta/IVC')
    if between(l5-20 if l5 is not None else None,l4) and near(*[n for n in NEIGHBOURS if n.startswith('iliac_')]) < 50:
        add('Common_Iliac', .45, 'L4–L5 region beside iliac vessels', True)
        flags.append('Common/internal/external iliac boundary is not segmented by total')
    sacrum = landmarks.get('sacrum')
    if sacrum and between(*sacrum['z_extent_mm']) and offset < 30 and 0 < y-sacrum['centroid_ras_mm'][1] < 100 and near('sacrum') < 60:
        add('Presacral', .55, 'Anterior to sacrum near spinal midline')
    if l5 is not None and z < l5 and side in ('Left','Right'):
        suffix=side.lower(); hip=landmarks.get('hip_'+suffix); femur=landmarks.get('femur_'+suffix)
        vessels=('iliac_artery_'+suffix,'iliac_vena_'+suffix)
        if hip and near('hip_'+suffix,*vessels) < 70:
            anterior = y > hip['centroid_ras_mm'][1]
            add('External_Iliac', .4 if anterior else .25, 'Pelvic vessel/hip proximity; anterior hip proxy', True)
            add('Internal_Iliac', .35 if not anterior else .25, 'Pelvic vessel/hip proximity; posterior hip proxy', True)
            femur_top = femur['z_extent_mm'][1] if femur else None
            medial = abs(x - row.get('midline_x_at_centroid_mm',x)) < abs(hip['centroid_ras_mm'][0]-row.get('midline_x_at_centroid_mm',x))
            add('Obturator', .4 if medial and femur_top is not None and z <= femur_top+20 else .2,
                'Medial hip at proximal femur level is only an obturator proxy', True)
            flags.append('External/internal iliac versus obturator unresolved by total anatomy alone')
        if femur and z < femur['z_extent_mm'][1] and y > femur['proximal_centroid_ras_mm'][1] and near('femur_'+suffix,*vessels) < 80:
            add('Inguinal', .55, 'Anterior to proximal femur below its superior extent', True)
            flags.append('Inguinal versus low external iliac/obturator needs review of inguinal ligament')
    if row['side']['midline_extrapolated']:
        flags.append('Spinal midline unavailable at part of label extent')
        for item in candidates: item['score']=min(item['score'],.35)
    if row['voxels'] < 20:
        flags.append('Fewer than 20 voxels')
        for item in candidates: item['score']=min(item['score'],.2)
    components=row.get('largest_components',[])
    if row.get('largest_component_fraction',1)<.8 and len(components)>1:
        separation=np.linalg.norm(np.array(components[0]['centroid_ras_mm'])-components[1]['centroid_ras_mm'])
        if separation>80:
            flags.append('Substantial disconnected components are >80 mm apart; the aggregate centroid may obscure their anatomy')
            for item in candidates:item['score']=min(item['score'],.35)
    candidates.sort(key=lambda item: (-item['score'], item['region']))
    if len(candidates)>1 and candidates[0]['score']-candidates[1]['score']<.2:
        flags.append('Competing anatomical rules')
    if not candidates: flags.append('Insufficient landmarks or no anatomical rule matched')
    return candidates,flags


def vertebral_extent(row, landmarks):
    levels={n:s for n,s in landmarks.items() if n.startswith('vertebrae_')}
    if not levels or not row['voxels']: return None
    lo,hi=row['z_extent_mm']
    nearest=lambda z: min(levels,key=lambda n:abs(levels[n]['centroid_ras_mm'][2]-z))
    return {'caudal_nearest_centroid':nearest(lo), 'cranial_nearest_centroid':nearest(hi),
            'overlapping_z_ranges':[n for n,s in sorted(levels.items(),key=lambda p:p[1]['centroid_ras_mm'][2])
                                    if s['z_extent_mm'][0]<=hi and s['z_extent_mm'][1]>=lo],
            'note':'Nearest vertebral centroids and overlapping z ranges; not vertebral boundary assignment'}


def outline_panel(ct, pred, affine, label, centre, bounds, plane):
    # Sample a true world-axis plane. This also handles oblique source affines.
    low,high=np.array(bounds)
    a,b=(0,1) if plane=='axial' else (0,2)
    lower=np.array([low[a]-65,low[b]-65]);upper=np.array([high[a]+65,high[b]+65])
    size=np.maximum(2,np.ceil((upper-lower)/1.5).astype(int))
    scale=max(1,float(size.max())/650);size=np.maximum(2,np.ceil(size/scale).astype(int))
    horizontal=np.linspace(upper[0],lower[0],size[0])  # patient right at image left
    vertical=np.linspace(upper[1],lower[1],size[1])
    grid_a,grid_b=np.meshgrid(horizontal,vertical)
    world=np.broadcast_to(np.asarray(centre)[:,None,None],(3,*grid_a.shape)).copy()
    world[a]=grid_a;world[b]=grid_b
    inverse=np.linalg.inv(affine)
    vox=np.einsum('ij,jkl->ikl',inverse[:3,:3],world)+inverse[:3,3,None,None]
    grey=np.clip((map_coordinates(ct,vox,order=1,mode='constant',cval=-1000)+160)/400*255,0,255).astype(np.uint8)
    mask=map_coordinates(pred,vox,order=0,mode='constant',cval=0)==label
    rgb=np.repeat(grey[:,:,None],3,axis=2);rgb[mask & ~binary_erosion(mask)]=[255,65,45]
    panel=Image.new('RGB',(520,520),'black')
    resized=ImageOps.contain(Image.fromarray(rgb),(500,470))
    panel.paste(resized,((520-resized.width)//2,40+(470-resized.height)//2))
    draw=ImageDraw.Draw(panel)
    draw.text((12,8),f'{plane.title()}   R (patient)                           L (patient)',fill='white')
    draw.text((12,24),f'Top: {"anterior" if plane=="axial" else "superior"}; plane {"z" if plane=="axial" else "y"}={centre[2 if plane=="axial" else 1]:.1f} RAS mm',fill='white')
    return panel


def review_png(ct,pred,affine,row,points,path):
    sheet=Image.new('RGB',(1040,565),'black'); draw=ImageDraw.Draw(sheet)
    draw.text((15,8),f'Choi label {row["label"]:02d} | {row["volume_ml"]:.2f} mL | Proposed: {row["proposal"] or "unresolved"} | score {row["confidence"]:.2f}',fill='white')
    if len(points):
        # Review the largest component closely; full-label extents and other
        # component centroids remain in the JSON/HTML for full-volume review.
        centre=points[np.argmin(np.sum((points-np.array(row['centroid_ras_mm']))**2,axis=1))]
        if row.get('largest_components'):
            component=row['largest_components'][0]
            centre=np.array(row['review_largest_component_point_ras_mm'])
            bounds=component['bounds_ras_mm']
            row['review_component_note']='Largest 6-neighbour component; inspect all components in the original volume'
            draw.text((15,22),f'Largest component: {100*row["largest_component_fraction"]:.1f}% of label; {row["component_count"]} component(s). Full extent is in the report.',fill='white')
        else:bounds=row['bounds_ras_mm']
        row['review_plane_point_ras_mm']=centre.tolist()
        for i,plane in enumerate(('axial','coronal')):
            sheet.paste(outline_panel(ct,pred,affine,row['label'],centre,bounds,plane),(520*i,35))
    else: draw.text((30,70),'ABSENT: no contour to review. Use another fold or case; no name can be inferred here.',fill='white')
    sheet.save(path)
    return base64.b64encode(path.read_bytes()).decode()


def write_review(out,report,images):
    options='<option value="">Choose after review</option>'+''.join(f'<option>{n}</option>' for n in REGIONS)
    cards=[]
    for row,png in zip(report['labels'],images):
        facts={k:v for k,v in row.items() if k not in ('neighbours',)}
        neighbours=sorted(row['neighbours'],key=lambda n:n['minimum_voxel_centre_distance_mm'])[:10]
        cards.append(f'<section><h2>Label {row["label"]:02d}</h2><img alt="Label {row["label"]} axial and coronal outlines" src="data:image/png;base64,{png}"><pre>{html.escape(json.dumps(facts,indent=2))}</pre><details><summary>Nearest anatomy (mm)</summary><pre>{html.escape(json.dumps(neighbours,indent=2))}</pre></details><label>Confirmed region <select data-label="{row["label"]}">{options}</select></label><p>Review notes: <input data-note="{row["label"]}" size="90"></p></section>')
    evidence=html.escape('prediction-sha256:'+report['inputs']['prediction_sha256'],quote=True)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Choi label review</title>
<style>body{font:16px sans-serif;margin:2em;max-width:1100px}img{max-width:100%}section{break-inside:avoid;border-top:2px solid #555;padding:1em 0}pre{white-space:pre-wrap;font-size:13px}input,select{font:inherit}label{display:block}button{padding:.7em}</style>
<h1>Choi numeric label review</h1><p>These are anatomical proposals. Confirm every label using the original volume and, where a label is absent or ambiguous, other folds or cases. One slice pair cannot show the full extent. This sheet does not establish clinical or manuscript readiness.</p>'''
    page+='<p>'+html.escape(CONVENTION)+'</p><details><summary>Explicit rules and limitations</summary><pre>'+html.escape(RULES)+'</pre></details>'
    page+=''.join(cards)+f'<h2>Review record</h2><label>Clinician reviewer <input id="reviewer"></label><label>Evidence identifiers (one per line; include additional cases if used)<textarea id="evidence" rows="4" cols="90">{evidence}</textarea></label><label><input type="checkbox" id="confirm"> I reviewed all 18 assignments and resolved missing labels and ambiguities.</label><p><button id="export">Download confirmed mapping JSON</button></p><p id="message"></p>'
    page+='''<script>document.getElementById('export').onclick=()=>{const labels={},notes={};document.querySelectorAll('[data-label]').forEach(e=>labels[e.dataset.label]=e.value);document.querySelectorAll('[data-note]').forEach(e=>notes[e.dataset.note]=e.value);const reviewer=document.getElementById('reviewer').value.trim(),evidence=document.getElementById('evidence').value.split('\\n').map(s=>s.trim()).filter(Boolean);if(!document.getElementById('confirm').checked||!reviewer||!evidence.length||Object.values(labels).some(x=>!x)||new Set(Object.values(labels)).size!==18){document.getElementById('message').textContent='Complete all 18 distinct regions, reviewer, evidence and confirmation.';return;}const record={schema_version:1,confirmed:true,reviewer,reviewed_utc:new Date().toISOString(),evidence,labels,notes,source:'https://zenodo.org/records/7839889'};const u=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)+'\\n'],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='choi-tmli-reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};</script></html>'''
    (out/'review.html').write_text(page)


def analyze(prediction,ct_path,total_dir,out):
    ct_img,ct=load_image(ct_path)
    pred_img,pred=load_image(prediction,ct_img,labels=True)
    if np.any(pred>18): raise ValueError('Choi prediction has labels outside 0–18')
    pred=pred.astype(np.uint8);affine=ct_img.affine
    volume=abs(np.linalg.det(affine[:3,:3]));rows=[];points_by_label={};surface_by_label={}
    for label in range(1,19):
        mask=pred==label;indices=np.argwhere(mask);points=nib.affines.apply_affine(affine,indices)
        row={'label':label, 'neighbours':[], 'voxels':0, 'volume_ml':0.0}
        if len(points):
            row.update(describe(points,volume))
            # Cropping avoids a full-volume erosion for each small target.
            lo=indices.min(axis=0);hi=indices.max(axis=0)+1;slices=tuple(slice(a,b) for a,b in zip(lo,hi))
            crop=mask[slices];surface=np.argwhere(crop & ~binary_erosion(crop))+lo
            surface_by_label[label]=nib.affines.apply_affine(affine,surface)
            components,count=connected_components(crop)
            ids=components[tuple((indices-lo).T)];counts=np.bincount(ids)
            largest=np.argsort(counts[1:])[::-1][:3]+1
            row['component_count']=int(count)
            row['largest_component_fraction']=float(counts[largest[0]]/len(points))
            row['largest_components']=[]
            for rank,component_id in enumerate(largest):
                component_points=points[ids==component_id]
                stats=describe(component_points,volume);row['largest_components'].append(stats)
                if rank==0:
                    distance=np.sum((component_points-np.array(stats['centroid_ras_mm']))**2,axis=1)
                    row['review_largest_component_point_ras_mm']=component_points[distance.argmin()].tolist()
            del components,ids,counts
        points_by_label[label]=points;rows.append(row)
    landmarks={};vertebral_midline=[];cord_midline=[];input_masks={}
    inferred_units=['prediction'] if pred_img.header.get_xyzt_units()[0]=='unknown' else []
    for path in sorted(Path(total_dir).glob('*.nii.gz')):
        name=path.name[:-7]
        if name not in NEIGHBOURS and not name.startswith('vertebrae_'): continue
        mask_img,arr=load_image(path,ct_img,labels=True)
        if mask_img.header.get_xyzt_units()[0]=='unknown': inferred_units.append(name)
        if np.any(arr>1): raise ValueError(f'Expected separate binary TotalSegmentator masks: {name}')
        indices=np.argwhere(arr>0)
        input_masks[name]=digest(path)
        if not len(indices): continue
        points=nib.affines.apply_affine(affine,indices)
        stats=describe(points,volume);landmarks[name]=stats
        if name.startswith('vertebrae_'): vertebral_midline.append(stats['centroid_ras_mm'])
        if name=='spinal_cord':
            _,groups=np.unique(np.round(points[:,2]/5).astype(int),return_inverse=True)
            counts=np.bincount(groups)
            cord_midline=np.column_stack([np.bincount(groups,weights=points[:,axis])/counts for axis in range(3)]).tolist()
        if name.startswith('femur_'):
            stats['proximal_centroid_ras_mm']=points[points[:,2]>=np.quantile(points[:,2],.9)].mean(axis=0).tolist()
            stats['proxy_note']='Superior 10% of femur voxels; not a segmented femoral head'
        if name=='trachea': stats['carina_proxy_ras_z_mm']=stats['z_extent_mm'][0]
        tree=cKDTree(points)
        overlapping_labels=set(np.unique(pred[tuple(indices.T)]).tolist())
        for row in rows:
            if not row['voxels']: continue
            surface=surface_by_label[row['label']]
            # Zero for any overlap, including a completely enclosed mask.
            overlap=row['label'] in overlapping_labels
            minimum=0.0 if overlap else float(tree.query(surface,workers=1)[0].min())
            row['neighbours'].append({'structure':name,'minimum_voxel_centre_distance_mm':minimum,
                                      'centroid_to_mask_mm':float(tree.query(row['centroid_ras_mm'])[0])})
        del tree,points,indices,arr
    # Cord provides a local midline. Vertebral centroids extend it where the cord is absent.
    midpoints=cord_midline[:]
    if cord_midline:
        lo=min(p[2] for p in cord_midline);hi=max(p[2] for p in cord_midline)
        midpoints += [p for p in vertebral_midline if p[2]<lo or p[2]>hi]
    else: midpoints=vertebral_midline
    midline=np.array(sorted(midpoints,key=lambda p:p[2])).reshape(-1,3)
    if len(midline):
        # Repeated z coordinates can occur in deformed spines; use one mean per z.
        midline=np.array([midline[midline[:,2]==z].mean(axis=0) for z in np.unique(midline[:,2])])
    for row in rows:
        row['side']=side_evidence(points_by_label[row['label']],midline) if row['voxels'] else {'side':'absent'}
        if row['voxels'] and len(midline): row['midline_x_at_centroid_mm']=float(np.interp(row['centroid_ras_mm'][2],midline[:,2],midline[:,0]))
        row['vertebral_extent']=vertebral_extent(row,landmarks)
        row['candidates'],row['flags']=propose(row,landmarks)
        row['proposal']=row['candidates'][0]['region'] if row['candidates'] else None
        row['confidence']=row['candidates'][0]['score'] if row['candidates'] else 0.0
    counts=Counter(r['proposal'] for r in rows if r['proposal'])
    for row in rows:
        if counts.get(row['proposal'],0)>1:
            row['flags'].append('Duplicate region proposed for several labels; no bijection forced')
            row['confidence']=min(row['confidence'],.35)
    report={'schema_version':1,'confirmed':False,'coordinate_convention':CONVENTION,'rules':RULES,
            'distance_definition':'Euclidean distances between voxel centres in physical RAS mm; no sampling or registration',
            'inputs':{'ct_sha256':digest(ct_path),'prediction_sha256':digest(prediction),'total_masks_sha256':input_masks},
            'grid':{'shape':list(ct.shape),'affine_ras_mm':affine.tolist()},
            'mask_units_inferred_from_identical_ct_grid':inferred_units,
            'missing_landmarks':[n for n in NEIGHBOURS if n not in landmarks],
            'midline_ras_mm':midline.tolist(),'landmarks':landmarks,'labels':rows}
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    images=[review_png(ct,pred,affine,row,points_by_label[row['label']],out/f'label_{row["label"]:02d}.png') for row in rows]
    (out/'anatomy.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_review(out,report,images)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',type=Path,required=True)
    parser.add_argument('--ct',type=Path,required=True,help='Original CT in HU')
    parser.add_argument('--total',type=Path,required=True,help='Separate TotalSegmentator total binary masks on the same grid')
    parser.add_argument('--output',type=Path,required=True,help='New review directory')
    args=parser.parse_args(argv)
    if not args.total.is_dir(): parser.error('TotalSegmentator mask directory does not exist')
    report=analyze(args.prediction,args.ct,args.total,args.output)
    print(json.dumps({'labels_present':sum(bool(r['voxels']) for r in report['labels']),
                      'proposed':sum(bool(r['proposal']) for r in report['labels']),
                      'confirmed':False,'review':str(args.output/'review.html')}))


if __name__=='__main__': main()

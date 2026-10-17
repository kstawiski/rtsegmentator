"""CPU-only contracts and rendering for research CT draft reports.

Coverage is inferred from TotalSegmentator total masks, not from a clinical reading.
All report inputs use an allowlist: no DICOM demographics or identifiers are copied.
"""
from __future__ import annotations

import base64
import fnmatch
import html
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

TASK = 'nv_reason_ct_report'
REVISION = '386b93e034983f6c1fc841a43833a1b6a0cd9c13'
BANNER = 'RESEARCH ONLY — AI DRAFT. Requires review by a radiologist. Not a diagnostic device.'
LUNGS = ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
         'lung_middle_lobe_right', 'lung_lower_lobe_right']
REGION_ORGANS = {'chest': [*LUNGS, 'heart'],
                 'abdomen': ['liver', 'spleen', 'kidney_left', 'kidney_right', 'pancreas']}
MAPPING = json.loads((Path(__file__).parent / 'labels/nv-reason-ct-organs.json').read_text())


def wait_for_free_memory(probe, timeout, required_gib=14, clock=time.monotonic, sleep=time.sleep):
    """Bounded resource gate; probe returns free bytes or None when CUDA is unavailable."""
    deadline = clock() + timeout
    while True:
        free = probe()
        if free is not None and free >= required_gib * 2**30: return round(free / 2**30, 3)
        remaining = deadline - clock()
        if remaining <= 0: return None
        sleep(min(5, remaining))


def detect_cbct(ds) -> dict:
    """Known device names, explicit CBCT text, or a conservative acquisition heuristic.

    Heuristic: Varian/Elekta manufacturer AND Ram-Lak kernel AND either 100–150 kV
    or a 450–480 mm reconstruction diameter. Missing tags alone never imply CBCT.
    Never echo arbitrary DICOM text into the report.
    """
    def value(key): return str(getattr(ds, key, '')).lower()
    model = value('ManufacturerModelName')
    names = ('patient verification', 'advanced reconstruction', 'obi cone-beam ct',
             'acuity cone-beam ct', 'halcyon - pva', 'xvi')
    if any(name in model for name in names):
        return {'is_cbct': True, 'reason': 'CBCT device model (diagnostic CT required)'}
    text = ' '.join(value(k) for k in ('Manufacturer', 'ManufacturerModelName',
                                      'SeriesDescription', 'ProtocolName', 'ImageType'))
    if re.search(r'cone[ -]?beam|\bcbct\b|\bxvi\b', text):
        return {'is_cbct': True, 'reason': 'CBCT acquisition keyword (diagnostic CT required)'}
    def number(key):
        try: return float(getattr(ds, key, 0))
        except (TypeError, ValueError): return 0
    if (any(x in value('Manufacturer') for x in ('varian', 'elekta'))
            and 'ram' in value('ConvolutionKernel') and 'lak' in value('ConvolutionKernel')
            and (100 <= number('KVP') <= 150 or 450 <= number('ReconstructionDiameter') <= 480)):
        return {'is_cbct': True, 'reason': 'CBCT acquisition heuristic (manufacturer, kernel and geometry/kV)'}
    return {'is_cbct': False, 'reason': 'No CBCT header indicator detected; this does not establish image quality'}


def hu_check(data) -> dict:
    data = np.asarray(data)
    if data.size == 0 or not np.isfinite(data).all():
        return {'looks_like_hu': False, 'reason': 'Empty or nonfinite intensities'}
    p = np.percentile(data, [0.1, 1, 50, 99, 99.9])
    # Reuse the reference input-QC rule; air and high-density tissue must be visible.
    return {'looks_like_hu': bool(-1100 <= p[0] <= -900 and data.max() > 300),
            'hu_percentiles': [round(float(x), 2) for x in p],
            'hu_min': float(data.min()), 'hu_max': float(data.max()),
            'rule': '0.1 percentile between -1100 and -900 HU and maximum >300 HU'}


def mask_stats(mask, voxel_mm3, slice_axis=2) -> dict:
    mask = np.asarray(mask, dtype=bool)
    axes = tuple(i for i in range(3) if i != slice_axis)
    slices = np.flatnonzero(mask.any(axis=axes))
    return {'volume_ml': round(float(mask.sum()) * voxel_mm3 / 1000, 4),
            'slice_range': [int(slices[0]), int(slices[-1])] if slices.size else None,
            'touches_first_slice': bool(slices.size and slices[0] == 0),
            'touches_last_slice': bool(slices.size and slices[-1] == mask.shape[slice_axis] - 1)}


def covered_regions(coverage) -> list[str]:
    def volume(key): return coverage.get(key, {}).get('volume_ml', 0)
    result = []
    if sum(volume(k) for k in LUNGS) >= 1000 and volume('heart') >= 100:
        result.append('chest')
    if volume('liver') >= 500 and max(volume('kidney_left'), volume('kidney_right')) >= 50:
        result.append('abdomen')
    return result


def organ_status(coverage, spec, region=None) -> str:
    classes = spec.get('regional_classes', {}).get(region, spec['classes'])
    matched = [v for k, v in coverage.items() if any(fnmatch.fnmatchcase(k, p) for p in classes)]
    if not matched or sum(v['volume_ml'] for v in matched) <= 0: return 'absent'
    # Regional bones are expected to meet the scan boundary; presence is the relevant check.
    if spec.get('regional_presence'): return 'present'
    if (sum(v['volume_ml'] for v in matched) < spec['minimum_ml']
            or any(v.get('touches_first_slice') or v.get('touches_last_slice') for v in matched)
            or (spec.get('require_all') and any(not any(fnmatch.fnmatchcase(k, p) and v['volume_ml'] > 0
                for k, v in coverage.items()) for p in spec['classes']))):
        return 'partial'
    return 'present'


def parse_sections(raw) -> dict:
    """Accept uppercase/Markdown section names and multiline organ statements."""
    body = raw.split('</think>')[-1] if '</think>' in raw else re.sub(r'<think>.*', '', raw, flags=re.S)
    answers = re.findall(r'<answer>(.*?)</answer>', body, flags=re.S | re.I)
    body = re.sub(r'<answer>.*?(?:</answer>|$)', '', body, flags=re.S | re.I)
    result = {'technique': [], 'findings': [], 'impression': [], 'labels': []}
    section = 'findings'
    for line in body.splitlines():
        line = line.strip().strip('#').strip()
        if not line: continue
        clean = line.replace('**', '').strip()
        header = re.match(r'^(TECHNIQUE|FINDINGS|IMPRESSION)\s*:?\s*(.*)$', clean, re.I)
        if header:
            section = header[1].lower()
            clean = header[2]
            if not clean: continue
        clean = re.sub(r'^[-*]\s+|^\d+[.)]\s*', '', clean)
        if section == 'findings':
            heading = re.match(r'^([^:]{1,100}):\s*(.*)', clean)
            if heading:
                result[section].append({'heading': heading[1], 'text': heading[2]})
            elif result[section]: result[section][-1]['text'] += ' ' + clean
            else: result[section].append({'heading': 'Unmapped finding', 'text': clean})
        else: result[section].append(clean)
    for answer in answers:
        result['labels'].extend(x.strip(' \n\t-\"[]') for x in re.split(r'[\n;,]+', answer) if x.strip())
    return result


def check_consistency(raw, coverage, region=None) -> dict:
    parsed = parse_sections(raw)
    statuses = {key: organ_status(coverage, spec, region) for key, spec in MAPPING.items()}
    aliases = {alias.casefold(): key for key, spec in MAPPING.items() for alias in spec['headings']}
    flags = []
    if not parsed['findings'] or not parsed['impression']:
        flags.append({'status': 'unstructured_output', 'reason': 'Expected findings or impression section is missing'})
    for finding in parsed['findings']:
        key = aliases.get(finding['heading'].casefold())
        status = statuses[key] if key else 'unmapped'
        finding['organ_status'] = status
        if status == 'absent':
            finding['text'] = 'Not in the scanned volume (segmentation) — model statement withheld'
        elif status == 'partial':
            finding['text'] += ' [partly in the scanned volume]'
        if status != 'present': flags.append({'section': 'findings', 'heading': finding['heading'], 'status': status})
    for section in ('impression', 'labels'):
        for index, text in enumerate(parsed[section]):
            absent = [key for key, spec in MAPPING.items() if statuses[key] == 'absent'
                      and any(re.search(r'\b' + re.escape(term) + r'\b', text, re.I) for term in spec['terms'])]
            if absent:
                parsed[section][index] = 'ABSENT ANATOMY FLAG (' + ', '.join(absent) + '): ' + text
                flags.append({'section': section, 'index': index, 'absent_organs': absent})
    return {'sections': parsed, 'flags': flags, 'organ_status': statuses,
            'unchecked_headings': list(dict.fromkeys(f['heading'] for f in parsed['findings']
                                                     if f['organ_status'] == 'unmapped'))}


def polygon_area(points) -> float:
    """Planar polygon area in mm², including oblique contours (Newell formula)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 3: return 0.0
    return float(np.linalg.norm(np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)) / 2)


def rtstruct_summary(path, slice_spacing=1.0, slice_positions=None, normal=(0, 0, 1)) -> dict:
    ds = pydicom.dcmread(path)
    names = {int(x.ROINumber): str(x.ROIName) for x in getattr(ds, 'StructureSetROISequence', [])}
    rois = []
    for roi in getattr(ds, 'ROIContourSequence', []):
        planes = {}; unsupported = False
        for c in getattr(roi, 'ContourSequence', []):
            if str(getattr(c, 'ContourGeometricType', '')) != 'CLOSED_PLANAR':
                unsupported = True; continue
            points = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            z = round(float(points.mean(axis=0) @ np.asarray(normal)), 4)
            planes[z] = planes.get(z, 0) + polygon_area(points)
        positions = sorted(planes)
        spacing = float(np.median(np.diff(positions))) if len(positions) > 1 else slice_spacing
        # Prefer source spacing: gaps between contoured slices are not slice thickness.
        if slice_positions is not None and len(slice_positions) > 1:
            spacing = float(np.median(np.diff(sorted(slice_positions))))
        rois.append({'name': names.get(int(roi.ReferencedROINumber), 'Unnamed ROI'),
                     'approx_volume_ml': round(sum(planes.values()) * abs(spacing) / 1000, 3) if planes else None,
                     'slice_range_mm': [positions[0], positions[-1]] if positions else None,
                     'slice_range': [int(np.argmin(np.abs(np.asarray(sorted(slice_positions))-z)))
                                     for z in (positions[0], positions[-1])]
                                    if positions and slice_positions else None,
                     'unsupported_contours': unsupported})
    return {'model': Path(path).name.removesuffix('_RTSTRUCT.dcm'), 'rois': rois,
            'method': 'Sum of absolute planar polygon areas × source slice spacing / 1000. '
                      'Approximate: holes and overlapping contours are not subtracted; non-CLOSED_PLANAR contours omitted.'}


def thumbnail(volume) -> str:
    """Base64 PNG: axial, coronal and sagittal midplanes; soft tissue [-160,240] HU."""
    z, y, x = (s // 2 for s in volume.shape)
    planes = [volume[z], volume[::-1, y, :], volume[::-1, :, x]]
    tiles = [np.clip((p + 160) / 400 * 255, 0, 255).astype(np.uint8) for p in planes]
    buffer = io.BytesIO(); Image.fromarray(np.concatenate(tiles, axis=1)).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def rendered_text(region) -> str:
    s = region['sections']
    lines = ['TECHNIQUE', *s['technique'], 'FINDINGS']
    lines += [f"{f['heading']}: {f['text']}" for f in s['findings']]
    return '\n'.join([*lines, 'IMPRESSION', *s['impression'], 'ABNORMALITY LABELS', *s['labels']])


def unchecked_note(report):
    headings = list(dict.fromkeys(h for region in report['regions'] for h in region.get('unchecked_headings', [])))
    return 'Not checked against the segmentation: ' + '; '.join(headings) + '.' if headings else ''


def render_markdown(report, include_banner=True) -> str:
    lines = ['# NV-Reason-CT draft report', '', *(['**' + BANNER + '**', ''] if include_banner else []),
             'Status: ' + report['status'], report.get('reason', ''), '',
             'Segmentation can be wrong. Coverage checks do not verify diagnostic findings or image quality.']
    for region in report['regions']:
        lines += ['', '## ' + region['region'].capitalize(), '', rendered_text(region)]
        for flag in region.get('flags', []):
            if flag.get('status') == 'crop_incomplete':
                lines += ['', 'CROP WARNING: Less than 90% of the segmented regional organ volume is in the model input.']
            elif flag.get('status') == 'generation_truncated':
                lines += ['', 'INCOMPLETE DRAFT: Generation reached the token limit.']
            elif flag.get('status') == 'unstructured_output':
                lines += ['', 'INCOMPLETE DRAFT: Expected findings or impression section is missing. Review the raw output.']
    if unchecked_note(report): lines += ['', unchecked_note(report)]
    lines += ['', '## Anatomical coverage', '', '| Organ | mL | Slice range (zero based) | Truncated |', '|---|---:|---|---|']
    for organ, row in report['coverage'].items():
        lines.append(f"| {organ} | {row['volume_ml']} | {row['slice_range']} | {row['touches_first_slice'] or row['touches_last_slice']} |")
    lines += ['', '## Segmentation summary']
    for result in report['segmentation_summary']:
        lines += ['', result['model'], result['method']]
        for roi in result['rois']:
            lines.append(f"- {roi['name']}: approximately {roi['approx_volume_ml']} mL; slices {roi['slice_range']}; position {roi['slice_range_mm']} mm")
    lines += ['', '## Provenance', '', 'Model: nvidia/NV-Reason-CT', 'Revision: ' + REVISION]
    return '\n'.join(lines) + '\n'


def render_html(report) -> str:
    esc = html.escape
    def table(headings, rows):
        return ('<table><thead><tr>' + ''.join('<th>' + esc(str(h)) + '</th>' for h in headings)
                + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(str(v)) + '</td>'
                    for v in row) + '</tr>' for row in rows) + '</tbody></table>')
    content = '<h1>NV-Reason-CT draft report</h1><p>Status: <b>' + esc(report['status']) + '</b></p>'
    if report.get('reason'): content += '<p>' + esc(report['reason']) + '</p>'
    content += '<p>Segmentation can be wrong. Coverage checks do not verify diagnostic findings or image quality.</p>'
    for region in report['regions']:
        content += '<section><h2>' + esc(region['region'].capitalize()) + '</h2><pre>' + esc(rendered_text(region)) + '</pre>'
        for flag in region.get('flags', []):
            if flag.get('status') in ('crop_incomplete', 'generation_truncated', 'unstructured_output'):
                content += '<p class="warning">' + esc(flag.get('reason', flag['status']).replace('_', ' ')) + '</p>'
        if region.get('thumbnail_png'):
            content += '<figure><img alt="Axial, coronal and sagittal crop; soft tissue window" src="data:image/png;base64,' + region['thumbnail_png'] + '">'
            content += '<figcaption>Model input: axial, coronal and sagittal midplanes; window −160 to 240 HU.</figcaption></figure>'
        content += '<details><summary>Crop details: ' + esc(region['crop_path'].replace('_', ' ')) + '</summary><pre>' + esc(json.dumps(region['crop_qc'], indent=2)) + '</pre></details>'
        content += '<details class="raw-output"><summary>Unfiltered model output and reasoning — may contain false statements</summary><pre>' + esc(region['raw_text']) + '</pre></details></section>'
    if unchecked_note(report): content += '<p>' + esc(unchecked_note(report)) + '</p>'
    content += '<h2>Anatomical coverage</h2>' + table(['Organ', 'Volume (mL)', 'Slices (zero based)', 'Truncated'],
        [[k, v['volume_ml'], v['slice_range'], v['touches_first_slice'] or v['touches_last_slice']]
         for k, v in report['coverage'].items()])
    content += '<h2>Segmentation summary</h2>'
    if not report['segmentation_summary']: content += '<p>No RTSTRUCT was produced by this job.</p>'
    for result in report['segmentation_summary']:
        content += '<h3>' + esc(result['model']) + '</h3><p>' + esc(result['method']) + '</p>'
        content += table(['ROI', 'Approximate volume (mL)', 'Slices (zero based)', 'Position range (mm)'],
            [[r['name'], r['approx_volume_ml'], r['slice_range'], r['slice_range_mm']] for r in result['rois']])
    content += '<p>Model: nvidia/NV-Reason-CT · Revision: ' + REVISION + '</p>'
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>NV-Reason-CT research draft</title><style>body{font:16px/1.6 "Open Sans",system-ui,-apple-system,"Segoe UI",Arial,sans-serif;'
            'color:#142226;margin:0 auto;padding:2rem 1.5rem;max-width:1100px}h1,h2,h3{color:#00525c;letter-spacing:-.03em;line-height:1.2;font-weight:650}'
            '.banner,.warning{background:#fff2f2;border:2px solid #a3323d;border-left-width:6px;border-radius:.65rem;padding:1rem 1.25rem;font-weight:bold}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f9f9;border:1px solid #d5e4e4;border-radius:.65rem;padding:1rem}'
            'img{max-width:100%;border-radius:.65rem}figure{margin:1rem 0}figcaption{font-size:.85rem}'
            'details{border:1px solid #d5e4e4;border-radius:.8rem;padding:.75rem 1rem;margin:.5rem 0}summary{cursor:pointer;color:#00525c;font-weight:bold}'
            'table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #d5e4e4;padding:.5rem;text-align:left}'
            'th{background:#f5f9f9;color:#00525c}section{border-bottom:1px solid #d5e4e4;padding-bottom:1rem}</style>'
            '<body><div class="banner">' + esc(BANNER) + '</div>' + content + '</body></html>')


def write_report(directory, report):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for region in report['regions']: region['rendered_text'] = rendered_text(region)
    (directory / (TASK + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    (directory / (TASK + '.md')).write_text(render_markdown(report))
    (directory / (TASK + '.html')).write_text(render_html(report))

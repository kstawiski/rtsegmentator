#!/usr/bin/env python3
"""Synthetic CPU write/read/validate comparison with frozen 3600f83 functions.

Fresh process per mode. Default --shape is 300 slices x 512 rows x 512 columns,
117 disjoint labels. No inference or source DICOM pixel I/O. Temporary RTSTRUCTs
are removed after measurement. Set TMPDIR or --scratch-dir to project scratch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


def benchmark(mode, zyx, count, scratch):
    import nibabel as nib
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from rtstruct_geometry import add_roi, slice_geometry
    from rtstruct_roundtrip import CroppedMask, label_boxes, read_rtstruct, source_plane_geometry, validate_roi_roundtrip
    from roundtrip_reference import full_volume_reference
    import writer_reference

    phases = {}
    started = step = time.perf_counter()
    shape = tuple(reversed(zyx))
    data = np.zeros(shape, np.uint16, order='F')
    nx = int(np.ceil(np.sqrt(count * shape[0] / shape[1])))
    ny = int(np.ceil(count / nx))
    if min(shape[0] // nx, shape[1] // ny) < 6 or shape[2] < 6:
        raise ValueError('shape too small for requested ROI count')
    for i in range(count):
        x0, x1 = (i % nx) * shape[0] // nx, (i % nx + 1) * shape[0] // nx
        y0, y1 = (i // nx) * shape[1] // ny, (i // nx + 1) * shape[1] // ny
        xx, yy = np.ogrid[x0:x1, y0:y1]
        x, y = (xx - (x0+x1-1)/2) / ((x1-x0)/2-1), (yy - (y0+y1-1)/2) / ((y1-y0)/2-1)
        for k in range(1 + i % 3, shape[2] - 1 - i % 3):
            radius = .72 + .15 * np.sin(k / 7 + i)
            mask = x*x + y*y <= radius*radius
            if i % 3 == 0: mask &= x*x + y*y >= .14
            data[x0:x1, y0:y1, k][mask] = i + 1
    image = nib.Nifti1Image(data, np.diag([-1., -1., 2., 1.]))
    slices = []
    for k in range(shape[2]):
        ds = Dataset()
        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.SOPInstanceUID = f'1.2.826.0.1.3680043.10.543.1.{k+1}'
        ds.FrameOfReferenceUID = '1.2.826.0.1.3680043.10.543.2'
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, k * 2]
        ds.PixelSpacing = [1, 1]
        slices.append(ds)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    meta.MediaStorageSOPInstanceUID = '1.2.826.0.1.3680043.10.543.3'
    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = 'RTSTRUCT'
    ds.StructureSetROISequence = []
    ds.ROIContourSequence = []
    ds.RTROIObservationsSequence = []
    rt = SimpleNamespace(ds=ds, series_data=slices)
    phases['synthetic_input_s'] = time.perf_counter() - step
    step = time.perf_counter()
    if mode == 'before':
        expected = {label: data == label for label in range(1, count + 1)}
        boxes = None
    else:
        boxes = label_boxes(data)
        expected = None
    phases['label_geometry_s'] = time.perf_counter() - step
    step = time.perf_counter()
    bounds = {}
    plane_geometry = [slice_geometry(ds) for ds in slices] if mode == 'after' else None
    for label in range(1, count + 1):
        if mode == 'before':
            mask = expected[label]
            writer_reference.add_roi(rt, mask, f'ROI {label}', [100, 150, 200])
            bounds[label] = writer_reference.mask_bounds(mask, slices)
        else:
            box = boxes[label]
            mask = data[box] == label
            bound = []
            add_roi(rt, mask, f'ROI {label}', [100, 150, 200],
                    offset=tuple(s.start for s in box), bounds_out=bound, plane_geometry=plane_geometry)
            bounds[label] = bound
        del mask
    phases['contours_and_bounds_s'] = time.perf_counter() - step
    with tempfile.TemporaryDirectory(prefix='rtstruct-benchmark-', dir=scratch) as temporary:
        path = Path(temporary) / 'synthetic.dcm'
        step = time.perf_counter()
        ds.save_as(path, enforce_file_format=True)
        phases['save_s'] = time.perf_counter() - step
        step = time.perf_counter()
        saved = pydicom.dcmread(path) if mode == 'before' else read_rtstruct(path)
        # Conventional pydicom decoding is lazy: include conversion in read time.
        for roi in saved.ROIContourSequence:
            for contour in roi.ContourSequence:
                contour.ContourData
        phases['read_and_decode_s'] = time.perf_counter() - step
        step = time.perf_counter()
        reports = {}
        source_geometry = source_plane_geometry(slices) if mode == 'after' else None
        for label, roi in enumerate(saved.ROIContourSequence, 1):
            if mode == 'before':
                reports[label] = full_volume_reference(roi, image, expected[label], source_slices=slices)
            else:
                box = boxes[label]
                crop = CroppedMask(data[box] == label, tuple(s.start for s in box), data.shape)
                reports[label] = validate_roi_roundtrip(roi, image, crop, source_geometry=source_geometry)
        phases['validation_s'] = time.perf_counter() - step
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
    elapsed = time.perf_counter() - started
    payload = {'reports': reports, 'bounds': {k: np.asarray(v).tolist() for k, v in bounds.items()}}
    return {'mode': mode, 'shape_zyx': zyx, 'rois': count,
            'algorithm_s': round(elapsed, 3), 'phases_s': {k: round(v, 3) for k, v in phases.items()},
            'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            'rtstruct_bytes': size, 'file_sha256': digest,
            'reports_and_bounds_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs=3, default=[300, 512, 512], metavar=('Z', 'Y', 'X'))
    parser.add_argument('--rois', type=int, default=117)
    parser.add_argument('--mode', choices=['before', 'after'])
    parser.add_argument('--scratch-dir')
    args = parser.parse_args()
    if min(args.shape) <= 0 or not 1 <= args.rois <= 32767:
        parser.error('positive shape and 1..32767 ROIs required')
    if args.mode:
        print(json.dumps(benchmark(args.mode, args.shape, args.rois, args.scratch_dir)))
        return
    results = []
    for mode in ('before', 'after'):
        command = [sys.executable, __file__, '--mode', mode, '--shape', *map(str, args.shape), '--rois', str(args.rois)]
        if args.scratch_dir: command.extend(['--scratch-dir', args.scratch_dir])
        started = time.perf_counter()
        result = json.loads(subprocess.check_output(command, text=True))
        result['wall_s'] = round(time.perf_counter() - started, 3)
        results.append(result)
    equal = all(results[0][key] == results[1][key] for key in
                ('file_sha256', 'reports_and_bounds_sha256', 'rtstruct_bytes'))
    print(json.dumps({'results': results, 'saved_bytes_and_reports_equal': equal}, indent=2))
    if not equal: raise SystemExit(1)


if __name__ == '__main__': main()

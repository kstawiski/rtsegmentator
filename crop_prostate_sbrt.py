"""Crop an axial CT series and its RTSTRUCTs along patient z, without resampling.

The returned directory is published only after DICOM and rt_utils validation.
Inputs are never changed; an existing non-empty output directory is rejected.

Default UID policy: the cropped CT is a new DICOM study (new Study, Series,
SOP and FrameOfReference UIDs) so treatment-planning systems import it as a
separate study instead of merging it with the source. RTSTRUCT study/frame/
series/image references are rewritten to match. Pass keep_uids=True only to
preserve the source UIDs for debugging.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

TOL_MM = 1e-3


class NoSelectedROIError(ValueError):
    """No selected ROI has contour data (the only skippable postprocess error)."""


def discover_dicom(input_dir):
    """Find CT and RTSTRUCT files recursively, including extensionless DICOM."""
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError("Input directory does not exist")
    ct_files, rt_files = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True,
                                 specific_tags=["Modality"])
        except (pydicom.errors.InvalidDicomError, OSError, ValueError):
            continue
        if ds.get("Modality") == "CT":
            ct_files.append(path)
        elif ds.get("Modality") == "RTSTRUCT":
            rt_files.append(path)
    return ct_files, rt_files


def _series_references(ds):
    for frame in ds.get("ReferencedFrameOfReferenceSequence", []):
        for study in frame.get("RTReferencedStudySequence", []):
            yield from study.get("RTReferencedSeriesSequence", [])


def _references(ds, series_uid, frame_uid):
    explicit = {str(s.SeriesInstanceUID) for s in _series_references(ds)
                if s.get("SeriesInstanceUID")}
    if explicit:
        return series_uid in explicit
    frames = {str(f.get("FrameOfReferenceUID", ""))
              for f in ds.get("ReferencedFrameOfReferenceSequence", [])}
    frames.add(str(ds.get("FrameOfReferenceUID", "")))
    return bool(frame_uid) and frame_uid in frames


def _selected(name, mode):
    """prostate mode: the ROI named "prostate". ptvctv mode: names starting with PTV or CTV (PTV*, CTV*),
    so subtraction helpers such as "Bowel - PTV" are not treated as target volumes."""
    name = name.strip().casefold()
    return name == "prostate" if mode == "prostate" else name.startswith(("ptv", "ctv"))


def _contour_z(contour):
    values = np.asarray(contour.get("ContourData", []), dtype=float)
    if values.size % 3 or not np.isfinite(values).all():
        raise ValueError("Malformed or non-finite ContourData")
    return values[2::3]


def _nodes(ds):
    yield ds
    for element in ds:
        if element.VR == "SQ":
            for item in element.value:
                yield from _nodes(item)


def _remap_references(ds, uid_map, series_uid, output_series_uid,
                      study_uid=None, output_study_uid=None,
                      frame_uid=None, output_frame_uid=None):
    """Point RTSTRUCT study/frame/series/image references at the cropped CT.

    The cropped CT carries new Study/Frame/Series/SOP UIDs, so every reference
    that identified the source must be rewritten for the RTSTRUCT to stay
    valid. References to any other series are dropped, as before.
    """
    if output_study_uid and "StudyInstanceUID" in ds:
        ds.StudyInstanceUID = output_study_uid
    # ARIA requires the top-level Frame of Reference UID (0020,0052) to match
    # the entry in ReferencedFrameOfReferenceSequence (3006,0010).
    if output_frame_uid and "FrameOfReferenceUID" in ds:
        ds.FrameOfReferenceUID = output_frame_uid
    for frame in ds.get("ReferencedFrameOfReferenceSequence", []):
        if (output_frame_uid and "FrameOfReferenceUID" in frame
                and (not frame_uid or str(frame.FrameOfReferenceUID) == frame_uid)):
            frame.FrameOfReferenceUID = output_frame_uid
        for study in frame.get("RTReferencedStudySequence", []):
            # In this context ReferencedSOPInstanceUID carries the Study UID.
            if (output_study_uid and "ReferencedSOPInstanceUID" in study
                    and (not study_uid or str(study.ReferencedSOPInstanceUID) == study_uid)):
                study.ReferencedSOPInstanceUID = output_study_uid
    for node in _nodes(ds):
        if "RTReferencedSeriesSequence" in node:
            # Do not relabel references to other series as the selected series.
            node.RTReferencedSeriesSequence = Sequence([
                item for item in node.RTReferencedSeriesSequence
                if not item.get("SeriesInstanceUID") or str(item.SeriesInstanceUID) == series_uid])
            for item in node.RTReferencedSeriesSequence:
                item.SeriesInstanceUID = output_series_uid
                if "ContourImageSequence" not in item:
                    item.ContourImageSequence = Sequence()
        if "ContourImageSequence" in node:
            kept = []
            for image in node.ContourImageSequence:
                old = str(image.get("ReferencedSOPInstanceUID", ""))
                if old in uid_map:
                    image.ReferencedSOPInstanceUID = uid_map[old]
                    kept.append(image)
            node.ContourImageSequence = Sequence(kept)


def _describe(ds, margin_mm, rtstruct=False):
    # ASCII suffix is valid in every SpecificCharacterSet, so the source encoding stays untouched.
    suffix = f" crop +/-{margin_mm / 10:g} cm"
    ds.SeriesDescription = (str(ds.get("SeriesDescription", "")) + suffix)[:64]
    if rtstruct and "StructureSetDescription" in ds:
        ds.StructureSetDescription = (str(ds.StructureSetDescription) + suffix)[:1024]


def _validate(output, rt_names, kept, uid_map, mode, study_uid=None, frame_uid=None):
    from rt_utils import RTStructBuilder

    output_uids = set(uid_map.values())
    actual = {str(pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID)
              for p in (output / "CT").glob("*.dcm")}
    if actual != output_uids:
        raise ValueError("Validation failed: output CT UID inventory mismatch")
    if study_uid is not None or frame_uid is not None:
        headers = [pydicom.dcmread(p, stop_before_pixels=True)
                   for p in sorted((output / "CT").glob("*.dcm"))]
        if study_uid is not None and {str(h.get("StudyInstanceUID", "")) for h in headers} != {study_uid}:
            raise ValueError("Validation failed: output CT StudyInstanceUID mismatch")
        if frame_uid is not None and {str(h.get("FrameOfReferenceUID", "")) for h in headers} != {frame_uid}:
            raise ValueError("Validation failed: output CT FrameOfReferenceUID mismatch")
    zs = [float(ds.ImagePositionPatient[2]) for _, ds in kept]
    spacing = float(np.median(np.diff(zs))) if len(zs) > 1 else abs(float(kept[0][1].get("SliceThickness", 1)))
    validated_masks = []
    mask_error = "no selected ROI retains contours in the kept slice range"
    for name in rt_names:
        path = output / name
        ds = pydicom.dcmread(path)
        if study_uid is not None and str(ds.get("StudyInstanceUID", "")) != study_uid:
            raise ValueError("Validation failed: RTSTRUCT StudyInstanceUID does not match output CT")
        if (frame_uid is not None and "FrameOfReferenceUID" in ds
                and str(ds.FrameOfReferenceUID) != frame_uid):
            raise ValueError("Validation failed: RTSTRUCT top-level FrameOfReferenceUID does not match output CT")
        for frame in ds.get("ReferencedFrameOfReferenceSequence", []):
            if frame_uid is not None and str(frame.get("FrameOfReferenceUID", "")) != frame_uid:
                raise ValueError("Validation failed: RTSTRUCT FrameOfReferenceUID does not match output CT")
            for study in frame.get("RTReferencedStudySequence", []):
                if (study_uid is not None and "ReferencedSOPInstanceUID" in study
                        and str(study.ReferencedSOPInstanceUID) != study_uid):
                    raise ValueError("Validation failed: RTSTRUCT referenced study does not match output CT")
        for node in _nodes(ds):
            for image in node.get("ContourImageSequence", []):
                if str(image.ReferencedSOPInstanceUID) not in output_uids:
                    raise ValueError("Validation failed: RTSTRUCT references a missing CT slice")
            if "ContourData" in node:
                z = _contour_z(node)
                if z.size and (z.min() < min(zs) - spacing / 2 - TOL_MM or
                               z.max() > max(zs) + spacing / 2 + TOL_MM):
                    raise ValueError("Validation failed: contour outside output CT range")
        try:
            rt = RTStructBuilder.create_from(dicom_series_path=str(output / "CT"),
                                            rt_struct_path=str(path))
            # rt_utils silently skips images it cannot decode. Require all slices.
            if {str(s.SOPInstanceUID) for s in rt.series_data} != output_uids:
                raise ValueError("rt_utils could not decode every output CT slice")
            names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
            candidates = [names[int(r.ReferencedROINumber)] for r in ds.ROIContourSequence
                          if r.get("ContourSequence") and _selected(names[int(r.ReferencedROINumber)], mode)]
            for candidate in candidates:
                try:
                    mask = rt.get_roi_mask_by_name(candidate)
                    if mask.ndim != 3 or mask.shape[2] != len(kept):
                        raise ValueError("rt_utils returned an unexpected mask shape")
                except Exception as exc:
                    mask_error = str(exc)
                    # A selected POINT ROI may not be rasterizable by rt_utils.
                    # The contract requires one working selected mask, not all.
                    continue
                validated_masks.append({"rtstruct": name, "roi": candidate})
                del mask
                break
            del rt
        except Exception as exc:
            raise ValueError(f"Validation failed: rt_utils reload/mask: {exc}") from exc
    if not validated_masks:
        raise ValueError(f"Validation failed: no working rt_utils selected ROI mask: {mask_error}")
    return validated_masks


def crop_ct_rtstruct(ct_files, rtstruct_files, out_dir, roi_mode="ptvctv",
                     margin_mm=150.0, keep_uids=False) -> dict:
    """Crop one identifiable axial CT series and all supplied matching RTSTRUCTs.

    Several uploaded CT series are accepted only when the RTSTRUCT references
    identify exactly one. Explicit series references take precedence over frames.
    Contours are retained whole, never geometrically clipped or interpolated.
    """
    if roi_mode not in {"ptvctv", "prostate"}:
        raise ValueError("roi_mode must be ptvctv or prostate")
    margin_mm = float(margin_mm)
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and non-negative")
    ct_files, rtstruct_files = list(ct_files), list(rtstruct_files)
    groups = {}
    for path in ct_files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if ds.get("Modality") != "CT":
            continue
        if not ds.get("SeriesInstanceUID"):
            raise ValueError("CT is missing SeriesInstanceUID")
        groups.setdefault(str(ds.SeriesInstanceUID), []).append((Path(path), ds))
    if not groups:
        raise ValueError("No CT image series found")
    structs = []
    for path in rtstruct_files:
        ds = pydicom.dcmread(path)
        if ds.get("Modality") != "RTSTRUCT":
            raise ValueError("An RTSTRUCT input is not Modality RTSTRUCT")
        structs.append((Path(path), ds))
    matches = {uid for uid, slices in groups.items()
               if any(_references(rt, uid, str(slices[0][1].get("FrameOfReferenceUID", "")))
                      for _, rt in structs)}
    if len(groups) > 1 and len(matches) != 1:
        raise ValueError("Several CT series found; RTSTRUCTs must identify exactly one CT series")
    if not matches:
        raise ValueError("No RTSTRUCT referencing the CT series was found")
    series_uid = next(iter(matches))
    slices = groups[series_uid]
    frame_uid = str(slices[0][1].get("FrameOfReferenceUID", ""))
    if any(not _references(rt, series_uid, frame_uid) for _, rt in structs):
        raise ValueError("An input RTSTRUCT does not reference the selected CT series")
    study_uid = str(slices[0][1].get("StudyInstanceUID", ""))
    seen = set()
    for _, ds in slices:
        try:
            orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
            position = np.asarray(ds.ImagePositionPatient, dtype=float)
            normal = np.cross(orientation[:3], orientation[3:])
            valid = (orientation.size == 6 and position.size == 3 and
                     np.isfinite(orientation).all() and np.isfinite(position).all() and
                     np.isclose(np.linalg.norm(normal), 1, atol=1e-3) and
                     abs(float(normal[2])) / np.linalg.norm(normal) >= math.cos(math.radians(1)))
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Non-axial CT or invalid geometry: slice normal must be parallel to patient z within 1 degree")
        if int(ds.get("NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced/multiframe CT is not supported; provide single-frame axial CT")
        if (str(ds.get("FrameOfReferenceUID", "")) != frame_uid or
                str(ds.get("StudyInstanceUID", "")) != study_uid):
            raise ValueError("CT series contains inconsistent Study/FrameOfReferenceUID values")
        uid = str(ds.get("SOPInstanceUID", ""))
        if not uid or uid in seen:
            raise ValueError("CT slices have missing or duplicate SOPInstanceUID values")
        seen.add(uid)
    slices.sort(key=lambda item: float(item[1].ImagePositionPatient[2]))
    if any(float(b[1].ImagePositionPatient[2]) - float(a[1].ImagePositionPatient[2]) <= TOL_MM
           for a, b in zip(slices, slices[1:])):
        raise ValueError("CT series contains duplicate slice z positions")
    selected_names, selected_z = set(), []
    for _, ds in structs:
        names = {int(r.ROINumber): str(r.ROIName) for r in ds.get("StructureSetROISequence", [])}
        for roi in ds.get("ROIContourSequence", []):
            name = names.get(int(roi.ReferencedROINumber), "")
            if _selected(name, roi_mode):
                selected_names.add(name)
                for contour in roi.get("ContourSequence", []):
                    selected_z.extend(_contour_z(contour).tolist())
    if not selected_z:
        label = "prostate" if roi_mode == "prostate" else "PTV/CTV"
        raise NoSelectedROIError(f"No selected {label} ROI with contour data was found")
    zmin, zmax = min(selected_z), max(selected_z)
    requested = [zmin - margin_mm, zmax + margin_mm]
    kept = [(p, ds) for p, ds in slices
            if requested[0] - TOL_MM <= float(ds.ImagePositionPatient[2]) <= requested[1] + TOL_MM]
    if not kept:
        raise ValueError("Crop range contains no CT slices")
    low, high = float(kept[0][1].ImagePositionPatient[2]), float(kept[-1][1].ImagePositionPatient[2])
    rt_names = [f"{p.stem}_cropped.dcm" for p, _ in structs]
    if len(set(rt_names)) != len(rt_names):
        raise ValueError("RTSTRUCT filenames must have distinct stems to avoid output collisions")
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError("Output directory must be absent or empty")
    # Prevent publication from replacing an input directory, even if it is empty.
    if any(out_dir.resolve() == p.resolve().parent or out_dir.resolve() in p.resolve().parents
           for p in [*map(Path, ct_files), *map(Path, rtstruct_files)]):
        raise ValueError("Output directory must not contain input files")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    output_study_uid = study_uid if keep_uids else generate_uid()
    output_frame_uid = frame_uid if keep_uids else generate_uid()
    output_series_uid = series_uid if keep_uids else generate_uid()
    uid_map = {str(ds.SOPInstanceUID): str(ds.SOPInstanceUID) if keep_uids else generate_uid()
               for _, ds in kept}
    removed = []
    with tempfile.TemporaryDirectory(prefix=".crop-", dir=out_dir.parent) as tmp:
        output = Path(tmp) / "result"
        (output / "CT").mkdir(parents=True)
        for path, header in kept:
            ds = pydicom.dcmread(path)
            if not keep_uids:
                ds.StudyInstanceUID = output_study_uid
                ds.FrameOfReferenceUID = output_frame_uid
            ds.SeriesInstanceUID = output_series_uid
            ds.SOPInstanceUID = uid_map[str(header.SOPInstanceUID)]
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            _describe(ds, margin_mm)
            ds.save_as(output / "CT" / f"CT.{ds.SOPInstanceUID}.dcm", enforce_file_format=True)
        for (_, source), name in zip(structs, rt_names):
            ds = copy.deepcopy(source)
            if not keep_uids:
                ds.StudyInstanceUID = output_study_uid
            ds.SOPInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            names = {int(r.ROINumber): str(r.ROIName) for r in ds.get("StructureSetROISequence", [])}
            counts = {number: 0 for number in names}
            for roi in ds.get("ROIContourSequence", []):
                contours = []
                for contour in roi.get("ContourSequence", []):
                    z = _contour_z(contour)
                    if z.size and z.min() >= low - TOL_MM and z.max() <= high + TOL_MM:
                        contours.append(contour)
                    else:
                        number = int(roi.ReferencedROINumber)
                        counts[number] = counts.get(number, 0) + 1
                if contours:
                    roi.ContourSequence = Sequence(contours)
                elif "ContourSequence" in roi:
                    del roi.ContourSequence
            removed.extend({"rtstruct": name, "roi_number": number, "roi_name": names.get(number, ""),
                            "contours_removed": count} for number, count in counts.items())
            _remap_references(ds, uid_map, series_uid, output_series_uid,
                              study_uid, output_study_uid, frame_uid, output_frame_uid)
            _describe(ds, margin_mm, rtstruct=True)
            ds.save_as(output / name, enforce_file_format=True)
        masks = _validate(output, rt_names, kept, uid_map, roi_mode,
                          study_uid=output_study_uid, frame_uid=output_frame_uid)
        report = {"roi_mode": roi_mode, "selected_roi_names": sorted(selected_names),
                  "zmin": zmin, "zmax": zmax, "margin_mm": margin_mm,
                  "requested_z_range": requested, "kept_z_range": [low, high],
                  "slices_kept": len(kept), "slices_total": len(slices),
                  "contours_removed_per_roi": removed,
                  "uid_policy": ("original Study/Series/SOP/Frame UIDs; new RTSTRUCT UIDs"
                                 if keep_uids else
                                 "new Study/Series/SOP/Frame UIDs for CT with matching RTSTRUCT references"),
                  "keep_uids": bool(keep_uids), "ct_series_uid": output_series_uid,
                  "ct_study_uid": output_study_uid, "ct_frame_uid": output_frame_uid,
                  "file_counts": {"CT": len(kept), "RTSTRUCT": len(structs), "report": 1,
                                  "total": len(kept) + len(structs) + 1},
                  "validation": {"pydicom": "passed", "references": "passed", "contour_range": "passed",
                                 "rt_utils": "passed", "masks_checked": masks}}
        lines = ["Prostate SBRT CT + RTSTRUCT crop report", "",
                 f"Selected ROIs ({roi_mode}): {', '.join(report['selected_roi_names'])}",
                 f"Selected contour z: {zmin:g} to {zmax:g} mm",
                 f"Margin: {margin_mm:g} mm ({margin_mm / 10:g} cm)",
                 f"Requested z range: {requested[0]:g} to {requested[1]:g} mm",
                 f"Kept slice z range: {low:g} to {high:g} mm",
                 f"Slices kept: {len(kept)} / {len(slices)}",
                 f"UID policy: {report['uid_policy']}",
                 "Pixel data unchanged; whole contours outside the kept slice range removed.",
                 "Contours removed per ROI:"]
        lines.extend(f"  {r['rtstruct']} / {r['roi_number']} / {r['roi_name']}: {r['contours_removed']}" for r in removed)
        lines.extend([f"File counts: {json.dumps(report['file_counts'])}",
                      "Validation: pydicom, image references, contour z range and rt_utils reload/mask passed.",
                      "", "Machine-readable report:", json.dumps(report, indent=2, ensure_ascii=False)])
        (output / "crop_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if out_dir.exists():
            out_dir.rmdir()
        output.replace(out_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-mm", type=float, default=150.0)
    parser.add_argument("--keep-uids", action="store_true")
    parser.add_argument("--roi-mode", choices=["ptvctv", "prostate"], default="ptvctv")
    args = parser.parse_args(argv)
    try:
        ct, rt = discover_dicom(args.input)
        report = crop_ct_rtstruct(ct, rt, args.output, args.roi_mode, args.margin_mm, args.keep_uids)
    except ValueError as exc:
        parser.exit(2, f"Crop failed: {exc}\n")
    print(f"Kept {report['slices_kept']}/{report['slices_total']} slices, "
          f"z {report['kept_z_range'][0]:g} to {report['kept_z_range'][1]:g} mm. "
          f"Report: {args.output / 'crop_report.txt'}")


if __name__ == "__main__":
    main()

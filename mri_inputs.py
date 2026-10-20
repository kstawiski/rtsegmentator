"""DICOM sequence assessment and the reviewed multi-series input contract.

Only standard diffusion tags are supported. No private-tag or description-based
b-value inference is performed. Pixel data and rescale tags are never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import re


B_VALUE_SOURCE = "(0018,9087)"
ROLE_PATTERN = r"[a-z0-9_]+"
MR_FIELDS = ("study_uid", "frame_uid", "image_type", "protocol_name", "acquisition_time",
             "acquisition_datetime", "orientation", "pixel_spacing", "slice_thickness",
             "spacing_between_slices", "dimensions", "instances", "b_values", "b_value_sources",
             "is_adc", "hbv_options", "hbv_unavailable_reason", "metadata_consistent")


def numbers(value):
    try:
        result = [float(v) for v in value]
        return result if all(math.isfinite(v) for v in result) else []
    except (TypeError, ValueError):
        return []


def first(sequence):
    return sequence[0] if sequence else None


def attribute(ds, name, default=None):
    return getattr(ds, name, default) if ds is not None else default


def image_types(ds):
    value = attribute(ds, "ImageType", [])
    return value.split("\\") if isinstance(value, str) else list(map(str, value))


def b_value(ds):
    try:
        value = float(attribute(ds, "DiffusionBValue"))
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def functional_value(frame, shared, ds, sequence, keyword, default=None):
    for container in (frame, shared):
        item = first(attribute(container, sequence, []))
        value = attribute(item, keyword)
        if value is not None:
            return value
    return attribute(ds, keyword, default)


def mr_instance(ds, filename):
    shared = first(attribute(ds, "SharedFunctionalGroupsSequence", []))
    frames = attribute(ds, "PerFrameFunctionalGroupsSequence", [])
    count = int(attribute(ds, "NumberOfFrames", 1))
    if count < 1:
        raise ValueError("Invalid NumberOfFrames")
    records = []
    for index in range(count):
        frame = frames[index] if index < len(frames) else None
        diffusion = first(attribute(frame, "MRDiffusionSequence", []))
        if diffusion is None:
            diffusion = first(attribute(shared, "MRDiffusionSequence", []))
        value = b_value(diffusion if diffusion is not None else ds)
        path = ("PerFrameFunctionalGroupsSequence/MRDiffusionSequence" if
                attribute(frame, "MRDiffusionSequence") else
                "SharedFunctionalGroupsSequence/MRDiffusionSequence" if
                attribute(shared, "MRDiffusionSequence") else "DiffusionBValue")
        records.append({"frame": index + 1, "b_value": value,
                        "b_value_source": B_VALUE_SOURCE if value is not None else None,
                        "b_value_path": path if value is not None else None,
                        "position": numbers(functional_value(frame, shared, ds, "PlanePositionSequence", "ImagePositionPatient", [])),
                        "orientation": numbers(functional_value(frame, shared, ds, "PlaneOrientationSequence", "ImageOrientationPatient", [])),
                        "pixel_spacing": numbers(functional_value(frame, shared, ds, "PixelMeasuresSequence", "PixelSpacing", []))})
    values = sorted({r["b_value"] for r in records if r["b_value"] is not None})
    complete = all(r["b_value"] is not None for r in records)
    return {"file": filename, "sop_uid": str(attribute(ds, "SOPInstanceUID", "")),
            "study_uid": str(attribute(ds, "StudyInstanceUID", "")),
            "frame_uid": str(attribute(ds, "FrameOfReferenceUID", "")),
            "image_type": image_types(ds),
            "description": str(attribute(ds, "SeriesDescription", "")),
            "protocol_name": str(attribute(ds, "ProtocolName", "")),
            "acquisition_time": str(attribute(ds, "AcquisitionTime", "")),
            "acquisition_datetime": str(attribute(ds, "AcquisitionDateTime", "")),
            "rows": int(ds.Rows), "columns": int(ds.Columns), "number_of_frames": count,
            "orientation": records[0]["orientation"], "pixel_spacing": records[0]["pixel_spacing"],
            "slice_thickness": numbers([functional_value(None, shared, ds, "PixelMeasuresSequence", "SliceThickness")]),
            "spacing_between_slices": numbers([functional_value(None, shared, ds, "PixelMeasuresSequence", "SpacingBetweenSlices")]),
            "b_value": values[0] if complete and len(values) == 1 else None,
            "b_values": values, "b_value_complete": complete,
            "b_value_source": B_VALUE_SOURCE if complete else None, "frames": records}


def volume_problem(instances):
    """Reject known mixed geometry and repeated planes (e.g. diffusion directions)."""
    geometry = {(i["rows"], i["columns"], tuple(f["orientation"]), tuple(f["pixel_spacing"]))
                for i in instances for f in i["frames"]}
    if len(geometry) != 1:
        return "Selected volume has inconsistent dimensions, orientation or spacing"
    _, _, orientation, spacing = next(iter(geometry))
    if len(orientation) != 6 or len(spacing) != 2 or min(spacing) <= 0:
        return "Selected volume has missing orientation or pixel spacing"
    positions = [tuple(f["position"]) for i in instances for f in i["frames"]]
    if any(len(p) != 3 for p in positions):
        return "Selected volume has missing ImagePositionPatient geometry"
    if len(positions) != len(set(positions)):
        return "Selected volume has repeated slice positions; choose a single 3-D volume"
    return ""


def finish_mr_series(series):
    instances = series["instances"]
    first_instance = instances[0]
    for name in ("protocol_name", "acquisition_time", "acquisition_datetime", "orientation",
                 "pixel_spacing", "slice_thickness", "spacing_between_slices"):
        series[name] = first_instance[name]
    series["dimensions"] = [series["columns"], series["rows"], sum(i["number_of_frames"] for i in instances)]
    series["b_values"] = sorted({v for i in instances for v in i["b_values"]})
    series["b_value_sources"] = [B_VALUE_SOURCE] if series["b_values"] else []
    series["is_adc"] = any("ADC" in value.upper() for value in series["image_type"])
    series["metadata_consistent"] = all(i["study_uid"] == series["study_uid"] and
                                         i["frame_uid"] == series["frame_uid"] for i in instances)
    reason = ""
    if not series["metadata_consistent"]:
        reason = "Series contains different Study or Frame of Reference identifiers"
    elif series["is_adc"]:
        reason = "ADC map is not a high b-value DWI volume"
    elif any(not i["b_value_complete"] for i in instances):
        reason = "Diffusion b-value is missing from a documented tag in one or more instances/frames"
    elif any(len(i["b_values"]) != 1 for i in instances):
        reason = "Mixed-b Enhanced MR object requires frame extraction; upload separate single-b instances"
    options = []
    if not reason:
        for value in reversed(series["b_values"]):
            if value < 800:
                continue
            selected = [i for i in instances if i["b_value"] == value]
            problem = volume_problem(selected)
            if problem:
                reason = problem
                continue
            options.append({"b_value": value, "b_value_source": B_VALUE_SOURCE,
                            "dimensions": [series["columns"], series["rows"], sum(i["number_of_frames"] for i in selected)],
                            "files": [i["file"] for i in selected]})
    series["hbv_options"] = options
    series["hbv_unavailable_reason"] = "" if options else reason or "No documented b-value ≥ 800 s/mm²"


def public_mr_fields(series):
    result = {key: series.get(key) for key in MR_FIELDS if key in series}
    result["instances"] = [{k: v for k, v in i.items() if k != "file"} for i in series.get("instances", [])]
    result["hbv_options"] = [{k: v for k, v in o.items() if k != "files"} for o in series.get("hbv_options", [])]
    return result


def role_contract(task):
    roles = task.get("input_roles") or []
    if not roles:
        return [], None
    reference = task.get("reference_role")
    if (not isinstance(roles, list) or any(not isinstance(r, dict) or
            not isinstance(r.get("role"), str) or not re.fullmatch(ROLE_PATTERN, r["role"]) for r in roles)):
        raise ValueError("Invalid catalogue input_roles")
    names = [r["role"] for r in roles]
    if len(names) != len(set(names)) or reference not in names:
        raise ValueError("Invalid catalogue reference_role or duplicate roles")
    return roles, reference


def role_candidates(primary, series, role):
    candidates = []
    for candidate in series:
        if (candidate["key"] == primary["key"] or not primary.get("study_uid") or
                candidate.get("study_uid") != primary["study_uid"] or candidate.get("modality") != role.get("modality", "MR") or
                candidate.get("metadata_consistent") is False):
            continue
        if role["role"] == "hbv":
            for option in candidate.get("hbv_options", []):
                candidates.append({"series_key": candidate["key"], "b_value": option["b_value"],
                                   "score": option["b_value"]})
        else:
            score = (2 if candidate.get("is_adc") else 1 if re.search(r"\badc\b|apparent diffusion", " ".join(
                [candidate.get("description", ""), candidate.get("protocol_name", "")]), re.I) else 0) if role["role"] == "adc" else 0
            candidates.append({"series_key": candidate["key"], "score": score})
    return sorted(candidates, key=lambda c: (-c["score"], c["series_key"]))


def suggest_role(primary, series, role):
    candidates = role_candidates(primary, series, role)
    if not candidates or candidates[0]["score"] <= 0:
        return None, f"Missing {role.get('label', role['role'])}; select a series"
    if len(candidates) > 1 and candidates[0]["score"] == candidates[1]["score"]:
        return None, f"Ambiguous {role.get('label', role['role'])}; select a series"
    return {k: v for k, v in candidates[0].items() if k != "score"}, ""


def resolve_roles(primary, all_series, task, selections):
    roles, reference = role_contract(task)
    if not roles:
        return {}, {}, ""
    by_key = {s["key"]: s for s in all_series}
    records, sources = {}, {}
    for role in roles:
        name = role["role"]
        if name == reference:
            choice, error = {"series_key": primary["key"]}, ""
        elif name in selections:
            choice, error = selections[name], f"Missing {role.get('label', name)}; select a series"
        else:
            choice, error = suggest_role(primary, all_series, role)
        if choice is None:
            if role.get("required", True):
                return {}, {}, error
            continue
        if isinstance(choice, str):
            choice = {"series_key": choice}
        if not isinstance(choice, dict) or set(choice) - {"series_key", "b_value"}:
            raise ValueError(f"Invalid selection for role {name}")
        candidate = by_key.get(choice.get("series_key"))
        if candidate is None:
            raise ValueError(f"Unknown series for role {name}")
        if (not primary.get("study_uid") or candidate.get("study_uid") != primary["study_uid"]):
            raise ValueError(f"Role {name} must share the primary StudyInstanceUID")
        if candidate.get("modality") != role.get("modality", "MR"):
            raise ValueError(f"Role {name} requires {role.get('modality', 'MR')} images")
        if candidate.get("metadata_consistent") is False:
            raise ValueError(f"Role {name} has inconsistent Study or Frame of Reference identifiers")
        if name != reference and candidate["key"] == primary["key"]:
            raise ValueError(f"Role {name} must use a companion series")
        record = {"series_key": candidate["key"], "series_uid": candidate["series_uid"],
                  "description": candidate.get("description", ""), "dir": "input" if name == reference else f"input_{name}",
                  "study_uid": candidate.get("study_uid"), "frame_uid": candidate.get("frame_uid"),
                  "frame_of_reference_differs": candidate.get("frame_uid") != primary.get("frame_uid")}
        files = candidate["files"]
        if name == "hbv":
            options = candidate.get("hbv_options", [])
            value = choice.get("b_value")
            if value is None and len(options) == 1:
                value = options[0]["b_value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("HBV requires an explicit documented b_value ≥ 800 s/mm²")
            option = next((o for o in options if o["b_value"] == value), None)
            if option is None:
                raise ValueError("Invalid HBV b_value: " + (candidate.get("hbv_unavailable_reason") or "b-value not present or below 800 s/mm²"))
            files = option["files"]
            record.update(b_value=value, b_value_source=option["b_value_source"])
        elif "b_value" in choice:
            raise ValueError(f"b_value is only supported for the hbv role, not {name}")
        if name != reference and candidate.get("instances"):
            problem = volume_problem([i for i in candidate["instances"] if i["file"] in files])
            if problem:
                raise ValueError(f"Role {name}: {problem}")
        records[name] = record
        sources[record["dir"]] = files
    keys = [r["series_key"] for r in records.values()]
    if len(keys) != len(set(keys)):
        raise ValueError("Each MRI role must use a different series")
    return records, sources, ""


def input_directories(data):
    names = {"input", "input_ct"}
    roles = data.get("input_roles", {})
    if not isinstance(roles, dict):
        raise ValueError("Invalid stored input_roles")
    for role, record in roles.items():
        if (not re.fullmatch(ROLE_PATTERN, role) or not isinstance(record, dict) or
                record.get("dir") not in {"input", f"input_{role}"}):
            raise ValueError("Unsafe stored role directory")
        names.add(record["dir"])
    if roles and sum(r["dir"] == "input" for r in roles.values()) != 1:
        raise ValueError("Exactly one reference input role is required")
    return ["input"] + sorted(names - {"input"})


def input_identity(data):
    payload = {key: data.get(key) for key in ("input_roles", "input_manifest")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

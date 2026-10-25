"""Machine-readable manifest packaged with every segmentation result.

The manifest lets downstream software verify a result ZIP without trusting file
names. It lists every packaged file with its size and SHA-256, identifies the
source image series, and records for each requested model the exact ROI names
written to its RTSTRUCT, the catalog label contract, and the licence and
limitation text shown in the portal. ``schemas/result-manifest-1.0.schema.json``
is the published JSON Schema for this structure.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

import pydicom

SCHEMA = "rtsegmentator.result-manifest"
SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "rtsegmentator_manifest.json"
RTSTRUCT_SUFFIX = "_RTSTRUCT.dcm"
# Descriptive catalog fields copied verbatim. None of them grants availability.
MODEL_FIELDS = (
    "description", "reference", "license", "license_note", "code_license", "weights_license",
    "noncommercial", "licensed", "limitations", "output", "modality",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    try:
        moment = dt.datetime.fromtimestamp(float(timestamp), tz=dt.timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def rtstruct_record(path: Path, arcname: str) -> dict:
    """Identity and ROI inventory of one packaged RTSTRUCT, read from its header."""
    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    names = [str(getattr(roi, "ROIName", "")).strip()
             for roi in getattr(dataset, "StructureSetROISequence", None) or []]
    frames, series = set(), set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", None) or []:
        uid = getattr(frame, "FrameOfReferenceUID", None)
        if uid:
            frames.add(str(uid))
        for study in getattr(frame, "RTReferencedStudySequence", None) or []:
            for item in getattr(study, "RTReferencedSeriesSequence", None) or []:
                uid = getattr(item, "SeriesInstanceUID", None)
                if uid:
                    series.add(str(uid))
    return {
        "path": arcname,
        "sop_instance_uid": str(getattr(dataset, "SOPInstanceUID", "") or "") or None,
        "roi_names": names,
        "roi_count": len(names),
        "frame_of_reference_uids": sorted(frames),
        "referenced_series_instance_uids": sorted(series),
    }


def _member_bytes(source: bytes | Path) -> tuple[int, str]:
    if isinstance(source, (bytes, bytearray)):
        return len(source), sha256_bytes(bytes(source))
    path = Path(source)
    return path.stat().st_size, sha256_file(path)


def _source_series(job: Mapping) -> dict | None:
    series = job.get("series")
    if not isinstance(series, Mapping):
        return None
    record = {
        "series_instance_uid": series.get("series_uid"),
        "modality": series.get("modality"),
        "slices": series.get("slices"),
        "rows": series.get("rows"),
        "columns": series.get("columns"),
        "cbct": series.get("cbct"),
    }
    return {key: value for key, value in record.items() if value is not None}


def _input_roles(job: Mapping) -> dict:
    roles = {}
    for name, record in (job.get("input_roles") or {}).items():
        if isinstance(record, Mapping):
            entry = {"series_instance_uid": record.get("series_uid"),
                     "selection_method": record.get("selection_method")}
            if record.get("b_value") is not None:
                entry["b_value"] = record.get("b_value")
            roles[str(name)] = {key: value for key, value in entry.items() if value is not None}
    return roles


def build_manifest(job: Mapping, members: Iterable[tuple[str, bytes | Path]], *, version: str,
                   catalog: Mapping[str, Mapping], structure_catalog: Mapping[str, Mapping],
                   empty_models: Iterable[str] = (), notes: Iterable[str] = (),
                   created: float | None = None) -> dict:
    """Describe a result package whose other members are ``members`` (arcname, bytes or path)."""
    members = sorted(members, key=lambda item: item[0])
    names = [name for name, _ in members]
    if len(set(names)) != len(names) or MANIFEST_NAME in names:
        raise ValueError("Result package member names must be unique and must not include the manifest")
    files, rtstructs = [], {}
    for name, source in members:
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError(f"Result package member escapes the package root: {name}")
        size, digest = _member_bytes(source)
        files.append({"path": name, "size": size, "sha256": digest})
        if "/" not in name and name.endswith(RTSTRUCT_SUFFIX) and isinstance(source, Path):
            try:
                rtstructs[name.removesuffix(RTSTRUCT_SUFFIX)] = rtstruct_record(source, name)
            except Exception as exc:  # an unreadable header must not fail a finished job
                rtstructs[name.removesuffix(RTSTRUCT_SUFFIX)] = {
                    "path": name, "sop_instance_uid": None, "roi_names": [], "roi_count": 0,
                    "frame_of_reference_uids": [], "referenced_series_instance_uids": [],
                    "error": f"RTSTRUCT header unreadable: {type(exc).__name__}: {exc}"[:500]}
    empty = set(empty_models)
    models = []
    requested = list(job.get("requested_models") or job.get("models") or [])
    for name in requested:
        details = catalog.get(name, {})
        contract = structure_catalog.get(name, {})
        entry = {
            "name": name,
            "structure_kind": contract.get("kind", details.get("structure_kind", "unresolved")),
            "catalog_structures": list(contract.get("names", details.get("structure_names", [])) or []),
            **{key: details[key] for key in MODEL_FIELDS if details.get(key) not in (None, "")},
        }
        if name in rtstructs:
            entry["outcome"] = "rtstruct"
            entry["rtstruct"] = rtstructs[name]
        elif details.get("output") == "report":
            entry["outcome"] = "report"
        elif name in empty:
            entry["outcome"] = "no_structures_found"
        else:
            entry["outcome"] = "no_rtstruct"
        # Orientation evidence, lesion tables, provenance and spatial maps share the
        # ``<task>_RTSTRUCT.dcm.`` prefix.
        present = [path for path in names if path.startswith(f"{name}{RTSTRUCT_SUFFIX}.")]
        if present:
            entry["sidecars"] = present
        models.append(entry)
    # An RTSTRUCT whose task is not in the job record is still a packaged output;
    # it is described rather than hidden, and marked as not requested.
    for name in sorted(set(rtstructs) - {model["name"] for model in models}):
        contract = structure_catalog.get(name, {})
        models.append({"name": name, "structure_kind": contract.get("kind", "unresolved"),
                       "catalog_structures": list(contract.get("names", []) or []),
                       "outcome": "rtstruct", "rtstruct": rtstructs[name], "requested": False})
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "RTsegmentator", "version": version},
        "created_utc": utc_iso(created),
        "job": {
            "id": job.get("id"),
            "created_utc": utc_iso(job.get("created")),
            "batch_id": job.get("batch_id"),
            "batch_index": job.get("batch_index"),
            "batch_size": job.get("batch_size"),
            "models": requested,
            "postprocess": list(job.get("postprocess") or []),
        },
        "source_series": _source_series(job),
        "models": models,
        "files": files,
        "notes": [str(note) for note in notes],
    }
    roles = _input_roles(job)
    if roles:
        manifest["input_roles"] = roles
    if job.get("paired_series_key"):
        manifest["paired_ct_series_key"] = job["paired_series_key"]
    return manifest


def dumps(manifest: Mapping) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode("utf-8")


def verify_package(manifest: Mapping, read_member) -> list[str]:
    """Return integrity problems; ``read_member(path)`` returns a member's bytes or None."""
    problems = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {manifest.get('schema')!r}")
    if str(manifest.get("schema_version", "")).split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        problems.append(f"unsupported schema version {manifest.get('schema_version')!r}")
    for record in manifest.get("files") or []:
        data = read_member(record.get("path"))
        if data is None:
            problems.append(f"missing member {record.get('path')}")
        elif len(data) != record.get("size") or sha256_bytes(data) != record.get("sha256"):
            problems.append(f"checksum mismatch for {record.get('path')}")
    return problems

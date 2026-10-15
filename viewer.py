"""Portal-local DICOM viewing. Coordinates always describe the stored pixel grid."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import numpy as np
from PIL import Image
import pydicom
from pydicom.uid import RTStructureSetStorage


CT_WINDOWS = [
    {"id": "soft-tissue", "label": "Soft tissue", "center": 40, "width": 400},
    {"id": "lung", "label": "Lung", "center": -600, "width": 1500},
    {"id": "bone", "label": "Bone", "center": 400, "width": 1800},
    {"id": "brain", "label": "Brain", "center": 40, "width": 80},
]
PALETTE = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6")
PRIVATE_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class Cache:
    """Bounded, synchronized caches contain geometry, never decoded pixel volumes."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key, build):
        with self.lock:
            if key not in self.items:
                self.items[key] = build()
            self.items.move_to_end(key)
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)
            return self.items[key]

    def discard(self, root):
        with self.lock:
            for key in list(self.items):
                if Path(key[0]).is_relative_to(root):
                    del self.items[key]


def contained(root: Path, path: Path) -> Path:
    """Reject symlinks as well as traversal, including metadata-supplied filenames."""
    try:
        valid = path.is_relative_to(root) and path.resolve() == root.resolve() / path.relative_to(root)
    except (OSError, RuntimeError):
        valid = False
    if not valid:
        raise HTTPException(404, "Data not found or expired")
    return path


def signature(path: Path):
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)


def vector(value, length):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError("Missing or invalid DICOM image geometry")
    return result


def first_float(value):
    return float(value[0] if isinstance(value, (list, tuple, pydicom.multival.MultiValue)) else value)


@dataclass
class Slice:
    path: Path
    uid: str
    ipp: np.ndarray
    column_direction: np.ndarray  # IOP first triplet: direction of increasing column.
    row_direction: np.ndarray     # IOP second triplet: direction of increasing row.
    spacing: np.ndarray           # DICOM order: row, column.
    position: float


@dataclass
class Series:
    slices: list[Slice]
    normal: np.ndarray
    info: dict
    tolerance: float
    frame_uid: str

    def check_index(self, index):
        if not 0 <= index < len(self.slices):
            raise HTTPException(404, "Slice not found")


def index_series(paths):
    slices = []
    first = None
    try:
        for path in paths:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except pydicom.errors.InvalidDicomError:
                continue
            if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                continue
            if int(getattr(ds, "NumberOfFrames", 1)) != 1 or hasattr(ds, "PerFrameFunctionalGroupsSequence"):
                raise ValueError("Multi-frame DICOM is not supported; upload a single-frame series")
            if int(getattr(ds, "SamplesPerPixel", 1)) != 1 or str(ds.PhotometricInterpretation) not in {"MONOCHROME1", "MONOCHROME2"}:
                raise ValueError("Only monochrome DICOM images are supported")
            orientation = vector(ds.ImageOrientationPatient, 6)
            column, row = orientation[:3], orientation[3:]
            if not (np.isclose(np.linalg.norm(column), 1, atol=1e-4)
                    and np.isclose(np.linalg.norm(row), 1, atol=1e-4)
                    and abs(float(column @ row)) < 1e-4):
                raise ValueError("DICOM direction cosines are not orthonormal")
            normal = np.cross(column, row)
            normal /= np.linalg.norm(normal)
            ipp, spacing = vector(ds.ImagePositionPatient, 3), vector(ds.PixelSpacing, 2)
            if np.any(spacing <= 0) or int(ds.Rows) < 1 or int(ds.Columns) < 1:
                raise ValueError("Invalid DICOM dimensions or pixel spacing")
            if first is None:
                first = ds
                reference_orientation, reference_spacing = orientation, spacing
                reference_normal = normal
            elif (str(ds.SeriesInstanceUID) != str(first.SeriesInstanceUID)
                  or str(getattr(ds, "FrameOfReferenceUID", "")) != str(getattr(first, "FrameOfReferenceUID", ""))
                  or (int(ds.Rows), int(ds.Columns)) != (int(first.Rows), int(first.Columns))
                  or not np.allclose(orientation, reference_orientation, atol=1e-4, rtol=0)
                  or not np.allclose(spacing, reference_spacing, atol=1e-5, rtol=0)):
                raise ValueError("Viewer requires one series with consistent orientation, dimensions and spacing")
            slices.append(Slice(path, str(ds.SOPInstanceUID), ipp, column, row, spacing, float(ipp @ reference_normal)))
        if not slices:
            raise ValueError("Series has no renderable image slices")
        slices.sort(key=lambda item: item.position)
        gaps = np.diff([item.position for item in slices])
        if len({item.uid for item in slices}) != len(slices) or np.any(gaps < 1e-4):
            raise ValueError("Duplicate image UIDs or coincident slice planes are not supported")
        nominal = abs(float(getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1))))
        if not math.isfinite(nominal) or nominal <= 0:
            nominal = 1.0
        slice_spacing = float(np.median(gaps)) if len(gaps) else nominal
        # Do not stretch matching tolerance across missing slices. No interpolation.
        matching_spacing = float(gaps.min()) if len(gaps) else nominal
        if hasattr(first, "SpacingBetweenSlices"):
            matching_spacing = min(matching_spacing, nominal)
        tolerance = matching_spacing / 2
        dominant = int(np.argmax(np.abs(reference_normal)))
        orientation_name = ("sagittal", "coronal", "axial")[dominant] if abs(reference_normal[dominant]) >= math.cos(math.radians(1)) else "oblique"
        presets = [dict(item) for item in CT_WINDOWS] if str(first.Modality) == "CT" else []
        try:
            center, width = first_float(first.WindowCenter), first_float(first.WindowWidth)
            if not math.isfinite(center) or not math.isfinite(width) or width < 1:
                raise ValueError()
            presets.insert(0, {"id": "dicom", "label": "DICOM window", "center": center, "width": width})
        except (AttributeError, TypeError, ValueError, IndexError):
            pass
        if not presets:
            # One representative slice supplies a stable series-wide MR/PT window.
            pixels = decoded_pixels(slices[len(slices) // 2].path)[0]
            low, high = np.percentile(pixels, [1, 99])
            presets.append({"id": "auto", "label": "Series window", "center": float((low + high) / 2),
                            "width": max(1.0, float(high - low))})
        info = {"modality": str(first.Modality), "rows": int(first.Rows), "columns": int(first.Columns),
                "slices": len(slices), "pixel_spacing": reference_spacing.tolist(), "slice_spacing": slice_spacing,
                "orientation": orientation_name, "plane": orientation_name,
                "column_direction": reference_orientation[:3].tolist(),
                "row_direction": reference_orientation[3:].tolist(),
                "window_presets": presets, "default_window": presets[0]["id"]}
        return Series(slices, reference_normal, info, tolerance, str(getattr(first, "FrameOfReferenceUID", "")))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(422, str(exc)) from exc


def decoded_pixels(path):
    try:
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array.astype(np.float64)
        pixels = pixels * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if pixels.ndim != 2 or not np.isfinite(pixels).all():
            raise ValueError()
        return pixels, str(ds.PhotometricInterpretation)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Series not found or expired") from exc
    except Exception as exc:
        raise HTTPException(422, "DICOM pixel decoding failed; check the transfer syntax and installed pixel decoder") from exc


def render_slice(series, index, window=None, center=None, width=None):
    series.check_index(index)
    if center is not None or width is not None:
        if window is not None or center is None or width is None:
            raise HTTPException(422, "Specify either window or both center and width")
    else:
        preset = next((p for p in series.info["window_presets"] if p["id"] == (window or series.info["default_window"])), None)
        if preset is None:
            raise HTTPException(422, "Unknown window preset")
        center, width = preset["center"], preset["width"]
    if not math.isfinite(center) or not math.isfinite(width) or width < 1:
        raise HTTPException(422, "Window center must be finite and width must be finite and at least 1")
    pixels, photometric = decoded_pixels(series.slices[index].path)
    # DICOM LINEAR window: pixel centres, including the width=1 threshold case.
    if width == 1:
        rendered = np.where(pixels > center - 0.5, 255, 0).astype(np.uint8)
    else:
        rendered = np.clip(((pixels - (center - 0.5)) / (width - 1) + 0.5) * 255, 0, 255).astype(np.uint8)
    if photometric == "MONOCHROME1":
        rendered = 255 - rendered
    payload = io.BytesIO()
    Image.fromarray(rendered).save(payload, format="PNG")
    return Response(payload.getvalue(), media_type="image/png", headers={**PRIVATE_HEADERS,
                    "X-Slice-Index": str(index), "X-Slice-Count": str(len(series.slices)),
                    "X-Columns": str(series.info["columns"]), "X-Rows": str(series.info["rows"])})


def read_structure(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except pydicom.errors.InvalidDicomError:
        return None
    if str(getattr(ds, "SOPClassUID", "")) != RTStructureSetStorage:
        return None
    # Keep only overlay metadata; do not cache demographics or the source dataset.
    names = {int(roi.ROINumber): (str(roi.ROIName), str(getattr(roi, "ReferencedFrameOfReferenceUID", "")))
             for roi in getattr(ds, "StructureSetROISequence", [])}
    contours = {int(roi.ReferencedROINumber): roi for roi in getattr(ds, "ROIContourSequence", [])}
    rois = []
    for number, (name, frame) in names.items():
        roi = contours.get(number)
        color = PALETTE[(number - 1) % len(PALETTE)]
        try:
            rgb = [int(v) for v in roi.ROIDisplayColor]
            if len(rgb) == 3 and all(0 <= v <= 255 for v in rgb):
                color = "#" + "".join(f"{v:02x}" for v in rgb)
        except (AttributeError, TypeError, ValueError):
            pass
        paths = []
        for contour in getattr(roi, "ContourSequence", []):
            kind = str(getattr(contour, "ContourGeometricType", ""))
            if kind not in {"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"}:
                raise HTTPException(422, "Viewer supports only CLOSED_PLANAR or CLOSEDPLANAR_XOR contours")
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            # rt-utils keeps single-pixel and line components as degenerate
            # closed contours. Preserve them in the independent pixel mapping.
            if not len(points) or not np.isfinite(points).all():
                raise HTTPException(422, "RTSTRUCT contains invalid contour coordinates")
            if int(getattr(contour, "NumberOfContourPoints", 0)) != len(points):
                raise HTTPException(422, "RTSTRUCT contour point count does not match coordinates")
            refs = tuple(str(im.ReferencedSOPInstanceUID) for im in getattr(contour, "ContourImageSequence", []))
            paths.append((refs, points))
        rois.append(({"number": number, "name": name, "color": color}, frame, paths))
    return str(getattr(ds, "StructureSetLabel", "")), rois


def map_structure(raw, path, output, series):
    label, rois = raw
    relative = path.relative_to(output).as_posix()
    prefix = hashlib.sha256(relative.encode()).hexdigest()[:16]
    model = path.stem.removesuffix("_RTSTRUCT")
    structure = {"model": model, "label": label or model, "rois": []}
    by_uid = {item.uid: index for index, item in enumerate(series.slices)}
    positions = np.asarray([item.position for item in series.slices])
    mapped = {}
    for info, frame, paths in rois:
        key = f"{prefix}:{info['number']}"
        planes = {}
        for refs, points in paths:
            if frame and series.frame_uid and frame != series.frame_uid:
                continue
            if refs:
                matches = {by_uid[uid] for uid in refs if uid in by_uid}
                if not matches:
                    continue  # A reference to another series must never fall back to distance.
                if len(matches) != 1:
                    raise HTTPException(422, "One RTSTRUCT contour references multiple source planes")
                index = matches.pop()
            else:
                position = float(np.mean(points @ series.normal))
                distances = np.abs(positions - position)
                index = int(np.argmin(distances))
                if distances[index] > series.tolerance + 1e-4:
                    continue
                if np.count_nonzero(np.isclose(distances, distances[index], atol=1e-4, rtol=0)) > 1:
                    continue  # Exactly halfway between two planes is ambiguous.
            item = series.slices[index]
            delta = points - item.ipp
            if np.max(np.abs(delta @ series.normal)) > series.tolerance + 1e-4:
                if refs:
                    raise HTTPException(422, "RTSTRUCT contour is off its referenced slice plane")
                continue
            if np.ptp(points @ series.normal) > 1e-3:
                raise HTTPException(422, "RTSTRUCT contour is not planar on the source image grid")
            xy = np.column_stack((delta @ item.column_direction / item.spacing[1],
                                  delta @ item.row_direction / item.spacing[0]))
            planes.setdefault(index, []).append(xy.tolist())
        structure["rois"].append({"key": key, **info, "slices": sorted(planes)})
        for index, polygons in planes.items():
            mapped.setdefault(index, []).append({"key": key, "paths": polygons})
    return structure, mapped


class ViewerService:
    """The live app namespace supplies existing lifecycle/SSH helpers without circular imports."""

    def __init__(self, portal):
        self.portal = portal
        self.series_cache, self.rt_cache, self.overlay_cache = Cache(16), Cache(128), Cache(16)
        self.mutation_lock = threading.RLock()
        self.router = APIRouter(tags=["viewer"])
        self.add_routes()

    def serialize_job_mutation(self, function):
        @wraps(function)
        def locked(*args, **kwargs):
            with self.mutation_lock:
                return function(*args, **kwargs)
        return locked

    def resource(self, identifier, upload=False, allow_expired=False):
        pattern = r"upload-[A-Za-z0-9-]+" if upload else r"[A-Za-z0-9-]+"
        if len(identifier) > 128 or not re.fullmatch(pattern, identifier) or (not upload and identifier.startswith(("upload-", "session-"))):
            raise HTTPException(404, "Data not found or expired")
        root = self.portal["BASE"]
        path = contained(root, root / identifier)
        try:
            meta = contained(path, path / ("upload.json" if upload else "job.json"))
            data = json.loads(meta.read_text())
            if not allow_expired and float(data.get("created", path.stat().st_mtime)) < time.time() - self.portal["RETENTION_HOURS"] * 3600:
                raise HTTPException(404, "Data not found or expired")
            return path, data
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(404, "Data not found or expired") from exc

    def series(self, root, names=None):
        try:
            contained(root.parent, root)
            if not root.is_dir():
                raise FileNotFoundError()
            paths = sorted(contained(root, root / name) for name in names) if names is not None else sorted(
                contained(root, path) for path in root.iterdir() if path.is_file())
            key = (str(root), tuple(signature(path) for path in paths))
            return key, self.series_cache.get(key, lambda: index_series(paths))
        except FileNotFoundError as exc:
            raise HTTPException(404, "Series not found or expired") from exc

    def upload_series(self, uid, series_key):
        root, data = self.resource(uid, upload=True)
        selected = next((item for item in data["series"] if item["key"] == series_key), None)
        if selected is None:
            raise HTTPException(404, "Series not found")
        return self.series(root / "input", selected["files"])[1]

    def job_series(self, jid):
        root, data = self.resource(jid)
        key, series = self.series(root / "input")
        return root, data, key, series

    def overlays(self, root, series_key, series):
        output = contained(root, root / "output")
        try:
            files = sorted(contained(root, path) for path in output.rglob("*") if path.is_file())
            versions = tuple(signature(path) for path in files)
            def build():
                structures, slices = [], {}
                for path, version in zip(files, versions):
                    raw = self.rt_cache.get(version, lambda: read_structure(path))
                    if raw is None:
                        continue
                    structure, mapped = map_structure(raw, path, output, series)
                    structures.append(structure)
                    for index, rois in mapped.items():
                        slices.setdefault(index, []).extend(rois)
                return structures, slices
            return self.overlay_cache.get((str(root), series_key, versions), build)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Job output not found or expired") from exc
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(422, "Invalid RTSTRUCT geometry or metadata") from exc

    def preview(self, uid, series_key, index=0):
        series = self.upload_series(uid, series_key)
        return render_slice(series, max(0, min(index, len(series.slices) - 1)))

    def delete(self, jid: str):
        with self.mutation_lock:
            root, data = self.resource(jid, allow_expired=True)
            if data.get("status") not in {"complete", "failed", "cancelled"}:
                raise HTTPException(409, "Job is queued or running")
            portal = self.portal
            remote = (portal["WORKER_MODE"] == "ssh" or data.get("remote_results_retained")
                      or data.get("execution_backend") == "ssh_task"
                      or any(portal["task_uses_remote"](task) for task in data.get("models", []))
                      or (portal["WORKER_MODE"] == "colab" and portal["COLAB_ORCHESTRATOR"] == "worker"))
            if remote:
                try:
                    portal["remove_worker_job"](root, f"{portal['REMOTE_BASE']}/{jid}")
                except (OSError, subprocess.SubprocessError) as exc:
                    raise HTTPException(502, "Remote deletion failed; local job retained so deletion can be retried") from exc
            try:
                shutil.rmtree(root)
            except FileNotFoundError as exc:
                raise HTTPException(404, "Job not found or expired") from exc
            for cache in (self.series_cache, self.rt_cache, self.overlay_cache):
                cache.discard(root)
            return {"id": jid, "deleted": True}

    def add_routes(self):
        @self.router.get("/api/v1/jobs/{jid}/viewer")
        def job_viewer(jid: str, response: Response):
            root, data, key, series = self.job_series(jid)
            structures, _ = self.overlays(root, key, series)
            response.headers.update(PRIVATE_HEADERS)
            report = contained(root, root / "output/report" / (self.portal["REPORT_TASK"] + ".json"))
            return {"series": series.info, "structure_sets": structures,
                    "report": data.get("status") == "complete" and report.is_file()}

        @self.router.get("/api/v1/jobs/{jid}/viewer/slices/{index}.png")
        def job_slice(jid: str, index: int, window: str | None = None, center: float | None = None, width: float | None = None):
            return render_slice(self.job_series(jid)[3], index, window, center, width)

        @self.router.get("/api/v1/jobs/{jid}/viewer/contours/{index}")
        def job_contours(jid: str, index: int, response: Response):
            root, _, key, series = self.job_series(jid)
            series.check_index(index)
            _, slices = self.overlays(root, key, series)
            response.headers.update(PRIVATE_HEADERS)
            return {"index": index, "rois": slices.get(index, [])}

        @self.router.get("/api/v1/uploads/{uid}/series/{series_key}/viewer")
        def upload_viewer(uid: str, series_key: str, response: Response):
            response.headers.update(PRIVATE_HEADERS)
            return {"series": self.upload_series(uid, series_key).info, "structure_sets": [], "report": False}

        @self.router.get("/api/v1/uploads/{uid}/series/{series_key}/viewer/slices/{index}.png")
        def upload_slice(uid: str, series_key: str, index: int, window: str | None = None, center: float | None = None, width: float | None = None):
            return render_slice(self.upload_series(uid, series_key), index, window, center, width)

        self.router.add_api_route("/api/v1/jobs/{jid}", self.delete, methods=["DELETE"])

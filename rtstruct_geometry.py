"""RTSTRUCT polygons on each acquired DICOM plane, without a volume affine.

rt-utils supplies the DICOM container. OpenCV uses its existing contour
convention (pixel centres, RETR_TREE, CHAIN_APPROX_SIMPLE). Patient coordinates
and SOP references come from the particular source image, in GDCM array order.
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.dataelem import RawDataElement
from pydicom.tag import Tag
from pydicom.valuerep import format_number_as_ds
from rt_utils import RTStructBuilder, ds_helper
from rt_utils.utils import ROIData

from dicom_series import ordered_dicom_files
from structure_labels import roi_name, unique_roi_names


def slice_geometry(ds):
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
    origin = np.asarray(ds.ImagePositionPatient, dtype=float)
    spacing = np.asarray(ds.PixelSpacing, dtype=float)
    if (orientation.shape != (6,) or origin.shape != (3,) or spacing.shape != (2,)
            or not all(np.isfinite(a).all() for a in (orientation, origin, spacing))
            or np.any(spacing <= 0)):
        raise ValueError("Invalid DICOM slice geometry")
    column, row = orientation[:3], orientation[3:]
    if (not np.isclose(np.linalg.norm(column), 1, atol=1e-4)
            or not np.isclose(np.linalg.norm(row), 1, atol=1e-4)
            or abs(column @ row) > 1e-4):
        raise ValueError("DICOM direction cosines are not orthonormal")
    return origin, column * spacing[1], row * spacing[0]


def dicom_grid(source, reference):
    """Return the original GDCM grid and acquired planes without reading pixels."""
    files = ordered_dicom_files(source)
    slices = [pydicom.dcmread(path, stop_before_pixels=True) for path in files]
    first = slices[0]
    shape = (int(first.Columns), int(first.Rows), len(slices))
    uids = [str(ds.SOPInstanceUID) for ds in slices]
    if len(set(uids)) != len(uids):
        raise ValueError("Duplicate source SOP Instance UIDs")
    for ds in slices:
        if (int(getattr(ds, "NumberOfFrames", 1)) != 1
                or hasattr(ds, "PerFrameFunctionalGroupsSequence")
                or (int(ds.Columns), int(ds.Rows)) != shape[:2]
                or ds.FrameOfReferenceUID != first.FrameOfReferenceUID):
            raise ValueError("Inconsistent or multi-frame DICOM source images")
        slice_geometry(ds)

    # Reconstruct the original GDCM source grid. GDCM stores
    # irregular acquired planes at successive array indices, with the mean
    # endpoint distance in the NIfTI affine. That affine must not position RT
    # contours. Only an exact permutation/flip may restore a prediction here.
    origin, column, row = slice_geometry(first)
    normal = np.cross(column, row)
    normal /= np.linalg.norm(normal)
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1], affine[:3, 3] = column, row, origin
    if len(slices) > 1:
        step = np.linalg.norm(np.asarray(slices[-1].ImagePositionPatient) - origin) / (len(slices) - 1)
        if step <= 0:
            raise ValueError("Coincident DICOM source planes")
        affine[:3, 2] = normal * step
    else:
        # For a single plane only its in-plane grid and origin are determined.
        axes = np.asarray(reference.affine[:3, :3]) * np.array([-1, -1, 1])[:, None]
        axis = int(np.argmax(np.abs(normal @ axes) / np.linalg.norm(axes, axis=0)))
        affine[:3, 2] = normal * np.linalg.norm(axes[:, axis])
    affine[:2] *= -1  # LPS -> RAS
    return shape, affine, slices


def create_rtstruct(source, model_input):
    shape, affine, slices = dicom_grid(source, model_input)
    if model_input.shape != shape:
        raise ValueError(f"model input shape {model_input.shape} does not match DICOM {shape}")
    if not np.allclose(model_input.affine, affine, atol=1e-3, rtol=0):
        raise ValueError("model input affine does not match the DICOM/GDCM source grid")

    uids = [str(ds.SOPInstanceUID) for ds in slices]
    rt = RTStructBuilder.create_new(dicom_series_path=str(source))
    if {str(ds.SOPInstanceUID) for ds in rt.series_data} != set(uids):
        raise ValueError("rt-utils did not load every DICOM source image")
    # Do not depend on rt-utils sorting: use the exact reader used for inference.
    rt.series_data = slices
    return rt



def add_roi(rt, mask, name, color, *, offset=(0, 0, 0), bounds_out=None, plane_geometry=None):
    """Add a [column,row,slice] box at its source offset; False if no contour exists."""
    contours = []
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    ox, oy, oz = offset
    for k in range(mask.shape[2]):
        ds = rt.series_data[k + oz]
        polygons, _ = cv2.findContours(mask[:, :, k].T.astype(np.uint8),
                                      cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        origin, column, row = (slice_geometry(ds) if plane_geometry is None
                               else plane_geometry[k + oz])
        for polygon in polygons:
            xy = polygon[:, 0, :] + np.array([ox, oy])
            # Match rt-utils' complete pixel-centre contour set, including
            # single points and zero-area lines. Removing these pieces loses
            # source voxels and can shrink an ROI's physical bounds.
            contour = Dataset()
            syntax = rt.ds.file_meta.TransferSyntaxUID
            contour.set_original_encoding(syntax.is_implicit_VR, syntax.is_little_endian,
                                          contour._character_set)
            contour.ContourGeometricType = "CLOSED_PLANAR"
            contour.NumberOfContourPoints = len(xy)
            points = origin + xy[:, :1] * column + xy[:, 1:] * row
            # Store exactly the original DS strings as an unconverted element.
            # pydicom writes raw elements directly; callers inspecting this
            # dataset still get the usual lazy DS conversion on first access.
            # No global validation settings are changed.
            encoded = "\\".join(format_number_as_ds(float(value))
                                for value in points.ravel()).encode("ascii")
            if len(encoded) % 2:
                encoded += b" "
            tag = Tag(0x3006, 0x0050)
            contour[tag] = RawDataElement(tag, "DS", len(encoded), encoded,
                                          0, syntax.is_implicit_VR, syntax.is_little_endian)
            lower = np.minimum(lower, points.min(0))
            upper = np.maximum(upper, points.max(0))
            reference = Dataset()
            reference.ReferencedSOPClassUID = ds.SOPClassUID
            reference.ReferencedSOPInstanceUID = ds.SOPInstanceUID
            contour.ContourImageSequence = [reference]
            contours.append(contour)
    if not contours:
        return False

    number = len(rt.ds.StructureSetROISequence) + 1
    roi_data = ROIData(mask.transpose(1, 0, 2), color, number, str(name),
                       rt.series_data[0].FrameOfReferenceUID)
    rt.ds.StructureSetROISequence.append(ds_helper.create_structure_set_roi(roi_data))
    geometry = Dataset()
    geometry.ReferencedROINumber = number
    geometry.ROIDisplayColor = color
    geometry.ContourSequence = contours
    rt.ds.ROIContourSequence.append(geometry)
    rt.ds.RTROIObservationsSequence.append(ds_helper.create_rtroi_observation(roi_data))
    if bounds_out is not None:
        # Every linear-coordinate extremum of foreground pixels lies on its
        # contour, including points/lines and interior/exterior boundaries.
        bounds_out.extend((lower, upper))
    return True


def mask_bounds(mask, slices, *, offset=(0, 0, 0)):
    """Physical bounds of a voxel box using its source offset and acquired planes."""
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    ox, oy, oz = offset
    for k in range(mask.shape[2]):
        ds = slices[k + oz]
        xy = np.argwhere(mask[:, :, k])
        if not len(xy):
            continue
        xy += np.array([ox, oy])
        origin, column, row = slice_geometry(ds)
        points = origin + xy[:, :1] * column + xy[:, 1:] * row
        lower = np.minimum(lower, points.min(0))
        upper = np.maximum(upper, points.max(0))
    return lower, upper


def report_omissions(output, omitted):
    if omitted:
        # Both local and SSH workers retain stderr in the job log. The portal
        # publishes these structured notes in metadata and the result ZIP.
        print("RTSTRUCT_CONVERSION_NOTE " + json.dumps({
            "output": Path(output).name, "omitted_rois": omitted}), file=sys.stderr, flush=True)

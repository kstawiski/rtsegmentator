"""Frozen writer and physical bounds from 3600f83; synthetic test/benchmark reference."""
import cv2
import numpy as np
from pydicom.dataset import Dataset
from pydicom.valuerep import format_number_as_ds
from rt_utils import ds_helper
from rt_utils.utils import ROIData

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

def add_roi(rt, mask, name, color):
    """Add a [column,row,slice] mask; return False only if no contour exists."""
    contours = []
    for k, ds in enumerate(rt.series_data):
        polygons, _ = cv2.findContours(mask[:, :, k].T.astype(np.uint8),
                                      cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        origin, column, row = slice_geometry(ds)
        for polygon in polygons:
            xy = polygon[:, 0, :]
            # Match rt-utils' complete pixel-centre contour set, including
            # single points and zero-area lines. Removing these pieces loses
            # source voxels and can shrink an ROI's physical bounds.
            contour = Dataset()
            contour.ContourGeometricType = "CLOSED_PLANAR"
            contour.NumberOfContourPoints = len(xy)
            points = origin + xy[:, :1] * column + xy[:, 1:] * row
            contour.ContourData = [format_number_as_ds(float(value)) for value in points.ravel()]
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
    return True

def mask_bounds(mask, slices):
    """Physical bounds of source voxels using their acquired plane geometry."""
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for k, ds in enumerate(slices):
        xy = np.argwhere(mask[:, :, k])
        if not len(xy):
            continue
        origin, column, row = slice_geometry(ds)
        points = origin + xy[:, :1] * column + xy[:, 1:] * row
        lower = np.minimum(lower, points.min(0))
        upper = np.maximum(upper, points.max(0))
    return lower, upper

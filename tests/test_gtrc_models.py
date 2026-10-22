"""Offline synthetic unit tests for GTRC-Net PET/CT tasks and QIBA SUVbw converter."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    PositronEmissionTomographyImageStorage,
    generate_uid,
)
import pytest
import SimpleITK as sitk

import gtrc_models
from pet_to_suvbw import (
    MIN_PLAUSIBLE_DOSE_BQ,
    RWVM_MAX_SUV_ERROR_TOL,
    RWVM_SLOPE_RTOL,
    compute_suvbw_from_datasets,
    convert_dicom_series_to_suvbw_nifti,
)
try:
    import run_gtrc
except ImportError:  # pandas and scikit-image are runtime-only; CI installs requirements-dev.txt
    run_gtrc = None

requires_run_gtrc = pytest.mark.skipif(
    run_gtrc is None, reason="run_gtrc needs pandas and scikit-image, which are not installed"
)

ROOT = Path(__file__).resolve().parents[1]


def make_pet_header(
    *,
    units: str = "BQML",
    suv_type: str | None = None,
    rescale_slope: float | None = 1.0,
    rescale_intercept: float | None = 0.0,
    decay_correction: str | None = "START",
    corrected_image: list[str] | None = ("DECY", "ATTN"),
    patient_weight: float | None = 80.0,
    dose_bq: float | None = 160_000_000.0,
    half_life_s: float | None = 6586.2,
    start_time: str | None = "100000.00",
    start_datetime: str | None = None,
    series_date: str | None = "20250115",
    series_time: str | None = "114946.20",
    acquisition_date: str | None = "20250115",
    acquisition_time: str | None = "114946.20",
    study_date: str | None = "20250115",
    timezone_offset_from_utc: str | None = None,
    study_uid: str = "1.2.826.0.1.3680043.10.100",
    series_uid: str = "1.2.826.0.1.3680043.10.101",
    rwvm_slope: float | None = None,
    rwvm_intercept: float | None = 0.0,
    rwvm_ucum: str | None = None,
    rwvm_scheme: str = "UCUM",
    rwvm_omit_units: bool = False,
    rwvm_lut: list[float] | None = None,
    rwvm_lut_explanation: str | None = None,
    rwvm_lut_label: str | None = None,
    rwvm_first_mapped: int | None = 0,
    rwvm_last_mapped: int | None = 65535,
    rwvm_omit_mapped_range: bool = False,
    pixel_values: list[int] | None = None,
    pixel_representation: int = 0,
    smallest_pixel_value: int | None = None,
    largest_pixel_value: int | None = None,
    frame_reference_time: float | None = None,
    actual_frame_duration: float | None = None,
    ge_scan_datetime: str | None = None,
    ge_private_creator: str | None = "GEMS_PETD_01",
    siemens_decay_datetime: str | None = None,
    siemens_private_creator: str | None = "SIEMENS MED PT",
) -> Dataset:
    ds = Dataset()
    ds.Modality = "PT"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    if units is not None:
        ds.Units = units
    if suv_type is not None:
        ds.SUVType = suv_type
    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept
    if decay_correction is not None:
        ds.DecayCorrection = decay_correction
    if corrected_image is not None:
        ds.CorrectedImage = list(corrected_image)
    if patient_weight is not None:
        ds.PatientWeight = patient_weight
    if study_date is not None:
        ds.StudyDate = study_date
    if series_date is not None:
        ds.SeriesDate = series_date
    if series_time is not None:
        ds.SeriesTime = series_time
    if acquisition_date is not None:
        ds.AcquisitionDate = acquisition_date
    if acquisition_time is not None:
        ds.AcquisitionTime = acquisition_time
    if timezone_offset_from_utc is not None:
        ds.TimezoneOffsetFromUTC = timezone_offset_from_utc
    if frame_reference_time is not None:
        ds.FrameReferenceTime = frame_reference_time
    if actual_frame_duration is not None:
        ds.ActualFrameDuration = actual_frame_duration
    if ge_scan_datetime is not None:
        if ge_private_creator is not None:
            ds.add_new((0x0009, 0x0010), "LO", ge_private_creator)
        ds.add_new((0x0009, 0x100D), "DT", ge_scan_datetime)
    if siemens_decay_datetime is not None:
        if siemens_private_creator is not None:
            ds.add_new((0x0071, 0x0010), "LO", siemens_private_creator)
        ds.add_new((0x0071, 0x1022), "DT", siemens_decay_datetime)

    if smallest_pixel_value is not None:
        ds.SmallestImagePixelValue = int(smallest_pixel_value)
    if largest_pixel_value is not None:
        ds.LargestImagePixelValue = int(largest_pixel_value)
    if pixel_values is not None:
        arr = np.asarray(pixel_values, dtype=np.int16 if pixel_representation == 1 else np.uint16)
        ds.Rows = 1
        ds.Columns = int(arr.size)
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = int(pixel_representation)
        ds.PixelData = arr.tobytes()

    rph = Dataset()
    if dose_bq is not None:
        rph.RadionuclideTotalDose = dose_bq
    if half_life_s is not None:
        rph.RadionuclideHalfLife = half_life_s
    if start_time is not None:
        rph.RadiopharmaceuticalStartTime = start_time
    if start_datetime is not None:
        rph.RadiopharmaceuticalStartDateTime = start_datetime
    ds.RadiopharmaceuticalInformationSequence = DicomSequence([rph])

    if (
        rwvm_slope is not None
        or rwvm_ucum is not None
        or rwvm_lut is not None
        or rwvm_lut_explanation is not None
        or rwvm_lut_label is not None
        or rwvm_omit_units
    ):
        rw = Dataset()
        if rwvm_slope is not None:
            rw.RealWorldValueSlope = rwvm_slope
        if rwvm_intercept is not None:
            rw.RealWorldValueIntercept = rwvm_intercept
        if rwvm_lut is not None:
            rw.LUTData = list(rwvm_lut)
        if rwvm_lut_explanation is not None:
            rw.LUTExplanation = rwvm_lut_explanation
        if rwvm_lut_label is not None:
            rw.LUTLabel = rwvm_lut_label
        if not rwvm_omit_mapped_range:
            if rwvm_first_mapped is not None:
                rw.RealWorldValueFirstValueMapped = int(rwvm_first_mapped)
            if rwvm_last_mapped is not None:
                rw.RealWorldValueLastValueMapped = int(rwvm_last_mapped)
        effective_ucum = rwvm_ucum
        if effective_ucum is None and not rwvm_omit_units:
            effective_ucum = "Bq/ml" if (units or "").upper() == "BQML" else "g/ml{SUVbw}"
        if effective_ucum is not None and not rwvm_omit_units:
            unit_item = Dataset()
            unit_item.CodeValue = effective_ucum
            unit_item.CodingSchemeDesignator = rwvm_scheme
            unit_item.CodeMeaning = effective_ucum
            rw.MeasurementUnitsCodeSequence = DicomSequence([unit_item])
        ds.RealWorldValueMappingSequence = DicomSequence([rw])

    return ds


def write_synthetic_dicom_series(
    directory: Path,
    volume_zyx: np.ndarray,
    *,
    modality: str,
    study_uid: str,
    frame_uid: str,
    series_uid: str | None = None,
    origin_xyz: tuple[float, float, float] = (-100.0, -100.0, -50.0),
    spacing_xyz: tuple[float, float, float] = (2.5, 2.5, 3.0),
    pet_kwargs: dict | None = None,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = series_uid or generate_uid()
    sop_class = PositronEmissionTomographyImageStorage if modality == "PT" else CTImageStorage
    nz, ny, nx = volume_zyx.shape
    pet_kwargs = pet_kwargs or {}

    for iz in range(nz):
        path = directory / f"{iz:03d}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = sop_class
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PatientName = "SYNTHETIC^GTRC"
        ds.PatientID = "SYNTHETIC01"
        ds.StudyID = "1"
        ds.StudyDate = pet_kwargs.get("study_date", "20250115")
        ds.SeriesDate = pet_kwargs.get("series_date", "20250115")
        ds.AcquisitionDate = pet_kwargs.get("acquisition_date", "20250115")
        ds.StudyTime = "100000"
        ds.SeriesTime = pet_kwargs.get("series_time", "100000.00")
        ds.AcquisitionTime = pet_kwargs.get("acquisition_time", "100000.00")
        ds.Modality = modality
        ds.Rows, ds.Columns = int(ny), int(nx)
        ds.InstanceNumber = iz + 1
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [
            float(origin_xyz[0]),
            float(origin_xyz[1]),
            float(origin_xyz[2] + iz * spacing_xyz[2]),
        ]
        ds.PixelSpacing = [float(spacing_xyz[1]), float(spacing_xyz[0])]
        ds.SliceThickness = float(spacing_xyz[2])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        if modality == "PT":
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.Units = pet_kwargs.get("units", "BQML")
            ds.DecayCorrection = pet_kwargs.get("decay_correction", "START")
            ds.CorrectedImage = ["DECY", "ATTN"]
            ds.PatientWeight = float(pet_kwargs.get("patient_weight", 100.0))
            slope = float(pet_kwargs.get("rescale_slope", 1.0))
            intercept = float(pet_kwargs.get("rescale_intercept", 0.0))
            ds.RescaleSlope = slope
            ds.RescaleIntercept = intercept
            rph = Dataset()
            rph.RadionuclideTotalDose = float(pet_kwargs.get("dose_bq", 100_000.0))
            rph.RadionuclideHalfLife = float(pet_kwargs.get("half_life_s", 6586.2))
            rph.RadiopharmaceuticalStartTime = pet_kwargs.get("start_time", "100000.00")
            rph.RadiopharmaceuticalStartDateTime = pet_kwargs.get(
                "start_datetime", "20250115100000.00"
            )
            ds.RadiopharmaceuticalInformationSequence = DicomSequence([rph])
            stored = np.clip(np.rint((volume_zyx[iz] - intercept) / slope), 0, 65535).astype(np.uint16)
            ds.PixelData = stored.tobytes()
        else:
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = 0.0
            stored = np.clip(np.rint(volume_zyx[iz]), -32768, 32767).astype(np.int16)
            ds.PixelData = stored.tobytes()

        ds.save_as(path, enforce_file_format=True)
    return series_uid


def test_suvbw_converter_exact_qiba_calculation():
    ds1 = make_pet_header()
    ds2 = make_pet_header()
    meta = compute_suvbw_from_datasets([ds1, ds2])
    assert meta["conversion_mode"] == "bqml_to_suvbw"
    assert meta["elapsed_seconds"] == pytest.approx(6586.2, abs=1e-3)
    assert meta["decay_factor"] == pytest.approx(0.5, rel=1e-6)
    assert meta["suv_scale_factor"] == pytest.approx(0.001, rel=1e-6)
    assert meta["dose_floor_heuristic_bq"] == MIN_PLAUSIBLE_DOSE_BQ == 10_000.0


def test_suvbw_converter_qiba_scan_start_resolutions_and_conflicts():
    # 1. SeriesTime > AcquisitionTime resolved via GE private tag (0009,100D) with GEMS_PETD_01
    ds_ge = make_pet_header(
        series_time="130000.00",
        acquisition_time="120000.00",
        ge_scan_datetime="20250115114946.20",
        ge_private_creator="GEMS_PETD_01",
    )
    meta_ge = compute_suvbw_from_datasets([ds_ge])
    assert "GE" in meta_ge["reference_time_source"] and "0009,100d" in meta_ge["reference_time_source"].lower()
    assert meta_ge["elapsed_seconds"] == pytest.approx(6586.2, abs=1e-3)

    # Reviewer probe (Finding 5): GE (0009,100D) without GEMS_PETD_01 private creator must be rejected
    ds_ge_bad_creator = make_pet_header(
        series_time="130000.00",
        acquisition_time="120000.00",
        ge_scan_datetime="20250115114946.20",
        ge_private_creator="OTHER_CREATOR",
    )
    with pytest.raises(ValueError, match="GEMS_PETD_01"):
        compute_suvbw_from_datasets([ds_ge_bad_creator])

    ds_ge_missing_creator = make_pet_header(
        series_time="130000.00",
        acquisition_time="120000.00",
        ge_scan_datetime="20250115114946.20",
        ge_private_creator=None,
    )
    with pytest.raises(ValueError, match="GEMS_PETD_01"):
        compute_suvbw_from_datasets([ds_ge_missing_creator])

    # 2. SeriesTime > AcquisitionTime resolved via Siemens private tag (0071,1022)
    ds_siemens = make_pet_header(
        series_time="130000.00",
        acquisition_time="120000.00",
        siemens_decay_datetime="20250115114946.20",
    )
    meta_siemens = compute_suvbw_from_datasets([ds_siemens])
    assert "Siemens" in meta_siemens["reference_time_source"] and "0071,1022" in meta_siemens["reference_time_source"]
    assert meta_siemens["elapsed_seconds"] == pytest.approx(6586.2, abs=1e-3)

    # 3. Multi-bed acquisition where SeriesTime is absent or overwritten, resolved via
    # QIBA FrameReferenceTime (0054,1300) and ActualFrameDuration (0018,1242).
    slice_a = make_pet_header(
        series_time="140000.00",
        acquisition_time="114946.20",
        frame_reference_time=59936.855,
        actual_frame_duration=120000.0,
    )
    slice_b = make_pet_header(
        series_time="140000.00",
        acquisition_time="115146.20",
        frame_reference_time=179936.855,
        actual_frame_duration=120000.0,
    )
    meta_qiba = compute_suvbw_from_datasets([slice_a, slice_b])
    assert "QIBA FrameReferenceTime" in meta_qiba["reference_time_source"]
    assert meta_qiba["elapsed_seconds"] == pytest.approx(6586.2, abs=0.1)

    # Reviewer probe (Finding 5): SeriesTime <= AcquisitionTime that contradicts FrameReferenceTime / ActualFrameDuration
    ds_conflict = make_pet_header(
        series_time="101000.00",
        acquisition_time="114946.20",
        frame_reference_time=59936.855,
        actual_frame_duration=120000.0,
    )
    with pytest.raises(ValueError, match="Contradictory SeriesTime.*FrameReferenceTime"):
        compute_suvbw_from_datasets([ds_conflict])

    # Reviewer probe (Finding 5): FrameReferenceTime=1 ms, ActualFrameDuration=120000 ms
    # yields a back-computed scan start ~59.94 s AFTER AcquisitionTime -> must be rejected
    ds_after_acq = make_pet_header(
        series_time="103000.00",
        acquisition_time="110000.00",
        frame_reference_time=1.0,
        actual_frame_duration=120000.0,
    )
    with pytest.raises(ValueError, match="later than AcquisitionTime"):
        compute_suvbw_from_datasets([ds_after_acq])


def test_suvbw_converter_timezone_aware_timestamps():
    # Reviewer probe (Finding 5): injection 20250115100000+0100 (09:00 UTC),
    # acquisition 20250115110000+0000 (11:00 UTC), absent SeriesTime/SeriesDate and series timezone,
    # FrameReferenceTime 60000 ms, ActualFrameDuration 120000 ms, half-life 6586.2 s -> 7199.936855 s.
    ds_tz = make_pet_header(
        start_time=None,
        start_datetime="20250115100000+0100",
        series_date=None,
        series_time=None,
        acquisition_date=None,
        acquisition_time=None,
        timezone_offset_from_utc=None,
        frame_reference_time=60000.0,
        actual_frame_duration=120000.0,
        half_life_s=6586.2,
    )
    ds_tz.AcquisitionDateTime = "20250115110000+0000"
    meta = compute_suvbw_from_datasets([ds_tz])
    assert meta["elapsed_seconds"] == pytest.approx(7199.936855, abs=1e-3)

    # Consistent explicit DT offset (+0200) and TimezoneOffsetFromUTC (+0200) -> +3600 s elapsed
    ds_consistent_tz = make_pet_header(
        start_time=None,
        start_datetime="20250115120000.00+0200",
        series_date="20250115",
        series_time="130000.00",
        acquisition_date="20250115",
        acquisition_time="130000.00",
        timezone_offset_from_utc="+0200",
        half_life_s=3600.0,
    )
    meta_consistent = compute_suvbw_from_datasets([ds_consistent_tz])
    assert meta_consistent["elapsed_seconds"] == pytest.approx(3600.0, abs=1e-3)
    assert meta_consistent["decay_factor"] == pytest.approx(0.5, rel=1e-6)

    # Unresolved partial timezone offset (DT has +0200, SeriesTime is naive without TimezoneOffsetFromUTC) -> rejected
    ds_unresolved = make_pet_header(
        start_time=None,
        start_datetime="20250115120000.00+0200",
        series_date="20250115",
        series_time="110000.00",
        acquisition_date="20250115",
        acquisition_time="110000.00",
        timezone_offset_from_utc=None,
    )
    with pytest.raises(ValueError, match="Unresolved.*timezone offset"):
        compute_suvbw_from_datasets([ds_unresolved])

    # Round-3 Reviewer Finding 3: Study/Series/Acquisition dates 20250115, injection TM 100000,
    # SeriesTime and AcquisitionTime 110000, AcquisitionDateTime 20250115110000+0100,
    # no TimezoneOffsetFromUTC, no RadiopharmaceuticalStartDateTime, and no frame/vendor timing tags -> rejected!
    ds_series_vs_acq_tz = make_pet_header(
        start_time="100000",
        start_datetime=None,
        study_date="20250115",
        series_date="20250115",
        series_time="110000",
        acquisition_date="20250115",
        acquisition_time="110000",
        timezone_offset_from_utc=None,
    )
    ds_series_vs_acq_tz.AcquisitionDateTime = "20250115110000+0100"
    with pytest.raises(ValueError, match="Unresolved.*timezone offset"):
        compute_suvbw_from_datasets([ds_series_vs_acq_tz])

    # Conflicting TimezoneOffsetFromUTC (+0000) vs StartDateTime offset (+0200) -> rejected
    ds_bad_tz = make_pet_header(
        start_time=None,
        start_datetime="20250115120000.00+0200",
        timezone_offset_from_utc="+0000",
    )
    with pytest.raises(ValueError, match="Contradictory timezone between RadiopharmaceuticalStartDateTime"):
        compute_suvbw_from_datasets([ds_bad_tz])


def test_suvbw_converter_rwvm_linear_explanation_and_calibration_bounds(tmp_path):
    # Reviewer probe (Finding 4): Valid linear RWVM with descriptive LUTExplanation and LUTLabel is accepted
    ds_valid_linear = make_pet_header(
        units="BQML",
        rescale_slope=2.5,
        rescale_intercept=0.0,
        rwvm_slope=2.5,
        rwvm_intercept=0.0,
        rwvm_ucum="Bq/ml",
        rwvm_lut_explanation="Linear activity concentration mapping",
        rwvm_lut_label="BQML",
        rwvm_first_mapped=0,
        rwvm_last_mapped=65535,
        pixel_values=[0, 100, 1000],
    )
    meta = compute_suvbw_from_datasets([ds_valid_linear])
    assert meta["conversion_mode"] == "bqml_to_suvbw"
    assert RWVM_SLOPE_RTOL == 1e-5
    assert RWVM_MAX_SUV_ERROR_TOL == 1e-4

    # Round-3 Reviewer Finding 2 Case 1: Explicit GML/BW dataset with integer stored pixels [0, 1000]
    # and NO RealWorldValueFirstValueMapped / RealWorldValueLastValueMapped -> rejected
    ds_no_mapped_range = make_pet_header(
        units="GML",
        suv_type="BW",
        rescale_slope=0.01,
        rescale_intercept=0.0,
        rwvm_slope=0.01,
        rwvm_intercept=0.0,
        rwvm_ucum="g/ml{SUVbw}",
        rwvm_omit_mapped_range=True,
        pixel_values=[0, 1000],
        patient_weight=None,
        dose_bq=None,
        half_life_s=None,
    )
    with pytest.raises(
        ValueError,
        match="Missing required RealWorldValueFirstValueMapped.*or RealWorldValueLastValueMapped",
    ):
        compute_suvbw_from_datasets([ds_no_mapped_range])

    # Round-3 Reviewer Finding 2 Case 2: Same dataset with RealWorldValueFirstValueMapped=0 and
    # RealWorldValueLastValueMapped=10 while actual stored pixels are [0, 1000] -> rejected
    ds_out_of_mapped_range = make_pet_header(
        units="GML",
        suv_type="BW",
        rescale_slope=0.01,
        rescale_intercept=0.0,
        rwvm_slope=0.01,
        rwvm_intercept=0.0,
        rwvm_ucum="g/ml{SUVbw}",
        rwvm_first_mapped=0,
        rwvm_last_mapped=10,
        pixel_values=[0, 1000],
        patient_weight=None,
        dose_bq=None,
        half_life_s=None,
    )
    with pytest.raises(
        ValueError,
        match=r"Decoded stored pixel values \[0, 1000\].*lie outside.*mapped range \[0, 10\]",
    ):
        compute_suvbw_from_datasets([ds_out_of_mapped_range])

    # Round-3 Reviewer Finding 2 Case 3: RescaleSlope=0.01, RealWorldValueSlope=0.01000009 (rel_err=9e-6 <= 1e-5),
    # stored uint16 pixel 4000 (40.0 SUV -> 0.00036 SUV error > 0.0001 SUV bound):
    # (a) with mapped range [0, 100] -> rejected because pixel 4000 is outside [0, 100]
    ds_calib_narrow_map = make_pet_header(
        units="GML",
        suv_type="BW",
        rescale_slope=0.01,
        rescale_intercept=0.0,
        rwvm_slope=0.01000009,
        rwvm_intercept=0.0,
        rwvm_ucum="g/ml{SUVbw}",
        rwvm_first_mapped=0,
        rwvm_last_mapped=100,
        pixel_values=[0, 4000],
        patient_weight=None,
        dose_bq=None,
        half_life_s=None,
    )
    with pytest.raises(
        ValueError,
        match=r"Decoded stored pixel values \[0, 4000\].*lie outside.*mapped range \[0, 100\]",
    ):
        compute_suvbw_from_datasets([ds_calib_narrow_map])

    # (b) with LargestImagePixelValue=100 and mapped range [0, 4000] -> rejected because pixel 4000 > LargestImagePixelValue=100
    ds_calib_bad_largest = make_pet_header(
        units="GML",
        suv_type="BW",
        rescale_slope=0.01,
        rescale_intercept=0.0,
        rwvm_slope=0.01000009,
        rwvm_intercept=0.0,
        rwvm_ucum="g/ml{SUVbw}",
        rwvm_first_mapped=0,
        rwvm_last_mapped=4000,
        smallest_pixel_value=0,
        largest_pixel_value=100,
        pixel_values=[0, 4000],
        patient_weight=None,
        dose_bq=None,
        half_life_s=None,
    )
    with pytest.raises(ValueError, match=r"Contradictory LargestImagePixelValue \(100\)"):
        compute_suvbw_from_datasets([ds_calib_bad_largest])

    # (c) with mapped range [0, 4000] covering pixel 4000 -> rejected because max_suv_err = 0.00036 SUV > 0.0001 SUV
    ds_calib_err = make_pet_header(
        units="GML",
        suv_type="BW",
        rescale_slope=0.01,
        rescale_intercept=0.0,
        rwvm_slope=0.01000009,
        rwvm_intercept=0.0,
        rwvm_ucum="g/ml{SUVbw}",
        rwvm_first_mapped=0,
        rwvm_last_mapped=4000,
        pixel_values=[0, 4000],
        patient_weight=None,
        dose_bq=None,
        half_life_s=None,
    )
    with pytest.raises(ValueError, match="max calibration discrepancy 0.00036"):
        compute_suvbw_from_datasets([ds_calib_err])

    # (d) Full conversion path on disk (convert_dicom_series_to_suvbw_nifti) decodes stored pixels
    # in a single pass and rejects both out-of-range pixels and calibration error > 0.0001 SUV.
    series_dir = tmp_path / "pet_rwvm_disk"
    vol = np.zeros((2, 4, 4), dtype=np.float32)
    vol[1, 2, 2] = 40.0  # stored pixel = 4000 at slope 0.01
    write_synthetic_dicom_series(
        series_dir,
        vol,
        modality="PT",
        study_uid=generate_uid(),
        frame_uid=generate_uid(),
        pet_kwargs={
            "units": "GML",
            "rescale_slope": 0.01,
            "rescale_intercept": 0.0,
        },
    )
    for dcm_file in sorted(series_dir.glob("*.dcm")):
        ds_disk = pydicom.dcmread(dcm_file)
        ds_disk.SUVType = "BW"
        rw = Dataset()
        rw.RealWorldValueSlope = 0.01000009
        rw.RealWorldValueIntercept = 0.0
        rw.RealWorldValueFirstValueMapped = 0
        rw.RealWorldValueLastValueMapped = 100
        unit_item = Dataset()
        unit_item.CodeValue = "g/ml{SUVbw}"
        unit_item.CodingSchemeDesignator = "UCUM"
        unit_item.CodeMeaning = "g/ml{SUVbw}"
        rw.MeasurementUnitsCodeSequence = DicomSequence([unit_item])
        ds_disk.RealWorldValueMappingSequence = DicomSequence([rw])
        ds_disk.save_as(dcm_file)

    with pytest.raises(
        ValueError,
        match=r"Decoded stored pixel values \[0, 4000\].*lie outside.*mapped range \[0, 100\]",
    ):
        convert_dicom_series_to_suvbw_nifti(series_dir, tmp_path / "out_suv.nii.gz")

    for dcm_file in sorted(series_dir.glob("*.dcm")):
        ds_disk = pydicom.dcmread(dcm_file)
        ds_disk.RealWorldValueMappingSequence[0].RealWorldValueLastValueMapped = 4000
        ds_disk.save_as(dcm_file)

    with pytest.raises(ValueError, match="max calibration discrepancy 0.00036"):
        convert_dicom_series_to_suvbw_nifti(series_dir, tmp_path / "out_suv2.nii.gz")

    # Reviewer probe (Finding 4): Ninefold slope discrepancy (1e-6 vs 9e-6) must be rejected
    ds_ninefold = make_pet_header(
        units="GML",
        suv_type="BW",
        rescale_slope=1e-6,
        rescale_intercept=0.0,
        rwvm_slope=9e-6,
        rwvm_intercept=0.0,
        rwvm_ucum="g/ml{SUVbw}",
        patient_weight=None,
        dose_bq=None,
        half_life_s=None,
    )
    with pytest.raises(ValueError, match="Contradictory.*RealWorldValueMappingSequence"):
        compute_suvbw_from_datasets([ds_ninefold])

    # Reviewer probe (Finding 4): Missing MeasurementUnitsCodeSequence on RWVM item must be rejected
    ds_missing_units = make_pet_header(
        units="BQML",
        rescale_slope=1.0,
        rwvm_slope=1.0,
        rwvm_omit_units=True,
    )
    with pytest.raises(ValueError, match="Missing required MeasurementUnitsCodeSequence"):
        compute_suvbw_from_datasets([ds_missing_units])

    # Mapped range check: FirstValueMapped > LastValueMapped rejected
    ds_bad_range = make_pet_header(
        units="BQML",
        rescale_slope=1.0,
        rwvm_slope=1.0,
        rwvm_ucum="Bq/ml",
        rwvm_first_mapped=1000,
        rwvm_last_mapped=100,
    )
    with pytest.raises(ValueError, match="RealWorldValueFirstValueMapped"):
        compute_suvbw_from_datasets([ds_bad_range])


def test_suvbw_converter_verified_midnight_crossing_with_explicit_dates():
    ds = make_pet_header(
        start_time="233000.00",
        start_datetime="20250115233000.00",
        series_date="20250116",
        series_time="003000.00",
        acquisition_date="20250116",
        acquisition_time="003000.00",
        study_date="20250115",
        half_life_s=3600.0,
        dose_bq=100_000_000.0,
        patient_weight=50.0,
    )
    meta = compute_suvbw_from_datasets([ds])
    assert meta["elapsed_seconds"] == pytest.approx(3600.0, abs=1e-3)
    assert meta["decay_factor"] == pytest.approx(0.5, rel=1e-6)
    assert meta["suv_scale_factor"] == pytest.approx(50_000.0 / 50_000_000.0, rel=1e-6)


@pytest.mark.parametrize(
    "units,suv_type",
    [
        ("GML", "BW"),
        ("GML", None),
        ("SUVBW", None),
        ("SUV", "BW"),
    ],
)
def test_suvbw_converter_already_suvbw_never_converts_twice(units, suv_type):
    ds = make_pet_header(
        units=units,
        suv_type=suv_type,
        patient_weight=None,
        dose_bq=None,
        half_life_s=None,
    )
    meta = compute_suvbw_from_datasets([ds])
    assert meta["conversion_mode"] == "already_suvbw"
    assert meta["suv_scale_factor"] == 1.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"units": ""}, "Missing required PET Units"),
        ({"units": "CNTS"}, "Unsupported PET Units"),
        ({"units": "PROPCNTS"}, "Unsupported PET Units"),
        ({"units": "SUV", "suv_type": None}, "Unsupported PET Units"),
        ({"units": "GML", "suv_type": "BSA"}, "non-body-weight SUVType='BSA'"),
        ({"units": "GML", "suv_type": "LBM"}, "non-body-weight SUVType='LBM'"),
        ({"units": "GML", "suv_type": "IBW"}, "non-body-weight SUVType='IBW'"),
        (
            {"units": "GML", "rwvm_ucum": "g/ml{SUVlbm}", "rwvm_slope": 1.0},
            "RealWorldValueMapping.*SUVlbm",
        ),
        (
            {"units": "BQML", "rwvm_ucum": "1", "rwvm_scheme": "99VENDOR", "rwvm_slope": 1.0},
            "CodingSchemeDesignator",
        ),
        (
            {"units": "GML", "rwvm_ucum": "g/ml{SUVbw}", "rwvm_lut": [1.0, 2.0, 3.0], "rwvm_slope": None},
            "non-linear LUT.*RealWorldValueMappingSequence",
        ),
        (
            {"units": "GML", "rwvm_ucum": "g/ml{SUVbw}", "rwvm_slope": 0.001, "rescale_slope": 1.0},
            "Contradictory.*RealWorldValueMappingSequence",
        ),
        ({"rescale_slope": None}, "Missing required.*RescaleSlope"),
        ({"rescale_intercept": None}, "Missing required.*RescaleIntercept"),
        ({"units": "BQML", "rwvm_ucum": "g/ml{SUVbw}", "rwvm_slope": 1.0}, "Contradictory PET Units='BQML'"),
        ({"rescale_slope": 0.0}, "RescaleSlope.*must be > 0"),
        ({"rescale_slope": -1.5}, "RescaleSlope.*must be > 0"),
        ({"rescale_slope": 2.0, "rwvm_slope": 3.0}, "Contradictory.*RealWorldValueMappingSequence"),
        ({"decay_correction": "NONE"}, "Unsupported or missing DecayCorrection"),
        ({"decay_correction": "ADMIN"}, "Unsupported or missing DecayCorrection"),
        ({"corrected_image": ["DECY"]}, "must include both 'DECY' and 'ATTN'"),
        ({"patient_weight": None}, "PatientWeight"),
        ({"patient_weight": 0.0}, "PatientWeight.*must be in"),
        ({"patient_weight": 650.0}, "PatientWeight.*must be in"),
        ({"dose_bq": None}, "RadionuclideTotalDose"),
        ({"dose_bq": -100.0}, "RadionuclideTotalDose.*must be > 0"),
        ({"dose_bq": 250.0}, "below.*minimum plausible Bq"),
        ({"half_life_s": 0.0}, "RadionuclideHalfLife.*must be > 0"),
        ({"start_time": None, "start_datetime": None}, "Missing both RadiopharmaceuticalStartDateTime"),
        (
            {"start_time": "100000.00", "start_datetime": "20250115103000.00"},
            "Contradictory RadiopharmaceuticalStartDateTime",
        ),
        (
            {"series_time": "120000.00", "acquisition_time": "115000.00"},
            "later than earliest AcquisitionTime",
        ),
        (
            {
                "start_time": "233000.00",
                "start_datetime": None,
                "series_time": "003000.00",
                "acquisition_time": "003000.00",
            },
            "Ambiguous midnight crossing or negative decay elapsed time",
        ),
        (
            {
                "start_time": "100000.00",
                "start_datetime": None,
                "study_date": "20250114",
                "series_date": "20250115",
            },
            "Ambiguous date crossing",
        ),
        (
            {
                "start_time": "110000.00",
                "start_datetime": "20250115110000.00",
                "series_date": "20250115",
                "series_time": "103000.00",
                "acquisition_time": "103000.00",
            },
            "Negative decay elapsed time",
        ),
    ],
)
def test_suvbw_converter_rejects_invalid_or_ambiguous_headers(kwargs, match):
    ds = make_pet_header(**kwargs)
    with pytest.raises(ValueError, match=match):
        compute_suvbw_from_datasets([ds])


def test_suvbw_converter_rejects_slice2_units_or_varying_acq_time_without_frame_ref():
    ds1 = make_pet_header(units="BQML")
    ds2 = make_pet_header(units="CNTS")
    with pytest.raises(ValueError, match="CNTS"):
        compute_suvbw_from_datasets([ds1, ds2])

    slice_a = make_pet_header(series_time=None, acquisition_time="114946.20")
    slice_b = make_pet_header(series_time=None, acquisition_time="115246.20")
    with pytest.raises(ValueError, match="Missing SeriesTime.*without QIBA"):
        compute_suvbw_from_datasets([slice_a, slice_b])


def test_suvbw_converter_rejects_contradictory_slices():
    ds1 = make_pet_header(patient_weight=75.0)
    ds2 = make_pet_header(patient_weight=82.0)
    with pytest.raises(ValueError, match="Contradictory PatientWeight"):
        compute_suvbw_from_datasets([ds1, ds2])


def test_suv_threshold_resolution_and_bounds():
    assert gtrc_models.GTRC_TASKS["gtrc_fdg_petct_ttb"]["default_suv_threshold"] is None
    assert gtrc_models.GTRC_TASKS["gtrc_fdg_petct_ttb"]["suv_threshold_required"] is True
    with pytest.raises(ValueError, match="requires a patient-specific FDG SUV threshold"):
        gtrc_models.resolve_job_suv_threshold("gtrc_fdg_petct_ttb", {})

    val, src = gtrc_models.resolve_job_suv_threshold(
        "gtrc_fdg_petct_ttb",
        {"task_options": {"gtrc_fdg_petct_ttb": {"suv_threshold": 2.9650267106980666}}},
    )
    assert val == pytest.approx(2.9650267106980666, rel=0, abs=1e-15)
    assert src == "user-entered"

    for bad in (0.0, -1.0, 0.05, 100.1, float("nan"), float("inf"), "abc", True):
        with pytest.raises(ValueError):
            gtrc_models.validate_suv_threshold("gtrc_fdg_petct_ttb", bad)

    # PSMA defaults to 3.0 ("published-default") and allows user override ("user-entered")
    psma_val, psma_src = gtrc_models.resolve_job_suv_threshold("gtrc_psma_petct_ttb", {})
    assert psma_val == 3.0
    assert psma_src == "published-default"

    psma_def, psma_def_src = gtrc_models.resolve_job_suv_threshold(
        "gtrc_psma_petct_ttb",
        {"task_options": {"gtrc_psma_petct_ttb": {"suv_threshold": 3.0, "source": "published-default"}}},
    )
    assert psma_def == 3.0
    assert psma_def_src == "published-default"

    # Finding 9: published-default is rejected for non-3.0 PSMA or any FDG threshold
    with pytest.raises(ValueError, match="published-default.*only valid for gtrc_psma_petct_ttb at 3.0"):
        gtrc_models.resolve_job_suv_threshold(
            "gtrc_psma_petct_ttb",
            {"task_options": {"gtrc_psma_petct_ttb": {"suv_threshold": 2.5, "source": "published-default"}}},
        )
    with pytest.raises(ValueError, match="published-default.*only valid for gtrc_psma_petct_ttb at 3.0"):
        gtrc_models.resolve_job_suv_threshold(
            "gtrc_fdg_petct_ttb",
            {"task_options": {"gtrc_fdg_petct_ttb": {"suv_threshold": 3.0, "source": "published-default"}}},
        )

    psma_over, psma_over_src = gtrc_models.resolve_job_suv_threshold(
        "gtrc_psma_petct_ttb",
        {"task_options": {"gtrc_psma_petct_ttb": {"suv_threshold": 2.999862, "source": "user-entered"}}},
    )
    assert psma_over == pytest.approx(2.999862, rel=0, abs=1e-12)
    assert psma_over_src == "user-entered"


@requires_run_gtrc
def test_preprocess_pet_ct_threshold_division_and_resampling():
    pet_arr = np.full((6, 8, 8), 6.0, dtype=np.float32)
    pet_im = sitk.GetImageFromArray(pet_arr)
    pet_im.SetSpacing((3.0, 3.0, 3.0))
    pet_im.SetOrigin((-12.0, -12.0, -9.0))

    ct_arr = np.full((6, 16, 16), 50.0, dtype=np.int16)
    ct_im = sitk.GetImageFromArray(ct_arr)
    ct_im.SetSpacing((1.5, 1.5, 3.0))
    ct_im.SetOrigin((-12.0, -12.0, -9.0))

    pt_rs_psma, ct_rs = run_gtrc.preprocess_pet_ct(pet_im, ct_im, 3.0)
    assert pt_rs_psma.GetSize() == pet_im.GetSize()
    assert ct_rs.GetSize() == pet_im.GetSize()
    assert ct_rs.GetSpacing() == pet_im.GetSpacing()
    assert np.allclose(sitk.GetArrayFromImage(pt_rs_psma), 2.0)
    assert np.allclose(sitk.GetArrayFromImage(ct_rs), 50.0)

    fdg_thresh = 2.9650267106980666
    pt_rs_fdg, _ = run_gtrc.preprocess_pet_ct(pet_im, ct_im, fdg_thresh)
    assert np.allclose(sitk.GetArrayFromImage(pt_rs_fdg), 6.0 / fdg_thresh)


@requires_run_gtrc
def test_ct_axial_coverage_and_linear_support_contract(tmp_path):
    # CT must cover PET axial extent within max(5.0, pet_spacing_z, ct_spacing_z) mm
    # AND every PET voxel >= threshold must lie inside SimpleITK's sitkLinear CT support
    # (verified directly via the resampled all-ones CT image).
    pet_arr = np.zeros((10, 8, 8), dtype=np.float32)
    pet_arr[4, 4, 4] = 5.0
    pet_im = sitk.GetImageFromArray(pet_arr)
    pet_im.SetSpacing((2.0, 2.0, 3.0))
    pet_im.SetOrigin((0.0, 0.0, 0.0))  # z from 0.0 to 27.0 mm
    pet_path = tmp_path / "pet.nii.gz"
    sitk.WriteImage(pet_im, str(pet_path))

    # Case 1: CT covers z=[6.0, 27.0] -> missing 6.0 mm at inferior end (> 5.0 mm tolerance)
    ct_short = sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.int16))
    ct_short.SetSpacing((2.0, 2.0, 3.0))
    ct_short.SetOrigin((0.0, 0.0, 6.0))
    ct_short_path = tmp_path / "ct_short.nii.gz"
    sitk.WriteImage(ct_short, str(ct_short_path))
    with pytest.raises(ValueError, match="does not cover PET axial extent"):
        gtrc_models.check_physical_overlap(pet_path, ct_short_path, suv_threshold=3.0)

    # Case 2: Above-threshold PET voxel at z=27.0 mm while CT last slice center is at z=25.0 mm
    # (within the 5.0 mm axial coverage tolerance, but continuous index 9.67 > 9.5 is outside
    # SimpleITK sitkLinear support so sitkLinear assigns the -1000 HU default fill value).
    pet_top = np.zeros((10, 8, 8), dtype=np.float32)
    pet_top[9, 4, 4] = 5.0  # z = 27.0 mm
    pet_top_im = sitk.GetImageFromArray(pet_top)
    pet_top_im.SetSpacing((2.0, 2.0, 3.0))
    pet_top_im.SetOrigin((0.0, 0.0, 0.0))
    pet_top_path = tmp_path / "pet_top.nii.gz"
    sitk.WriteImage(pet_top_im, str(pet_top_path))

    ct_half_voxel_short = sitk.GetImageFromArray(np.zeros((10, 8, 8), dtype=np.int16))
    ct_half_voxel_short.SetSpacing((2.0, 2.0, 3.0))
    ct_half_voxel_short.SetOrigin((0.0, 0.0, -2.0))  # z from -2.0 to 25.0 mm -> index 9.67 > 9.5
    ct_half_path = tmp_path / "ct_half_short.nii.gz"
    sitk.WriteImage(ct_half_voxel_short, str(ct_half_path))
    with pytest.raises(ValueError, match="1 of 1 above-threshold PET voxels.*outside the physical CT support"):
        gtrc_models.check_physical_overlap(pet_top_path, ct_half_path, suv_threshold=3.0)

    # Case 3 (Round-3 Reviewer Finding 6): 10x10x10 1 mm grids, CT origin (0,0,0) with uniform 100 HU,
    # PET all 1000 voxels SUV=4.0 >= 3.0:
    # - Supported fractional-edge shifts (+0.25, +0.25, +0.25) and (-0.25, -0.25, -0.25) lie inside
    #   SimpleITK's sitkLinear support (resampled all-ones == 1.0, preprocess_pet_ct assigns 100 HU to
    #   all 1000 voxels with zero -1000 HU fill voxels) and MUST be accepted.
    # - Unsupported half-voxel shift (0.5, 0.5, 0.5) places 271 voxels at index 9.5 (outside sitkLinear
    #   support, receiving -1000 HU) and MUST be rejected.
    ct_10 = sitk.GetImageFromArray(np.full((10, 10, 10), 100, dtype=np.int16))
    ct_10.SetSpacing((1.0, 1.0, 1.0))
    ct_10.SetOrigin((0.0, 0.0, 0.0))
    ct_10_path = tmp_path / "ct_10.nii.gz"
    sitk.WriteImage(ct_10, str(ct_10_path))

    for idx_shift, origin_shift in enumerate(((0.25, 0.25, 0.25), (-0.25, -0.25, -0.25))):
        pet_frac = sitk.GetImageFromArray(np.full((10, 10, 10), 4.0, dtype=np.float32))
        pet_frac.SetSpacing((1.0, 1.0, 1.0))
        pet_frac.SetOrigin(origin_shift)
        pet_frac_path = tmp_path / f"pet_frac_{idx_shift}.nii.gz"
        sitk.WriteImage(pet_frac, str(pet_frac_path))

        _, ct_rs_frac = run_gtrc.preprocess_pet_ct(pet_frac, ct_10, 3.0)
        ct_rs_arr = sitk.GetArrayFromImage(ct_rs_frac)
        assert int((ct_rs_arr == -1000).sum()) == 0
        assert np.allclose(ct_rs_arr, 100.0)

        overlap_res = gtrc_models.check_physical_overlap(pet_frac_path, ct_10_path, suv_threshold=3.0)
        assert overlap_res["above_threshold_voxel_count"] == 1000
        assert overlap_res["above_threshold_outside_ct_voxels"] == 0
        assert overlap_res["all_above_threshold_inside_ct"] is True

    pet_10 = sitk.GetImageFromArray(np.full((10, 10, 10), 4.0, dtype=np.float32))
    pet_10.SetSpacing((1.0, 1.0, 1.0))
    pet_10.SetOrigin((0.5, 0.5, 0.5))
    pet_10_path = tmp_path / "pet_10.nii.gz"
    sitk.WriteImage(pet_10, str(pet_10_path))
    with pytest.raises(ValueError, match="271 of 1000 above-threshold PET voxels.*outside the physical CT support"):
        gtrc_models.check_physical_overlap(pet_10_path, ct_10_path, suv_threshold=3.0)

    # Case 4 (Finding 8): Oblique CT grid rotated in XY where an above-threshold PET corner voxel lies
    # inside the axis-aligned bounding box but outside the rotated CT linear-interpolation support.
    pet_obl_arr = np.zeros((10, 10, 10), dtype=np.float32)
    pet_obl_arr[5, 0, 9] = 5.0  # corner (x=9, y=0, z=5)
    pet_obl = sitk.GetImageFromArray(pet_obl_arr)
    pet_obl.SetSpacing((1.0, 1.0, 1.0))
    pet_obl.SetOrigin((0.0, 0.0, 0.0))
    pet_obl_path = tmp_path / "pet_obl.nii.gz"
    sitk.WriteImage(pet_obl, str(pet_obl_path))

    theta = np.deg2rad(15.0)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    ct_obl = sitk.GetImageFromArray(np.full((10, 10, 10), 100, dtype=np.int16))
    ct_obl.SetSpacing((1.0, 1.0, 1.0))
    ct_obl.SetOrigin((0.0, 0.0, 0.0))
    ct_obl.SetDirection((c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
    ct_obl_path = tmp_path / "ct_obl.nii.gz"
    sitk.WriteImage(ct_obl, str(ct_obl_path))
    with pytest.raises(ValueError, match="1 of 1 above-threshold PET voxels.*outside the physical CT support"):
        gtrc_models.check_physical_overlap(pet_obl_path, ct_obl_path, suv_threshold=3.0)


def test_inspect_ct_series_metadata_checks_every_slice(tmp_path):
    ct_dir = tmp_path / "ct_mixed"
    write_synthetic_dicom_series(
        ct_dir,
        np.zeros((4, 8, 8), dtype=np.float32),
        modality="CT",
        study_uid="1.2.3.4",
        frame_uid="1.2.3.5",
    )
    last_dcm = sorted(ct_dir.glob("*.dcm"))[-1]
    ds = pydicom.dcmread(last_dcm)
    ds.FrameOfReferenceUID = "1.2.3.999"
    ds.save_as(last_dcm)
    with pytest.raises(ValueError, match="Mixed FrameOfReferenceUID"):
        gtrc_models.inspect_ct_series_metadata(ct_dir)


@requires_run_gtrc
def test_subregion_and_consensus_equivalence_with_upstream():
    pt_ar = np.full((16, 16, 16), 0.2, dtype=np.float32)
    pt_ar[2:6, 2:6, 2:6] = 1.2
    pt_ar[3:5, 3:5, 3:5] = 2.5

    pt_ar[10:14, 10:14, 10:14] = 1.2
    pt_ar[11:13, 11:13, 11:13] = 1.8

    pt_ar[2:4, 6:8, 2:4] = 0.95
    pt_ar[2:4, 8:10, 2:4] = 1.05

    ct_ar = np.full((16, 16, 16), 40.0, dtype=np.float32)
    pred_ar = np.zeros((16, 16, 16), dtype=np.int8)
    pred_ar[2:6, 2:6, 2:6] = 1
    pred_ar[10:14, 10:14, 10:14] = 2
    pred_ar[10, 10, 10] = 1

    pt_rs = sitk.GetImageFromArray(pt_ar)
    pt_rs.SetSpacing((2.86, 2.86, 3.27))
    ct_rs = sitk.GetImageFromArray(ct_ar)
    ct_rs.CopyInformation(pt_rs)
    pred_im = sitk.GetImageFromArray(pred_ar)
    pred_im.CopyInformation(pt_rs)

    tar = (pt_ar >= 1.0).astype("int8")
    ws_verbatim = run_gtrc.create_subregion_labels_verbatim(pt_rs, tar)

    upstream_source = gtrc_models.MODEL_ROOT / gtrc_models.GTRC_SOURCE["source_dir"]
    if (upstream_source / "gtrc_utils.py").is_file():
        upstream_fn = run_gtrc.load_upstream_subregion_fn(upstream_source)
        ws_upstream = upstream_fn(
            pt_rs,
            tar,
            local_maxima_threshold=run_gtrc.LOCAL_MAXIMA_THRESHOLD,
            sphere_radius=run_gtrc.SPHERE_RADIUS_MM,
        )
        assert np.array_equal(
            sitk.GetArrayFromImage(ws_verbatim),
            sitk.GetArrayFromImage(ws_upstream),
        )

    def make_voter(positive_when_ttb: bool):
        def _model(x: np.ndarray) -> np.ndarray:
            ttb_overlap = float(x[0, 4])
            if positive_when_ttb and ttb_overlap > 0.3:
                return np.array([[0.9]], dtype=np.float32)
            return np.array([[0.1]], dtype=np.float32)

        return _model

    models_3_of_5 = [
        make_voter(True),
        make_voter(True),
        make_voter(True),
        make_voter(False),
        make_voter(False),
    ]
    out_im, sub_im, records, counts = run_gtrc.evaluate_consensus_and_paint(
        pt_rs,
        ct_rs,
        pred_im,
        models_3_of_5,
        suv_threshold=3.0,
    )
    out_ar = sitk.GetArrayFromImage(out_im)
    assert counts["accepted_subregion_count"] == 1
    assert counts["rejected_subregion_count"] == 1
    assert counts["inpainted_ttb_additional_voxels"] == 1
    assert np.all(out_ar[2:6, 2:6, 2:6] == 1)
    assert out_ar[10, 10, 10] == 1
    assert int(out_ar[10:14, 10:14, 10:14].sum()) == 1
    assert any(r["votes_positive"] == 3 and r["accepted"] is True for r in records)
    assert any(r["votes_positive"] == 0 and r["accepted"] is False for r in records)


@requires_run_gtrc
def test_production_entrypoints_reject_test_flags_and_env_overrides():
    # Finding 1: --precomputed-nnunet-pred, --nnunet-predict-exe, GTRC_NNUNET_PREDICT_EXE, and
    # GTRC_ALLOW_TEST_STUB are removed from production modules.
    src_run = (ROOT / "run_gtrc.py").read_text()
    src_mod = (ROOT / "gtrc_models.py").read_text()
    for forbidden in (
        "--precomputed-nnunet-pred",
        "--nnunet-predict-exe",
        "GTRC_NNUNET_PREDICT_EXE",
        "GTRC_ALLOW_TEST_STUB",
    ):
        assert forbidden not in src_run
        assert forbidden not in src_mod

    with pytest.raises(SystemExit):
        run_gtrc.main(["--precomputed-nnunet-pred", "/tmp/x.nii.gz"])
    with pytest.raises(SystemExit):
        run_gtrc.main(["--nnunet-predict-exe", "/tmp/exe"])


@requires_run_gtrc
def test_end_to_end_synthetic_job_positive_and_empty_and_execution_gates(tmp_path, monkeypatch):
    study_uid = generate_uid()
    frame_uid = generate_uid()
    pet_vol = np.zeros((10, 24, 24), dtype=np.float32)
    pet_vol[3:7, 8:16, 8:16] = 12.0
    ct_vol = np.full((10, 24, 24), 35.0, dtype=np.float32)

    job = tmp_path / "job_pos"
    write_synthetic_dicom_series(
        job / "input",
        pet_vol,
        modality="PT",
        study_uid=study_uid,
        frame_uid=frame_uid,
        pet_kwargs={"patient_weight": 100.0, "dose_bq": 100_000.0, "rescale_slope": 1.0},
    )
    write_synthetic_dicom_series(
        job / "input_ct",
        ct_vol,
        modality="CT",
        study_uid=study_uid,
        frame_uid=frame_uid,
    )

    fake_model_root = tmp_path / "model_root"
    fake_bin = fake_model_root / "runtime-envs/gtrc/bin"
    fake_bin.mkdir(parents=True)
    py_wrapper = fake_bin / "python"
    py_wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py_wrapper.chmod(0o755)

    fake_app_root = tmp_path / "app_root"
    fake_app_root.mkdir(parents=True)
    for name in (
        "dicom_series_to_nifti.py",
        "dicom_series.py",
        "orientation_contract.py",
        "pet_to_suvbw.py",
        "convert_validate_rtstruct.py",
        "rtstruct_geometry.py",
        "rtstruct_roundtrip.py",
        "task_progress.py",
    ):
        (fake_app_root / name).symlink_to(ROOT / name)
    (fake_app_root / "labels").symlink_to(ROOT / "labels")

    stub_runner = fake_app_root / "run_gtrc.py"
    stub_runner.write_text(
        """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--pt-suv", required=True, type=Path)
p.add_argument("--ct", required=True, type=Path)
p.add_argument("--suv-threshold", required=True, type=float)
p.add_argument("--suv-threshold-source", default="published-default")
p.add_argument("--work-dir", required=True, type=Path)
p.add_argument("--output-mask", required=True, type=Path)
p.add_argument("--output-nnunet-labels", required=True, type=Path)
p.add_argument("--output-subregions", required=True, type=Path)
p.add_argument("--output-lesions-json", required=True, type=Path)
p.add_argument("--output-provenance-json", required=True, type=Path)
p.add_argument("--job-context-json", required=True, type=Path)
p.add_argument("--gtrc-acceptance-bootstrap", action="store_true")
a = p.parse_args()

pt = sitk.ReadImage(str(a.pt_suv))
arr = sitk.GetArrayFromImage(pt)
mask = (arr >= a.suv_threshold).astype(np.int8)
im = sitk.GetImageFromArray(mask)
im.CopyInformation(pt)
for out in (a.output_mask, a.output_nnunet_labels, a.output_subregions):
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(im, str(out), True)
vox = int(mask.sum())
a.output_lesions_json.write_text(json.dumps({
    "task": a.task,
    "roi_name": "GTRC_PSMA_TTB" if "psma" in a.task else "GTRC_FDG_TTB",
    "empty_prediction": vox == 0,
    "ttb_voxel_count": vox,
    "ttb_volume_ml": float(vox * 0.01875),
    "suv_threshold": a.suv_threshold,
    "suv_threshold_source": a.suv_threshold_source,
}) + "\\n")
a.output_provenance_json.write_text(json.dumps({"task": a.task}) + "\\n")
"""
    )

    # Without valid bundle/runtime/markers, run_task fails closed in both normal and bootstrap modes!
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        gtrc_models.run_task(
            "gtrc_psma_petct_ttb",
            job,
            app_root=fake_app_root,
            model_root=fake_model_root,
        )
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        gtrc_models.run_task(
            "gtrc_psma_petct_ttb",
            job,
            app_root=fake_app_root,
            model_root=fake_model_root,
            _acceptance_bootstrap=True,
        )

    # Monkeypatch internal gate functions in pytest to test the orchestration pipeline
    host_py_wrapper = tmp_path / "host_python"
    py_path_str = ":".join(p for p in sys.path if p)
    host_py_wrapper.write_text(
        f'#!/bin/sh\nPYTHONPATH="{py_path_str}" exec "{sys.executable}" "$@"\n'
    )
    host_py_wrapper.chmod(0o755)
    host_python = str(host_py_wrapper)
    monkeypatch.setattr(
        gtrc_models,
        "installed_source_commit",
        lambda *a, **kw: gtrc_models.GTRC_SOURCE["source_commit"],
    )
    monkeypatch.setattr(
        gtrc_models,
        "verify_weight_inventory",
        lambda *a, **kw: (True, "ok", []),
    )
    monkeypatch.setattr(
        gtrc_models,
        "expected_runtime_lock_sha256",
        lambda *a, **kw: "a" * 64,
    )
    monkeypatch.setattr(
        gtrc_models,
        "verify_runtime_cuda_compatibility",
        lambda *a, **kw: (
            True,
            "ok",
            {"device_name": "NVIDIA H100L-47C", "cuda_device_capability": [9, 0]},
        ),
    )
    monkeypatch.setattr(gtrc_models, "validate_acceptance", lambda *a, **kw: True)
    monkeypatch.setattr(gtrc_models, "validate_orientation", lambda *a, **kw: True)

    report = gtrc_models.run_task(
        "gtrc_psma_petct_ttb",
        job,
        app_root=fake_app_root,
        model_root=fake_model_root,
        python_exe=host_python,
    )
    assert report["empty_prediction"] is False
    assert report["ttb_voxel_count"] > 0
    rtstruct_path = job / "output/gtrc_psma_petct_ttb_RTSTRUCT.dcm"
    mask_path = job / "output/gtrc_psma_petct_ttb_RTSTRUCT.dcm.ttb_mask.nii.gz"
    lesions_path = job / "output/gtrc_psma_petct_ttb_RTSTRUCT.dcm.lesions.json"
    assert rtstruct_path.is_file()
    assert mask_path.is_file()
    assert lesions_path.is_file()
    assert not (job / "output/gtrc_psma_petct_ttb_pet_suvbw.json").exists()

    ds = pydicom.dcmread(rtstruct_path)
    roi_names = [str(r.ROIName) for r in ds.StructureSetROISequence]
    assert roi_names == ["GTRC_PSMA_TTB"]

    # Empty prediction test for FDG:
    job_empty = tmp_path / "job_empty"
    write_synthetic_dicom_series(
        job_empty / "input",
        np.full((10, 24, 24), 0.5, dtype=np.float32),
        modality="PT",
        study_uid=study_uid,
        frame_uid=frame_uid,
        pet_kwargs={"patient_weight": 100.0, "dose_bq": 100_000.0, "rescale_slope": 1.0},
    )
    write_synthetic_dicom_series(
        job_empty / "input_ct",
        ct_vol,
        modality="CT",
        study_uid=study_uid,
        frame_uid=frame_uid,
    )
    with pytest.raises(ValueError, match="requires a patient-specific FDG SUV threshold"):
        gtrc_models.run_task(
            "gtrc_fdg_petct_ttb",
            job_empty,
            app_root=fake_app_root,
            model_root=fake_model_root,
        )

    (job_empty / "job.json").write_text(
        json.dumps(
            {
                "task": "gtrc_fdg_petct_ttb",
                "task_options": {
                    "gtrc_fdg_petct_ttb": {"suv_threshold": 2.97, "source": "user-entered"}
                },
            }
        )
        + "\n"
    )
    with pytest.raises(SystemExit, match="empty prediction: no RTSTRUCT created"):
        gtrc_models.run_task(
            "gtrc_fdg_petct_ttb",
            job_empty,
            app_root=fake_app_root,
            model_root=fake_model_root,
            python_exe=host_python,
        )
    assert not (job_empty / "output/gtrc_fdg_petct_ttb_RTSTRUCT.dcm").exists()
    lesions_sidecar = job_empty / "output/gtrc_fdg_petct_ttb_RTSTRUCT.dcm.lesions.json"
    mask_sidecar = job_empty / "output/gtrc_fdg_petct_ttb_RTSTRUCT.dcm.ttb_mask.nii.gz"
    assert lesions_sidecar.is_file()
    assert mask_sidecar.is_file()
    empty_data = json.loads(lesions_sidecar.read_text())
    assert empty_data["empty_prediction"] is True
    assert empty_data["ttb_voxel_count"] == 0
    assert empty_data["suv_threshold"] == pytest.approx(2.97)
    assert empty_data["suv_threshold_source"] == "user-entered"


def test_catalogue_and_structure_catalog_contracts():
    gtrc_cat = json.loads((ROOT / "catalog/gtrc.json").read_text())
    structures_cat = json.loads((ROOT / "catalog/structures.json").read_text())

    for task, expected_roi in (
        ("gtrc_psma_petct_ttb", "GTRC_PSMA_TTB"),
        ("gtrc_fdg_petct_ttb", "GTRC_FDG_TTB"),
    ):
        entry = gtrc_cat[task]
        assert entry["modality"] == "PT"
        assert entry["requires_ct"] is True
        assert entry["requires_preflight"] is True
        assert entry["enabled"] is True
        assert entry["status"] == "acceptance_pending"
        assert entry["output_artifacts"] == [
            f"{task}_RTSTRUCT.dcm",
            f"{task}_RTSTRUCT.dcm.lesions.json",
            f"{task}_RTSTRUCT.dcm.ttb_mask.nii.gz",
        ]
        assert all(a.startswith(f"{task}_RTSTRUCT.dcm") for a in entry["output_artifacts"])
        assert "No licence stated by the authors" in entry["license"]["code"]
        assert "Owner accepted these terms on 2026-10-22" in entry["license"]["note"]
        assert "10.2967/jnumed.125.270077" in entry["reference"]
        assert "1.5x mean + 2x SD of liver background for [18F]FDG PET imaging" in entry["description"]
        assert "DEEP-PSMA challenge pages describe liver mean + 2 SD" in entry["description"]

        label_map = json.loads((ROOT / entry["label_map"]).read_text())
        assert label_map == {"1": expected_roi}

        struct_entry = structures_cat[task]
        assert struct_entry["kind"] == "fixed"
        assert struct_entry["names"] == [expected_roi]
        assert struct_entry["count"] == 1

    assert gtrc_cat["gtrc_psma_petct_ttb"]["suv_threshold_required"] is False
    assert gtrc_cat["gtrc_psma_petct_ttb"]["default_suv_threshold"] == 3.0
    assert gtrc_cat["gtrc_fdg_petct_ttb"]["suv_threshold_required"] is True
    assert gtrc_cat["gtrc_fdg_petct_ttb"]["default_suv_threshold"] is None


def _build_valid_test_bundle_and_records(
    bundle: Path,
    task: str = "gtrc_psma_petct_ttb",
    device: str = "H100",
    run_id: str = "gtrc-h100-20261022T190000Z-1234",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = gtrc_models.GTRC_TASKS[task]
    source_dir = bundle / spec["source_dir"]
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "RTSEG_MANIFEST.json").write_text(
        json.dumps({"commit": spec["source_commit"]}) + "\n"
    )
    for fname in ("GTRC.py", "GTRC_Infer.py", "gtrc_utils.py", "nnunet_config_paths.py"):
        (source_dir / fname).write_text("# source\n")

    for item in gtrc_models.GTRC_WEIGHT_FILES[task]:
        p = bundle / item["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.truncate(item["bytes"])
        stat = p.stat()
        gtrc_models._SHA256_CACHE[
            (
                str(p.resolve()),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        ] = item["sha256"]

    lock_file = bundle / "runtime-envs/gtrc/requirements.lock.txt"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("torch==2.7.1+cu118\n")

    code_hashes = gtrc_models.expected_code_hashes(ROOT)
    code_sha256 = gtrc_models.expected_code_digest(ROOT)
    lock_sha256 = gtrc_models.expected_runtime_lock_sha256(bundle)
    ckpt_hashes = gtrc_models.expected_checkpoint_hashes(task)
    ckpt_sha256 = gtrc_models.expected_checkpoint_digest(task)

    if device == "H100":
        live_gpu = {
            "cuda_available": True,
            "name": "NVIDIA H100L-47C",
            "device_name": "NVIDIA H100L-47C",
            "compute_capability": [9, 0],
            "cuda_device_capability": [9, 0],
            "cuda": "11.8",
            "cuda_version": "11.8",
            "compiled_architectures": ["sm_60", "sm_70", "sm_80", "sm_90"],
            "cuda_arch_list": ["sm_60", "sm_70", "sm_80", "sm_90"],
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "inference_mode": "production",
            "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
        }
    else:
        live_gpu = {
            "cuda_available": True,
            "name": "Quadro P6000",
            "device_name": "Quadro P6000",
            "compute_capability": [6, 1],
            "cuda_device_capability": [6, 1],
            "cuda": "11.8",
            "cuda_version": "11.8",
            "compiled_architectures": ["sm_60", "sm_70", "sm_80", "sm_90"],
            "cuda_arch_list": ["sm_60", "sm_70", "sm_80", "sm_90"],
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "inference_mode": "production",
            "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
        }

    nn_flags = {
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
        "pid": 4321,
        "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
        "inference_mode": "production",
        "device_name": live_gpu["device_name"],
        "cuda_device_capability": list(live_gpu["cuda_device_capability"]),
    }
    grid_metrics = {
        "passed": True,
        "size_equal": True,
        "cand_size": [64, 64, 32],
        "ref_size": [64, 64, 32],
        "spacing_max_delta_mm": 0.0,
        "origin_max_delta_mm": 0.0,
        "direction_max_delta": 0.0,
    }
    biomarker_rec = {
        "passed": True,
        "ttb_voxel_count": 120,
        "ttb_volume_ml": 3.24,
        "suv_mean": 8.5,
        "suv_max": 22.1,
        "suv_min": 3.1,
        "suv_median": 7.4,
        "suv_std": 2.3,
        "hu_mean": 42.0,
        "lesion_count": 2,
        "max_stat_abs_err": 0.0,
    }
    thr = 2.999862 if "psma" in task else 2.9650267106980666
    roi_name = spec["roi_name"]

    val_record = {
        "task": task,
        "family": "gtrc",
        "device_key": device,
        "run_id": run_id,
        "tracer": spec["tracer"],
        "dataset_id": spec["dataset_id"],
        "result": "passed",
        "status": "validated",
        "passed": True,
        "valid": True,
        "inference_mode": "production",
        "cpu_smoke": False,
        "validated_at_utc": "2026-10-22T19:00:00+00:00",
        "source_commit": spec["source_commit"],
        "runtime_lock_sha256": lock_sha256,
        "runtime_provenance": copy.deepcopy(live_gpu),
        "gpu": copy.deepcopy(live_gpu),
        "suv_threshold": thr,
        "suv_threshold_source": "user-entered",
        "input_binding": {
            "archive_sha256": "a" * 64,
            "fixture_manifest_sha256": "f" * 64,
            "pet_nifti_sha256": "b" * 64,
            "ct_nifti_sha256": "c" * 64,
            "ttb_nifti_sha256": "d" * 64,
            "threshold_json_sha256": "e" * 64,
            "pet_dicom_sha256": "1" * 64,
            "ct_dicom_sha256": "2" * 64,
            "acceptance_harness_sha256": "3" * 64,
            "fixture_builder_sha256": "4" * 64,
            "suv_threshold": thr,
            "suv_threshold_source": "user-entered",
        },
        "numerical_tolerances": dict(gtrc_models.GTRC_TOLERANCE_CONTRACT),
        "acceptance_modes": ["reference", "dicom"],
        "checks": [
            "gpu_inference",
            "suvbw_conversion",
            "exact_input_grid",
            "rtstruct_roundtrip",
            "orientation",
            "both_acceptance_modes",
            "author_reference_exact_match",
        ],
        "reference_mode": {
            "passed": True,
            "valid": True,
            "status": "passed",
            "inference_mode": "production",
            "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
            "nnunet_runtime_flags": copy.deepcopy(nn_flags),
            "upstream_nnunet_runtime_flags": copy.deepcopy(nn_flags),
            "exact_mask_agreement": True,
            "subregion_label_agreement": True,
            "grid_agreement": True,
            "grid_check": copy.deepcopy(grid_metrics),
            "upstream_grid_check": copy.deepcopy(grid_metrics),
            "biomarker_recomputation": copy.deepcopy(biomarker_rec),
            "suv_threshold": thr,
            "suv_threshold_source": "user-entered",
            "dice": 1.0,
            "mismatch_voxels": 0,
            "nnunet_label_mismatch_voxels": 0,
            "subregion_label_mismatch_voxels": 0,
            "ref_voxels": 120,
            "cand_voxels": 120,
            "result_sha256": "a" * 64,
        },
        "dicom_mode": {
            "passed": True,
            "valid": True,
            "status": "passed",
            "dispatch_entrypoint": "run_task.py",
            "inference_mode": "production",
            "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
            "nnunet_runtime_flags": copy.deepcopy(nn_flags),
            "rtstruct_valid": True,
            "exact_reference_agreement": True,
            "reference_mismatch_voxels": 0,
            "nnunet_label_mismatch_voxels": 0,
            "subregion_label_mismatch_voxels": 0,
            "reference_dice": 1.0,
            "grid_agreement": True,
            "grid_check": copy.deepcopy(grid_metrics),
            "biomarker_recomputation": copy.deepcopy(biomarker_rec),
            "suv_threshold": thr,
            "suv_threshold_source": "user-entered",
            "roi_name": roi_name,
            "roi_names": [roi_name],
            "referenced_pet_series_uid": "1.2.3.4",
            "referenced_pet_series_uid_match": True,
            "roundtrip_valid": True,
            "roundtrip_min_dice": 1.0,
            "roundtrip_median_dice": 1.0,
            "roundtrip_nonempty_rois": 1,
            "nonempty_rois": 1,
            "ttb_voxel_count": 120,
            "ttb_volume_ml": 3.24,
            "vs_reference_mode": {
                "exact_match": True,
                "mismatch_voxels": 0,
                "dice": 1.0,
            },
            "pet_dicom_sha256": "1" * 64,
            "ct_dicom_sha256": "2" * 64,
            "fixture_roundtrip_verification": {
                "passed": True,
                "conversion_mode": "bqml_to_suvbw",
                "pet_dicom_sha256": "1" * 64,
                "ct_dicom_sha256": "2" * 64,
                "pet_max_abs_diff_suv": 0.00033,
                "pet_mean_abs_diff_suv": 0.000002,
                "pet_p999_abs_diff_suv": 0.00021,
                "ct_max_abs_diff_hu": 0.0,
                "ct_mean_abs_diff_hu": 0.0,
                "suvbw_scale_factor_rel_error": 0.0,
                "decay_elapsed_seconds_abs_error": 0.0,
                "fixture_threshold": thr,
                "threshold_3_0_exact_match": True,
                "fixture_threshold_exact_match": True,
                "pet_grid": copy.deepcopy(grid_metrics),
                "ct_grid": copy.deepcopy(grid_metrics),
            },
            "rtstruct_sha256": "5" * 64,
            "ttb_mask_sha256": "6" * 64,
            "artifact_sha256": {
                "rtstruct_sha256": "5" * 64,
                "ttb_mask_sha256": "6" * 64,
                "lesions_json_sha256": "7" * 64,
            },
        },
        "suvbw_converter_check": {
            "passed": True,
            "conversion_mode": "bqml_to_suvbw",
            "threshold_3_0_exact_match": True,
            "fixture_threshold_exact_match": True,
            "suvbw_max_abs_err": 0.00033,
            "suvbw_mean_abs_err": 0.000002,
            "suvbw_p999_abs_err": 0.00021,
            "ct_hu_max_abs_err": 0.0,
            "ct_hu_mean_abs_err": 0.0,
            "suvbw_scale_factor_rel_error": 0.0,
            "decay_elapsed_seconds_abs_error": 0.0,
            "pet_grid": copy.deepcopy(grid_metrics),
            "ct_grid": copy.deepcopy(grid_metrics),
        },
        "checkpoint_hashes": ckpt_hashes,
        "checkpoint_sha256": ckpt_sha256,
        "code_hashes": code_hashes,
        "code_sha256": code_sha256,
    }
    val_sha256 = gtrc_models.canonical_json_sha256(val_record)
    ori_record = {
        "task": task,
        "device_key": device,
        "run_id": run_id,
        "result": "passed",
        "status": "validated",
        "passed": True,
        "valid": True,
        "roundtrip_valid": True,
        "inference_mode": "production",
        "cpu_smoke": False,
        "gpu": copy.deepcopy(live_gpu),
        "roi_name": roi_name,
        "roi_names": [roi_name],
        "referenced_pet_series_uid": "1.2.3.4",
        "referenced_pet_series_uid_match": True,
        "nonempty_rois": 1,
        "min_dice": 1.0,
        "median_dice": 1.0,
        "source_commit": spec["source_commit"],
        "checkpoint_sha256": ckpt_sha256,
        "code_sha256": code_sha256,
        "runtime_lock_sha256": lock_sha256,
        "acceptance_sha256": val_sha256,
        "artifact_sha256": {
            "rtstruct_sha256": "5" * 64,
            "ttb_mask_sha256": "6" * 64,
            "lesions_json_sha256": "7" * 64,
        },
        "validated_at_utc": "2026-10-22T19:00:00+00:00",
    }
    return val_record, ori_record, live_gpu


def _seed_bundle_device_records(
    bundle: Path,
    device: str = "H100",
    run_id: str = "run-h100",
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    val_records: dict[str, dict[str, Any]] = {}
    ori_records: dict[str, dict[str, Any]] = {}
    task_records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    live_gpu: dict[str, Any] = {}
    for t in gtrc_models.GTRC_TASKS:
        vr, orr, gpu = _build_valid_test_bundle_and_records(bundle, t, device, run_id)
        val_records[t] = vr
        ori_records[t] = orr
        task_records[t] = (vr, orr)
        live_gpu = gpu
        gtrc_models.store_device_marker(bundle / "validation" / f"{t}.json", device, vr)
        gtrc_models.store_device_marker(bundle / "orientation-validation" / f"{t}.json", device, orr)
    first_val = next(iter(val_records.values()))
    comp = gtrc_models.build_completion_record(
        device=device,
        run_id=run_id,
        gpu=live_gpu,
        source_commit=first_val["source_commit"],
        code_sha256=first_val["code_sha256"],
        runtime_lock_sha256=first_val["runtime_lock_sha256"],
        task_records=task_records,
        completed_at_utc="2026-10-22T19:00:00+00:00",
    )
    gtrc_models.store_completion_marker(bundle / "validation", device, comp)
    return val_records, ori_records, comp, live_gpu


def test_acceptance_and_orientation_schema2_device_envelopes_and_strict_bindings(tmp_path):
    task = "gtrc_psma_petct_ttb"
    bundle = tmp_path / "bundle"
    val_h100_map, ori_h100_map, comp_h100, gpu_h100 = _seed_bundle_device_records(bundle, "H100", "run-h100")
    val_h100 = val_h100_map[task]
    ori_h100 = ori_h100_map[task]

    val_path = bundle / "validation" / f"{task}.json"
    ori_path = bundle / "orientation-validation" / f"{task}.json"
    comp_path = bundle / gtrc_models.GTRC_COMPLETION_REL

    assert gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_h100) is True
    assert gtrc_models.validate_orientation(task, bundle, app_root=ROOT, gpu=gpu_h100) is True

    # Round-3 Finding 1: Missing completion record or missing peer-task record must fail closed!
    saved_comp = comp_path.read_text()
    comp_path.unlink()
    with pytest.raises(ValueError, match="Missing GTRC completion marker"):
        gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_h100)
    with pytest.raises(ValueError, match="Missing GTRC completion marker"):
        gtrc_models.validate_orientation(task, bundle, app_root=ROOT, gpu=gpu_h100)
    comp_path.write_text(saved_comp)

    fdg_val_path = bundle / "validation/gtrc_fdg_petct_ttb.json"
    saved_fdg = fdg_val_path.read_text()
    fdg_val_path.unlink()
    with pytest.raises(ValueError, match="Missing validation marker"):
        gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_h100)
    fdg_val_path.write_text(saved_fdg)

    # Corrupted completion record (mismatched run_id or validation_sha256) fails closed
    bad_comp_run = copy.deepcopy(comp_h100)
    bad_comp_run["run_id"] = "wrong-run-id"
    gtrc_models.store_completion_marker(bundle / "validation", "H100", bad_comp_run)
    with pytest.raises(ValueError, match="Completion marker run_id.*mismatch"):
        gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_h100)
    comp_path.write_text(saved_comp)

    bad_comp_sha = copy.deepcopy(comp_h100)
    bad_comp_sha["tasks"][task]["validation_sha256"] = "9" * 64
    gtrc_models.store_completion_marker(bundle / "validation", "H100", bad_comp_sha)
    with pytest.raises(ValueError, match="validation_sha256 mismatch"):
        gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_h100)
    comp_path.write_text(saved_comp)

    # On P6000, H100-only envelope must fail closed
    _, _, gpu_p6000 = _build_valid_test_bundle_and_records(bundle, task, "P6000", "run-p6000")
    with pytest.raises(ValueError, match="Missing P6000 acceptance marker record"):
        gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_p6000)
    with pytest.raises(ValueError, match="Missing P6000 orientation marker record"):
        gtrc_models.validate_orientation(task, bundle, app_root=ROOT, gpu=gpu_p6000)

    # Publish P6000 records + completion atomically; verify H100 record is preserved
    _seed_bundle_device_records(bundle, "P6000", "run-p6000")
    assert gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_h100) is True
    assert gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_p6000) is True
    assert gtrc_models.validate_orientation(task, bundle, app_root=ROOT, gpu=gpu_h100) is True
    assert gtrc_models.validate_orientation(task, bundle, app_root=ROOT, gpu=gpu_p6000) is True

    # Round-3 Finding 4: Deleting runtime-envs/gtrc/requirements.lock.txt MUST fail even when a
    # bundle-root <bundle>/requirements.lock.txt exists with matching contents!
    lock_file = bundle / "runtime-envs/gtrc/requirements.lock.txt"
    bundle_root_lock = bundle / "requirements.lock.txt"
    saved_lock = lock_file.read_text()
    bundle_root_lock.write_text(saved_lock)
    lock_file.unlink()
    with pytest.raises(ValueError, match="Missing required installed runtime lock file"):
        gtrc_models.expected_runtime_lock_sha256(bundle)
    with pytest.raises(ValueError, match="Missing required installed runtime lock file"):
        gtrc_models.validate_acceptance(task, bundle, app_root=ROOT, gpu=gpu_h100)
    with pytest.raises(ValueError, match="Missing required installed runtime lock file"):
        gtrc_models.validate_orientation(task, bundle, app_root=ROOT, gpu=gpu_h100)
    lock_file.write_text(saved_lock)
    bundle_root_lock.unlink()

    # Finding 6: Orientation validation requires an existing, valid acceptance marker on disk
    saved_val = val_path.read_text()
    val_path.unlink()
    with pytest.raises(ValueError, match="requires existing validation marker"):
        gtrc_models.validate_orientation(task, bundle, app_root=ROOT, gpu=gpu_h100)
    val_path.write_text(saved_val)

    # Test rejection of every invalid marker mutation (including Round-3 Finding 5 omissions)
    for mutate, msg in [
        (lambda m: m.update(waived=True), "waived"),
        (lambda m: m.update(cpu_smoke=True), "smoke/stub"),
        (lambda m: m["runtime_provenance"].update(inference_mode="cpu_smoke_test"), "inference_mode"),
        (lambda m: m["runtime_provenance"].update(cudnn_benchmark=True), "deterministic cuDNN"),
        (lambda m: m["runtime_provenance"].update(cudnn_deterministic=False), "deterministic cuDNN"),
        (lambda m: m["runtime_provenance"].update(device_name="", name=""), "missing GPU device_name"),
        (lambda m: m["runtime_provenance"].update(device_name="Quadro P6000", name="Quadro P6000"), "Unsupported acceptance GPU"),
        (lambda m: m.update(source_commit="0" * 40), "Source commit mismatch"),
        (lambda m: m.update(acceptance_modes=["reference"]), "Both reference and dicom"),
        (lambda m: m.pop("suvbw_converter_check"), "suvbw_converter_check"),
        (lambda m: m["suvbw_converter_check"].update(suvbw_max_abs_err=0.05), "suvbw_max_abs_err"),
        (lambda m: m["suvbw_converter_check"].pop("pet_grid"), "suvbw_converter_check.pet_grid"),
        (lambda m: m["suvbw_converter_check"].pop("ct_grid"), "suvbw_converter_check.ct_grid"),
        (lambda m: m["suvbw_converter_check"]["pet_grid"].update(spacing_max_delta_mm=0.05), "spacing_max_delta_mm"),
        (lambda m: m["reference_mode"].pop("biomarker_recomputation"), "biomarker_recomputation"),
        (lambda m: m["reference_mode"].pop("grid_check"), "grid_check"),
        (lambda m: m["reference_mode"].pop("upstream_grid_check"), "upstream_grid_check"),
        (lambda m: m["reference_mode"].pop("nnunet_runtime_flags"), "nnunet_runtime_flags"),
        (lambda m: m["reference_mode"]["nnunet_runtime_flags"].update(cudnn_benchmark=True), "deterministic cuDNN"),
        (lambda m: m["reference_mode"].pop("upstream_nnunet_runtime_flags"), "upstream_nnunet_runtime_flags"),
        (
            lambda m: m["reference_mode"]["upstream_nnunet_runtime_flags"].update(
                cudnn_benchmark=True, cudnn_deterministic=False
            ),
            "upstream_nnunet_runtime_flags.*deterministic cuDNN",
        ),
        (lambda m: m["reference_mode"].update(nnunet_label_mismatch_voxels=1), "nnunet_label_mismatch_voxels"),
        (lambda m: m["reference_mode"].update(dice=1.2), "exceeds maximum 1.0"),
        (lambda m: m["reference_mode"].update(suv_threshold=4.0), "suv_threshold"),
        (lambda m: m["dicom_mode"].update(suv_threshold_source="published-default"), "published-default"),
        (lambda m: m["dicom_mode"].update(rtstruct_valid=False), "failure|DICOM mode"),
        (lambda m: m["dicom_mode"].update(roi_name="WRONG_ROI"), "DICOM mode failed"),
        (lambda m: m["dicom_mode"].update(reference_mismatch_voxels=1), "reference_mismatch_voxels"),
        (lambda m: m["dicom_mode"].pop("vs_reference_mode"), "vs_reference_mode"),
        (lambda m: m["dicom_mode"]["vs_reference_mode"].update(mismatch_voxels=2), "vs_reference_mode"),
        (lambda m: m["dicom_mode"].pop("fixture_roundtrip_verification"), "fixture_roundtrip_verification"),
        (
            lambda m: m["dicom_mode"]["fixture_roundtrip_verification"].pop("pet_grid"),
            "fixture_roundtrip_verification.pet_grid",
        ),
        (
            lambda m: m["dicom_mode"]["fixture_roundtrip_verification"].update(pet_dicom_sha256="9" * 64),
            "fixture_roundtrip_verification.*pet_dicom_sha256",
        ),
        (lambda m: m["dicom_mode"].pop("nnunet_label_mismatch_voxels"), "nnunet_label_mismatch_voxels"),
        (lambda m: m["dicom_mode"].update(nnunet_label_mismatch_voxels=-1), "nnunet_label_mismatch_voxels"),
        (lambda m: m["dicom_mode"].update(subregion_label_mismatch_voxels=1.5), "subregion_label_mismatch_voxels"),
        (lambda m: m.update(checkpoint_sha256="b" * 64), "checkpoint_sha256 digest mismatch"),
        (lambda m: m.update(code_sha256="c" * 64), "code_sha256 mismatch"),
        (lambda m: m.update(runtime_lock_sha256="d" * 64), "runtime_lock_sha256 mismatch"),
        (lambda m: m["input_binding"].pop("pet_dicom_sha256"), "pet_dicom_sha256"),
    ]:
        bad = copy.deepcopy(val_h100)
        mutate(bad)
        env = {"schema_version": 2, "devices": {"H100": bad}}
        with pytest.raises(ValueError, match=msg):
            gtrc_models.validate_acceptance(
                env, task, bundle, app_root=ROOT, gpu=gpu_h100, require_completion=False
            )

    # DICOM-mode inputs carry the int16 SUV encoding error: raw label and subregion differences are
    # recorded diagnostics, while the exported TTB mask must still match reference mode exactly.
    diag = copy.deepcopy(val_h100)
    diag["dicom_mode"].update(nnunet_label_mismatch_voxels=4, subregion_label_mismatch_voxels=10)
    gtrc_models.validate_acceptance(
        {"schema_version": 2, "devices": {"H100": diag}},
        task,
        bundle,
        app_root=ROOT,
        gpu=gpu_h100,
        require_completion=False,
    )

    # Orientation validator probes
    for mutate_ori, msg_ori in [
        (lambda o: o.update(nonempty_rois=0), "nonempty_rois"),
        (lambda o: o.update(min_dice=0.1), "min_dice"),
        (lambda o: o.update(min_dice=1.2), "exceeds maximum 1.0"),
        (lambda o: o.update(median_dice=0.5), "median_dice"),
        (lambda o: o.update(roi_name="WRONG"), "roi_name"),
        (lambda o: o.update(roi_names=["WRONG"]), "roi_names"),
        (lambda o: o.update(referenced_pet_series_uid_match=False), "referenced_pet_series_uid_match"),
        (lambda o: o.update(acceptance_sha256="f" * 64), "acceptance_sha256"),
        (lambda o: o.update(run_id="wrong-run"), "run_id"),
        (lambda o: o.pop("runtime_lock_sha256"), "runtime_lock_sha256"),
        (lambda o: o.update(cpu_smoke=True), "smoke/stub"),
    ]:
        bad_ori = copy.deepcopy(ori_h100)
        mutate_ori(bad_ori)
        ori_env = {"schema_version": 2, "devices": {"H100": bad_ori}}
        with pytest.raises(ValueError, match=msg_ori):
            gtrc_models.validate_orientation(
                ori_env, task, bundle, app_root=ROOT, gpu=gpu_h100, require_completion=False
            )


@requires_run_gtrc
def test_preflight_and_runner_require_completion_marker(tmp_path, monkeypatch):
    import model_preflight

    task = "gtrc_psma_petct_ttb"
    bundle = tmp_path / "bundle_pf"
    _, _, _, gpu_h100 = _seed_bundle_device_records(bundle, "H100", "run-pf-h100")

    fake_bin = bundle / "runtime-envs/gtrc/bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    for exe_name in ("python", "nnUNetv2_predict"):
        exe = fake_bin / exe_name
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)

    gpu_prov = copy.deepcopy(gpu_h100)
    gpu_prov["cuda_arch_compatible"] = True
    gpu_prov["gpu_name"] = gpu_h100["device_name"]
    gpu_prov["device_capability"] = gpu_h100["cuda_device_capability"]
    gpu_prov["cuda_arch_flags"] = gpu_h100["cuda_arch_list"]

    monkeypatch.setattr(model_preflight, "MODEL_ROOT", bundle)
    monkeypatch.setattr(model_preflight, "APP_ROOT", ROOT)
    monkeypatch.setattr(
        model_preflight,
        "gtrc_verify_cuda",
        lambda *a, **kw: (True, "ok", gpu_prov),
    )
    monkeypatch.setattr(
        gtrc_models,
        "verify_runtime_cuda_compatibility",
        lambda *a, **kw: (True, "ok", gpu_prov),
    )
    monkeypatch.setattr(run_gtrc, "runtime_provenance", lambda: copy.deepcopy(gpu_prov))
    monkeypatch.setenv("RTSEG_MODEL_ROOT", str(bundle))
    monkeypatch.setenv("RTSEG_APP_ROOT", str(ROOT))

    res_ok = model_preflight.audit([task])[task]
    assert res_ok["ready"] is True, res_ok

    # Removing or invalidating validation/gtrc_completion.json causes both preflight and runner start to fail closed
    comp_path = bundle / gtrc_models.GTRC_COMPLETION_REL
    saved_comp = comp_path.read_text()
    comp_path.unlink()
    res_missing = model_preflight.audit([task])[task]
    assert res_missing["ready"] is False
    assert any(gtrc_models.GTRC_COMPLETION_REL in m for m in res_missing["missing"])

    with pytest.raises(SystemExit, match="GTRC completion gate refused"):
        run_gtrc.main(
            [
                "--task",
                task,
                "--pt-suv",
                str(tmp_path / "pt.nii.gz"),
                "--ct",
                str(tmp_path / "ct.nii.gz"),
                "--work-dir",
                str(tmp_path / "w_run"),
                "--output-mask",
                str(tmp_path / "out.nii.gz"),
            ]
        )
    comp_path.write_text(saved_comp)

    gtrc_models.invalidate_device(bundle, task, "H100")
    res_inv = model_preflight.audit([task])[task]
    assert res_inv["ready"] is False
    with pytest.raises(SystemExit, match="GTRC completion gate refused"):
        run_gtrc.main(
            [
                "--task",
                task,
                "--pt-suv",
                str(tmp_path / "pt.nii.gz"),
                "--ct",
                str(tmp_path / "ct.nii.gz"),
                "--work-dir",
                str(tmp_path / "w_run2"),
                "--output-mask",
                str(tmp_path / "out2.nii.gz"),
            ]
        )


def test_cuda_arch_compatibility_helper():
    assert gtrc_models.parse_sm_version("sm_60") == (6, 0)
    assert gtrc_models.parse_sm_version("compute_90a") == (9, 0)
    compiled = ["sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert gtrc_models.runtime_supports_device_arch(compiled, (6, 1)) is True
    assert gtrc_models.runtime_supports_device_arch(compiled, (9, 0)) is True
    assert gtrc_models.runtime_supports_device_arch(compiled, (5, 2)) is False


def test_api_gtrc_threshold_provenance_and_multiple_ct_fail_closed(tmp_path, monkeypatch):
    # Findings 8 and 9: API tests for FDG threshold requirement, PSMA published-default vs
    # user-entered provenance preservation, and multiple matching CTs.
    from fastapi.testclient import TestClient
    from test_api import load_app

    app = load_app(tmp_path)
    module = sys.modules["app"]
    queued = []
    monkeypatch.setattr(module, "enqueue_jobs", lambda jobs, **kw: queued.extend(jobs))
    for t_name in ("gtrc_psma_petct_ttb", "gtrc_fdg_petct_ttb"):
        module.EXTERNAL_TASKS[t_name]["enabled"] = True
        module.EXTERNAL_TASKS[t_name]["status"] = "available"
    monkeypatch.setattr(
        module,
        "tasks",
        lambda: {
            "tasks": [
                {"name": t_name, **module.model_details(t_name), "enabled": True}
                for t_name in ("gtrc_psma_petct_ttb", "gtrc_fdg_petct_ttb")
            ]
        },
    )

    study_uid = generate_uid()
    frame_uid = generate_uid()

    def stage_upload(uid: str, *, two_cts: bool = False):
        stage = module.BASE / uid
        inp = stage / "input"
        inp.mkdir(parents=True, exist_ok=True)
        pet_tmp = tmp_path / f"{uid}_pet"
        ct1_tmp = tmp_path / f"{uid}_ct1"
        write_synthetic_dicom_series(
            pet_tmp,
            np.ones((4, 8, 8), dtype=np.float32) * 5.0,
            modality="PT",
            study_uid=study_uid,
            frame_uid=frame_uid,
            series_uid="1.2.826.0.1.3680043.10.9001",
        )
        write_synthetic_dicom_series(
            ct1_tmp,
            np.zeros((4, 8, 8), dtype=np.float32),
            modality="CT",
            study_uid=study_uid,
            frame_uid=frame_uid,
            series_uid="1.2.826.0.1.3680043.10.9002",
        )
        for p in sorted(pet_tmp.glob("*.dcm")):
            (inp / f"pet_{p.name}").write_bytes(p.read_bytes())
        for p in sorted(ct1_tmp.glob("*.dcm")):
            (inp / f"ct1_{p.name}").write_bytes(p.read_bytes())
        if two_cts:
            ct2_tmp = tmp_path / f"{uid}_ct2"
            write_synthetic_dicom_series(
                ct2_tmp,
                np.zeros((4, 8, 8), dtype=np.float32),
                modality="CT",
                study_uid=study_uid,
                frame_uid=frame_uid,
                series_uid="1.2.826.0.1.3680043.10.9003",
            )
            for p in sorted(ct2_tmp.glob("*.dcm")):
                (inp / f"ct2_{p.name}").write_bytes(p.read_bytes())
        report = module.inspect_upload(inp)
        (stage / "upload.json").write_text(json.dumps({"id": uid, **report}, indent=2))
        return report

    report1 = stage_upload("upload-gtrc01", two_cts=False)
    pet_key = next(s["key"] for s in report1["series"] if s["modality"] == "PT")

    client = TestClient(app)
    try:
        # 1. FDG without threshold -> HTTP 400
        res_missing = client.post(
            "/api/v1/uploads/upload-gtrc01/jobs",
            data={"series_keys": [pet_key], "models": ["gtrc_fdg_petct_ttb"]},
        )
        assert res_missing.status_code == 400
        assert "requires a patient-specific FDG SUV threshold" in res_missing.json()["detail"]

        # 2. Finding 9: FDG with published-default or PSMA 2.5 with published-default -> HTTP 400
        res_bad_fdg_src = client.post(
            "/api/v1/uploads/upload-gtrc01/jobs",
            data={
                "series_keys": [pet_key],
                "models": ["gtrc_fdg_petct_ttb"],
                "task_options": json.dumps(
                    {"gtrc_fdg_petct_ttb": {"suv_threshold": 3.0, "source": "published-default"}}
                ),
            },
        )
        assert res_bad_fdg_src.status_code == 400
        assert "published-default" in res_bad_fdg_src.json()["detail"]

        res_bad_psma_src = client.post(
            "/api/v1/uploads/upload-gtrc01/jobs",
            data={
                "series_keys": [pet_key],
                "models": ["gtrc_psma_petct_ttb"],
                "task_options": json.dumps(
                    {"gtrc_psma_petct_ttb": {"suv_threshold": 2.5, "source": "published-default"}}
                ),
            },
        )
        assert res_bad_psma_src.status_code == 400
        assert "published-default" in res_bad_psma_src.json()["detail"]

        # 3. Finding 9: UI default payload {suv_threshold: 3.0, source: "published-default"} preserves "published-default"
        stage_upload("upload-gtrc01b", two_cts=False)
        res_ok = client.post(
            "/api/v1/uploads/upload-gtrc01b/jobs",
            data={
                "series_keys": [pet_key],
                "models": ["gtrc_fdg_petct_ttb", "gtrc_psma_petct_ttb"],
                "task_options": json.dumps(
                    {
                        "gtrc_fdg_petct_ttb": {"suv_threshold": 2.97, "source": "user-entered"},
                        "gtrc_psma_petct_ttb": {"suv_threshold": 3.0, "source": "published-default"},
                    }
                ),
            },
        )
        assert res_ok.status_code == 202, res_ok.text
        job_id = res_ok.json()["jobs"][0]["id"]
        job_meta = json.loads((module.BASE / job_id / "job.json").read_text())
        assert job_meta["task_options"]["gtrc_fdg_petct_ttb"] == {
            "suv_threshold": 2.97,
            "source": "user-entered",
        }
        assert job_meta["task_options"]["gtrc_psma_petct_ttb"] == {
            "suv_threshold": 3.0,
            "source": "published-default",
        }

        # 4. Upload with two matching CT series in the same Study and Frame of Reference -> ambiguous without choice
        report2 = stage_upload("upload-gtrc02", two_cts=True)
        pet_key2 = next(s["key"] for s in report2["series"] if s["modality"] == "PT")
        ct_keys = [s["key"] for s in report2["series"] if s["modality"] == "CT"]
        assert len(ct_keys) == 2

        res_ambig = client.post(
            "/api/v1/uploads/upload-gtrc02/jobs",
            data={"series_keys": [pet_key2], "models": ["gtrc_psma_petct_ttb"]},
        )
        assert res_ambig.status_code == 400
        assert "CT series match StudyInstanceUID and FrameOfReferenceUID" in res_ambig.json()["detail"]

        # Explicit ct_series_key resolves the ambiguity and preserves "published-default"
        res_explicit = client.post(
            "/api/v1/uploads/upload-gtrc02/jobs",
            data={
                "series_keys": [pet_key2],
                "models": ["gtrc_psma_petct_ttb"],
                "task_options": json.dumps(
                    {
                        "gtrc_psma_petct_ttb": {
                            "suv_threshold": 3.0,
                            "source": "published-default",
                            "ct_series_key": ct_keys[1],
                        }
                    }
                ),
            },
        )
        assert res_explicit.status_code == 202, res_explicit.text
        job_id2 = res_explicit.json()["jobs"][0]["id"]
        job_meta2 = json.loads((module.BASE / job_id2 / "job.json").read_text())
        assert job_meta2["paired_series_key"] == ct_keys[1]
        assert job_meta2["task_options"]["gtrc_psma_petct_ttb"] == {
            "suv_threshold": 3.0,
            "source": "published-default",
        }
    finally:
        client.close()

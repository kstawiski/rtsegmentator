#!/usr/bin/env python3
"""Vendor-neutral PET-to-SUVbw converter with explicit DICOM header verification.

Implements the RSNA Quantitative Imaging Biomarker Alliance (QIBA) SUVbw
calculation convention:
  - QIBA Profile: FDG-PET/CT as an Imaging Biomarker Measuring Response to
    Cancer Therapy (RSNA QIBA FDG-PET/CT Technical Committee SUV Pseudocode).
  - DICOM PS3.3 C.8.9.1 (PET Series Module), C.8.9.2 (PET Isotope Module),
    C.8.9.4 (PET Image Module), and C.7.6.16.2.11 (Real World Value Mapping).

For decay-corrected (DecayCorrection == "START") PET images in Units == "BQML":
    decay_factor    = 2 ** (-(t_ref - t_inj) / half_life_s)
    decayed_dose_bq = injected_dose_bq * decay_factor
    suv_scale       = (patient_weight_kg * 1000.0) / decayed_dose_bq
    SUVbw [g/mL]    = activity_bqml [Bq/mL] * suv_scale

Images that explicitly declare body-weight SUV units (Units == "GML" with
SUVType == "BW" or absent, or Units == "SUVBW", or a matching UCUM
"g/ml{SUVbw}" RealWorldValueMapping) are validated and retained in float32
SUVbw without applying radiopharmaceutical scaling a second time.

Missing or contradictory tags, non-body-weight SUV types, unsupported decay
corrections, and unverified midnight/date crossings raise ValueError.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import pydicom
import SimpleITK as sitk

from dicom_series import ordered_dicom_files
from orientation_contract import input_sidecar_path, normalize_file, sidecar_path

QIBA_CITATION = (
    "RSNA QIBA FDG-PET/CT Technical Committee SUVbw Pseudocode "
    "(QIBA Profile: FDG-PET/CT as an Imaging Biomarker Measuring Response to Cancer Therapy; "
    "DICOM PS3.3 C.8.9.1/C.8.9.2)"
)
# Implementation heuristic (not a QIBA cutoff): 10,000 Bq (10 kBq) minimum injected activity
# catches accidental MBq or mCi values stored in RadionuclideTotalDose (0018,1074).
MIN_PLAUSIBLE_DOSE_BQ = 1e4
MAX_PATIENT_WEIGHT_KG = 500.0
# Calibration agreement tolerance between RescaleSlope/Intercept and RealWorldValueMappingSequence:
# relative slope tolerance 1e-5 and maximum resulting SUVbw discrepancy <= 1e-4 g/mL across the
# slice's stored-value range.
RWVM_SLOPE_RTOL = 1e-5
RWVM_MAX_SUV_ERROR_TOL = 1e-4
NON_BW_SUV_TYPES = {"BSA", "LBM", "LBMJAMES", "LBMJANMA", "IBW"}
SUVBW_EXPLICIT_UCUM_CODES = {"g/ml{suvbw}", "{suvbw}g/ml", "{suvbw}"}
SUVBW_GML_UCUM_CODES = {"g/ml"}
BQML_UCUM_CODES = {"bq/ml"}


def parse_dicom_tz_offset_seconds(raw: Any, field_name: str = "TimezoneOffset") -> int | None:
    """Parse a DICOM timezone offset (+HHMM or -HHMM) into signed seconds from UTC."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) != 5 or text[0] not in ("+", "-") or not text[1:].isdigit():
        raise ValueError(f"Malformed timezone offset in {field_name}: {raw!r}")
    sign = 1 if text[0] == "+" else -1
    hours = int(text[1:3])
    minutes = int(text[3:5])
    if not (0 <= hours <= 14 and 0 <= minutes < 60):
        raise ValueError(f"Out-of-range timezone offset in {field_name}: {raw!r}")
    return sign * (hours * 3600 + minutes * 60)


def _timestamp_diff_seconds(
    date_a: date | None,
    time_a_s: float,
    tz_a_s: int | None,
    label_a: str,
    date_b: date | None,
    time_b_s: float,
    tz_b_s: int | None,
    label_b: str,
) -> float:
    """Compute (A - B) in seconds, converting explicit timezone offsets to UTC and rejecting partial offsets."""
    if (tz_a_s is None) != (tz_b_s is None):
        raise ValueError(
            f"Unresolved or partial timezone offset combination between {label_a} "
            f"(tz={tz_a_s}) and {label_b} (tz={tz_b_s})"
        )
    day_delta = (date_a - date_b).days if (date_a is not None and date_b is not None) else 0
    tz_delta = (tz_a_s - tz_b_s) if (tz_a_s is not None and tz_b_s is not None) else 0
    return day_delta * 86400.0 + (time_a_s - time_b_s) - float(tz_delta)


def parse_dicom_time_seconds(raw: Any, field_name: str = "TM") -> float:
    """Parse a DICOM TM string (HH[MM[SS[.FFFFFF]]]) into seconds since midnight."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"Missing or empty DICOM time for {field_name}")
    if any(ch in text for ch in ("+", "-", "&", " ")):
        raise ValueError(f"Malformed DICOM time for {field_name}: {text!r}")
    digits = text.split(".", 1)[0]
    if len(digits) not in (2, 4, 6) or not digits.isdigit():
        raise ValueError(f"Malformed DICOM time for {field_name}: {text!r}")
    hours = int(digits[0:2])
    minutes = int(digits[2:4]) if len(digits) >= 4 else 0
    whole_seconds = int(digits[4:6]) if len(digits) == 6 else 0
    frac_str = text[4:] if len(digits) == 4 else (text[2:] if len(digits) == 2 else text[4:])
    try:
        seconds = float(frac_str) if len(digits) == 6 else float(whole_seconds)
    except ValueError as exc:
        raise ValueError(f"Malformed DICOM seconds in {field_name}: {text!r}") from exc
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0.0 <= seconds < 60.0):
        raise ValueError(f"Out-of-range DICOM time in {field_name}: {text!r}")
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_dicom_date(raw: Any, field_name: str = "DA") -> date:
    """Parse a DICOM DA string (YYYYMMDD) into a calendar date."""
    text = str(raw or "").strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"Malformed DICOM date for {field_name}: {text!r}")
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date for {field_name}: {text!r}") from exc


def parse_dicom_datetime(raw: Any, field_name: str = "DT") -> tuple[date, float, str | None]:
    """Parse a DICOM DT string (YYYYMMDDHH[MM[SS[.FFFFFF]]][&ZZXX]) into (date, time_s, tz_offset)."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"Missing or empty DICOM datetime for {field_name}")
    tz_offset: str | None = None
    for idx in range(1, len(text)):
        if text[idx] in ("+", "-"):
            tz_offset = text[idx:]
            text = text[:idx]
            parse_dicom_tz_offset_seconds(tz_offset, field_name=field_name)
            break
    if len(text) < 10:
        raise ValueError(
            f"DICOM datetime {field_name}={raw!r} must include at least YYYYMMDDHH"
        )
    cal_date = parse_dicom_date(text[:8], field_name=f"{field_name}[date]")
    time_s = parse_dicom_time_seconds(text[8:], field_name=f"{field_name}[time]")
    return cal_date, time_s, tz_offset


def _finite_float(raw: Any, field_name: str) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or non-numeric {field_name}: {raw!r}") from exc
    if not math.isfinite(val):
        raise ValueError(f"Non-finite {field_name}: {raw!r}")
    return val


def _extract_rwvm(ds: pydicom.Dataset, index: int) -> dict[str, Any] | None:
    if "RealWorldValueMappingSequence" not in ds:
        return None
    seq = getattr(ds, "RealWorldValueMappingSequence", None)
    if not seq or len(seq) != 1:
        count = len(seq) if seq else 0
        raise ValueError(
            f"Invalid RealWorldValueMappingSequence (0040,9096) on slice {index} with {count} items; expected 1"
        )
    item = seq[0]
    if (
        getattr(item, "LUTData", None) is not None
        or getattr(item, "RealWorldValueLUTData", None) is not None
    ):
        raise ValueError(
            f"Unsupported non-linear LUT mapping in RealWorldValueMappingSequence (0040,9096) on slice {index}"
        )
    lut_explanation = (
        str(getattr(item, "LUTExplanation", "") or "").strip()
        if getattr(item, "LUTExplanation", None) is not None
        else None
    )
    lut_label = (
        str(getattr(item, "LUTLabel", "") or "").strip()
        if getattr(item, "LUTLabel", None) is not None
        else None
    )

    has_rw_slope = "RealWorldValueSlope" in item and getattr(item, "RealWorldValueSlope", None) not in (None, "")
    has_rw_intercept = (
        "RealWorldValueIntercept" in item
        and getattr(item, "RealWorldValueIntercept", None) not in (None, "")
    )
    if has_rw_slope != has_rw_intercept:
        raise ValueError(
            f"Incomplete RealWorldValueMappingSequence linear mapping on slice {index}: "
            "both RealWorldValueSlope (0040,9225) and RealWorldValueIntercept (0040,9224) are required"
        )
    if not has_rw_slope:
        raise ValueError(
            f"RealWorldValueMappingSequence on slice {index} is missing linear "
            "RealWorldValueSlope (0040,9225) and RealWorldValueIntercept (0040,9224)"
        )
    slope = _finite_float(
        getattr(item, "RealWorldValueSlope"), f"slice[{index}].RealWorldValueSlope (0040,9225)"
    )
    intercept = _finite_float(
        getattr(item, "RealWorldValueIntercept"),
        f"slice[{index}].RealWorldValueIntercept (0040,9224)",
    )
    if slope <= 0.0:
        raise ValueError(f"slice[{index}].RealWorldValueSlope (0040,9225) must be > 0, got {slope}")

    if "MeasurementUnitsCodeSequence" not in item:
        raise ValueError(
            f"Missing required MeasurementUnitsCodeSequence (0040,08EA) in "
            f"RealWorldValueMappingSequence on slice {index}"
        )
    units_seq = getattr(item, "MeasurementUnitsCodeSequence", None)
    if not units_seq or len(units_seq) != 1:
        raise ValueError(
            f"Invalid MeasurementUnitsCodeSequence (0040,08EA) on slice {index}; expected 1 item"
        )
    u_item = units_seq[0]
    coding_scheme = str(getattr(u_item, "CodingSchemeDesignator", "") or "").strip().upper()
    ucum_code = str(getattr(u_item, "CodeValue", "") or "").strip()
    if coding_scheme != "UCUM":
        raise ValueError(
            f"Unsupported MeasurementUnitsCodeSequence CodingSchemeDesignator={coding_scheme!r} "
            f"on slice {index}; expected 'UCUM'"
        )
    if not ucum_code:
        raise ValueError(
            f"Missing MeasurementUnitsCodeSequence CodeValue (0008,0100) on slice {index}"
        )

    is_float_pixel_data = ("FloatPixelData" in ds) or ("DoubleFloatPixelData" in ds)
    has_first_mapped = (
        "RealWorldValueFirstValueMapped" in item
        and getattr(item, "RealWorldValueFirstValueMapped", None) not in (None, "")
    )
    has_last_mapped = (
        "RealWorldValueLastValueMapped" in item
        and getattr(item, "RealWorldValueLastValueMapped", None) not in (None, "")
    )
    if has_first_mapped != has_last_mapped:
        raise ValueError(
            f"Incomplete RealWorldValueMappingSequence mapped range on slice {index}: "
            "both RealWorldValueFirstValueMapped (0040,9216) and RealWorldValueLastValueMapped (0040,9211) "
            "are required when either is present"
        )
    first_mapped = (
        _finite_float(
            getattr(item, "RealWorldValueFirstValueMapped"),
            f"slice[{index}].RealWorldValueFirstValueMapped (0040,9216)",
        )
        if has_first_mapped
        else None
    )
    last_mapped = (
        _finite_float(
            getattr(item, "RealWorldValueLastValueMapped"),
            f"slice[{index}].RealWorldValueLastValueMapped (0040,9211)",
        )
        if has_last_mapped
        else None
    )
    if first_mapped is not None and last_mapped is not None and first_mapped > last_mapped:
        raise ValueError(
            f"Invalid RealWorldValueMappingSequence mapped range on slice {index}: "
            f"RealWorldValueFirstValueMapped ({first_mapped}) > RealWorldValueLastValueMapped ({last_mapped})"
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "ucum_code": ucum_code,
        "coding_scheme": coding_scheme,
        "lut_explanation": lut_explanation,
        "lut_label": lut_label,
        "first_value_mapped": first_mapped,
        "last_value_mapped": last_mapped,
        "is_float_pixel_data": is_float_pixel_data,
    }


def _decode_slice_stored_pixel_extrema(
    ds: pydicom.Dataset,
    index: int,
    *,
    bits_stored: int,
    pix_rep: int,
    require_pixel_data: bool = False,
) -> tuple[float, float] | None:
    """Decode a slice's stored pixel values and return `(min_stored, max_stored)`."""
    has_pixel_data = "PixelData" in ds and getattr(ds, "PixelData", None) not in (None, b"")
    has_float_data = "FloatPixelData" in ds and getattr(ds, "FloatPixelData", None) not in (None, b"")
    has_double_data = "DoubleFloatPixelData" in ds and getattr(ds, "DoubleFloatPixelData", None) not in (
        None,
        b"",
    )
    if not (has_pixel_data or has_float_data or has_double_data):
        if require_pixel_data:
            raise ValueError(f"Missing required PixelData (7FE0,0010) on slice {index}")
        return None

    arr_px: np.ndarray | None = None
    if has_pixel_data:
        try:
            arr_px = np.asarray(ds.pixel_array)
        except Exception:
            raw_bytes = bytes(getattr(ds, "PixelData"))
            bits_alloc_raw = getattr(ds, "BitsAllocated", None)
            bits_alloc = (
                int(bits_alloc_raw)
                if bits_alloc_raw not in (None, "")
                else (8 if bits_stored <= 8 else (16 if bits_stored <= 16 else 32))
            )
            dtype_map = {
                (8, 0): np.dtype("<u1"),
                (8, 1): np.dtype("<i1"),
                (16, 0): np.dtype("<u2"),
                (16, 1): np.dtype("<i2"),
                (32, 0): np.dtype("<u4"),
                (32, 1): np.dtype("<i4"),
            }
            dt = dtype_map.get((bits_alloc, pix_rep))
            if dt is None or len(raw_bytes) == 0 or len(raw_bytes) % dt.itemsize != 0:
                raise ValueError(f"Cannot decode PixelData on slice {index}")
            arr_px = np.frombuffer(raw_bytes, dtype=dt)
    elif has_float_data:
        raw_bytes = bytes(getattr(ds, "FloatPixelData"))
        if len(raw_bytes) == 0 or len(raw_bytes) % 4 != 0:
            raise ValueError(f"Cannot decode FloatPixelData on slice {index}")
        arr_px = np.frombuffer(raw_bytes, dtype=np.dtype("<f4"))
    else:
        raw_bytes = bytes(getattr(ds, "DoubleFloatPixelData"))
        if len(raw_bytes) == 0 or len(raw_bytes) % 8 != 0:
            raise ValueError(f"Cannot decode DoubleFloatPixelData on slice {index}")
        arr_px = np.frombuffer(raw_bytes, dtype=np.dtype("<f8"))

    if arr_px is None or arr_px.size == 0 or not np.isfinite(arr_px).all():
        raise ValueError(f"Empty or non-finite decoded PixelData on slice {index}")
    return float(np.min(arr_px)), float(np.max(arr_px))


def _slice_stored_value_range(
    ds: pydicom.Dataset,
    index: int,
    rwvm: dict[str, Any] | None,
    *,
    decoded_extrema: tuple[float, float] | None = None,
    require_pixel_data: bool = False,
) -> tuple[float, float]:
    """Determine and validate the slice's stored-value domain for RWVM calibration checks."""
    bits_raw = getattr(ds, "BitsStored", None)
    bits_stored = int(bits_raw) if bits_raw not in (None, "") else 16
    if not (1 <= bits_stored <= 32):
        raise ValueError(f"Invalid BitsStored ({bits_raw!r}) on slice {index}")
    pix_rep_raw = getattr(ds, "PixelRepresentation", None)
    pix_rep = int(pix_rep_raw) if pix_rep_raw not in (None, "") else 0
    if pix_rep not in (0, 1):
        raise ValueError(f"Invalid PixelRepresentation ({pix_rep_raw!r}) on slice {index}")
    if pix_rep == 0:
        rep_min = 0.0
        rep_max = float((1 << bits_stored) - 1)
    else:
        rep_min = float(-(1 << (bits_stored - 1)))
        rep_max = float((1 << (bits_stored - 1)) - 1)

    obs_min: float | None = None
    obs_max: float | None = None
    if getattr(ds, "SmallestImagePixelValue", None) not in (None, ""):
        obs_min = _finite_float(
            getattr(ds, "SmallestImagePixelValue"),
            f"slice[{index}].SmallestImagePixelValue (0028,0106)",
        )
        if obs_min < rep_min or obs_min > rep_max:
            raise ValueError(
                f"SmallestImagePixelValue ({obs_min:g}) on slice {index} lies outside "
                f"representable stored-value range [{rep_min:g}, {rep_max:g}]"
            )
    if getattr(ds, "LargestImagePixelValue", None) not in (None, ""):
        obs_max = _finite_float(
            getattr(ds, "LargestImagePixelValue"),
            f"slice[{index}].LargestImagePixelValue (0028,0107)",
        )
        if obs_max < rep_min or obs_max > rep_max:
            raise ValueError(
                f"LargestImagePixelValue ({obs_max:g}) on slice {index} lies outside "
                f"representable stored-value range [{rep_min:g}, {rep_max:g}]"
            )
    if obs_min is not None and obs_max is not None and obs_min > obs_max:
        raise ValueError(
            f"Contradictory SmallestImagePixelValue ({obs_min}) > LargestImagePixelValue ({obs_max}) on slice {index}"
        )

    px_extrema = _decode_slice_stored_pixel_extrema(
        ds,
        index,
        bits_stored=bits_stored,
        pix_rep=pix_rep,
        require_pixel_data=require_pixel_data and decoded_extrema is None,
    )
    if px_extrema is not None and decoded_extrema is not None:
        actual_min = min(px_extrema[0], decoded_extrema[0])
        actual_max = max(px_extrema[1], decoded_extrema[1])
    else:
        actual_min, actual_max = px_extrema if px_extrema is not None else (
            decoded_extrema if decoded_extrema is not None else (None, None)
        )

    if actual_min is not None and actual_max is not None:
        if actual_min < rep_min or actual_max > rep_max:
            raise ValueError(
                f"Decoded stored pixel range [{actual_min:g}, {actual_max:g}] on slice {index} "
                f"exceeds representable stored-value range [{rep_min:g}, {rep_max:g}]"
            )
        if obs_min is not None and actual_min != obs_min:
            raise ValueError(
                f"Contradictory SmallestImagePixelValue ({obs_min:g}) vs actual decoded minimum "
                f"stored pixel ({actual_min:g}) on slice {index}"
            )
        if obs_max is not None and actual_max != obs_max:
            raise ValueError(
                f"Contradictory LargestImagePixelValue ({obs_max:g}) vs actual decoded maximum "
                f"stored pixel ({actual_max:g}) on slice {index}"
            )
        sv_min = actual_min
        sv_max = actual_max
    else:
        sv_min = obs_min if obs_min is not None else rep_min
        sv_max = obs_max if obs_max is not None else rep_max

    if rwvm is not None and rwvm["first_value_mapped"] is not None and rwvm["last_value_mapped"] is not None:
        first_m = float(rwvm["first_value_mapped"])
        last_m = float(rwvm["last_value_mapped"])
        if first_m < rep_min or last_m > rep_max:
            raise ValueError(
                f"RealWorldValueMappingSequence mapped range [{first_m}, {last_m}] on slice {index} "
                f"exceeds representable stored-value range [{rep_min}, {rep_max}]"
            )
        if obs_min is not None and (obs_min < first_m or obs_min > last_m):
            raise ValueError(
                f"SmallestImagePixelValue ({obs_min:g}) on slice {index} lies outside "
                f"RealWorldValueMappingSequence mapped range [{first_m:g}, {last_m:g}]"
            )
        if obs_max is not None and (obs_max < first_m or obs_max > last_m):
            raise ValueError(
                f"LargestImagePixelValue ({obs_max:g}) on slice {index} lies outside "
                f"RealWorldValueMappingSequence mapped range [{first_m:g}, {last_m:g}]"
            )
        if actual_min is not None and actual_max is not None:
            if actual_min < first_m or actual_max > last_m:
                raise ValueError(
                    f"Decoded stored pixel values [{actual_min:g}, {actual_max:g}] on slice {index} "
                    f"lie outside RealWorldValueMappingSequence mapped range [{first_m:g}, {last_m:g}]"
                )
        elif obs_min is None and obs_max is None and (first_m > 0.0 or last_m <= 0.0):
            raise ValueError(
                f"RealWorldValueMappingSequence mapped range [{first_m}, {last_m}] on slice {index} "
                "does not cover stored zero/positive pixel values"
            )
        sv_min = min(sv_min, first_m)
        sv_max = max(sv_max, last_m)

    return sv_min, sv_max


def _check_slice_rescale_and_rwvm(
    ds: pydicom.Dataset,
    index: int,
    *,
    expect_mode: str,
    units: str,
    suv_type: str | None,
    decoded_extrema: tuple[float, float] | None = None,
    require_pixel_data: bool = False,
) -> dict[str, Any]:
    has_rescale_slope = "RescaleSlope" in ds and getattr(ds, "RescaleSlope", None) not in (None, "")
    has_rescale_intercept = (
        "RescaleIntercept" in ds and getattr(ds, "RescaleIntercept", None) not in (None, "")
    )
    if not has_rescale_slope or not has_rescale_intercept:
        raise ValueError(
            f"Missing required explicit RescaleSlope (0028,1053) or RescaleIntercept (0028,1052) on slice {index}; "
            "SimpleITK series conversion requires explicit linear rescale tags on every slice"
        )
    slope = _finite_float(getattr(ds, "RescaleSlope"), f"slice[{index}].RescaleSlope (0028,1053)")
    intercept = _finite_float(
        getattr(ds, "RescaleIntercept"), f"slice[{index}].RescaleIntercept (0028,1052)"
    )
    if slope <= 0.0:
        raise ValueError(f"slice[{index}].RescaleSlope (0028,1053) must be > 0, got {slope}")

    rwvm = _extract_rwvm(ds, index)
    if rwvm is not None:
        code_low = rwvm["ucum_code"].lower()
        if expect_mode == "bqml":
            if code_low in SUVBW_EXPLICIT_UCUM_CODES or code_low in SUVBW_GML_UCUM_CODES:
                raise ValueError(
                    f"Contradictory PET Units='BQML' with SUV RealWorldValueMapping UCUM code "
                    f"{rwvm['ucum_code']!r} on slice {index}"
                )
            if code_low not in BQML_UCUM_CODES:
                raise ValueError(
                    f"Unsupported RealWorldValueMapping UCUM CodeValue={rwvm['ucum_code']!r} for "
                    f"Units='BQML' on slice {index}; expected 'Bq/ml'"
                )
        elif expect_mode == "already_suvbw":
            allowed_suvbw = set(SUVBW_EXPLICIT_UCUM_CODES)
            if units in ("GML", "SUVBW") and (suv_type is None or suv_type == "BW"):
                allowed_suvbw |= SUVBW_GML_UCUM_CODES
            if code_low not in allowed_suvbw:
                raise ValueError(
                    f"Contradictory or unsupported RealWorldValueMapping UCUM CodeValue={rwvm['ucum_code']!r} "
                    f"for body-weight SUV series (Units={units!r}, SUVType={suv_type!r}) on slice {index}"
                )

        rw_slope = rwvm["slope"]
        rw_intercept = rwvm["intercept"]
        tol_s = RWVM_SLOPE_RTOL * min(abs(slope), abs(rw_slope))
        if abs(rw_slope - slope) > tol_s:
            raise ValueError(
                f"Contradictory RescaleSlope/Intercept ({slope}, {intercept}) and "
                f"RealWorldValueMappingSequence ({rw_slope}, {rw_intercept}) on slice {index}"
            )
        if not rwvm.get("is_float_pixel_data", False) and (
            rwvm["first_value_mapped"] is None or rwvm["last_value_mapped"] is None
        ):
            raise ValueError(
                f"Missing required RealWorldValueFirstValueMapped (0040,9216) or "
                f"RealWorldValueLastValueMapped (0040,9211) in RealWorldValueMappingSequence on slice {index}; "
                "DICOM PS3.3 Table C.7.6.16-12b requires mapped-range endpoints for integer Pixel Data"
            )

    sv_min, sv_max = _slice_stored_value_range(
        ds,
        index,
        rwvm,
        decoded_extrema=decoded_extrema,
        require_pixel_data=require_pixel_data,
    )
    max_cal_diff = 0.0
    if rwvm is not None:
        rw_slope = rwvm["slope"]
        rw_intercept = rwvm["intercept"]
        tol_s = RWVM_SLOPE_RTOL * min(abs(slope), abs(rw_slope))
        max_cal_diff = max(
            abs((rw_slope - slope) * sv + (rw_intercept - intercept))
            for sv in (sv_min, sv_max)
        )
        max_rescaled_span = max(abs(slope * sv_max + intercept), abs(slope * sv_min + intercept), abs(slope))
        cal_bound = (
            RWVM_MAX_SUV_ERROR_TOL
            if expect_mode == "already_suvbw"
            else RWVM_MAX_SUV_ERROR_TOL * max_rescaled_span
        )
        if abs(rw_slope - slope) > tol_s or max_cal_diff > cal_bound:
            raise ValueError(
                f"Contradictory RescaleSlope/Intercept ({slope}, {intercept}) and "
                f"RealWorldValueMappingSequence ({rw_slope}, {rw_intercept}) on slice {index} "
                f"(max calibration discrepancy {max_cal_diff:.6g} over stored range [{sv_min:g}, {sv_max:g}])"
            )

    return {
        "has_rescale": True,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "rwvm": rwvm,
        "stored_value_min": sv_min,
        "stored_value_max": sv_max,
        "max_calibration_diff": max_cal_diff,
    }


def _corrected_image_tokens(ds: pydicom.Dataset) -> list[str]:
    raw = getattr(ds, "CorrectedImage", None)
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        text = raw.decode("ascii", errors="ignore") if isinstance(raw, bytes) else raw
        return [tok.strip().upper() for tok in text.replace("\\", " ").split() if tok.strip()]
    tokens: list[str] = []
    for item in raw:
        for tok in str(item).replace("\\", " ").split():
            if tok.strip():
                tokens.append(tok.strip().upper())
    return tokens


def compute_suvbw_from_datasets(
    datasets: Sequence[pydicom.Dataset],
    *,
    min_dose_bq: float = MIN_PLAUSIBLE_DOSE_BQ,
    require_corrected_image: bool = True,
    slice_pixel_extrema: Sequence[tuple[float, float]] | None = None,
    require_pixel_data: bool = False,
) -> dict[str, Any]:
    """Validate PET DICOM headers across a series and compute the SUVbw scaling contract.

    Returns a dictionary recording every checked tag, conversion mode
    ('bqml_to_suvbw' or 'already_suvbw'), scale factor, and post-GDCM linear
    adjustment if RealWorldValueMappingSequence was used without RescaleSlope.
    """
    if not datasets:
        raise ValueError("PET series contains no DICOM headers")
    if slice_pixel_extrema is not None and len(slice_pixel_extrema) != len(datasets):
        raise ValueError(
            f"slice_pixel_extrema length ({len(slice_pixel_extrema)}) does not match slice count ({len(datasets)})"
        )

    first = datasets[0]
    modalities = {str(getattr(ds, "Modality", "") or "").strip().upper() for ds in datasets}
    if modalities != {"PT"}:
        raise ValueError(f"Expected Modality (0008,0060) == 'PT' across all slices, got {sorted(modalities)}")

    series_uids = {str(getattr(ds, "SeriesInstanceUID", "") or "").strip() for ds in datasets}
    if len(series_uids) > 1:
        raise ValueError(f"Contradictory SeriesInstanceUID (0020,000E) across slices: {sorted(series_uids)}")
    study_uids = {str(getattr(ds, "StudyInstanceUID", "") or "").strip() for ds in datasets}
    if len(study_uids) > 1:
        raise ValueError(f"Contradictory StudyInstanceUID (0020,000D) across slices: {sorted(study_uids)}")

    units_set = {str(getattr(ds, "Units", "") or "").strip().upper() for ds in datasets}
    if len(units_set) != 1:
        raise ValueError(f"Contradictory Units (0054,1001) across slices: {sorted(units_set)}")
    units = next(iter(units_set))
    if not units:
        raise ValueError("Missing required PET Units (0054,1001) tag")

    suv_types = {
        str(getattr(ds, "SUVType", "") or "").strip().upper()
        for ds in datasets
        if getattr(ds, "SUVType", None) is not None and str(getattr(ds, "SUVType", "")).strip() != ""
    }
    if len(suv_types) > 1:
        raise ValueError(f"Contradictory SUVType (0054,1006) across slices: {sorted(suv_types)}")
    suv_type = next(iter(suv_types)) if suv_types else None

    is_already_suvbw = units in ("GML", "SUVBW") or (units == "SUV" and suv_type == "BW")
    if is_already_suvbw:
        if suv_type is not None and suv_type in NON_BW_SUV_TYPES:
            raise ValueError(
                f"PET series with Units={units!r} has non-body-weight SUVType={suv_type!r}; "
                "SUVbw requires SUVType='BW'"
            )
        if suv_type is not None and suv_type != "BW":
            raise ValueError(
                f"PET series with Units={units!r} has unrecognized SUVType={suv_type!r}; expected 'BW'"
            )
        expect_mode = "already_suvbw"
    elif units == "BQML":
        if suv_type is not None and suv_type not in ("BW",):
            raise ValueError(
                f"PET series with Units='BQML' has contradictory SUVType={suv_type!r}"
            )
        expect_mode = "bqml"
    else:
        raise ValueError(
            f"Unsupported PET Units (0054,1001)={units!r}: expected 'BQML' or explicit body-weight SUV "
            "('GML' with SUVType='BW' or 'SUVBW')"
        )

    rescale_records = [
        _check_slice_rescale_and_rwvm(
            ds,
            idx,
            expect_mode=expect_mode,
            units=units,
            suv_type=suv_type,
            decoded_extrema=slice_pixel_extrema[idx] if slice_pixel_extrema is not None else None,
            require_pixel_data=require_pixel_data,
        )
        for idx, ds in enumerate(datasets)
    ]
    slopes = [rec["rescale_slope"] for rec in rescale_records]
    intercepts = [rec["rescale_intercept"] for rec in rescale_records]
    stored_mins = [float(rec["stored_value_min"]) for rec in rescale_records]
    stored_maxs = [float(rec["stored_value_max"]) for rec in rescale_records]
    max_cal_diffs = [float(rec["max_calibration_diff"]) for rec in rescale_records]
    rwvm_codes = {
        rec["rwvm"]["ucum_code"].lower()
        for rec in rescale_records
        if rec["rwvm"] is not None and rec["rwvm"].get("ucum_code")
    }

    # Case 1: Image is already explicitly in body-weight SUV (never convert twice).
    if is_already_suvbw:
        return {
            "citation": QIBA_CITATION,
            "conversion_mode": "already_suvbw",
            "units": units,
            "suv_type": suv_type or "BW",
            "slice_count": len(datasets),
            "rescale_slope_min": float(min(slopes)),
            "rescale_slope_max": float(max(slopes)),
            "rescale_intercept_min": float(min(intercepts)),
            "rescale_intercept_max": float(max(intercepts)),
            "stored_value_min": float(min(stored_mins)),
            "stored_value_max": float(max(stored_maxs)),
            "max_calibration_diff": float(max(max_cal_diffs)),
            "rwvm_ucum_codes": sorted(rwvm_codes),
            "post_gdcm_slope": 1.0,
            "post_gdcm_intercept": 0.0,
            "suv_scale_factor": 1.0,
        }

    # Case 2: Units == "BQML" -> verify QIBA decay, dose, weight, and timing contract.
    decay_corrs = {str(getattr(ds, "DecayCorrection", "") or "").strip().upper() for ds in datasets}
    if len(decay_corrs) != 1:
        raise ValueError(
            f"Contradictory DecayCorrection (0054,1102) across slices: {sorted(decay_corrs)}"
        )
    decay_correction = next(iter(decay_corrs))
    if decay_correction != "START":
        raise ValueError(
            f"Unsupported or missing DecayCorrection (0054,1102)={decay_correction!r}; "
            "QIBA SUVbw requires 'START'"
        )

    corrected_tokens_per_slice = [_corrected_image_tokens(ds) for ds in datasets]
    if require_corrected_image or any(corrected_tokens_per_slice):
        for idx, tokens in enumerate(corrected_tokens_per_slice):
            if "DECY" not in tokens or "ATTN" not in tokens:
                raise ValueError(
                    f"PET CorrectedImage (0028,0051) on slice {idx} ({tokens}) must include both 'DECY' and 'ATTN'"
                )

    # PatientWeight (0010,1030) in kg
    weights = [
        _finite_float(getattr(ds, "PatientWeight", None), f"slice[{idx}].PatientWeight (0010,1030)")
        for idx, ds in enumerate(datasets)
    ]
    weight_kg = weights[0]
    if not (0.0 < weight_kg <= MAX_PATIENT_WEIGHT_KG):
        raise ValueError(
            f"PatientWeight (0010,1030) must be in (0, {MAX_PATIENT_WEIGHT_KG}] kg, got {weight_kg}"
        )
    if any(abs(w - weight_kg) > 1e-3 for w in weights):
        raise ValueError(
            f"Contradictory PatientWeight (0010,1030) across slices: min={min(weights)}, max={max(weights)}"
        )

    # TimezoneOffsetFromUTC (0008,0201) consistency
    tz_offsets = {
        str(getattr(ds, "TimezoneOffsetFromUTC", "") or "").strip()
        for ds in datasets
        if getattr(ds, "TimezoneOffsetFromUTC", None) is not None
        and str(getattr(ds, "TimezoneOffsetFromUTC", "")).strip() != ""
    }
    if len(tz_offsets) > 1:
        raise ValueError(f"Contradictory TimezoneOffsetFromUTC (0008,0201) across slices: {sorted(tz_offsets)}")
    series_tz = next(iter(tz_offsets)) if tz_offsets else None
    series_tz_s = parse_dicom_tz_offset_seconds(series_tz, "TimezoneOffsetFromUTC (0008,0201)")

    # RadiopharmaceuticalInformationSequence (0054,0016)
    doses_bq: list[float] = []
    half_lives_s: list[float] = []
    inj_dates: list[date | None] = []
    inj_times_s: list[float] = []
    inj_tzs_s: list[int | None] = []
    inj_time_sources: set[str] = set()
    radionuclide_names: set[str] = set()

    for idx, ds in enumerate(datasets):
        rph_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
        if not rph_seq or len(rph_seq) == 0:
            raise ValueError(
                f"Missing RadiopharmaceuticalInformationSequence (0054,0016) on slice {idx}"
            )
        if len(rph_seq) > 1:
            raise ValueError(
                f"Ambiguous RadiopharmaceuticalInformationSequence (0054,0016) with {len(rph_seq)} items on slice {idx}"
            )
        rph = rph_seq[0]
        dose = _finite_float(
            getattr(rph, "RadionuclideTotalDose", None),
            f"slice[{idx}].RadionuclideTotalDose (0018,1074)",
        )
        if dose <= 0.0:
            raise ValueError(f"RadionuclideTotalDose (0018,1074) must be > 0, got {dose}")
        if dose < min_dose_bq:
            raise ValueError(
                f"RadionuclideTotalDose (0018,1074)={dose} is below minimum plausible Bq threshold "
                f"{min_dose_bq} (suspect non-Bq unit such as MBq or mCi)"
            )
        half_life = _finite_float(
            getattr(rph, "RadionuclideHalfLife", None),
            f"slice[{idx}].RadionuclideHalfLife (0018,1075)",
        )
        if half_life <= 0.0:
            raise ValueError(f"RadionuclideHalfLife (0018,1075) must be > 0, got {half_life}")

        dt_raw = getattr(rph, "RadiopharmaceuticalStartDateTime", None)
        tm_raw = getattr(rph, "RadiopharmaceuticalStartTime", None)
        has_dt = dt_raw is not None and str(dt_raw).strip() != ""
        has_tm = tm_raw is not None and str(tm_raw).strip() != ""
        if not has_dt and not has_tm:
            raise ValueError(
                f"Missing both RadiopharmaceuticalStartDateTime (0018,1078) and "
                f"RadiopharmaceuticalStartTime (0018,1072) on slice {idx}"
            )
        inj_date: date | None = None
        inj_tz_s: int | None = series_tz_s
        if has_dt:
            inj_date, dt_time_s, dt_tz = parse_dicom_datetime(
                dt_raw, field_name=f"slice[{idx}].RadiopharmaceuticalStartDateTime (0018,1078)"
            )
            dt_tz_s = parse_dicom_tz_offset_seconds(
                dt_tz, field_name=f"slice[{idx}].RadiopharmaceuticalStartDateTime (0018,1078)"
            )
            if dt_tz_s is not None and series_tz_s is not None and dt_tz_s != series_tz_s:
                raise ValueError(
                    f"Contradictory timezone between RadiopharmaceuticalStartDateTime ({dt_tz}) "
                    f"and TimezoneOffsetFromUTC ({series_tz})"
                )
            if dt_tz_s is not None:
                inj_tz_s = dt_tz_s
            if has_tm:
                tm_time_s = parse_dicom_time_seconds(
                    tm_raw, field_name=f"slice[{idx}].RadiopharmaceuticalStartTime (0018,1072)"
                )
                if abs(dt_time_s - tm_time_s) > 1.0:
                    raise ValueError(
                        f"Contradictory RadiopharmaceuticalStartDateTime ({dt_raw!r}) and "
                        f"RadiopharmaceuticalStartTime ({tm_raw!r}) on slice {idx}"
                    )
            inj_time_s = dt_time_s
            inj_time_sources.add("RadiopharmaceuticalStartDateTime (0018,1078)")
        else:
            inj_time_s = parse_dicom_time_seconds(
                tm_raw, field_name=f"slice[{idx}].RadiopharmaceuticalStartTime (0018,1072)"
            )
            inj_time_sources.add("RadiopharmaceuticalStartTime (0018,1072)")

        rn_seq = getattr(rph, "RadionuclideCodeSequence", None)
        if rn_seq and len(rn_seq) >= 1:
            meaning = str(getattr(rn_seq[0], "CodeMeaning", "") or "").strip()
            if meaning:
                radionuclide_names.add(meaning)

        doses_bq.append(dose)
        half_lives_s.append(half_life)
        inj_dates.append(inj_date)
        inj_times_s.append(inj_time_s)
        inj_tzs_s.append(inj_tz_s)

    dose_bq = doses_bq[0]
    if any(abs(d - dose_bq) > 1e-3 * max(1.0, dose_bq) for d in doses_bq):
        raise ValueError(
            f"Contradictory RadionuclideTotalDose (0018,1074) across slices: min={min(doses_bq)}, max={max(doses_bq)}"
        )
    half_life_s = half_lives_s[0]
    if any(abs(h - half_life_s) > 1e-2 for h in half_lives_s):
        raise ValueError(
            f"Contradictory RadionuclideHalfLife (0018,1075) across slices: min={min(half_lives_s)}, max={max(half_lives_s)}"
        )
    if len(set(inj_dates)) > 1 or len(set(inj_tzs_s)) > 1 or any(abs(t - inj_times_s[0]) > 1.0 for t in inj_times_s):
        raise ValueError("Contradictory radiopharmaceutical start time/date/timezone across slices")
    inj_date = inj_dates[0]
    inj_time_s = inj_times_s[0]
    inj_tz_s = inj_tzs_s[0]
    inj_time_source = sorted(inj_time_sources)[0]

    # Reference scan start date/time for DecayCorrection == "START"
    series_times_raw = {
        str(getattr(ds, "SeriesTime", "") or "").strip()
        for ds in datasets
        if getattr(ds, "SeriesTime", None) is not None and str(getattr(ds, "SeriesTime", "")).strip() != ""
    }
    if len(series_times_raw) > 1:
        raise ValueError(f"Contradictory SeriesTime (0008,0031) across slices: {sorted(series_times_raw)}")
    series_time_str = next(iter(series_times_raw)) if series_times_raw else None

    series_dates_raw = {
        str(getattr(ds, "SeriesDate", "") or "").strip()
        for ds in datasets
        if getattr(ds, "SeriesDate", None) is not None and str(getattr(ds, "SeriesDate", "")).strip() != ""
    }
    if len(series_dates_raw) > 1:
        raise ValueError(f"Contradictory SeriesDate (0008,0021) across slices: {sorted(series_dates_raw)}")
    series_date = parse_dicom_date(next(iter(series_dates_raw)), "SeriesDate (0008,0021)") if series_dates_raw else None

    study_dates_raw = {
        str(getattr(ds, "StudyDate", "") or "").strip()
        for ds in datasets
        if getattr(ds, "StudyDate", None) is not None and str(getattr(ds, "StudyDate", "")).strip() != ""
    }
    if len(study_dates_raw) > 1:
        raise ValueError(f"Contradictory StudyDate (0008,0020) across slices: {sorted(study_dates_raw)}")
    study_date = parse_dicom_date(next(iter(study_dates_raw)), "StudyDate (0008,0020)") if study_dates_raw else None

    acq_dates: set[date] = set()
    acq_times_s: list[float] = []
    slice_acq_records: list[tuple[int, date | None, float | None, int | None]] = []
    qiba_backcomputed_starts: list[tuple[date | None, float, int | None]] = []
    vendor_scan_starts: list[tuple[date, float, str, int | None]] = []
    decay_lambda = math.log(2.0) / half_life_s

    for idx, ds in enumerate(datasets):
        ad_raw = getattr(ds, "AcquisitionDate", None)
        s_acq_date = (
            parse_dicom_date(ad_raw, f"slice[{idx}].AcquisitionDate (0008,0022)")
            if ad_raw is not None and str(ad_raw).strip() != ""
            else None
        )
        at_raw = getattr(ds, "AcquisitionTime", None)
        s_acq_time = (
            parse_dicom_time_seconds(at_raw, f"slice[{idx}].AcquisitionTime (0008,0032)")
            if at_raw is not None and str(at_raw).strip() != ""
            else None
        )
        s_acq_tz_s: int | None = series_tz_s
        adt_raw = getattr(ds, "AcquisitionDateTime", None)
        if adt_raw is not None and str(adt_raw).strip() != "":
            adt_date, adt_time_s, adt_tz = parse_dicom_datetime(
                adt_raw, f"slice[{idx}].AcquisitionDateTime (0008,002A)"
            )
            adt_tz_s = parse_dicom_tz_offset_seconds(
                adt_tz, f"slice[{idx}].AcquisitionDateTime (0008,002A)"
            )
            if adt_tz_s is not None and series_tz_s is not None and adt_tz_s != series_tz_s:
                raise ValueError(
                    f"Contradictory timezone between AcquisitionDateTime ({adt_tz}) "
                    f"and TimezoneOffsetFromUTC ({series_tz}) on slice {idx}"
                )
            if adt_tz_s is not None:
                s_acq_tz_s = adt_tz_s
            if s_acq_date is not None and s_acq_date != adt_date:
                raise ValueError(
                    f"Contradictory AcquisitionDate ({s_acq_date}) and AcquisitionDateTime ({adt_raw!r}) on slice {idx}"
                )
            if s_acq_time is not None and abs(s_acq_time - adt_time_s) > 1.0:
                raise ValueError(
                    f"Contradictory AcquisitionTime ({at_raw!r}) and AcquisitionDateTime ({adt_raw!r}) on slice {idx}"
                )
            s_acq_date = s_acq_date or adt_date
            s_acq_time = s_acq_time if s_acq_time is not None else adt_time_s

        if s_acq_date is not None:
            acq_dates.add(s_acq_date)
        if s_acq_time is not None:
            acq_times_s.append(s_acq_time)
        slice_acq_records.append((idx, s_acq_date, s_acq_time, s_acq_tz_s))

        # QIBA FrameReferenceTime (0054,1300) + ActualFrameDuration (0018,1242) scan-start back-calculation
        frt_raw = getattr(ds, "FrameReferenceTime", None)
        afd_raw = getattr(ds, "ActualFrameDuration", None)
        has_frt = frt_raw is not None and str(frt_raw).strip() != ""
        has_afd = afd_raw is not None and str(afd_raw).strip() != ""
        if has_frt or has_afd:
            if not (has_frt and has_afd):
                raise ValueError(
                    f"Incomplete QIBA frame-reference timing on slice {idx}: both FrameReferenceTime (0054,1300) "
                    "and ActualFrameDuration (0018,1242) are required when either is present"
                )
            if s_acq_time is None:
                raise ValueError(
                    f"Missing AcquisitionTime (0008,0032) required for QIBA FrameReferenceTime back-calculation on slice {idx}"
                )
            frt_ms = _finite_float(frt_raw, f"slice[{idx}].FrameReferenceTime (0054,1300)")
            afd_ms = _finite_float(afd_raw, f"slice[{idx}].ActualFrameDuration (0018,1242)")
            if frt_ms <= 0.0 or afd_ms <= 0.0:
                raise ValueError(
                    f"FrameReferenceTime ({frt_ms} ms) and ActualFrameDuration ({afd_ms} ms) must be > 0 on slice {idx}"
                )
            frame_dur_s = afd_ms / 1000.0
            ave_decay = (decay_lambda * frame_dur_s) / (1.0 - math.exp(-decay_lambda * frame_dur_s))
            frame_decay_offset_s = (math.log(ave_decay) / decay_lambda) - (frt_ms / 1000.0)
            if frame_decay_offset_s > 1.0:
                raise ValueError(
                    f"Contradictory QIBA FrameReferenceTime ({frt_ms} ms) and ActualFrameDuration ({afd_ms} ms) "
                    f"on slice {idx}: back-calculated scan start is {frame_decay_offset_s:.2f} s later than AcquisitionTime"
                )
            start_s = s_acq_time + frame_decay_offset_s
            start_date = s_acq_date or series_date or study_date
            if start_date is not None:
                day_shift = math.floor(start_s / 86400.0)
                start_date = start_date + timedelta(days=int(day_shift))
                start_s = start_s - day_shift * 86400.0
            qiba_backcomputed_starts.append((start_date, start_s, s_acq_tz_s))

        # Documented vendor scan-start datetime tags: GE (0009,100d) with creator (0009,0010)=='GEMS_PETD_01',
        # Siemens (0071,1022) with creator (0071,0010)=='SIEMENS MED PT'.
        slice_vendor_candidates: list[tuple[date, float, str, int | None]] = []
        for tag, creator_tag, expected_creator, label in (
            (
                (0x0009, 0x100D),
                (0x0009, 0x0010),
                "GEMS_PETD_01",
                "GE Private ScanStartDateTime (0009,100d)",
            ),
            (
                (0x0071, 0x1022),
                (0x0071, 0x0010),
                "SIEMENS MED PT",
                "Siemens Private DecayCorrectionDateTime (0071,1022)",
            ),
        ):
            if tag in ds and ds[tag].value not in (None, ""):
                creator_val = (
                    str(ds[creator_tag].value or "").strip().upper()
                    if creator_tag in ds and ds[creator_tag].value is not None
                    else ""
                )
                if creator_val != expected_creator:
                    raise ValueError(
                        f"Unverified or mismatched private creator {creator_tag}="
                        f"{creator_val!r} for {label} on slice {idx}; expected {expected_creator!r}"
                    )
                v_date, v_time_s, v_tz = parse_dicom_datetime(ds[tag].value, f"slice[{idx}].{label}")
                v_tz_s = parse_dicom_tz_offset_seconds(v_tz, f"slice[{idx}].{label}")
                if v_tz_s is not None and series_tz_s is not None and v_tz_s != series_tz_s:
                    raise ValueError(f"Contradictory timezone in {label} on slice {idx}")
                effective_v_tz_s = v_tz_s if v_tz_s is not None else series_tz_s
                slice_vendor_candidates.append((v_date, v_time_s, label, effective_v_tz_s))
        if len(slice_vendor_candidates) > 1:
            c0 = slice_vendor_candidates[0]
            for c_other in slice_vendor_candidates[1:]:
                diff_v = _timestamp_diff_seconds(
                    c0[0], c0[1], c0[3], c0[2], c_other[0], c_other[1], c_other[3], c_other[2]
                )
                if abs(diff_v) > 1.0:
                    raise ValueError(
                        f"Contradictory vendor scan-start tags on slice {idx}: {c0[2]} vs {c_other[2]} "
                        f"differ by {diff_v:.2f} s"
                    )
        if slice_vendor_candidates:
            vendor_scan_starts.append(slice_vendor_candidates[0])

    if len(acq_dates) > 1 and inj_date is None:
        raise ValueError(
            f"Ambiguous midnight/date crossing across slice AcquisitionDate values {sorted(map(str, acq_dates))} "
            "without RadiopharmaceuticalStartDateTime (0018,1078)"
        )
    if qiba_backcomputed_starts and len(qiba_backcomputed_starts) != len(datasets):
        raise ValueError("FrameReferenceTime/ActualFrameDuration present on only a subset of slices")
    if vendor_scan_starts and len(vendor_scan_starts) != len(datasets):
        raise ValueError("Vendor scan-start datetime tag present on only a subset of slices")

    acq_date_earliest = min(acq_dates) if acq_dates else None

    qiba_start_resolved: tuple[date | None, float, int | None] | None = None
    if qiba_backcomputed_starts:
        q_date0, q_time0, q_tz0 = qiba_backcomputed_starts[0]
        for idx, (q_d, q_t, q_tz) in enumerate(qiba_backcomputed_starts[1:], start=1):
            delta_s = _timestamp_diff_seconds(
                q_d,
                q_t,
                q_tz,
                f"slice[{idx}].QIBA_FrameReferenceTime_start",
                q_date0,
                q_time0,
                q_tz0,
                "slice[0].QIBA_FrameReferenceTime_start",
            )
            if abs(delta_s) > 1.5:
                raise ValueError(
                    f"Inconsistent QIBA FrameReferenceTime back-calculated scan start across slices "
                    f"(slice 0 vs slice {idx} differ by {delta_s:.2f} s)"
                )
        for idx, s_acq_date, s_acq_time, s_acq_tz in slice_acq_records:
            if s_acq_time is None:
                continue
            q_vs_acq_s = _timestamp_diff_seconds(
                q_date0,
                q_time0,
                q_tz0,
                "QIBA FrameReferenceTime back-calculated scan start",
                s_acq_date,
                s_acq_time,
                s_acq_tz,
                f"slice[{idx}].AcquisitionTime",
            )
            if q_vs_acq_s > 1.0:
                raise ValueError(
                    f"Contradictory QIBA FrameReferenceTime scan start later than AcquisitionTime "
                    f"by {q_vs_acq_s:.1f} s on slice {idx}"
                )
        qiba_start_resolved = (q_date0, q_time0, q_tz0)

    vendor_start_resolved: tuple[date, float, str, int | None] | None = None
    if vendor_scan_starts:
        v_date0, v_time0, v_label0, v_tz0 = vendor_scan_starts[0]
        for idx, (v_d, v_t, v_lbl, v_tz) in enumerate(vendor_scan_starts[1:], start=1):
            delta_s = _timestamp_diff_seconds(
                v_d, v_t, v_tz, f"slice[{idx}].{v_lbl}", v_date0, v_time0, v_tz0, f"slice[0].{v_label0}"
            )
            if abs(delta_s) > 1.0:
                raise ValueError(
                    f"Inconsistent {v_label0} across slices (slice 0 vs slice {idx} differ by {delta_s:.2f} s)"
                )
        for idx, s_acq_date, s_acq_time, s_acq_tz in slice_acq_records:
            if s_acq_time is None:
                continue
            v_vs_acq_s = _timestamp_diff_seconds(
                v_date0,
                v_time0,
                v_tz0,
                v_label0,
                s_acq_date,
                s_acq_time,
                s_acq_tz,
                f"slice[{idx}].AcquisitionTime",
            )
            if v_vs_acq_s > 1.0:
                raise ValueError(
                    f"Contradictory {v_label0} later than AcquisitionTime by {v_vs_acq_s:.1f} s on slice {idx}"
                )
        vendor_start_resolved = (v_date0, v_time0, v_label0, v_tz0)

    # Whenever both vendor private scan start and QIBA FrameReferenceTime scan start are present,
    # require them to agree with each other rather than silently preferring one over the other.
    if vendor_start_resolved is not None and qiba_start_resolved is not None:
        v_date, v_time, v_label, v_tz = vendor_start_resolved
        q_date, q_time, q_tz = qiba_start_resolved
        vq_diff = _timestamp_diff_seconds(
            v_date,
            v_time,
            v_tz,
            v_label,
            q_date,
            q_time,
            q_tz,
            "QIBA FrameReferenceTime (0054,1300) + ActualFrameDuration (0018,1242)",
        )
        if abs(vq_diff) > 2.0:
            raise ValueError(
                f"Contradictory {v_label} ({v_time:.1f} s) vs QIBA FrameReferenceTime back-calculated "
                f"scan start ({q_time:.1f} s, delta={vq_diff:.2f} s)"
            )

    ref_tz_s: int | None = series_tz_s
    if series_time_str is not None:
        ref_time_s = parse_dicom_time_seconds(series_time_str, "SeriesTime (0008,0031)")
        ref_time_source = "SeriesTime (0008,0031)"
        if series_date is None and len(acq_dates | ({study_date} if study_date else set())) > 1:
            raise ValueError(
                "Missing SeriesDate (0008,0021) while StudyDate/AcquisitionDate span multiple calendar dates"
            )
        ref_date = series_date or acq_date_earliest or study_date
        if series_date is not None and study_date is not None and series_date < study_date:
            raise ValueError(
                f"Contradictory SeriesDate ({series_date}) earlier than StudyDate ({study_date})"
            )
        # Check every slice for post-acquisition / post-reconstruction SeriesDate+SeriesTime overwrite
        overwritten_slices: list[tuple[int, float]] = []
        non_overwritten_slices: list[tuple[int, float]] = []
        for idx, s_acq_date, s_acq_time, s_acq_tz in slice_acq_records:
            if s_acq_time is None:
                continue
            series_vs_acq_s = _timestamp_diff_seconds(
                series_date or ref_date,
                ref_time_s,
                series_tz_s,
                "SeriesTime (0008,0031)",
                s_acq_date or ref_date,
                s_acq_time,
                s_acq_tz,
                f"slice[{idx}].AcquisitionTime (0008,0032)",
            )
            if series_vs_acq_s > 1.0:
                overwritten_slices.append((idx, series_vs_acq_s))
            else:
                non_overwritten_slices.append((idx, series_vs_acq_s))
        if overwritten_slices:
            if non_overwritten_slices:
                bad_idx, bad_delta = overwritten_slices[0]
                raise ValueError(
                    f"Contradictory SeriesTime (0008,0031) later than earliest AcquisitionTime (0008,0032) "
                    f"by {bad_delta:.1f} s on slice {bad_idx} while preceding later slice acquisitions"
                )
            # Per QIBA SUVbw Profile, when SeriesTime was overwritten after acquisition,
            # fall back to vendor private scan-start datetime (GE 0009,100d / Siemens 0071,1022)
            # or QIBA FrameReferenceTime (0054,1300) + ActualFrameDuration (0018,1242) if available.
            if vendor_start_resolved is not None:
                ref_date, ref_time_s, ref_time_source, ref_tz_s = vendor_start_resolved
            elif qiba_start_resolved is not None:
                ref_date, ref_time_s, ref_tz_s = qiba_start_resolved
                ref_time_source = "QIBA FrameReferenceTime (0054,1300) + ActualFrameDuration (0018,1242)"
            else:
                bad_idx, bad_delta = overwritten_slices[0]
                raise ValueError(
                    f"Contradictory SeriesTime (0008,0031) later than earliest AcquisitionTime (0008,0032) "
                    f"by {bad_delta:.1f} s on slice {bad_idx} of DecayCorrection='START' series"
                )
        else:
            if qiba_start_resolved is not None:
                q_date, q_time, q_tz = qiba_start_resolved
                q_diff = _timestamp_diff_seconds(
                    ref_date,
                    ref_time_s,
                    ref_tz_s,
                    "SeriesTime (0008,0031)",
                    q_date,
                    q_time,
                    q_tz,
                    "QIBA FrameReferenceTime (0054,1300) + ActualFrameDuration (0018,1242)",
                )
                if abs(q_diff) > 2.0:
                    raise ValueError(
                        f"Contradictory SeriesTime ({ref_time_s:.1f} s) vs QIBA FrameReferenceTime back-calculated "
                        f"scan start ({q_time:.1f} s, delta={q_diff:.2f} s)"
                    )
            if vendor_start_resolved is not None:
                v_date, v_time, v_label, v_tz = vendor_start_resolved
                v_diff = _timestamp_diff_seconds(
                    ref_date,
                    ref_time_s,
                    ref_tz_s,
                    "SeriesTime (0008,0031)",
                    v_date,
                    v_time,
                    v_tz,
                    v_label,
                )
                if abs(v_diff) > 2.0:
                    raise ValueError(
                        f"Contradictory SeriesTime ({ref_time_s:.1f} s) vs {v_label} "
                        f"({v_time:.1f} s, delta={v_diff:.2f} s)"
                    )
    elif qiba_start_resolved is not None:
        ref_date, ref_time_s, ref_tz_s = qiba_start_resolved
        ref_time_source = "QIBA FrameReferenceTime (0054,1300) + ActualFrameDuration (0018,1242)"
    elif vendor_start_resolved is not None:
        ref_date, ref_time_s, ref_time_source, ref_tz_s = vendor_start_resolved
    else:
        raise ValueError(
            "Missing SeriesTime (0008,0031) on DecayCorrection='START' series without QIBA "
            "FrameReferenceTime (0054,1300) + ActualFrameDuration (0018,1242) or vendor scan-start datetime; "
            "AcquisitionTime (0008,0032) alone cannot verify scan start"
        )

    # Check date consistency, timezone alignment, and midnight crossing
    known_scan_dates = {d for d in (study_date, series_date, *acq_dates) if d is not None}
    if inj_date is not None:
        if ref_date is None:
            raise ValueError(
                "RadiopharmaceuticalStartDateTime (0018,1078) is present, but scan SeriesDate/AcquisitionDate/StudyDate "
                "is missing; cannot verify calendar date alignment"
            )
        elapsed_seconds = _timestamp_diff_seconds(
            ref_date,
            ref_time_s,
            ref_tz_s,
            ref_time_source,
            inj_date,
            inj_time_s,
            inj_tz_s,
            inj_time_source,
        )
        if elapsed_seconds < -1e-3:
            raise ValueError(
                f"Negative decay elapsed time ({elapsed_seconds:.2f} s): scan reference datetime "
                f"({ref_date} +{ref_time_s:.1f}s) precedes RadiopharmaceuticalStartDateTime ({inj_date} +{inj_time_s:.1f}s)"
            )
        elapsed_seconds = max(0.0, elapsed_seconds)
    else:
        if not known_scan_dates:
            raise ValueError(
                "Missing scan calendar date (SeriesDate/AcquisitionDate/StudyDate) required to verify decay timing"
            )
        if len(known_scan_dates) > 1:
            raise ValueError(
                f"Ambiguous date crossing between StudyDate/SeriesDate/AcquisitionDate "
                f"({sorted(map(str, known_scan_dates))}) without explicit RadiopharmaceuticalStartDateTime (0018,1078)"
            )
        elapsed_seconds = _timestamp_diff_seconds(
            None,
            ref_time_s,
            ref_tz_s,
            ref_time_source,
            None,
            inj_time_s,
            inj_tz_s,
            inj_time_source,
        )
        if elapsed_seconds < -1e-3:
            raise ValueError(
                f"Ambiguous midnight crossing or negative decay elapsed time ({elapsed_seconds:.2f} s): "
                f"scan reference time ({ref_time_s:.1f} s) precedes RadiopharmaceuticalStartTime ({inj_time_s:.1f} s) "
                "without explicit RadiopharmaceuticalStartDateTime (0018,1078) and SeriesDate (0008,0021)"
            )
        elapsed_seconds = max(0.0, elapsed_seconds)

    max_plausible_elapsed_s = max(86400.0, 10.0 * half_life_s)
    if elapsed_seconds > max_plausible_elapsed_s:
        raise ValueError(
            f"Implausible decay elapsed time ({elapsed_seconds:.1f} s > {max_plausible_elapsed_s:.1f} s)"
        )

    decay_factor = 2.0 ** (-elapsed_seconds / half_life_s)
    decayed_dose_bq = dose_bq * decay_factor
    weight_g = weight_kg * 1000.0
    suv_scale_factor = weight_g / decayed_dose_bq

    for idx, rec in enumerate(rescale_records):
        if rec["rwvm"] is not None:
            suv_cal_err = float(rec["max_calibration_diff"]) * suv_scale_factor
            if suv_cal_err > RWVM_MAX_SUV_ERROR_TOL:
                raise ValueError(
                    f"Contradictory RescaleSlope/Intercept and RealWorldValueMappingSequence on slice {idx}: "
                    f"resulting SUVbw error {suv_cal_err:.6g} exceeds {RWVM_MAX_SUV_ERROR_TOL} g/mL"
                )

    return {
        "citation": QIBA_CITATION,
        "conversion_mode": "bqml_to_suvbw",
        "units": units,
        "suv_type": "BW",
        "decay_correction": decay_correction,
        "corrected_image": corrected_tokens_per_slice[0] if corrected_tokens_per_slice else [],
        "slice_count": len(datasets),
        "patient_weight_kg": weight_kg,
        "injected_dose_bq": dose_bq,
        "dose_floor_heuristic_bq": float(min_dose_bq),
        "half_life_seconds": half_life_s,
        "radionuclide": sorted(radionuclide_names)[0] if len(radionuclide_names) == 1 else None,
        "injection_date": inj_date.isoformat() if inj_date else None,
        "injection_time_seconds": inj_time_s,
        "injection_time_source": inj_time_source,
        "reference_date": ref_date.isoformat() if ref_date else None,
        "reference_time_seconds": ref_time_s,
        "reference_time_source": ref_time_source,
        "elapsed_seconds": elapsed_seconds,
        "decay_factor": decay_factor,
        "decayed_dose_bq": decayed_dose_bq,
        "rescale_slope_min": float(min(slopes)),
        "rescale_slope_max": float(max(slopes)),
        "rescale_intercept_min": float(min(intercepts)),
        "rescale_intercept_max": float(max(intercepts)),
        "stored_value_min": float(min(stored_mins)),
        "stored_value_max": float(max(stored_maxs)),
        "max_calibration_diff": float(max(max_cal_diffs)),
        "post_gdcm_slope": 1.0,
        "post_gdcm_intercept": 0.0,
        "suv_scale_factor": suv_scale_factor,
    }


def load_series_headers(
    dicom_dir: Path | str,
    *,
    stop_before_pixels: bool = False,
) -> list[pydicom.Dataset]:
    """Read ordered DICOM datasets for a PET directory (including PixelData by default)."""
    files = ordered_dicom_files(Path(dicom_dir))
    return [pydicom.dcmread(f, stop_before_pixels=stop_before_pixels) for f in files]


def convert_pet_nifti_to_suvbw(
    input_nifti: Path | str,
    dicom_dir: Path | str,
    output_nifti: Path | str,
    *,
    metadata_out: Path | str | None = None,
    min_dose_bq: float = MIN_PLAUSIBLE_DOSE_BQ,
    require_corrected_image: bool = True,
) -> dict[str, Any]:
    """Convert a geometry-preserving PET NIfTI and its source DICOM directory to float32 SUVbw."""
    input_path = Path(input_nifti)
    output_path = Path(output_nifti)
    datasets = load_series_headers(dicom_dir, stop_before_pixels=False)
    meta = compute_suvbw_from_datasets(
        datasets,
        min_dose_bq=min_dose_bq,
        require_corrected_image=require_corrected_image,
        require_pixel_data=True,
    )

    image = sitk.ReadImage(str(input_path))
    arr = sitk.GetArrayFromImage(image).astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"PET image {input_path} contains non-finite voxel values")

    if meta["post_gdcm_slope"] != 1.0 or meta["post_gdcm_intercept"] != 0.0:
        arr = arr * float(meta["post_gdcm_slope"]) + float(meta["post_gdcm_intercept"])
    arr = (arr * float(meta["suv_scale_factor"])).astype(np.float32)

    out_im = sitk.GetImageFromArray(arr)
    out_im.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_im, str(output_path), True)

    # Preserve orientation contract sidecar from input_nifti if present so
    # output_nifti can serve directly as --model-input for RTSTRUCT export.
    for src_sc, dst_sc in (
        (input_sidecar_path(input_path), input_sidecar_path(output_path)),
        (sidecar_path(input_path), sidecar_path(output_path)),
    ):
        if src_sc.is_file() and src_sc.resolve() != dst_sc.resolve():
            shutil.copyfile(src_sc, dst_sc)

    meta["suv_min"] = float(arr.min()) if arr.size else 0.0
    meta["suv_max"] = float(arr.max()) if arr.size else 0.0
    meta["output_nifti"] = str(output_path.name)

    if metadata_out is not None:
        Path(metadata_out).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def convert_dicom_series_to_suvbw_nifti(
    dicom_dir: Path | str,
    output_nifti: Path | str,
    *,
    metadata_out: Path | str | None = None,
    min_dose_bq: float = MIN_PLAUSIBLE_DOSE_BQ,
    require_corrected_image: bool = True,
) -> dict[str, Any]:
    """Read a DICOM PET series directly, normalize orientation, and write float32 SUVbw NIfTI."""
    source = Path(dicom_dir)
    output_path = Path(output_nifti)
    files = ordered_dicom_files(source)
    datasets = [pydicom.dcmread(f) for f in files]
    meta = compute_suvbw_from_datasets(
        datasets,
        min_dose_bq=min_dose_bq,
        require_corrected_image=require_corrected_image,
        require_pixel_data=True,
    )

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image).astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"PET DICOM series {source} contains non-finite voxel values")
    if meta["post_gdcm_slope"] != 1.0 or meta["post_gdcm_intercept"] != 0.0:
        arr = arr * float(meta["post_gdcm_slope"]) + float(meta["post_gdcm_intercept"])
    arr = (arr * float(meta["suv_scale_factor"])).astype(np.float32)

    out_im = sitk.GetImageFromArray(arr)
    out_im.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_im, str(output_path), True)
    contract = normalize_file(output_path)

    meta["orientation_contract"] = {
        "source_axcodes": contract["source_axcodes"],
        "target_axcodes": contract["target_axcodes"],
        "changed": contract["changed"],
    }
    meta["suv_min"] = float(arr.min()) if arr.size else 0.0
    meta["suv_max"] = float(arr.max()) if arr.size else 0.0
    meta["output_nifti"] = str(output_path.name)

    if metadata_out is not None:
        Path(metadata_out).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Support positional (input_nifti, dicom_dir, output_nifti [metadata_json])
    # matching pet_bqml_to_suv1000.py, as well as explicit flags.
    if argv and not argv[0].startswith("-"):
        if len(argv) not in (3, 4):
            raise SystemExit(
                "Usage: pet_to_suvbw.py <input_nifti> <dicom_dir> <output_nifti> [metadata_json]"
            )
        meta = convert_pet_nifti_to_suvbw(
            argv[0],
            argv[1],
            argv[2],
            metadata_out=argv[3] if len(argv) == 4 else None,
        )
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--input-nifti", type=Path, default=None)
        parser.add_argument("--dicom-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--metadata-out", type=Path, default=None)
        args = parser.parse_args(argv)
        if args.input_nifti is not None:
            meta = convert_pet_nifti_to_suvbw(
                args.input_nifti,
                args.dicom_dir,
                args.output,
                metadata_out=args.metadata_out,
            )
        else:
            meta = convert_dicom_series_to_suvbw_nifti(
                args.dicom_dir,
                args.output,
                metadata_out=args.metadata_out,
            )
    print(
        f"PET SUVbw ({meta['conversion_mode']}): scale={meta['suv_scale_factor']:.6g} "
        f"max={meta['suv_max']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

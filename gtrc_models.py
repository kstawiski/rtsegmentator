#!/usr/bin/env python3
"""GTRC-Net PSMA PET/CT and FDG PET/CT model contracts, preflight gates, and job orchestration."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk

from dicom_series import ordered_dicom_files
from pet_to_suvbw import convert_pet_nifti_to_suvbw

APP_ROOT = Path(os.environ.get("RTSEG_APP_ROOT", Path(__file__).resolve().parent))
MODEL_ROOT = Path(os.environ.get("RTSEG_MODEL_ROOT", "/models"))

MARKER_SCHEMA_VERSION = 2
SUPPORTED_DEVICES: tuple[str, ...] = ("H100", "P6000")
GTRC_RUNTIME_LOCK_REL = "runtime-envs/gtrc/requirements.lock.txt"
GTRC_COMPLETION_FILENAME = "gtrc_completion.json"
GTRC_COMPLETION_REL = f"validation/{GTRC_COMPLETION_FILENAME}"

GTRC_TOLERANCE_CONTRACT: dict[str, float] = {
    "suvbw_max_abs_err": 1e-3,
    "suvbw_mean_abs_err": 1e-5,
    "suvbw_p999_abs_err": 1e-3,
    "ct_hu_max_abs_err": 0.1,
    "ct_hu_mean_abs_err": 0.05,
    "scale_factor_rtol": 1e-6,
    "elapsed_time_atol_s": 1e-3,
    "grid_origin_mm_tol": 1e-4,
    "grid_spacing_mm_tol": 1e-4,
    "grid_direction_cos_tol": 1e-5,
    "threshold_abs_tol": 1e-12,
    "volume_ml_abs_tol": 1e-4,
    "suv_stat_abs_tol": 1e-4,
    "hu_stat_abs_tol": 1e-3,
    "roundtrip_min_dice": 0.95,
    "roundtrip_median_dice": 0.95,
}

GTRC_SOURCE = {
    "family": "gtrc",
    "source_dir": "sources/gtrc_net",
    "source_url": "https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained",
    "source_commit": "9db02ddd569fd4e82a2caf8973aba6741d5cd4a8",
    "runtime_env": "gtrc",
    "weights_root": "weights/gtrc_net",
    "runner_script": "run_gtrc.py",
    "paper_doi": "10.2967/jnumed.125.270077",
    "fixture_zenodo_record": "18150034",
}

MIN_SUV_THRESHOLD = 0.1
MAX_SUV_THRESHOLD = 100.0
AXIAL_COVERAGE_MIN_TOLERANCE_MM = 5.0

GTRC_CODE_FILES: tuple[str, ...] = (
    "gtrc_models.py",
    "run_gtrc.py",
    "pet_to_suvbw.py",
    "run_task.py",
    "dicom_series_to_nifti.py",
    "convert_validate_rtstruct.py",
    "orientation_contract.py",
    "labels/gtrc_psma_petct_ttb.json",
    "labels/gtrc_fdg_petct_ttb.json",
)

GTRC_TASKS: dict[str, dict[str, Any]] = {
    "gtrc_psma_petct_ttb": {
        **GTRC_SOURCE,
        "tracer": "PSMA",
        "dataset_id": 881,
        "dataset_name": "Dataset881_PSMA_PET",
        "model_folder": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres",
        "consensus_subdir": "data/PSMA/subregion_metrics",
        "default_suv_threshold": 3.0,
        "suv_threshold_required": False,
        "suv_threshold_bounds": [MIN_SUV_THRESHOLD, MAX_SUV_THRESHOLD],
        "suv_threshold_source_default": "published-default",
        "suv_threshold_source": (
            "Published PSMA PET static SUV>=3.0 cutoff (Jackson et al., J Nucl Med "
            "2025;66(11):1811-1817; upstream GTRC.py default --suv_threshold 3.0)"
        ),
        "suv_threshold_definition": (
            "Cases were delineated by nuclear medicine physicians and showed the extent of "
            "whole-body metastatic disease using either a fixed threshold (SUV of >=3 for PSMA "
            "PET and SPECT imaging) or a variable liver-based threshold (1.5x mean + 2x SD of "
            "liver background for [18F]FDG PET imaging) (Jackson et al., J Nucl Med 2025;66(11):1811-1817)."
        ),
        "roi_name": "GTRC_PSMA_TTB",
        "label_map": "labels/gtrc_psma_petct_ttb.json",
        "ct_normalization": {
            "scheme": "CTNormalization",
            "percentile_00_5": -1000.0,
            "percentile_99_5": 920.0,
            "mean": 32.0718879699707,
            "std": 212.49000549316406,
        },
        "fixture_case": "train_0001/PSMA",
    },
    "gtrc_fdg_petct_ttb": {
        **GTRC_SOURCE,
        "tracer": "FDG",
        "dataset_id": 882,
        "dataset_name": "Dataset882_FDG_PET",
        "model_folder": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres",
        "consensus_subdir": "data/FDG/subregion_metrics",
        "default_suv_threshold": None,
        "suv_threshold_required": True,
        "suv_threshold_bounds": [MIN_SUV_THRESHOLD, MAX_SUV_THRESHOLD],
        "suv_threshold_source_default": None,
        "suv_threshold_source": (
            "User-entered patient-specific FDG SUVbw threshold required per scan "
            "(JNM 2025;66(11):1811-1817: 1.5x mean + 2x SD of liver background; "
            "DEEP-PSMA challenge pages: liver mean + 2x SD, typically 2.5-5)"
        ),
        "suv_threshold_definition": (
            "Cases were delineated by nuclear medicine physicians and showed the extent of "
            "whole-body metastatic disease using either a fixed threshold (SUV of >=3 for PSMA "
            "PET and SPECT imaging) or a variable liver-based threshold (1.5x mean + 2x SD of "
            "liver background for [18F]FDG PET imaging) (Jackson et al., J Nucl Med 2025;66(11):1811-1817). "
            "The DEEP-PSMA challenge pages describe liver mean + 2x SD (typically 2.5-5)."
        ),
        "roi_name": "GTRC_FDG_TTB",
        "label_map": "labels/gtrc_fdg_petct_ttb.json",
        "ct_normalization": {
            "scheme": "CTNormalization",
            "percentile_00_5": -1000.0,
            "percentile_99_5": 1039.0,
            "mean": 67.15141296386719,
            "std": 188.75750732421875,
        },
        "fixture_case": "train_0001/FDG",
    },
}
TASKS = GTRC_TASKS

GTRC_WEIGHT_FILES: dict[str, list[dict[str, Any]]] = {
    "gtrc_psma_petct_ttb": [
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json",
            "bytes": 211,
            "sha256": "af3b21d0dc2c1effe6d7fef518d825b3a888da1606c99288b4ea3627673ba357",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json",
            "bytes": 108346,
            "sha256": "1b71ca18b7c41295d335d926fdbd5880446e04994f5eebaf8ad89d977ba55a20",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json",
            "bytes": 10959,
            "sha256": "c218f54df10868415a8aaeadf2f27a9a27df2f9bc9873b0e95e2e8c5849627e1",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
            "bytes": 245174877,
            "sha256": "f3c6d3c71e49141d6b166faadeff6aa95355d5889d8886455e9f6ba53808bffb",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/debug.json",
            "bytes": 13687,
            "sha256": "8fe9ecd3f6ab1d3ba717baac088b6d8ce3e1e01f287d6476b77ed0d89a98f451",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth",
            "bytes": 245174749,
            "sha256": "af63ee6b05f6ad555e02a68c5af60ca8e67f2966fa52211afca680a6eb73d3a6",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/debug.json",
            "bytes": 13662,
            "sha256": "f8d0e4d97821ff76788636e06c2fec0df1b7dc5b8b8be4738792b49ab62dc385",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth",
            "bytes": 245174941,
            "sha256": "69fa2676e63219b30017fa91a6be912d22fb25e7866ac9ad18fcd5371cceff92",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/debug.json",
            "bytes": 13629,
            "sha256": "6c538acd99b7a1087df967f64a2494a06048886f1660100285a3d5b791fd5b52",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth",
            "bytes": 245174813,
            "sha256": "864c644f0fbcb38c4221dd1d784e3cfb47951ec75dec441507e566735ea7beb2",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/debug.json",
            "bytes": 13641,
            "sha256": "3b48458b0bb7b8586409081478d559214a7889055db7cc478d59ad2157faff23",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/checkpoint_final.pth",
            "bytes": 245174877,
            "sha256": "2ac1ea9bdcfea7b1b691978ef7a8380363db8da1643264a7a992e45f918592c1",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset881_PSMA_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/debug.json",
            "bytes": 13629,
            "sha256": "f34a52b85ea87cbbcdef0b9a138177aaace3329fbf05c0797df7765a1d0ed9ac",
        },
        {
            "path": "weights/gtrc_net/data/PSMA/subregion_metrics/fold_0/consensus_model.keras",
            "bytes": 74518,
            "sha256": "e36c4543b4a41736d9d0a0039394477334a93a5247cbf5bf4b24bd50c5c82a7a",
        },
        {
            "path": "weights/gtrc_net/data/PSMA/subregion_metrics/fold_1/consensus_model.keras",
            "bytes": 74518,
            "sha256": "ae6942f2e91d9ca85779fb073a97f96d700d20201912619e2fafe60af565df93",
        },
        {
            "path": "weights/gtrc_net/data/PSMA/subregion_metrics/fold_2/consensus_model.keras",
            "bytes": 74518,
            "sha256": "23eae281fe236b8d604fed8d02ff51f4c13000c40629910c3ffd442904f540ca",
        },
        {
            "path": "weights/gtrc_net/data/PSMA/subregion_metrics/fold_3/consensus_model.keras",
            "bytes": 74518,
            "sha256": "336e3ff942f4f8cfa94c2a900d08478d863a792538e89be5019c3f284e0ae0da",
        },
        {
            "path": "weights/gtrc_net/data/PSMA/subregion_metrics/fold_4/consensus_model.keras",
            "bytes": 74518,
            "sha256": "57d764f25bb0bbefc7ffdc8e180fa659233eb221d7e1dd5e574154a4ce3defcb",
        },
    ],
    "gtrc_fdg_petct_ttb": [
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json",
            "bytes": 211,
            "sha256": "99f65abe4e0b3d1c84c62020589f455ba18b31346350f8e56895a53b6c39f9cf",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json",
            "bytes": 59411,
            "sha256": "f7fb73ce9087ac46418510f38f00840a997fb33337f88048934daf2559254583",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json",
            "bytes": 10961,
            "sha256": "6a6c9ca4fba436333fc3d124424b2ed181d3eee1ca0baa12d833372546bd25d1",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
            "bytes": 245174877,
            "sha256": "ec260e12c8590cc91bd703917ce6f82f5ef8c519f3a814f48b837d76cd1cb4c7",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/debug.json",
            "bytes": 13640,
            "sha256": "ee9ca376ff2a12544d49f6dfafc5a70cd34081b57d9d214ae568d1d243ecaa42",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth",
            "bytes": 245174749,
            "sha256": "3cdbebfbaef463766329bfe7fc6766bd408c555b0d0f727ddefc2f4bb4130e32",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/debug.json",
            "bytes": 13628,
            "sha256": "6cf4cb9603cf59981390232f0b6248a6770e7be1acc55b4d51cb998e7f51886f",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth",
            "bytes": 245174813,
            "sha256": "fd94cf6a0ce66227bb0b922219f7ac1bf675232a126024e4c0ca6b6b44363a4e",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/debug.json",
            "bytes": 13640,
            "sha256": "cfbbd0b094896e2a89b5e5f5130a7ea830689368708f306f36379287b69acd4d",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth",
            "bytes": 245174813,
            "sha256": "a3d921d2dae37ad64a2fe8db8fedd88a7a27fcd1406f2993aa440c996677b0fc",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/debug.json",
            "bytes": 13628,
            "sha256": "922a4ff892873be8dfcc268847992f388706734b9ba4d895be809d3f56741309",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/checkpoint_final.pth",
            "bytes": 245174941,
            "sha256": "415230076589dd8db8f890869507487eb4b49c806490a5b4c3be5ed799b4b08b",
        },
        {
            "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset882_FDG_PET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/debug.json",
            "bytes": 13640,
            "sha256": "46c7ed0fc3e5e479c1d6d52b2598e25cebdbfce89e68cb597ad227281facc46f",
        },
        {
            "path": "weights/gtrc_net/data/FDG/subregion_metrics/fold_0/consensus_model.keras",
            "bytes": 74518,
            "sha256": "0993e78924116f65e318fe702606df60d2d7db698b941aefffbc1a3636e1c5b2",
        },
        {
            "path": "weights/gtrc_net/data/FDG/subregion_metrics/fold_1/consensus_model.keras",
            "bytes": 74518,
            "sha256": "5f910429995b0c1ed387bc186a350a3cee2dfd7616c34dd5ab2d9953e6f76599",
        },
        {
            "path": "weights/gtrc_net/data/FDG/subregion_metrics/fold_2/consensus_model.keras",
            "bytes": 74518,
            "sha256": "f6c6093c4c176489b3ffcf7888056e3a96ce97073537db4a2cac36882bbf86b0",
        },
        {
            "path": "weights/gtrc_net/data/FDG/subregion_metrics/fold_3/consensus_model.keras",
            "bytes": 74518,
            "sha256": "1b09e76910c609ac2704cb758525a847f9d56d9df7bbe55cc2571ff61fa5e234",
        },
        {
            "path": "weights/gtrc_net/data/FDG/subregion_metrics/fold_4/consensus_model.keras",
            "bytes": 74518,
            "sha256": "8e854b5fcf3db4a83959296ac73de63c97394c7c42b44dba80dcd94d8effa5b4",
        },
    ],
}

# Retained in layout proposal only; Lu-PSMA SPECT/CT is out of scope for active portal tasks.
LUPSMA_WEIGHT_FILES: list[dict[str, Any]] = [
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json",
        "bytes": 211,
        "sha256": "54d6351454d702cd9c62ab8383da0fcbfb1b6e4bd549cc5a1d9edcd455f017fb",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json",
        "bytes": 77938,
        "sha256": "d53f719853a686c77ed02a4db786434d36deb30f37f58fcd91ba23997727f7a7",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json",
        "bytes": 10940,
        "sha256": "bad7d1deee3c85712301c064d8123d24fe4a6abdf8916277c8a00fd2a5a3e383",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
        "bytes": 245174749,
        "sha256": "91d322263c6f15b0778026b1a91936bed2a46d204bb2ec75e2f7982210a6e607",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/debug.json",
        "bytes": 13620,
        "sha256": "5efa64a4cbe83a64189617b475a3e2e46ef06dcface66e5cd19de1cd8d5d52ab",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth",
        "bytes": 245174813,
        "sha256": "df763b3ddd65d97298dc668fb69c6aa4e30b04e1c7b6f887ff759e88ca5311cb",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/debug.json",
        "bytes": 13620,
        "sha256": "dd0b8d2b54bef091525116c5132303304ff4897a0fdb23524cbf6da49503276f",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth",
        "bytes": 245174749,
        "sha256": "50b96fbe2645f05f65004f01284b8a5db2d1443de52dd2f7f3d13731d7b96e93",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/debug.json",
        "bytes": 13620,
        "sha256": "d987b3acbb8097bb76fa03128c8797eb0c11b036da6ef176f7253ad5a8697379",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth",
        "bytes": 245174749,
        "sha256": "d074e84bc43c0066bba4bde63c51f1e837afe9ca09de4503ff7b68ca1471ff30",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/debug.json",
        "bytes": 13620,
        "sha256": "ad52ba2ac94351f313295778f634e197ae6e05c064cd7fc7cf235efd3cebdd88",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/checkpoint_final.pth",
        "bytes": 245174877,
        "sha256": "5aa30197104073c98ed7338e6fd0119651e87b5b8cccc8ede5cdcc51e56941f5",
    },
    {
        "path": "weights/gtrc_net/data/nnUNet_data/results/Dataset883_LUPSMA_SPECT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/debug.json",
        "bytes": 13620,
        "sha256": "301ce4b0ca912ef0be0026a3d368b4b1ca086b255334abf8ca79142c49f7e8ba",
    },
    {
        "path": "weights/gtrc_net/data/LUPSMA/subregion_metrics/fold_0/consensus_model.keras",
        "bytes": 74518,
        "sha256": "83498333638921cffaceb33e2696be8a5a24769c65bd68e56ebd7f3cc63ab2d1",
    },
    {
        "path": "weights/gtrc_net/data/LUPSMA/subregion_metrics/fold_1/consensus_model.keras",
        "bytes": 74518,
        "sha256": "6f70faa8feee9db1678134bc95cf2184e3938e32dfe60bd77a990998e60ada27",
    },
    {
        "path": "weights/gtrc_net/data/LUPSMA/subregion_metrics/fold_2/consensus_model.keras",
        "bytes": 74518,
        "sha256": "f8d41e95bd3fef010f7922db58539217500432f2831933ac2f92d9f30e424a45",
    },
    {
        "path": "weights/gtrc_net/data/LUPSMA/subregion_metrics/fold_3/consensus_model.keras",
        "bytes": 74518,
        "sha256": "973f29b426c4f2c251f0ef9e2384ce43a35803a450c2612067d925562b431858",
    },
    {
        "path": "weights/gtrc_net/data/LUPSMA/subregion_metrics/fold_4/consensus_model.keras",
        "bytes": 74518,
        "sha256": "3a598b19493a9d0e2c5a917c22575f553d09642fa0d7f120811b9b0e1ffc6dcc",
    },
]

_SHA256_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}
_CUDA_PROBE_CACHE: dict[tuple[str, str], tuple[bool, str, dict[str, Any]]] = {}
GTRC_DATASET_LABELS = {"background": 0, "ttb": 1, "norm": 2}
GTRC_CHANNEL_NAMES = {"0": "noNorm", "1": "CT"}


def validate_gtrc_dataset(data: dict[str, Any]) -> None:
    if data.get("labels") != GTRC_DATASET_LABELS or data.get("channel_names") != GTRC_CHANNEL_NAMES:
        raise ValueError("GTRC-Net dataset.json channel/label contract differs from pinned release")
    if data.get("file_ending") != ".nii.gz":
        raise ValueError("GTRC-Net dataset.json requires .nii.gz file_ending")


def is_gtrc_task(task: str) -> bool:
    return task in GTRC_TASKS


def validate_suv_threshold_value(raw: Any, task: str) -> float:
    """Validate that a user-supplied SUVbw threshold is a finite float in [MIN_SUV_THRESHOLD, MAX_SUV_THRESHOLD]."""
    if isinstance(raw, bool) or raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValueError(
            f"{task} requires a numeric SUVbw threshold in [{MIN_SUV_THRESHOLD}, {MAX_SUV_THRESHOLD}]"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid SUVbw threshold {raw!r} for {task}; expected a finite number in "
            f"[{MIN_SUV_THRESHOLD}, {MAX_SUV_THRESHOLD}]"
        ) from exc
    if not np.isfinite(value) or not (MIN_SUV_THRESHOLD <= value <= MAX_SUV_THRESHOLD):
        raise ValueError(
            f"SUVbw threshold for {task} must be finite and within [{MIN_SUV_THRESHOLD}, {MAX_SUV_THRESHOLD}], "
            f"got {raw!r}"
        )
    return float(value)


def validate_suv_threshold(task: str, raw: Any) -> float:
    """Validate that `raw` is a finite SUVbw threshold in `[MIN_SUV_THRESHOLD, MAX_SUV_THRESHOLD]`."""
    return validate_suv_threshold_value(raw, task)


def _validate_threshold_and_source(
    task: str,
    raw_threshold: Any,
    raw_source: Any = None,
) -> tuple[float, str]:
    """Validate `(suv_threshold, source)`, accepting `published-default` only for PSMA at 3.0."""
    val = validate_suv_threshold_value(raw_threshold, task)
    if raw_source is None or str(raw_source).strip() == "":
        src = "user-entered"
    else:
        src = str(raw_source).strip()
    if src not in ("user-entered", "published-default"):
        raise ValueError(f"Invalid SUV threshold source {src!r} for {task}")
    if src == "published-default":
        if task != "gtrc_psma_petct_ttb" or abs(val - 3.0) > GTRC_TOLERANCE_CONTRACT["threshold_abs_tol"]:
            raise ValueError(
                f"SUV threshold source 'published-default' is only valid for gtrc_psma_petct_ttb at 3.0, "
                f"got task={task!r}, suv_threshold={val!r}"
            )
    return val, src


def resolve_job_suv_threshold(
    task: str,
    job: Path | str | dict[str, Any] | None = None,
    *,
    explicit_threshold: Any = None,
    explicit_source: str | None = None,
) -> tuple[float, str]:
    """Resolve and validate `(suv_threshold, source)` for `task` from explicit argument or `job/job.json`.

    - For `gtrc_fdg_petct_ttb`: there is no valid fixed default; a user-entered threshold in
      `[MIN_SUV_THRESHOLD, MAX_SUV_THRESHOLD]` is required and recorded with `source="user-entered"`.
    - For `gtrc_psma_petct_ttb`: defaults to `3.0` with `source="published-default"`, or uses a
      validated override with `source="user-entered"` (or `published-default` only when `value == 3.0`).
    """
    if task not in GTRC_TASKS:
        raise ValueError(f"Unknown GTRC task: {task!r}")
    spec = GTRC_TASKS[task]

    if explicit_threshold is not None:
        return _validate_threshold_and_source(task, explicit_threshold, explicit_source)

    job_meta: dict[str, Any] | None = None
    if isinstance(job, dict):
        job_meta = job
    elif job is not None:
        job_json = Path(job) / "job.json"
        if job_json.is_file():
            try:
                loaded = json.loads(job_json.read_text())
            except Exception as exc:
                raise ValueError(f"Invalid job.json in {job}: {exc}") from exc
            if isinstance(loaded, dict):
                job_meta = loaded

    if isinstance(job_meta, dict):
        task_opts = job_meta.get("task_options")
        if isinstance(task_opts, dict) and task in task_opts:
            opt = task_opts[task]
            if isinstance(opt, dict) and "suv_threshold" in opt and opt["suv_threshold"] is not None:
                return _validate_threshold_and_source(task, opt["suv_threshold"], opt.get("source"))
        if "suv_thresholds" in job_meta and isinstance(job_meta["suv_thresholds"], dict):
            raw_entry = job_meta["suv_thresholds"].get(task)
            if isinstance(raw_entry, dict) and raw_entry.get("value") is not None:
                return _validate_threshold_and_source(task, raw_entry["value"], raw_entry.get("source"))
            elif raw_entry is not None and not isinstance(raw_entry, dict):
                return _validate_threshold_and_source(task, raw_entry, "user-entered")
        if "suv_threshold" in job_meta and job_meta["suv_threshold"] is not None:
            raw_top = job_meta["suv_threshold"]
            if isinstance(raw_top, dict):
                return _validate_threshold_and_source(task, raw_top.get("value"), raw_top.get("source"))
            return _validate_threshold_and_source(task, raw_top, job_meta.get("suv_threshold_source"))

    if spec["default_suv_threshold"] is not None and not spec["suv_threshold_required"]:
        if explicit_source is not None and str(explicit_source).strip() not in ("", "published-default"):
            raise ValueError(
                f"SUV threshold source {explicit_source!r} requires an explicit suv_threshold for {task}"
            )
        return float(spec["default_suv_threshold"]), "published-default"

    raise ValueError(
        f"{task} requires a patient-specific FDG SUV threshold (user-entered in "
        f"[{MIN_SUV_THRESHOLD}, {MAX_SUV_THRESHOLD}], JNM 2025;66(11):1811: 1.5x mean + 2x SD of liver "
        "background; DEEP-PSMA challenge: liver mean + 2x SD, typically 2.5-5); no fixed default exists"
    )


def artifact_names(task: str) -> list[str]:
    """Finite, task-owned download allowlist; all prefixed with `<task>_RTSTRUCT.dcm.` for SSH retrieval."""
    if task not in GTRC_TASKS:
        return []
    return [
        f"{task}_RTSTRUCT.dcm.lesions.json",
        f"{task}_RTSTRUCT.dcm.provenance.json",
        f"{task}_RTSTRUCT.dcm.ttb_mask.nii.gz",
        f"{task}_RTSTRUCT.dcm.nnunet_labels.nii.gz",
        f"{task}_RTSTRUCT.dcm.subregions.nii.gz",
    ]


def model_folder(task: str) -> Path:
    return Path(GTRC_TASKS[task]["model_folder"])


def runtime_python(env_name: str = "gtrc", model_root: Path | None = None) -> Path:
    root = Path(model_root or MODEL_ROOT)
    return root / "runtime-envs" / env_name / "bin/python"


def weight_files(task: str, model_root: Path | None = None) -> list[Path]:
    if model_root is None:
        return [Path(item["path"]) for item in GTRC_WEIGHT_FILES[task]]
    root = Path(model_root)
    return [root / item["path"] for item in GTRC_WEIGHT_FILES[task]]


def requirements(task: str) -> list[str]:
    """Return preflight requirement strings following `model_preflight.py` conventions."""
    spec = GTRC_TASKS[task]
    return [
        f"@{spec['runner_script']}",
        "@gtrc_models.py",
        "@pet_to_suvbw.py",
        "@dicom_series_to_nifti.py",
        "@convert_validate_rtstruct.py",
        f"@{spec['label_map']}",
        f"runtime-envs/{spec['runtime_env']}/bin/python",
        f"runtime-envs/{spec['runtime_env']}/bin/nnUNetv2_predict",
        GTRC_RUNTIME_LOCK_REL,
        f"{spec['source_dir']}/GTRC.py",
        f"{spec['source_dir']}/GTRC_Infer.py",
        f"{spec['source_dir']}/gtrc_utils.py",
        *(item["path"] for item in GTRC_WEIGHT_FILES[task]),
        f"validation/{task}.json",
        f"orientation-validation/{task}.json",
        GTRC_COMPLETION_REL,
    ]


def file_sha256(path: Path | str, *, uncached: bool = False) -> str:
    """SHA-256 of a file, cached per process by resolved path, device/inode, size, and mtime/ctime."""
    p = Path(path)
    stat = p.stat()
    key = (
        str(p.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if uncached or key not in _SHA256_CACHE:
        digest = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        _SHA256_CACHE[key] = digest.hexdigest()
    return _SHA256_CACHE[key]


sha256_file = file_sha256


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def device_key(gpu: dict[str, Any] | None) -> str:
    """Map live or recorded GPU metadata to a supported acceptance device key ('H100' or 'P6000')."""
    if not isinstance(gpu, dict):
        raise ValueError("Missing GPU device metadata")
    name = str(gpu.get("name") or gpu.get("gpu_name") or gpu.get("device_name") or "").strip()
    cap_raw = (
        gpu.get("compute_capability")
        if gpu.get("compute_capability") is not None
        else (
            gpu.get("device_capability")
            if gpu.get("device_capability") is not None
            else gpu.get("cuda_device_capability")
        )
    )
    try:
        cap = [int(x) for x in cap_raw] if isinstance(cap_raw, (list, tuple)) and len(cap_raw) == 2 else []
    except (TypeError, ValueError):
        cap = []
    if cap == [9, 0] and "H100" in name.upper():
        return "H100"
    if cap == [6, 1] and "P6000" in name.upper():
        return "P6000"
    raise ValueError(f"Unsupported acceptance GPU device {name!r} with compute capability {cap!r}")


@contextmanager
def marker_lock(folder: Path) -> Iterator[None]:
    """Acquire an exclusive file lock on `folder/.device-markers.lock` for atomic per-device updates."""
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = folder / ".device-markers.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_device_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read schema-2 per-device marker records from `path`, returning `{device_key: record}`."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != MARKER_SCHEMA_VERSION:
        return {}
    devices = data.get("devices")
    if not isinstance(devices, dict):
        return {}
    return {
        key: dict(val)
        for key, val in devices.items()
        if key in SUPPORTED_DEVICES and isinstance(val, dict)
    }


def write_device_records(path: Path, devices: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Atomically write schema-2 per-device marker envelope to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "devices": {key: devices[key] for key in SUPPORTED_DEVICES if key in devices},
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return envelope


def _remove_devices_from_envelope_file(path: Path, target_devices: tuple[str, ...]) -> None:
    devices = read_device_records(path)
    removed = False
    for dev in target_devices:
        if dev in devices:
            del devices[dev]
            removed = True
    if removed:
        if devices:
            write_device_records(path, devices)
        else:
            path.unlink(missing_ok=True)
    elif path.is_file() and not devices:
        path.unlink(missing_ok=True)


def invalidate_device(
    root: Path | str,
    task: str | None = None,
    device: str | None = None,
    *,
    validation_dir: Path | str | None = None,
    orientation_dir: Path | str | None = None,
) -> None:
    """Remove `device` (or all supported devices when `device is None`) from completion and task marker envelopes.

    Always invalidates `validation/gtrc_completion.json` first so the device is immediately non-runnable
    before individual task validation and orientation records are removed.
    """
    root_path = Path(root)
    val_folder = Path(validation_dir) if validation_dir is not None else (root_path / "validation")
    ori_folder = Path(orientation_dir) if orientation_dir is not None else (root_path / "orientation-validation")
    target_devices = (device,) if device is not None else SUPPORTED_DEVICES
    target_tasks = (task,) if task is not None else tuple(GTRC_TASKS.keys())

    if val_folder.exists():
        with marker_lock(val_folder):
            _remove_devices_from_envelope_file(val_folder / GTRC_COMPLETION_FILENAME, target_devices)
            for t_name in target_tasks:
                _remove_devices_from_envelope_file(val_folder / f"{t_name}.json", target_devices)

    if ori_folder.exists():
        with marker_lock(ori_folder):
            for t_name in target_tasks:
                _remove_devices_from_envelope_file(ori_folder / f"{t_name}.json", target_devices)


def store_device_marker(
    path: Path,
    device: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Atomically update `devices[device]` in `path` under `marker_lock`, preserving other devices."""
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device key {device!r}; expected one of {SUPPORTED_DEVICES}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with marker_lock(path.parent):
        devices = read_device_records(path)
        entry = dict(record)
        entry["device_key"] = device
        devices[device] = entry
        return write_device_records(path, devices)


def build_completion_record(
    *,
    device: str,
    run_id: str,
    gpu: dict[str, Any],
    source_commit: str,
    code_sha256: str,
    runtime_lock_sha256: str,
    task_records: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    completed_at_utc: str = "2026-10-22T00:00:00+00:00",
) -> dict[str, Any]:
    """Build a per-device completion record binding `run_id` and the SHA-256 of all 4 GTRC task records."""
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device key {device!r}; expected one of {SUPPORTED_DEVICES}")
    if set(task_records.keys()) != set(GTRC_TASKS.keys()):
        raise ValueError(
            f"Completion record requires all GTRC tasks {sorted(GTRC_TASKS.keys())}, "
            f"got {sorted(task_records.keys())}"
        )
    tasks_binding: dict[str, dict[str, str]] = {}
    for t_name in GTRC_TASKS:
        val_rec, ori_rec = task_records[t_name]
        v_entry = dict(val_rec)
        v_entry["device_key"] = device
        o_entry = dict(ori_rec)
        o_entry["device_key"] = device
        tasks_binding[t_name] = {
            "validation_sha256": canonical_json_sha256(v_entry),
            "orientation_sha256": canonical_json_sha256(o_entry),
        }
    return {
        "family": "gtrc",
        "device_key": device,
        "run_id": run_id,
        "result": "passed",
        "status": "validated",
        "passed": True,
        "valid": True,
        "inference_mode": "production",
        "cpu_smoke": False,
        "gpu": dict(gpu),
        "source_commit": source_commit,
        "code_sha256": code_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "tasks": tasks_binding,
        "completed_at_utc": completed_at_utc,
    }


def store_completion_marker(
    val_dir_or_path: Path,
    device: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Write `validation/gtrc_completion.json` for `device` under `marker_lock`."""
    path = (
        val_dir_or_path
        if val_dir_or_path.name.endswith(".json")
        else (val_dir_or_path / GTRC_COMPLETION_FILENAME)
    )
    return store_device_marker(path, device, record)


def marker_has_failure(value: Any) -> bool:
    """A passing marker cannot conceal failed checks, smoke/stub flags, errors, or waived deviations."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"passed", "predeclared_passed", "valid", "roundtrip_valid"} and item is not True:
                return True
            if key in {"result", "status"} and item not in {"passed", "validated", "nonempty"}:
                return True
            if (
                key
                in {
                    "cpu_smoke",
                    "smoke",
                    "stub",
                    "mock",
                    "synthetic_stub",
                    "precomputed_prediction",
                    "precomputed_nnunet_pred",
                    "error",
                    "errors",
                    "fatal_error",
                    "failure",
                    "failures",
                    "failed_checks",
                    "unresolved_failures",
                    "deviations",
                    "waived",
                }
                and item
            ):
                return True
            if marker_has_failure(item):
                return True
    elif isinstance(value, list):
        return any(marker_has_failure(item) for item in value)
    return False


def expected_code_hashes(
    task_or_root: str | Path | None = None,
    app_root: Path | str | None = None,
) -> dict[str, str]:
    """Compute SHA-256 digests of the GTRC runner, adapter, converter, and label map files."""
    if isinstance(task_or_root, str) and task_or_root in GTRC_TASKS:
        app = Path(app_root or APP_ROOT)
    elif task_or_root is not None:
        app = Path(task_or_root)
    else:
        app = Path(app_root or APP_ROOT)
    hashes: dict[str, str] = {}
    for rel in GTRC_CODE_FILES:
        target = app / rel
        if not target.is_file():
            raise ValueError(f"Missing required GTRC code/contract file: {target}")
        hashes[rel] = file_sha256(target)
    return hashes


def expected_code_digest(
    task_or_root: str | Path | None = None,
    app_root: Path | str | None = None,
) -> str:
    hashes = expected_code_hashes(task_or_root, app_root)
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_lock_file_path(model_root: Path | str | None = None) -> Path:
    root = Path(model_root or MODEL_ROOT)
    return root / GTRC_RUNTIME_LOCK_REL


def expected_runtime_lock_sha256(model_root: Path | str | None = None) -> str:
    """Return the SHA-256 of the installed GTRC runtime lock file, raising ValueError if missing."""
    lock_path = runtime_lock_file_path(model_root)
    if not lock_path.is_file():
        raise ValueError(f"Missing required installed runtime lock file: {lock_path}")
    return file_sha256(lock_path)


def parse_sm_version(token: str) -> tuple[int, int] | None:
    """Parse a CUDA arch token such as `sm_60` or `compute_90a` into `(major, minor)`."""
    tok = str(token or "").strip().lower()
    m = re.fullmatch(r"(?:sm|compute)_(\d+)(\d)[a-z]*", tok)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def is_cuda_arch_compatible(
    capability: tuple[int, int] | list[int] | None,
    arch_list: list[str] | tuple[str, ...],
) -> bool:
    """Return True iff compiled CUDA arch targets (`sm_XX` / `compute_XX`) support `capability`."""
    if not capability or len(capability) != 2:
        return False
    try:
        major = int(capability[0])
        minor = int(capability[1])
    except (TypeError, ValueError):
        return False
    if major <= 0 or minor < 0:
        return False

    targets: set[tuple[int, int]] = set()
    for token in arch_list or ():
        parsed = parse_sm_version(str(token))
        if parsed is not None:
            targets.add(parsed)

    if (major, minor) in targets:
        return True
    # Within the same major compute capability family, cubins compiled for sm_<major><m_minor>
    # with m_minor <= minor execute natively (e.g. sm_60 cubin on sm_61 Quadro P6000; sm_80 on sm_86).
    if any(t_maj == major and t_min <= minor for t_maj, t_min in targets):
        return True
    return False


def runtime_supports_device_arch(
    arch_list: list[str] | tuple[str, ...],
    capability: tuple[int, int] | list[int] | None,
) -> bool:
    """Alias for `is_cuda_arch_compatible(capability, arch_list)` with `(arch_list, capability)` order."""
    return is_cuda_arch_compatible(capability, arch_list)


def verify_runtime_cuda_compatibility(
    model_root: Path | str | None = None,
    *,
    env_name: str = "gtrc",
) -> tuple[bool, str, dict[str, Any]]:
    """Query the installed GTRC runtime for live CUDA availability, arch compatibility, and kernel execution."""
    root = Path(model_root or MODEL_ROOT)
    py_bin = runtime_python(env_name, root)
    if not py_bin.is_file():
        return False, f"missing runtime python {py_bin}", {}

    cache_key = (str(py_bin.resolve()), os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if cache_key in _CUDA_PROBE_CACHE:
        return _CUDA_PROBE_CACHE[cache_key]

    probe_code = (
        "import json, torch\n"
        "cuda_ok = bool(torch.cuda.is_available())\n"
        "arch_list = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, 'get_arch_list') "
        "else (torch._C._cuda_getArchFlags() or '').split()\n"
        "dev_name = None\n"
        "cap = None\n"
        "tensor_ok = False\n"
        "err = None\n"
        "cuda_ver = getattr(torch.version, 'cuda', None)\n"
        "if cuda_ok:\n"
        "    try:\n"
        "        dev_name = torch.cuda.get_device_name(0)\n"
        "        cap = [int(x) for x in torch.cuda.get_device_capability(0)]\n"
        "        x = torch.ones((2, 2), device='cuda', dtype=torch.float32)\n"
        "        tensor_ok = bool(float((x + x).sum().item()) == 8.0)\n"
        "    except Exception as exc:\n"
        "        err = str(exc)\n"
        "print(json.dumps({\n"
        "    'cuda_available': cuda_ok,\n"
        "    'name': dev_name,\n"
        "    'device_name': dev_name,\n"
        "    'compute_capability': cap,\n"
        "    'device_capability': cap,\n"
        "    'cuda': cuda_ver,\n"
        "    'cuda_version': cuda_ver,\n"
        "    'compiled_architectures': arch_list,\n"
        "    'arch_list': arch_list,\n"
        "    'tensor_probe_ok': tensor_ok,\n"
        "    'error': err,\n"
        "}))\n"
    )
    try:
        proc = subprocess.run(
            [str(py_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        res = (False, f"GTRC runtime CUDA probe failed: {exc}", {})
        _CUDA_PROBE_CACHE[cache_key] = res
        return res

    if not info.get("cuda_available"):
        res = (False, "CUDA is unavailable in the GTRC runtime environment", info)
        _CUDA_PROBE_CACHE[cache_key] = res
        return res
    cap = info.get("compute_capability") or info.get("device_capability")
    arch_list = info.get("compiled_architectures") or info.get("arch_list") or []
    if not is_cuda_arch_compatible(cap, arch_list):
        res = (
            False,
            f"GPU compute capability {cap} is not supported by GTRC runtime compiled targets {arch_list}",
            info,
        )
        _CUDA_PROBE_CACHE[cache_key] = res
        return res
    if not info.get("tensor_probe_ok"):
        res = (
            False,
            f"CUDA kernel execution probe failed on {info.get('device_name')}: {info.get('error')}",
            info,
        )
        _CUDA_PROBE_CACHE[cache_key] = res
        return res

    res = (True, "ok", info)
    _CUDA_PROBE_CACHE[cache_key] = res
    return res


def installed_source_commit(source_dir: Path | str) -> str:
    """Read the pinned commit of `source_dir` from `.git` or `RTSEG_MANIFEST.json` without running git."""
    source = Path(source_dir)
    git_entry = source / ".git"
    if not git_entry.exists():
        manifest = source / "RTSEG_MANIFEST.json"
        if manifest.is_file():
            head = str(json.loads(manifest.read_text()).get("commit", "")).strip()
            if re.fullmatch(r"[0-9a-f]{40}", head):
                return head
        raise ValueError(f"Cannot verify installed source commit in {source}")

    git_dir = git_entry
    try:
        if git_entry.is_file():
            text = git_entry.read_text().strip()
            if text.startswith("gitdir:"):
                git_dir = (source / text.split(":", 1)[1].strip()).resolve()
        head = (git_dir / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            loose = git_dir / ref
            if loose.is_file():
                head = loose.read_text().strip()
            else:
                packed = (git_dir / "packed-refs").read_text().splitlines()
                head = next(
                    line.split()[0]
                    for line in packed
                    if not line.startswith(("#", "^")) and line.split()[-1] == ref
                )
    except (OSError, StopIteration, IndexError) as exc:
        raise ValueError("Cannot verify installed source commit") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("Cannot verify installed source commit")
    return head


def verify_weight_inventory(
    task: str,
    model_root: Path | None = None,
    *,
    verify_hashes: bool = False,
) -> tuple[bool, str, list[dict[str, Any]]]:
    root = Path(model_root or MODEL_ROOT)
    records: list[dict[str, Any]] = []
    for expected in GTRC_WEIGHT_FILES[task]:
        path = root / expected["path"]
        if not path.is_file():
            return False, f"missing weight file {path}", records
        size = path.stat().st_size
        if size != expected["bytes"]:
            return (
                False,
                f"weight file size mismatch for {expected['path']}: got {size}, expected {expected['bytes']}",
                records,
            )
        digest = file_sha256(path) if verify_hashes else expected["sha256"]
        if verify_hashes and digest != expected["sha256"]:
            return (
                False,
                f"weight file sha256 mismatch for {expected['path']}: got {digest}, expected {expected['sha256']}",
                records,
            )
        records.append({"path": expected["path"], "bytes": size, "sha256": digest})
    return True, "ok", records


def expected_checkpoint_hashes(task: str) -> dict[str, str]:
    return {item["path"]: item["sha256"] for item in GTRC_WEIGHT_FILES[task]}


def expected_checkpoint_digest(task: str) -> str:
    hashes = expected_checkpoint_hashes(task)
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_finite_float(
    raw: Any,
    name: str,
    *,
    low: float | None = None,
    high: float | None = None,
    strict_low: bool = False,
) -> float:
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"Missing or boolean numeric field {name}: {raw!r}")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric field {name}: {raw!r}") from exc
    if not math.isfinite(val):
        raise ValueError(f"Non-finite numeric field {name}: {raw!r}")
    if low is not None:
        if strict_low and not (val > low):
            raise ValueError(f"Field {name}={val} must be > {low}")
        if not strict_low and not (val >= low):
            raise ValueError(f"Field {name}={val} is below minimum {low}")
    if high is not None and not (val <= high):
        raise ValueError(f"Field {name}={val} exceeds maximum {high}")
    return val


def _require_exact_int(
    raw: Any,
    name: str,
    expected: int | None = None,
    *,
    min_val: int | None = None,
) -> int:
    if type(raw) is not int:
        raise ValueError(f"Missing or non-integer field {name}: {raw!r}")
    if expected is not None and raw != expected:
        raise ValueError(f"Field {name}={raw} does not equal required {expected}")
    if min_val is not None and raw < min_val:
        raise ValueError(f"Field {name}={raw} is below minimum {min_val}")
    return raw


def _require_sha256_hex(raw: Any, name: str) -> str:
    if not isinstance(raw, str) or not re.fullmatch(r"[0-9a-f]{64}", raw):
        raise ValueError(f"Missing or malformed SHA-256 hex digest for {name}: {raw!r}")
    return raw


def _validate_grid_check_dict(grid: Any, prefix: str) -> None:
    if not isinstance(grid, dict) or grid.get("passed") is not True or grid.get("size_equal") is not True:
        raise ValueError(f"Missing or failed {prefix} evidence")
    cand_size = grid.get("cand_size")
    ref_size = grid.get("ref_size")
    if (
        not isinstance(cand_size, list)
        or len(cand_size) != 3
        or not all(type(x) is int and x > 0 for x in cand_size)
        or cand_size != ref_size
    ):
        raise ValueError(f"Invalid {prefix} cand_size/ref_size: {cand_size!r} vs {ref_size!r}")
    _require_finite_float(
        grid.get("spacing_max_delta_mm"),
        f"{prefix}.spacing_max_delta_mm",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["grid_spacing_mm_tol"],
    )
    _require_finite_float(
        grid.get("origin_max_delta_mm"),
        f"{prefix}.origin_max_delta_mm",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["grid_origin_mm_tol"],
    )
    _require_finite_float(
        grid.get("direction_max_delta"),
        f"{prefix}.direction_max_delta",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["grid_direction_cos_tol"],
    )


def _validate_biomarker_recomputation_dict(
    bio: Any,
    prefix: str,
    *,
    expected_voxels: int,
) -> None:
    if not isinstance(bio, dict) or bio.get("passed") is not True:
        raise ValueError(f"Missing or failed {prefix} evidence")
    _require_exact_int(bio.get("ttb_voxel_count"), f"{prefix}.ttb_voxel_count", expected=expected_voxels)
    _require_finite_float(bio.get("ttb_volume_ml"), f"{prefix}.ttb_volume_ml", low=0.0, strict_low=True)
    _require_finite_float(bio.get("suv_mean"), f"{prefix}.suv_mean", low=0.0, strict_low=True)
    _require_finite_float(bio.get("suv_max"), f"{prefix}.suv_max", low=0.0, strict_low=True)
    _require_finite_float(bio.get("hu_mean"), f"{prefix}.hu_mean")
    _require_exact_int(bio.get("lesion_count"), f"{prefix}.lesion_count", min_val=1)
    _require_finite_float(
        bio.get("max_stat_abs_err"),
        f"{prefix}.max_stat_abs_err",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["suv_stat_abs_tol"],
    )


def _validate_nnunet_runtime_flags(
    flags: Any,
    prefix: str,
    *,
    expected_gpu: dict[str, Any],
) -> None:
    if not isinstance(flags, dict):
        raise ValueError(f"Missing {prefix} child-process runtime flags evidence")
    if flags.get("cudnn_deterministic") is not True or flags.get("cudnn_benchmark") is not False:
        raise ValueError(f"{prefix} missing deterministic cuDNN flags (cudnn_deterministic=True, cudnn_benchmark=False)")
    if flags.get("inference_mode") != "production":
        raise ValueError(f"{prefix} inference_mode must be 'production', got {flags.get('inference_mode')!r}")
    if flags.get("nnunet_launch_path") not in ("DETERMINISTIC_NNUNET_ENTRYPOINT", "UPSTREAM_DETERMINISTIC_WRAPPER"):
        raise ValueError(f"{prefix} invalid nnunet_launch_path={flags.get('nnunet_launch_path')!r}")
    if device_key(flags) != device_key(expected_gpu):
        raise ValueError(f"{prefix} device mismatch with live GPU")


def _validate_gpu_identity(recorded_gpu: dict[str, Any], live_gpu: dict[str, Any], context: str) -> str:
    rec_key = device_key(recorded_gpu)
    live_key = device_key(live_gpu)
    if rec_key != live_key:
        raise ValueError(f"{context} GPU device key mismatch: marker has {rec_key!r}, live GPU is {live_key!r}")
    rec_name = str(recorded_gpu.get("name") or recorded_gpu.get("gpu_name") or recorded_gpu.get("device_name") or "").strip()
    live_name = str(live_gpu.get("name") or live_gpu.get("gpu_name") or live_gpu.get("device_name") or "").strip()
    if not rec_name or rec_name != live_name:
        raise ValueError(f"{context} GPU name mismatch: marker has {rec_name!r}, live GPU is {live_name!r}")
    rec_cap = (
        recorded_gpu.get("compute_capability")
        if recorded_gpu.get("compute_capability") is not None
        else (
            recorded_gpu.get("device_capability")
            if recorded_gpu.get("device_capability") is not None
            else recorded_gpu.get("cuda_device_capability")
        )
    )
    live_cap = (
        live_gpu.get("compute_capability")
        if live_gpu.get("compute_capability") is not None
        else (
            live_gpu.get("device_capability")
            if live_gpu.get("device_capability") is not None
            else live_gpu.get("cuda_device_capability")
        )
    )
    if [int(x) for x in (rec_cap or [])] != [int(x) for x in (live_cap or [])]:
        raise ValueError(f"{context} compute_capability mismatch: marker has {rec_cap!r}, live GPU is {live_cap!r}")
    rec_cuda = str(recorded_gpu.get("cuda") or recorded_gpu.get("cuda_version") or "").strip()
    live_cuda = str(live_gpu.get("cuda") or live_gpu.get("cuda_version") or "").strip()
    if not rec_cuda or (live_cuda and rec_cuda != live_cuda):
        raise ValueError(f"{context} CUDA version mismatch: marker has {rec_cuda!r}, live GPU is {live_cuda!r}")
    rec_arch = (
        recorded_gpu.get("compiled_architectures")
        if recorded_gpu.get("compiled_architectures") is not None
        else (
            recorded_gpu.get("cuda_arch_flags")
            if recorded_gpu.get("cuda_arch_flags") is not None
            else (
                recorded_gpu.get("cuda_arch_list")
                if recorded_gpu.get("cuda_arch_list") is not None
                else recorded_gpu.get("arch_list")
            )
        )
    )
    live_arch = (
        live_gpu.get("compiled_architectures")
        if live_gpu.get("compiled_architectures") is not None
        else (
            live_gpu.get("cuda_arch_flags")
            if live_gpu.get("cuda_arch_flags") is not None
            else (
                live_gpu.get("cuda_arch_list")
                if live_gpu.get("cuda_arch_list") is not None
                else live_gpu.get("arch_list")
            )
        )
    )
    if not isinstance(rec_arch, list) or not rec_arch:
        raise ValueError(f"{context} missing compiled_architectures")
    if isinstance(live_arch, list) and live_arch and [str(x) for x in rec_arch] != [str(x) for x in live_arch]:
        raise ValueError(f"{context} compiled_architectures mismatch: marker has {rec_arch!r}, live GPU is {live_arch!r}")
    if not is_cuda_arch_compatible(rec_cap, rec_arch):
        raise ValueError(f"{context} incompatible CUDA capability {rec_cap} vs {rec_arch}")
    return live_key


def validate_completion(
    root: Path | str | None = None,
    *,
    app_root: Path | str | None = None,
    gpu: dict[str, Any] | None = None,
    validation_dir: Path | str | None = None,
    orientation_dir: Path | str | None = None,
    completion_envelope: dict[str, Any] | None = None,
    task_validation_overrides: dict[str, dict[str, Any]] | None = None,
    task_orientation_overrides: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Validate the per-device GTRC completion record (`validation/gtrc_completion.json`).

    Binds the shared `run_id`, live GPU identity, installed source commit, code digest, runtime lock digest,
    and canonical SHA-256 of all four task records (`validation/<task>.json` and
    `orientation-validation/<task>.json` for both PSMA and FDG).
    """
    root_path = Path(root or MODEL_ROOT)
    app_path = Path(app_root or APP_ROOT)
    val_folder = Path(validation_dir) if validation_dir is not None else (root_path / "validation")
    ori_folder = Path(orientation_dir) if orientation_dir is not None else (root_path / "orientation-validation")

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    if gpu is None:
        cuda_ok, cuda_msg, live_gpu = verify_runtime_cuda_compatibility(root_path, env_name=GTRC_SOURCE["runtime_env"])
        if not cuda_ok:
            raise ValueError(f"GTRC runtime/device check failed: {cuda_msg}")
        gpu = live_gpu
    live_dev = device_key(gpu)

    if completion_envelope is not None:
        comp_env = completion_envelope
    else:
        comp_path = val_folder / GTRC_COMPLETION_FILENAME
        require(comp_path.is_file(), f"Missing GTRC completion marker {comp_path}")
        comp_env = json.loads(comp_path.read_text(encoding="utf-8"))

    require(
        isinstance(comp_env, dict)
        and comp_env.get("schema_version") == MARKER_SCHEMA_VERSION
        and isinstance(comp_env.get("devices"), dict),
        "Completion marker must use schema_version 2 per-device envelope",
    )
    comp_rec = comp_env["devices"].get(live_dev)
    require(
        isinstance(comp_rec, dict),
        f"Missing {live_dev} completion marker record in {val_folder / GTRC_COMPLETION_FILENAME}",
    )
    require(
        comp_rec.get("family") == "gtrc"
        and comp_rec.get("device_key") == live_dev
        and comp_rec.get("result") == "passed"
        and comp_rec.get("status") == "validated"
        and comp_rec.get("passed") is True
        and comp_rec.get("valid") is True
        and comp_rec.get("inference_mode") == "production"
        and comp_rec.get("cpu_smoke", False) is False
        and not marker_has_failure(comp_rec),
        "Completion marker device/status/inference_mode mismatch or failure/smoke flag",
    )
    run_id = comp_rec.get("run_id")
    require(isinstance(run_id, str) and bool(run_id.strip()), "Completion marker missing run_id")
    require(isinstance(comp_rec.get("gpu"), dict), "Completion marker missing gpu metadata")
    _validate_gpu_identity(comp_rec["gpu"], gpu, f"Completion {live_dev} gpu")

    actual_commit = installed_source_commit(root_path / GTRC_SOURCE["source_dir"])
    require(
        comp_rec.get("source_commit") == actual_commit == GTRC_SOURCE["source_commit"],
        "Completion marker source_commit mismatch",
    )
    require(
        comp_rec.get("code_sha256") == expected_code_digest(app_path),
        "Completion marker code_sha256 mismatch",
    )
    require(
        comp_rec.get("runtime_lock_sha256") == expected_runtime_lock_sha256(root_path),
        "Completion marker runtime_lock_sha256 mismatch",
    )

    tasks_map = comp_rec.get("tasks")
    require(
        isinstance(tasks_map, dict) and set(tasks_map.keys()) == set(GTRC_TASKS.keys()),
        f"Completion marker must bind all GTRC tasks {sorted(GTRC_TASKS.keys())}",
    )
    for t_name in GTRC_TASKS:
        t_entry = tasks_map.get(t_name)
        require(isinstance(t_entry, dict), f"Completion marker missing task entry for {t_name}")
        val_sha = _require_sha256_hex(
            t_entry.get("validation_sha256"), f"completion.tasks.{t_name}.validation_sha256"
        )
        ori_sha = _require_sha256_hex(
            t_entry.get("orientation_sha256"), f"completion.tasks.{t_name}.orientation_sha256"
        )

        if task_validation_overrides and t_name in task_validation_overrides:
            t_val_rec = dict(task_validation_overrides[t_name])
            t_val_rec["device_key"] = live_dev
        else:
            val_p = val_folder / f"{t_name}.json"
            require(val_p.is_file(), f"Missing validation marker {val_p} required by completion record")
            t_val_rec = read_device_records(val_p).get(live_dev)
            require(isinstance(t_val_rec, dict), f"Missing {live_dev} acceptance marker record for {t_name}")

        if task_orientation_overrides and t_name in task_orientation_overrides:
            t_ori_rec = dict(task_orientation_overrides[t_name])
            t_ori_rec["device_key"] = live_dev
        else:
            ori_p = ori_folder / f"{t_name}.json"
            require(ori_p.is_file(), f"Missing orientation marker {ori_p} required by completion record")
            t_ori_rec = read_device_records(ori_p).get(live_dev)
            require(isinstance(t_ori_rec, dict), f"Missing {live_dev} orientation marker record for {t_name}")

        require(
            t_val_rec.get("run_id") == run_id and t_ori_rec.get("run_id") == run_id,
            f"Completion marker run_id ({run_id!r}) mismatch with {t_name} task records",
        )
        require(
            canonical_json_sha256(t_val_rec) == val_sha,
            f"Completion marker validation_sha256 mismatch for {t_name} on {live_dev}",
        )
        require(
            canonical_json_sha256(t_ori_rec) == ori_sha,
            f"Completion marker orientation_sha256 mismatch for {t_name} on {live_dev}",
        )
        require(
            t_ori_rec.get("acceptance_sha256") == val_sha,
            f"Orientation marker acceptance_sha256 mismatch with completion validation_sha256 for {t_name}",
        )
    return True


def validate_acceptance(
    record: Any,
    task: str | None = None,
    root: Path | str | None = None,
    *,
    app_root: Path | str | None = None,
    gpu: dict[str, Any] | None = None,
    require_completion: bool = True,
) -> bool:
    """Validate the GTRC GPU acceptance marker against the installed source, weights, code, runtime lock, and live GPU.

    Requires a schema-2 per-device envelope (`{"schema_version": 2, "devices": {"H100": ..., "P6000": ...}}`),
    selects the record matching the live GPU device (`gpu`, or probed via `verify_runtime_cuda_compatibility`),
    and enforces the per-device completion gate (`validation/gtrc_completion.json`) binding all four task records.
    """
    if isinstance(record, str) and record in GTRC_TASKS:
        root_path = Path(task) if task is not None else Path(root or MODEL_ROOT)
        task_name = record
        marker_path = root_path / "validation" / f"{task_name}.json"
        if not marker_path.is_file():
            raise ValueError(f"Missing validation marker {marker_path}")
        envelope = json.loads(marker_path.read_text(encoding="utf-8"))
    else:
        if task is None or task not in GTRC_TASKS:
            raise ValueError(f"Unknown GTRC task: {task!r}")
        task_name = task
        root_path = Path(root or MODEL_ROOT)
        envelope = record

    app_path = Path(app_root or APP_ROOT)
    spec = GTRC_TASKS[task_name]

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    if gpu is None:
        cuda_ok, cuda_msg, live_gpu = verify_runtime_cuda_compatibility(root_path, env_name=spec["runtime_env"])
        if not cuda_ok:
            raise ValueError(f"GTRC runtime/device check failed for {task_name}: {cuda_msg}")
        gpu = live_gpu
    live_dev = device_key(gpu)

    require(
        isinstance(envelope, dict)
        and envelope.get("schema_version") == MARKER_SCHEMA_VERSION
        and isinstance(envelope.get("devices"), dict),
        "Acceptance marker must use schema_version 2 per-device envelope",
    )
    record_dict = envelope["devices"].get(live_dev)
    require(
        isinstance(record_dict, dict),
        f"Missing {live_dev} acceptance marker record for {task_name}",
    )

    require(
        record_dict.get("task") == task_name
        and record_dict.get("family") == "gtrc"
        and record_dict.get("device_key") == live_dev
        and record_dict.get("result") == "passed"
        and record_dict.get("status") == "validated"
        and record_dict.get("valid") is True
        and record_dict.get("inference_mode") == "production"
        and record_dict.get("cpu_smoke", False) is False,
        "Acceptance task/device/result/status/inference_mode mismatch or CPU smoke/stub marker",
    )
    require(
        not marker_has_failure(record_dict),
        "Acceptance marker contains a failure, error, smoke/stub flag, or waived deviation",
    )
    require(
        record_dict.get("tracer") == spec["tracer"]
        and record_dict.get("dataset_id") == spec["dataset_id"],
        "Acceptance tracer/dataset_id mismatch",
    )
    require(
        isinstance(record_dict.get("run_id"), str) and bool(record_dict["run_id"].strip()),
        "Acceptance marker missing run_id",
    )

    # Validate runtime provenance & live GPU device evidence
    rt_prov = record_dict.get("runtime_provenance")
    require(isinstance(rt_prov, dict), "Acceptance marker missing runtime_provenance")
    require(
        rt_prov.get("cpu_smoke", False) is False
        and rt_prov.get("inference_mode") == "production"
        and rt_prov.get("nnunet_launch_path") == "DETERMINISTIC_NNUNET_ENTRYPOINT",
        "Acceptance runtime_provenance missing production inference_mode or DETERMINISTIC_NNUNET_ENTRYPOINT",
    )
    dev_name = rt_prov.get("gpu_name") if "gpu_name" in rt_prov else rt_prov.get("device_name")
    require(
        rt_prov.get("cuda_available") is True
        and isinstance(dev_name, str)
        and bool(dev_name.strip())
        and not any(tok in dev_name.lower() for tok in ("cpu", "stub", "mock", "fake")),
        "Acceptance runtime_provenance missing GPU device_name or cuda_available",
    )
    _validate_gpu_identity(rt_prov, gpu, f"Acceptance {task_name} runtime_provenance")
    if "gpu" in record_dict:
        require(isinstance(record_dict["gpu"], dict), "Acceptance gpu field must be a dict")
        _validate_gpu_identity(record_dict["gpu"], gpu, f"Acceptance {task_name} gpu")

    cudnn_det = (
        rt_prov.get("cudnn_deterministic_enforced")
        if "cudnn_deterministic_enforced" in rt_prov
        else rt_prov.get("cudnn_deterministic")
    )
    cudnn_bench = (
        rt_prov.get("cudnn_benchmark_enforced")
        if "cudnn_benchmark_enforced" in rt_prov
        else rt_prov.get("cudnn_benchmark")
    )
    require(
        cudnn_det is True and cudnn_bench is False,
        "Acceptance runtime_provenance missing deterministic cuDNN evidence",
    )

    # Validate SUV threshold and source binding across top-level, input_binding, reference_mode, and dicom_mode
    thresh_val, thresh_src = _validate_threshold_and_source(
        task_name,
        record_dict.get("suv_threshold"),
        record_dict.get("suv_threshold_source"),
    )

    # Validate input fixture binding (all digests mandatory)
    input_binding = record_dict.get("input_binding")
    require(isinstance(input_binding, dict), "Acceptance marker missing input_binding")
    for req_digest_key in (
        "archive_sha256",
        "fixture_manifest_sha256",
        "pet_nifti_sha256",
        "ct_nifti_sha256",
        "ttb_nifti_sha256",
        "threshold_json_sha256",
        "pet_dicom_sha256",
        "ct_dicom_sha256",
        "acceptance_harness_sha256",
        "fixture_builder_sha256",
    ):
        _require_sha256_hex(input_binding.get(req_digest_key), f"input_binding.{req_digest_key}")

    ib_thresh, ib_src = _validate_threshold_and_source(
        task_name,
        input_binding.get("suv_threshold"),
        input_binding.get("suv_threshold_source"),
    )
    require(
        abs(ib_thresh - thresh_val) <= GTRC_TOLERANCE_CONTRACT["threshold_abs_tol"] and ib_src == thresh_src,
        "Acceptance input_binding suv_threshold/source does not match top-level record",
    )

    modes = record_dict.get("acceptance_modes")
    require(
        isinstance(modes, list)
        and len(modes) == 2
        and "reference" in modes
        and "dicom" in modes,
        "Both reference and dicom acceptance modes are required",
    )

    # Validate mandatory SUVbw converter numerical tolerances and input PET/CT grids
    suv_check = record_dict.get("suvbw_converter_check")
    require(
        isinstance(suv_check, dict)
        and suv_check.get("passed") is True
        and suv_check.get("threshold_3_0_exact_match") is True
        and suv_check.get("fixture_threshold_exact_match") is True,
        "Acceptance marker missing or failed mandatory suvbw_converter_check",
    )
    _require_finite_float(
        suv_check.get("suvbw_max_abs_err"),
        "suvbw_converter_check.suvbw_max_abs_err",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["suvbw_max_abs_err"],
    )
    _require_finite_float(
        suv_check.get("suvbw_mean_abs_err"),
        "suvbw_converter_check.suvbw_mean_abs_err",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["suvbw_mean_abs_err"],
    )
    _require_finite_float(
        suv_check.get("suvbw_p999_abs_err"),
        "suvbw_converter_check.suvbw_p999_abs_err",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["suvbw_p999_abs_err"],
    )
    _require_finite_float(
        suv_check.get("ct_hu_max_abs_err"),
        "suvbw_converter_check.ct_hu_max_abs_err",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["ct_hu_max_abs_err"],
    )
    _require_finite_float(
        suv_check.get("ct_hu_mean_abs_err"),
        "suvbw_converter_check.ct_hu_mean_abs_err",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["ct_hu_mean_abs_err"],
    )
    _require_finite_float(
        suv_check.get("suvbw_scale_factor_rel_error"),
        "suvbw_converter_check.suvbw_scale_factor_rel_error",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["scale_factor_rtol"],
    )
    _require_finite_float(
        suv_check.get("decay_elapsed_seconds_abs_error"),
        "suvbw_converter_check.decay_elapsed_seconds_abs_error",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["elapsed_time_atol_s"],
    )
    _validate_grid_check_dict(suv_check.get("pet_grid"), "suvbw_converter_check.pet_grid")
    _validate_grid_check_dict(suv_check.get("ct_grid"), "suvbw_converter_check.ct_grid")

    # Validate reference mode (all quantitative fields mandatory, no defaults)
    ref_mode = record_dict.get("reference_mode")
    require(
        isinstance(ref_mode, dict)
        and ref_mode.get("passed") is True
        and ref_mode.get("valid") is True
        and ref_mode.get("status") == "passed"
        and ref_mode.get("inference_mode") == "production"
        and ref_mode.get("exact_mask_agreement") is True
        and ref_mode.get("subregion_label_agreement") is True
        and ref_mode.get("grid_agreement") is True,
        "Reference mode failed or missing mandatory status/grid/mask/subregion agreement flags",
    )
    _require_exact_int(ref_mode.get("mismatch_voxels"), "reference_mode.mismatch_voxels", 0)
    _require_exact_int(
        ref_mode.get("subregion_label_mismatch_voxels"),
        "reference_mode.subregion_label_mismatch_voxels",
        0,
    )
    _require_exact_int(
        ref_mode.get("nnunet_label_mismatch_voxels"),
        "reference_mode.nnunet_label_mismatch_voxels",
        0,
    )
    ref_voxels = _require_exact_int(ref_mode.get("ref_voxels"), "reference_mode.ref_voxels", min_val=1)
    _require_exact_int(ref_mode.get("cand_voxels"), "reference_mode.cand_voxels", expected=ref_voxels)
    _require_finite_float(ref_mode.get("dice"), "reference_mode.dice", low=1.0, high=1.0)
    _require_sha256_hex(ref_mode.get("result_sha256"), "reference_mode.result_sha256")
    _validate_grid_check_dict(ref_mode.get("grid_check"), "reference_mode.grid_check")
    _validate_grid_check_dict(ref_mode.get("upstream_grid_check"), "reference_mode.upstream_grid_check")
    _validate_biomarker_recomputation_dict(
        ref_mode.get("biomarker_recomputation"),
        "reference_mode.biomarker_recomputation",
        expected_voxels=ref_voxels,
    )
    _validate_nnunet_runtime_flags(
        ref_mode.get("nnunet_runtime_flags"),
        "reference_mode.nnunet_runtime_flags",
        expected_gpu=gpu,
    )
    _validate_nnunet_runtime_flags(
        ref_mode.get("upstream_nnunet_runtime_flags"),
        "reference_mode.upstream_nnunet_runtime_flags",
        expected_gpu=gpu,
    )
    ref_thresh, ref_src = _validate_threshold_and_source(
        task_name,
        ref_mode.get("suv_threshold"),
        ref_mode.get("suv_threshold_source"),
    )
    require(
        abs(ref_thresh - thresh_val) <= GTRC_TOLERANCE_CONTRACT["threshold_abs_tol"] and ref_src == thresh_src,
        "reference_mode suv_threshold/source does not match top-level record",
    )

    # Validate dicom mode (all quantitative fields mandatory, no defaults)
    dicom_mode = record_dict.get("dicom_mode")
    require(
        isinstance(dicom_mode, dict)
        and dicom_mode.get("passed") is True
        and dicom_mode.get("valid") is True
        and dicom_mode.get("status") == "passed"
        and dicom_mode.get("inference_mode") == "production"
        and dicom_mode.get("rtstruct_valid") is True
        and dicom_mode.get("exact_reference_agreement") is True
        and dicom_mode.get("grid_agreement") is True
        and dicom_mode.get("roi_name") == spec["roi_name"],
        "DICOM mode failed or missing run_task/RTSTRUCT/roundtrip/grid/reference agreement",
    )
    _require_exact_int(dicom_mode.get("reference_mismatch_voxels"), "dicom_mode.reference_mismatch_voxels", 0)
    # DICOM-mode inputs are read back from int16 DICOM and differ from the reference-mode floats by the
    # declared SUV encoding error, so raw nnU-Net labels and subregions may differ at a few boundary voxels.
    # They are recorded diagnostics; the exported TTB mask above must still agree exactly.
    _require_exact_int(
        dicom_mode.get("subregion_label_mismatch_voxels"),
        "dicom_mode.subregion_label_mismatch_voxels",
        min_val=0,
    )
    _require_exact_int(
        dicom_mode.get("nnunet_label_mismatch_voxels"),
        "dicom_mode.nnunet_label_mismatch_voxels",
        min_val=0,
    )
    _require_exact_int(dicom_mode.get("roundtrip_nonempty_rois"), "dicom_mode.roundtrip_nonempty_rois", 1)
    _require_finite_float(dicom_mode.get("reference_dice"), "dicom_mode.reference_dice", low=1.0, high=1.0)
    vs_ref = dicom_mode.get("vs_reference_mode")
    require(
        isinstance(vs_ref, dict) and vs_ref.get("exact_match") is True,
        "Missing or failed dicom_mode.vs_reference_mode exact_match",
    )
    _require_exact_int(vs_ref.get("mismatch_voxels"), "dicom_mode.vs_reference_mode.mismatch_voxels", 0)
    _require_finite_float(vs_ref.get("dice"), "dicom_mode.vs_reference_mode.dice", low=1.0, high=1.0)
    _require_finite_float(
        dicom_mode.get("roundtrip_min_dice"),
        "dicom_mode.roundtrip_min_dice",
        low=GTRC_TOLERANCE_CONTRACT["roundtrip_min_dice"],
        high=1.0,
    )
    _require_finite_float(
        dicom_mode.get("roundtrip_median_dice"),
        "dicom_mode.roundtrip_median_dice",
        low=GTRC_TOLERANCE_CONTRACT["roundtrip_median_dice"],
        high=1.0,
    )
    _require_sha256_hex(dicom_mode.get("rtstruct_sha256"), "dicom_mode.rtstruct_sha256")
    _require_sha256_hex(dicom_mode.get("ttb_mask_sha256"), "dicom_mode.ttb_mask_sha256")
    require(
        _require_sha256_hex(dicom_mode.get("pet_dicom_sha256"), "dicom_mode.pet_dicom_sha256")
        == input_binding["pet_dicom_sha256"],
        "dicom_mode.pet_dicom_sha256 does not match input_binding.pet_dicom_sha256",
    )
    require(
        _require_sha256_hex(dicom_mode.get("ct_dicom_sha256"), "dicom_mode.ct_dicom_sha256")
        == input_binding["ct_dicom_sha256"],
        "dicom_mode.ct_dicom_sha256 does not match input_binding.ct_dicom_sha256",
    )
    fix_rt = dicom_mode.get("fixture_roundtrip_verification")
    require(
        isinstance(fix_rt, dict)
        and fix_rt.get("passed") is True
        and fix_rt.get("threshold_3_0_exact_match") is True
        and fix_rt.get("fixture_threshold_exact_match") is True,
        "Missing or failed dicom_mode.fixture_roundtrip_verification",
    )
    _validate_grid_check_dict(fix_rt.get("pet_grid"), "dicom_mode.fixture_roundtrip_verification.pet_grid")
    _validate_grid_check_dict(fix_rt.get("ct_grid"), "dicom_mode.fixture_roundtrip_verification.ct_grid")
    _require_finite_float(
        fix_rt.get("pet_max_abs_diff_suv"),
        "dicom_mode.fixture_roundtrip_verification.pet_max_abs_diff_suv",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["suvbw_max_abs_err"],
    )
    _require_finite_float(
        fix_rt.get("pet_mean_abs_diff_suv"),
        "dicom_mode.fixture_roundtrip_verification.pet_mean_abs_diff_suv",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["suvbw_mean_abs_err"],
    )
    _require_finite_float(
        fix_rt.get("pet_p999_abs_diff_suv"),
        "dicom_mode.fixture_roundtrip_verification.pet_p999_abs_diff_suv",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["suvbw_p999_abs_err"],
    )
    _require_finite_float(
        fix_rt.get("ct_max_abs_diff_hu"),
        "dicom_mode.fixture_roundtrip_verification.ct_max_abs_diff_hu",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["ct_hu_max_abs_err"],
    )
    _require_finite_float(
        fix_rt.get("ct_mean_abs_diff_hu"),
        "dicom_mode.fixture_roundtrip_verification.ct_mean_abs_diff_hu",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["ct_hu_mean_abs_err"],
    )
    _require_finite_float(
        fix_rt.get("suvbw_scale_factor_rel_error"),
        "dicom_mode.fixture_roundtrip_verification.suvbw_scale_factor_rel_error",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["scale_factor_rtol"],
    )
    _require_finite_float(
        fix_rt.get("decay_elapsed_seconds_abs_error"),
        "dicom_mode.fixture_roundtrip_verification.decay_elapsed_seconds_abs_error",
        low=0.0,
        high=GTRC_TOLERANCE_CONTRACT["elapsed_time_atol_s"],
    )
    require(
        _require_sha256_hex(
            fix_rt.get("pet_dicom_sha256"),
            "dicom_mode.fixture_roundtrip_verification.pet_dicom_sha256",
        )
        == input_binding["pet_dicom_sha256"],
        "dicom_mode.fixture_roundtrip_verification.pet_dicom_sha256 does not match input_binding",
    )
    require(
        _require_sha256_hex(
            fix_rt.get("ct_dicom_sha256"),
            "dicom_mode.fixture_roundtrip_verification.ct_dicom_sha256",
        )
        == input_binding["ct_dicom_sha256"],
        "dicom_mode.fixture_roundtrip_verification.ct_dicom_sha256 does not match input_binding",
    )
    fix_rt_thresh = _require_finite_float(
        fix_rt.get("fixture_threshold"),
        "dicom_mode.fixture_roundtrip_verification.fixture_threshold",
    )
    require(
        abs(fix_rt_thresh - thresh_val) <= GTRC_TOLERANCE_CONTRACT["threshold_abs_tol"],
        "dicom_mode.fixture_roundtrip_verification.fixture_threshold does not match top-level record",
    )
    _validate_grid_check_dict(dicom_mode.get("grid_check"), "dicom_mode.grid_check")
    _validate_biomarker_recomputation_dict(
        dicom_mode.get("biomarker_recomputation"),
        "dicom_mode.biomarker_recomputation",
        expected_voxels=ref_voxels,
    )
    _validate_nnunet_runtime_flags(
        dicom_mode.get("nnunet_runtime_flags"),
        "dicom_mode.nnunet_runtime_flags",
        expected_gpu=gpu,
    )
    dcm_thresh, dcm_src = _validate_threshold_and_source(
        task_name,
        dicom_mode.get("suv_threshold"),
        dicom_mode.get("suv_threshold_source"),
    )
    require(
        abs(dcm_thresh - thresh_val) <= GTRC_TOLERANCE_CONTRACT["threshold_abs_tol"] and dcm_src == thresh_src,
        "dicom_mode suv_threshold/source does not match top-level record",
    )

    checks = record_dict.get("checks")
    required_checks = {
        "gpu_inference",
        "suvbw_conversion",
        "exact_input_grid",
        "rtstruct_roundtrip",
        "orientation",
        "both_acceptance_modes",
        "author_reference_exact_match",
    }
    require(
        isinstance(checks, list)
        and all(isinstance(c, str) for c in checks)
        and required_checks <= set(checks),
        "Incomplete acceptance checks",
    )

    source_path = root_path / spec["source_dir"]
    actual_commit = installed_source_commit(source_path)
    require(
        record_dict.get("source_commit") == actual_commit == spec["source_commit"],
        "Source commit mismatch",
    )

    ok_w, msg_w, _ = verify_weight_inventory(task_name, root_path, verify_hashes=True)
    require(ok_w, f"Installed weight verification failed: {msg_w}")

    actual_hashes = {
        item["path"]: file_sha256(root_path / item["path"])
        for item in GTRC_WEIGHT_FILES[task_name]
    }
    expected_hashes = expected_checkpoint_hashes(task_name)
    require(actual_hashes == expected_hashes, "Installed checkpoint hashes do not match pinned release")
    require(record_dict.get("checkpoint_hashes") == expected_hashes, "Marker checkpoint_hashes mismatch")
    require(
        record_dict.get("checkpoint_sha256") == expected_checkpoint_digest(task_name),
        "Marker checkpoint_sha256 digest mismatch",
    )

    actual_code_hashes = expected_code_hashes(task_name, app_path)
    actual_code_digest = expected_code_digest(task_name, app_path)
    require(record_dict.get("code_hashes") == actual_code_hashes, "Marker code_hashes mismatch with worktree files")
    require(record_dict.get("code_sha256") == actual_code_digest, "Marker code_sha256 mismatch")

    actual_lock_sha256 = expected_runtime_lock_sha256(root_path)
    require(
        record_dict.get("runtime_lock_sha256") == actual_lock_sha256,
        "Marker runtime_lock_sha256 mismatch with installed requirements.lock.txt",
    )

    if require_completion:
        validate_completion(
            root_path,
            app_root=app_path,
            gpu=gpu,
            task_validation_overrides={task_name: record_dict},
        )
        for peer_task in GTRC_TASKS:
            if peer_task != task_name:
                validate_acceptance(
                    peer_task,
                    root_path,
                    app_root=app_path,
                    gpu=gpu,
                    require_completion=False,
                )
    return True


def check_acceptance(
    task: str,
    model_root: Path | None = None,
    *,
    app_root: Path | None = None,
    gpu: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Non-throwing wrapper around `validate_acceptance` returning `(ok, message)`."""
    root = Path(model_root or MODEL_ROOT)
    marker_path = root / "validation" / f"{task}.json"
    if not marker_path.is_file():
        return False, f"missing validation marker {marker_path}"
    try:
        record = json.loads(marker_path.read_text(encoding="utf-8"))
        validate_acceptance(record, task, root, app_root=app_root, gpu=gpu)
        return True, "validated"
    except Exception as exc:
        return False, str(exc)


def validate_orientation(
    record: Any,
    task: str | None = None,
    root: Path | str | None = None,
    *,
    app_root: Path | str | None = None,
    gpu: dict[str, Any] | None = None,
    acceptance_envelope: dict[str, Any] | None = None,
    require_completion: bool = True,
) -> bool:
    """Validate the orientation marker for `task` against live GPU, runtime lock, acceptance marker, and completion gate."""
    if isinstance(record, str) and record in GTRC_TASKS:
        root_path = Path(task) if task is not None else Path(root or MODEL_ROOT)
        task_name = record
        marker_path = root_path / "orientation-validation" / f"{task_name}.json"
        if not marker_path.is_file():
            raise ValueError(f"Missing orientation marker {marker_path}")
        envelope = json.loads(marker_path.read_text(encoding="utf-8"))
    else:
        if task is None or task not in GTRC_TASKS:
            raise ValueError(f"Unknown GTRC task: {task!r}")
        task_name = str(task)
        root_path = Path(root or MODEL_ROOT)
        envelope = record

    app_path = Path(app_root or APP_ROOT)
    spec = GTRC_TASKS[task_name]

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    if gpu is None:
        cuda_ok, cuda_msg, live_gpu = verify_runtime_cuda_compatibility(root_path, env_name=spec["runtime_env"])
        if not cuda_ok:
            raise ValueError(f"GTRC runtime/device check failed for {task_name}: {cuda_msg}")
        gpu = live_gpu
    live_dev = device_key(gpu)

    require(
        isinstance(envelope, dict)
        and envelope.get("schema_version") == MARKER_SCHEMA_VERSION
        and isinstance(envelope.get("devices"), dict),
        "Orientation marker must use schema_version 2 per-device envelope",
    )
    record_dict = envelope["devices"].get(live_dev)
    require(
        isinstance(record_dict, dict),
        f"Missing {live_dev} orientation marker record for {task_name}",
    )

    require(
        record_dict.get("task") == task_name
        and record_dict.get("device_key") == live_dev
        and record_dict.get("result") == "passed"
        and record_dict.get("status") == "validated"
        and record_dict.get("valid") is True
        and record_dict.get("roundtrip_valid") is True
        and record_dict.get("inference_mode") == "production"
        and record_dict.get("cpu_smoke", False) is False
        and not marker_has_failure(record_dict),
        "Orientation task/device/result/inference_mode mismatch, smoke/stub flag, or failed checks",
    )
    require(
        isinstance(record_dict.get("run_id"), str) and bool(record_dict["run_id"].strip()),
        "Orientation marker missing run_id",
    )
    require(isinstance(record_dict.get("gpu"), dict), "Orientation marker missing gpu metadata")
    _validate_gpu_identity(record_dict["gpu"], gpu, f"Orientation {task_name} gpu")

    _require_exact_int(record_dict.get("nonempty_rois"), "orientation.nonempty_rois", 1)
    _require_finite_float(
        record_dict.get("min_dice"),
        "orientation.min_dice",
        low=GTRC_TOLERANCE_CONTRACT["roundtrip_min_dice"],
        high=1.0,
    )
    _require_finite_float(
        record_dict.get("median_dice"),
        "orientation.median_dice",
        low=GTRC_TOLERANCE_CONTRACT["roundtrip_median_dice"],
        high=1.0,
    )
    require(
        record_dict.get("roi_name") == spec["roi_name"],
        f"Orientation marker roi_name mismatch: expected {spec['roi_name']!r}",
    )
    require(
        record_dict.get("roi_names") == [spec["roi_name"]],
        f"Orientation marker roi_names mismatch: expected {[spec['roi_name']]!r}",
    )
    require(
        isinstance(record_dict.get("referenced_pet_series_uid"), str)
        and bool(record_dict["referenced_pet_series_uid"].strip())
        and record_dict.get("referenced_pet_series_uid_match") is True,
        "Orientation marker referenced_pet_series_uid_match must be True",
    )

    actual_commit = installed_source_commit(root_path / spec["source_dir"])
    require(
        record_dict.get("source_commit") == actual_commit == spec["source_commit"],
        "Orientation marker source_commit mismatch",
    )
    require(
        record_dict.get("checkpoint_sha256") == expected_checkpoint_digest(task_name),
        "Orientation marker checkpoint_sha256 mismatch",
    )
    require(
        record_dict.get("code_sha256") == expected_code_digest(task_name, app_path),
        "Orientation marker code_sha256 mismatch",
    )
    actual_lock_sha256 = expected_runtime_lock_sha256(root_path)
    require(
        record_dict.get("runtime_lock_sha256") == actual_lock_sha256,
        "Orientation marker missing or mismatched runtime_lock_sha256",
    )

    art_sha = record_dict.get("artifact_sha256")
    valid_art_sha = (
        isinstance(art_sha, str) and bool(re.fullmatch(r"[0-9a-f]{64}", art_sha))
    ) or (
        isinstance(art_sha, dict)
        and bool(art_sha)
        and all(
            isinstance(v, str) and bool(re.fullmatch(r"[0-9a-f]{64}", v))
            for v in art_sha.values()
        )
    )
    acc_sha = _require_sha256_hex(record_dict.get("acceptance_sha256"), "orientation.acceptance_sha256")
    require(valid_art_sha, "Orientation marker missing artifact_sha256 binding")

    if acceptance_envelope is not None:
        val_envelope = acceptance_envelope
        val_file_sha = None
    else:
        val_marker_path = root_path / "validation" / f"{task_name}.json"
        require(
            val_marker_path.is_file(),
            f"Orientation marker requires existing validation marker {val_marker_path}",
        )
        val_envelope = json.loads(val_marker_path.read_text(encoding="utf-8"))
        val_file_sha = file_sha256(val_marker_path, uncached=True)

    validate_acceptance(
        val_envelope,
        task_name,
        root_path,
        app_root=app_path,
        gpu=gpu,
        require_completion=require_completion,
    )
    val_dev_record = val_envelope["devices"][live_dev]
    expected_acc_shas = {canonical_json_sha256(val_dev_record)}
    if val_file_sha is not None:
        expected_acc_shas.add(val_file_sha)
    require(
        acc_sha in expected_acc_shas,
        "Orientation marker acceptance_sha256 does not match validated acceptance record",
    )
    require(
        record_dict.get("run_id") == val_dev_record.get("run_id"),
        "Orientation marker run_id does not match validated acceptance record run_id",
    )

    if require_completion:
        validate_completion(
            root_path,
            app_root=app_path,
            gpu=gpu,
            task_orientation_overrides={task_name: record_dict},
        )
        for peer_task in GTRC_TASKS:
            if peer_task != task_name:
                validate_orientation(
                    peer_task,
                    root_path,
                    app_root=app_path,
                    gpu=gpu,
                    require_completion=False,
                )
    return True


def check_orientation(
    task: str,
    model_root: Path | None = None,
    *,
    app_root: Path | None = None,
    gpu: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Non-throwing wrapper around `validate_orientation` returning `(ok, message)`."""
    root = Path(model_root or MODEL_ROOT)
    marker_path = root / "orientation-validation" / f"{task}.json"
    if not marker_path.is_file():
        return False, f"missing orientation marker {marker_path}"
    try:
        record = json.loads(marker_path.read_text(encoding="utf-8"))
        validate_orientation(record, task, root, app_root=app_root, gpu=gpu)
        return True, "validated"
    except Exception as exc:
        return False, str(exc)


def _validate_series_headers(
    files: list[str],
    *,
    expected_modality: str,
    label: str,
    task: str,
) -> dict[str, Any]:
    """Validate Modality, StudyInstanceUID, SeriesInstanceUID, FrameOfReferenceUID, and unique SOPInstanceUID on every slice."""
    if not files:
        raise ValueError(f"{task} {label} series contains no DICOM files")
    modalities: set[str] = set()
    study_uids: set[str] = set()
    series_uids: set[str] = set()
    for_uids: set[str] = set()
    sop_uids: set[str] = set()
    for idx, fpath in enumerate(files):
        ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        mod = str(getattr(ds, "Modality", "") or "").strip().upper()
        study_uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        for_uid = str(getattr(ds, "FrameOfReferenceUID", "") or "").strip()
        sop_uid = str(getattr(ds, "SOPInstanceUID", "") or "").strip()
        image_type = [str(x).strip().upper() for x in (getattr(ds, "ImageType", None) or [])]
        if "LOCALIZER" in image_type or "SCOUT" in image_type:
            raise ValueError(
                f"{task} {label} slice {idx} ({Path(fpath).name}) is a LOCALIZER/SCOUT image; "
                "3D volumetric series required"
            )
        if not study_uid:
            raise ValueError(f"{task} {label} slice {idx} ({Path(fpath).name}) is missing StudyInstanceUID")
        if not series_uid:
            raise ValueError(f"{task} {label} slice {idx} ({Path(fpath).name}) is missing SeriesInstanceUID")
        if not for_uid:
            raise ValueError(f"{task} {label} slice {idx} ({Path(fpath).name}) is missing FrameOfReferenceUID")
        if not sop_uid:
            raise ValueError(f"{task} {label} slice {idx} ({Path(fpath).name}) is missing SOPInstanceUID")
        if sop_uid in sop_uids:
            raise ValueError(
                f"{task} {label} slice {idx} ({Path(fpath).name}) has duplicate SOPInstanceUID {sop_uid!r}"
            )
        modalities.add(mod)
        study_uids.add(study_uid)
        series_uids.add(series_uid)
        for_uids.add(for_uid)
        sop_uids.add(sop_uid)

    if modalities != {expected_modality}:
        raise ValueError(
            f"{task} {label} series must have Modality={expected_modality!r} on every slice, "
            f"got {sorted(modalities)}"
        )
    if len(study_uids) != 1:
        raise ValueError(
            f"{task} {label} series has inconsistent/Mixed StudyInstanceUID across slices: {sorted(study_uids)}"
        )
    if len(series_uids) != 1:
        raise ValueError(
            f"{task} {label} series has inconsistent/Mixed SeriesInstanceUID across slices: {sorted(series_uids)}"
        )
    if len(for_uids) != 1:
        raise ValueError(
            f"{task} {label} series has inconsistent/Mixed FrameOfReferenceUID across CT/PET slices: {sorted(for_uids)}"
        )
    return {
        "study_uid": next(iter(study_uids)),
        "series_uid": next(iter(series_uids)),
        "frame_of_reference_uid": next(iter(for_uids)),
        "slice_count": len(files),
    }


def inspect_ct_series_metadata(ct_dir: Path | str, task: str = "gtrc_psma_petct_ttb") -> dict[str, Any]:
    """Validate every slice in a paired CT DICOM directory and return series metadata."""
    ct_path = Path(ct_dir)
    if not ct_path.is_dir():
        raise ValueError(f"{task} requires paired CT DICOM series in {ct_path}")
    ct_files = ordered_dicom_files(ct_path)
    return _validate_series_headers(ct_files, expected_modality="CT", label="paired CT", task=task)


def check_grid(
    cand_nifti: Path | str | sitk.Image,
    ref_nifti: Path | str | sitk.Image,
    *,
    origin_tol_mm: float = GTRC_TOLERANCE_CONTRACT["grid_origin_mm_tol"],
    spacing_tol_mm: float = GTRC_TOLERANCE_CONTRACT["grid_spacing_mm_tol"],
    direction_tol: float = GTRC_TOLERANCE_CONTRACT["grid_direction_cos_tol"],
) -> dict[str, Any]:
    """Verify that `cand_nifti` matches `ref_nifti` on size, spacing, origin, and direction."""
    cand_im = cand_nifti if isinstance(cand_nifti, sitk.Image) else sitk.ReadImage(str(cand_nifti))
    ref_im = ref_nifti if isinstance(ref_nifti, sitk.Image) else sitk.ReadImage(str(ref_nifti))
    cand_size = list(map(int, cand_im.GetSize()))
    ref_size = list(map(int, ref_im.GetSize()))
    cand_spacing = [float(x) for x in cand_im.GetSpacing()]
    ref_spacing = [float(x) for x in ref_im.GetSpacing()]
    cand_origin = [float(x) for x in cand_im.GetOrigin()]
    ref_origin = [float(x) for x in ref_im.GetOrigin()]
    cand_dir = [float(x) for x in cand_im.GetDirection()]
    ref_dir = [float(x) for x in ref_im.GetDirection()]

    size_equal = cand_size == ref_size
    spacing_max_delta = float(np.max(np.abs(np.asarray(cand_spacing) - np.asarray(ref_spacing))))
    origin_max_delta = float(np.max(np.abs(np.asarray(cand_origin) - np.asarray(ref_origin))))
    direction_max_delta = float(np.max(np.abs(np.asarray(cand_dir) - np.asarray(ref_dir))))

    if (
        not size_equal
        or spacing_max_delta > spacing_tol_mm
        or origin_max_delta > origin_tol_mm
        or direction_max_delta > direction_tol
    ):
        raise ValueError(
            f"NIfTI grid mismatch: size {cand_size} vs {ref_size}, "
            f"spacing_max_delta={spacing_max_delta:.6g} mm, "
            f"origin_max_delta={origin_max_delta:.6g} mm, "
            f"direction_max_delta={direction_max_delta:.6g}"
        )
    return {
        "passed": True,
        "size_equal": True,
        "cand_size": cand_size,
        "ref_size": ref_size,
        "spacing_max_delta_mm": spacing_max_delta,
        "origin_max_delta_mm": origin_max_delta,
        "direction_max_delta": direction_max_delta,
    }


def validate_pet_ct_job(
    job: Path,
    task: str,
    *,
    require_same_study: bool = True,
    require_same_frame_of_reference: bool = True,
) -> dict[str, Any]:
    """Validate that `job/input` (PET) and `job/input_ct` (CT) exist and share Study/FoR on every slice."""
    if task not in GTRC_TASKS:
        raise ValueError(f"Unknown GTRC task: {task}")
    pet_dir = job / "input"
    ct_dir = job / "input_ct"
    if not pet_dir.is_dir():
        raise ValueError(f"{task} requires PET DICOM series in {pet_dir}")
    if not ct_dir.is_dir():
        raise ValueError(f"{task} requires paired CT DICOM series in {ct_dir}")

    pet_files = ordered_dicom_files(pet_dir)
    ct_files = ordered_dicom_files(ct_dir)
    pet_hdr = _validate_series_headers(pet_files, expected_modality="PT", label="reference PET", task=task)
    ct_hdr = _validate_series_headers(ct_files, expected_modality="CT", label="paired CT", task=task)

    pet_study = pet_hdr["study_uid"]
    ct_study = ct_hdr["study_uid"]
    if require_same_study and (not pet_study or pet_study != ct_study):
        raise ValueError(
            f"{task} requires PET and CT in the same StudyInstanceUID "
            f"(PET={pet_study!r}, CT={ct_study!r})"
        )

    pet_for = pet_hdr["frame_of_reference_uid"]
    ct_for = ct_hdr["frame_of_reference_uid"]
    same_for = bool(pet_for and pet_for == ct_for)
    if require_same_frame_of_reference and not same_for:
        raise ValueError(
            f"{task} requires PET and CT sharing one FrameOfReferenceUID for identity physical resampling "
            f"(PET={pet_for!r}, CT={ct_for!r})"
        )

    return {
        "pet_series_uid": pet_hdr["series_uid"],
        "ct_series_uid": ct_hdr["series_uid"],
        "study_uid": pet_study,
        "pet_frame_of_reference_uid": pet_for,
        "ct_frame_of_reference_uid": ct_for,
        "same_frame_of_reference": same_for,
        "pet_slices": pet_hdr["slice_count"],
        "ct_slices": ct_hdr["slice_count"],
    }


def _physical_bounds_lps(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    size = image.GetSize()
    corners = []
    for ix in (0, max(0, size[0] - 1)):
        for iy in (0, max(0, size[1] - 1)):
            for iz in (0, max(0, size[2] - 1)):
                corners.append(image.TransformIndexToPhysicalPoint((int(ix), int(iy), int(iz))))
    pts = np.asarray(corners, dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def check_physical_overlap(
    pet_nifti: Path | str,
    ct_nifti: Path | str,
    *,
    suv_threshold: float | None = None,
    axial_tolerance_mm: float | None = None,
    min_overlap_fraction: float = 0.25,
) -> dict[str, Any]:
    """Verify physical LPS coverage between PET and CT before identity resampling.

    Enforces:
      1. Bounding-box overlap fraction >= `min_overlap_fraction`.
      2. CT physical axial (Z / Inferior-Superior) extent covers the PET axial extent within
         `axial_tolerance_mm` (default: `max(5.0, pet_spacing_z, ct_spacing_z)` mm, i.e. one
         axial slice thickness or 5.0 mm at the superior/inferior scan boundaries).
      3. When `suv_threshold` is supplied, 100% of PET voxels with `SUV >= suv_threshold`
         must be supported by SimpleITK's `sitkLinear` resampler on the CT grid (evaluated
         directly by resampling an all-ones CT support image onto the PET grid with the exact
         identity transform, `sitkLinear` interpolator, and `0.0` outside default used in
         `preprocess_pet_ct`), so no above-threshold PET voxel receives the `-1000` HU
         outside-buffer default value.
    """
    pet_im = sitk.ReadImage(str(pet_nifti))
    ct_im = sitk.ReadImage(str(ct_nifti))
    pet_min, pet_max = _physical_bounds_lps(pet_im)
    ct_min, ct_max = _physical_bounds_lps(ct_im)

    inter_min = np.maximum(pet_min, ct_min)
    inter_max = np.minimum(pet_max, ct_max)
    inter_lengths = np.maximum(0.0, inter_max - inter_min)
    pet_lengths = np.maximum(1e-6, pet_max - pet_min)

    inter_vol = float(np.prod(inter_lengths))
    pet_vol = float(np.prod(pet_lengths))
    overlap_fraction = inter_vol / pet_vol
    if overlap_fraction < min_overlap_fraction:
        raise ValueError(
            f"PET and CT physical bounding boxes have insufficient overlap "
            f"({overlap_fraction:.3f} < {min_overlap_fraction:.3f}); "
            f"PET bounds=[{pet_min.tolist()}, {pet_max.tolist()}], "
            f"CT bounds=[{ct_min.tolist()}, {ct_max.tolist()}]"
        )

    pet_spacing = [float(x) for x in pet_im.GetSpacing()]
    ct_spacing = [float(x) for x in ct_im.GetSpacing()]
    tol_z = (
        float(axial_tolerance_mm)
        if axial_tolerance_mm is not None
        else max(AXIAL_COVERAGE_MIN_TOLERANCE_MM, pet_spacing[2], ct_spacing[2])
    )
    inferior_deficit_mm = max(0.0, float(ct_min[2] - pet_min[2]))
    superior_deficit_mm = max(0.0, float(pet_max[2] - ct_max[2]))
    if inferior_deficit_mm > tol_z or superior_deficit_mm > tol_z:
        raise ValueError(
            f"Paired CT axial extent [{ct_min[2]:.2f}, {ct_max[2]:.2f}] mm does not cover PET axial extent "
            f"[{pet_min[2]:.2f}, {pet_max[2]:.2f}] mm within tolerance {tol_z:.2f} mm "
            f"(inferior deficit={inferior_deficit_mm:.2f} mm, superior deficit={superior_deficit_mm:.2f} mm)"
        )

    above_threshold_voxels = None
    outside_ct_voxels = None
    all_above_inside_ct = True
    if suv_threshold is not None:
        pet_arr = sitk.GetArrayFromImage(pet_im)
        above_mask = pet_arr >= float(suv_threshold)
        above_threshold_voxels = int(np.count_nonzero(above_mask))
        outside_ct_voxels = 0
        if above_threshold_voxels > 0:
            # Decide support directly from the resampled all-ones CT image using the exact
            # identity transform, sitkLinear interpolator, and outside default behavior as
            # upstream CT resampling in preprocess_pet_ct.
            ct_support = sitk.Image(ct_im.GetSize(), sitk.sitkFloat32) + 1.0
            ct_support.CopyInformation(ct_im)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(pet_im)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0.0)
            resampled_support = sitk.GetArrayFromImage(resampler.Execute(ct_support))

            inside = resampled_support[above_mask] >= (1.0 - 1e-5)

            outside_ct_voxels = int(np.sum(~inside))
            all_above_inside_ct = outside_ct_voxels == 0
            if not all_above_inside_ct:
                raise ValueError(
                    f"{outside_ct_voxels} of {above_threshold_voxels} above-threshold PET voxels (SUV >= {suv_threshold}) "
                    "lie outside the physical CT support; GTRC-Net requires CT coverage of all "
                    "above-threshold PET uptake"
                )

    return {
        "pet_size": list(map(int, pet_im.GetSize())),
        "pet_spacing_mm": pet_spacing,
        "ct_size": list(map(int, ct_im.GetSize())),
        "ct_spacing_mm": ct_spacing,
        "pet_bounds_lps_mm": [pet_min.tolist(), pet_max.tolist()],
        "ct_bounds_lps_mm": [ct_min.tolist(), ct_max.tolist()],
        "pet_bbox_overlap_fraction": round(overlap_fraction, 6),
        "axial_tolerance_mm": round(tol_z, 4),
        "axial_inferior_deficit_mm": round(inferior_deficit_mm, 4),
        "axial_superior_deficit_mm": round(superior_deficit_mm, 4),
        "above_threshold_voxel_count": above_threshold_voxels,
        "above_threshold_outside_ct_voxels": outside_ct_voxels,
        "all_above_threshold_inside_ct": all_above_inside_ct,
    }


def _short_ipc_tmpdir(preferred_parent: Path | None = None) -> tempfile.TemporaryDirectory[str]:
    """Create a short directory path for PyTorch/nnU-Net multiprocessing AF_UNIX sockets."""
    base = preferred_parent if (preferred_parent and len(str(preferred_parent)) < 48) else Path("/tmp")
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="rtseg-gtrc-", dir=str(base))


def run_task(
    task: str,
    job: Path,
    app_root: Path | None = None,
    model_root: Path | None = None,
    python_exe: str | Path | None = None,
    runner: Callable[..., Any] | None = None,
    *,
    suv_threshold: float | None = None,
    suv_threshold_source: str | None = None,
    _acceptance_bootstrap: bool = False,
) -> dict[str, Any]:
    """Run a GTRC-Net PET/CT task end-to-end on a staged portal job."""
    if task not in GTRC_TASKS:
        raise ValueError(f"Unknown GTRC task: {task}")
    job = Path(job)
    app = Path(app_root or APP_ROOT)
    root = Path(model_root or MODEL_ROOT)
    host_py = str(python_exe or sys.executable)
    spec = GTRC_TASKS[task]

    def _exec(cmd: list[str], env: dict[str, str] | None = None) -> None:
        if runner is not None:
            runner(cmd, env=env)
        else:
            subprocess.run(cmd, env=env, check=True)

    # Resolve and validate patient/task SUVbw threshold first (fails closed if FDG threshold is missing)
    threshold, threshold_source = resolve_job_suv_threshold(
        task,
        job,
        explicit_threshold=suv_threshold,
        explicit_source=suv_threshold_source,
    )

    # Execution-boundary gates: always verify installed source commit, weight SHA-256 digests,
    # runtime lock file, and live CUDA runtime/device compatibility; outside acceptance bootstrap,
    # also verify per-device acceptance and orientation markers for the live GPU.
    actual_commit = installed_source_commit(root / spec["source_dir"])
    if actual_commit != spec["source_commit"]:
        raise RuntimeError(
            f"Source commit mismatch for {task}: got {actual_commit}, expected {spec['source_commit']}"
        )
    ok_w, msg_w, _ = verify_weight_inventory(task, root, verify_hashes=True)
    if not ok_w:
        raise RuntimeError(f"GTRC weight verification failed for {task}: {msg_w}")
    expected_runtime_lock_sha256(root)

    cuda_ok, cuda_msg, live_gpu = verify_runtime_cuda_compatibility(root, env_name=spec["runtime_env"])
    if not cuda_ok:
        raise RuntimeError(f"GTRC runtime/device check failed for {task}: {cuda_msg}")
    device_key(live_gpu)

    if not _acceptance_bootstrap:
        validate_acceptance(task, root, app_root=app, gpu=live_gpu)
        validate_orientation(task, root, app_root=app, gpu=live_gpu)

    pairing_info = validate_pet_ct_job(job, task)
    work = job / "external" / task
    data_dir = work / "data"
    out_dir = work / "output"
    job_output = job / "output"
    for d in (data_dir, out_dir, job_output):
        d.mkdir(parents=True, exist_ok=True)

    ct_nifti = data_dir / "ct_hu.nii.gz"
    pet_raw_nifti = data_dir / "pet_raw.nii.gz"
    pet_suv_nifti = data_dir / "pet_suvbw.nii.gz"
    suvbw_meta_json = data_dir / "suvbw_metadata.json"

    _exec([host_py, str(app / "dicom_series_to_nifti.py"), str(job / "input_ct"), str(ct_nifti)])
    _exec([host_py, str(app / "dicom_series_to_nifti.py"), str(job / "input"), str(pet_raw_nifti)])
    suvbw_meta = convert_pet_nifti_to_suvbw(
        pet_raw_nifti,
        job / "input",
        pet_suv_nifti,
        metadata_out=suvbw_meta_json,
    )
    spatial_info = check_physical_overlap(
        pet_suv_nifti,
        ct_nifti,
        suv_threshold=threshold,
    )

    pred_mask = out_dir / f"{task}_RTSTRUCT.dcm.ttb_mask.nii.gz"
    nnunet_labels = out_dir / f"{task}_RTSTRUCT.dcm.nnunet_labels.nii.gz"
    subregions_nifti = out_dir / f"{task}_RTSTRUCT.dcm.subregions.nii.gz"
    lesions_json = out_dir / f"{task}_RTSTRUCT.dcm.lesions.json"
    provenance_json = out_dir / f"{task}_RTSTRUCT.dcm.provenance.json"
    context_json = data_dir / "job_context.json"
    context_json.write_text(
        json.dumps(
            {
                "pairing": pairing_info,
                "suvbw_conversion": suvbw_meta,
                "spatial_alignment": spatial_info,
                "suv_threshold": {
                    "value": threshold,
                    "source": threshold_source,
                },
            },
            indent=2,
        )
        + "\n"
    )

    py_bin = runtime_python(spec["runtime_env"], root)
    with _short_ipc_tmpdir() as ipc_tmp:
        env = os.environ.copy()
        env["TMPDIR"] = ipc_tmp
        env["RTSEG_MODEL_ROOT"] = str(root)
        env["RTSEG_APP_ROOT"] = str(app)
        cmd = [
            str(py_bin),
            str(app / spec["runner_script"]),
            "--task",
            task,
            "--pt-suv",
            str(pet_suv_nifti),
            "--ct",
            str(ct_nifti),
            "--suv-threshold",
            repr(threshold),
            "--suv-threshold-source",
            threshold_source,
            "--work-dir",
            str(work / "infer_work"),
            "--output-mask",
            str(pred_mask),
            "--output-nnunet-labels",
            str(nnunet_labels),
            "--output-subregions",
            str(subregions_nifti),
            "--output-lesions-json",
            str(lesions_json),
            "--output-provenance-json",
            str(provenance_json),
            "--job-context-json",
            str(context_json),
        ]
        if _acceptance_bootstrap:
            cmd.append("--gtrc-acceptance-bootstrap")
        _exec(cmd, env=env)

    # Copy declared sidecar and mask artifacts (all named `<task>_RTSTRUCT.dcm.*`)
    # to job/output before RTSTRUCT export so empty predictions and SSH retrieval retain them.
    for artifact in (lesions_json, provenance_json, pred_mask, nnunet_labels, subregions_nifti):
        if artifact.is_file():
            shutil.copyfile(artifact, job_output / artifact.name)

    mask_arr = np.asanyarray(nib.load(pred_mask).dataobj)
    if not np.any(mask_arr > 0):
        raise SystemExit("empty prediction: no RTSTRUCT created")

    rtstruct_out = job_output / f"{task}_RTSTRUCT.dcm"
    _exec(
        [
            host_py,
            str(app / "convert_validate_rtstruct.py"),
            "--prediction",
            str(pred_mask),
            "--model-input",
            str(pet_suv_nifti),
            "--dicom",
            str(job / "input"),
            "--output",
            str(rtstruct_out),
            "--labels",
            str(app / spec["label_map"]),
        ]
    )
    return json.loads((job_output / lesions_json.name).read_text())

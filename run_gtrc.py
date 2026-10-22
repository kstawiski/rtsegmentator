#!/usr/bin/env python3
"""Isolated runner for GTRC-Net PSMA PET/CT and FDG PET/CT inference.

Reproduces the published GTRC-Net full inference pipeline (GTRC.py / GTRC_Infer.py /
gtrc_utils.py at commit 9db02ddd569fd4e82a2caf8973aba6741d5cd4a8) while fixing the
three upstream static blockers:
  1. Windows hardcoded `nn_predict_exe = r"C:\\GTRC\\GTRC_venv\\Scripts\\nnUNetv2_predict.exe"`
     and unchecked `os.system` invocation.
  2. Case-sensitive `data/LuPSMA` vs `data/LUPSMA` path mismatch on Linux.
  3. Import-time directory creation outside the job workspace in `nnunet_config_paths.py`.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
import SimpleITK as sitk

from gtrc_models import (
    GTRC_TASKS,
    device_key,
    expected_runtime_lock_sha256,
    installed_source_commit,
    is_cuda_arch_compatible,
    resolve_job_suv_threshold,
    validate_acceptance,
    validate_completion,
    validate_orientation,
    verify_weight_inventory,
)

TRAINING_COLUMNS = [
    "total_volume",
    "suv_max",
    "suv_mean",
    "ct_hu_mean",
    "pred_ttb_overlap",
    "pred_norm_overlap",
]
LOCAL_MAXIMA_THRESHOLD = 0.16666
SPHERE_RADIUS_MM = 8.0

DETERMINISTIC_NNUNET_ENTRYPOINT = """\
import json
import os
from pathlib import Path
import sys
import torch
import nnunetv2.inference.predict_from_raw_data as _p

_orig_init = _p.nnUNetPredictor.__init__


def _write_runtime_flags():
    out_path = os.environ.get("GTRC_NNUNET_FLAGS_OUT")
    if not out_path:
        return
    cuda_ok = bool(torch.cuda.is_available())
    arch_list = (
        list(torch.cuda.get_arch_list())
        if hasattr(torch.cuda, "get_arch_list")
        else (torch._C._cuda_getArchFlags() or "").split()
    )
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else None
    dev_cap = [int(x) for x in torch.cuda.get_device_capability(0)] if cuda_ok else None
    cuda_ver = getattr(torch.version, "cuda", None)
    payload = {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_available": cuda_ok,
        "name": dev_name,
        "device_name": dev_name,
        "gpu_name": dev_name,
        "compute_capability": dev_cap,
        "device_capability": dev_cap,
        "cuda": cuda_ver,
        "cuda_version": cuda_ver,
        "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "compiled_architectures": arch_list,
        "arch_list": arch_list,
        "torch_version": getattr(torch, "__version__", None),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "inference_mode": "production",
        "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")


def _deterministic_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _write_runtime_flags()


_p.nnUNetPredictor.__init__ = _deterministic_init
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    rc = _p.predict_entry_point()
    _write_runtime_flags()
    sys.exit(rc)
"""


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_provenance() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "packages": {
            pkg: _pkg_version(pkg)
            for pkg in (
                "torch",
                "nnunetv2",
                "tensorflow-cpu",
                "tensorflow",
                "keras",
                "numpy",
                "SimpleITK",
                "scikit-image",
                "scipy",
                "pandas",
                "scikit-learn",
                "nibabel",
                "pydicom",
            )
        },
        "inference_mode": "production",
        "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
        "cudnn_deterministic_enforced": True,
        "cudnn_benchmark_enforced": False,
    }
    try:
        import torch

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cuda_ok = bool(torch.cuda.is_available())
        arch_flags = (
            list(torch.cuda.get_arch_list())
            if hasattr(torch.cuda, "get_arch_list")
            else (torch._C._cuda_getArchFlags() or "").split()
        )
        cuda_ver = getattr(torch.version, "cuda", None)
        info["cuda_available"] = cuda_ok
        info["cuda"] = cuda_ver
        info["cuda_version"] = cuda_ver
        info["compiled_architectures"] = arch_flags
        info["cuda_arch_flags"] = arch_flags
        if cuda_ok:
            dev_name = torch.cuda.get_device_name(0)
            cap = [int(x) for x in torch.cuda.get_device_capability(0)]
            info["name"] = dev_name
            info["device_name"] = dev_name
            info["gpu_name"] = dev_name
            info["compute_capability"] = cap
            info["device_capability"] = cap
            info["cuda_arch_compatible"] = is_cuda_arch_compatible(cap, arch_flags)
            x = torch.ones((2, 2), device="cuda", dtype=torch.float32)
            info["tensor_probe_ok"] = bool(float((x + x).sum().item()) == 8.0)
        else:
            info["compute_capability"] = None
            info["device_capability"] = None
            info["cuda_arch_compatible"] = False
            info["tensor_probe_ok"] = False
    except Exception:
        info["cuda_available"] = False
        info["compiled_architectures"] = []
        info["cuda_arch_flags"] = []
        info["compute_capability"] = None
        info["device_capability"] = None
        info["cuda_arch_compatible"] = False
        info["tensor_probe_ok"] = False
    return info


def load_upstream_subregion_fn(source_dir: Path | None) -> Callable[..., sitk.Image]:
    """Load `create_subregion_labels` from the pinned upstream `gtrc_utils.py` if present."""
    if source_dir is not None:
        utils_py = Path(source_dir) / "gtrc_utils.py"
        if utils_py.is_file():
            prev_no_bytecode = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                spec = importlib.util.spec_from_file_location("gtrc_utils_upstream", utils_py)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod.create_subregion_labels
            finally:
                sys.dont_write_bytecode = prev_no_bytecode
    return create_subregion_labels_verbatim


def get_expansion_sphere(radius: float, spacing: Sequence[float]) -> np.ndarray:
    """Exact reproduction of upstream `gtrc_utils.get_expansion_sphere`."""
    xlim = np.ceil(radius / spacing[0])
    ylim = np.ceil(radius / spacing[1])
    zlim = np.ceil(radius / spacing[2])
    x = np.arange(-xlim, xlim + 1, 1)
    y = np.arange(-ylim, ylim + 1, 1)
    z = np.arange(-zlim, zlim + 1, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    sphere = (
        np.sqrt((xx * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (zz * spacing[2]) ** 2)
        <= radius
    ).astype(np.float32)
    sphere = np.swapaxes(sphere, 0, 2)
    return sphere


def create_subregion_labels_verbatim(
    qspect: sitk.Image,
    threshold_array: np.ndarray | sitk.Image,
    local_maxima_threshold: float = LOCAL_MAXIMA_THRESHOLD,
    sphere_radius: float = SPHERE_RADIUS_MM,
) -> sitk.Image:
    """Exact reproduction of upstream `gtrc_utils.create_subregion_labels`."""
    qs_ar = sitk.GetArrayFromImage(qspect)
    if isinstance(threshold_array, sitk.Image):
        threshold_array = sitk.GetArrayFromImage(threshold_array)

    spacing = qspect.GetSpacing()
    sphere = get_expansion_sphere(sphere_radius, spacing)

    peaks = h_maxima(qs_ar, local_maxima_threshold, sphere)
    numbered_peaks, n_peaks = ndimage.label(peaks)
    labels_ws = watershed(-qs_ar, numbered_peaks, mask=threshold_array)

    # Preserve upstream's exact missing-peaks catch (including `n_peaks + i` indexing
    # and sitkInt16 cast) for bit-exact numerical equivalence.
    missing_peaks_ar = threshold_array - (labels_ws > 0).astype("uint8")
    if missing_peaks_ar.sum() > 0:
        missing_peaks_ar_numbered, n_missing = ndimage.label(missing_peaks_ar)
        for i in range(n_missing):
            labels_ws[missing_peaks_ar_numbered == (i + 1)] = n_peaks + i

    ws_im = sitk.GetImageFromArray(labels_ws)
    ws_im.CopyInformation(qspect)
    ws_im = sitk.Cast(ws_im, sitk.sitkInt16)
    return ws_im


def preprocess_pet_ct(
    pt_suv: sitk.Image,
    ct: sitk.Image,
    suv_threshold: float,
) -> tuple[sitk.Image, sitk.Image]:
    """Resample CT to PET physical grid (linear, -1000 default) and divide PET by SUV threshold."""
    if suv_threshold <= 0.0 or not np.isfinite(suv_threshold):
        raise ValueError(f"suv_threshold must be finite and > 0, got {suv_threshold}")
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(pt_suv)
    rs.SetDefaultPixelValue(-1000)
    ct_rs = rs.Execute(ct)
    pt_rs = pt_suv / float(suv_threshold)
    return pt_rs, ct_rs


def run_nnunet_predict(
    nn_input_dir: Path,
    nn_output_dir: Path,
    *,
    dataset_id: int,
    nnunet_results_dir: Path,
    work_dir: Path,
    fold: int | str = "all",
) -> Path:
    """Invoke nnUNetv2_predict with deterministic cuDNN flags and paths inside `work_dir`."""
    raw_dir = work_dir / "nnUNet_raw"
    prep_dir = work_dir / "nnUNet_preprocessed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)
    nn_output_dir.mkdir(parents=True, exist_ok=True)

    flags_out = nn_output_dir / "nnunet_runtime_flags.json"
    cmd = [
        sys.executable,
        "-c",
        DETERMINISTIC_NNUNET_ENTRYPOINT,
        "-i",
        str(nn_input_dir),
        "-o",
        str(nn_output_dir),
        "-d",
        str(dataset_id),
        "-c",
        "3d_fullres",
    ]
    if fold in (0, 1, 2, 3, 4, "0", "1", "2", "3", "4"):
        cmd.extend(["-f", str(fold)])

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw_dir)
    env["nnUNet_preprocessed"] = str(prep_dir)
    env["nnUNet_results"] = str(nnunet_results_dir)
    env["GTRC_NNUNET_FLAGS_OUT"] = str(flags_out)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTHONHASHSEED", "0")

    subprocess.run(cmd, env=env, check=True)
    pred_path = nn_output_dir / "gtrc.nii.gz"
    if not pred_path.is_file():
        raise RuntimeError(f"nnUNetv2_predict did not write expected output {pred_path}")
    if not flags_out.is_file():
        raise RuntimeError(f"nnUNetv2_predict did not write expected runtime flags {flags_out}")
    return pred_path


def load_keras_consensus_models(
    consensus_dir: Path,
    *,
    fold: int | str = "all",
) -> list[Callable[[np.ndarray], Any]]:
    """Load the 5-fold Keras consensus MLP models on CPU (keeping TensorFlow off GPU)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    models = []
    if fold == "all":
        for fold_n in range(5):
            model_path = consensus_dir / f"fold_{fold_n}" / "consensus_model.keras"
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing Keras consensus model: {model_path}")
            models.append(load_model(str(model_path), compile=False))
    elif fold in (0, 1, 2, 3, 4, "0", "1", "2", "3", "4"):
        model_path = consensus_dir / f"fold_{fold}" / "consensus_model.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing Keras consensus model: {model_path}")
        models.append(load_model(str(model_path), compile=False))
    else:
        raise ValueError(f"Invalid fold specification: {fold!r}")
    return models


def evaluate_consensus_and_paint(
    pt_rs: sitk.Image,
    ct_rs: sitk.Image,
    pred_im: sitk.Image,
    models: Sequence[Callable[[np.ndarray], Any]],
    *,
    suv_threshold: float,
    subregion_fn: Callable[..., sitk.Image] = create_subregion_labels_verbatim,
    local_maxima_threshold: float = LOCAL_MAXIMA_THRESHOLD,
    sphere_radius: float = SPHERE_RADIUS_MM,
) -> tuple[sitk.Image, sitk.Image, list[dict[str, Any]], dict[str, int]]:
    """Execute the exact upstream subregion construction, 6-feature extraction, voting, and TTB painting."""
    spacing = pt_rs.GetSpacing()
    voxel_volume = float(np.prod(np.array(spacing)) / 1000.0)

    pt_ar = sitk.GetArrayFromImage(pt_rs)
    ct_ar = sitk.GetArrayFromImage(ct_rs)
    pred_ar = sitk.GetArrayFromImage(pred_im)
    pred_ttb_ar = (pred_ar == 1).astype("int8")
    pred_norm_ar = (pred_ar == 2).astype("int8")

    # Voxels >= 1.0 in threshold-normalized PET (i.e. SUV >= suv_threshold)
    tar = (pt_ar >= 1.0).astype("int8")

    subregion_im = subregion_fn(
        pt_rs,
        tar,
        local_maxima_threshold=local_maxima_threshold,
        sphere_radius=sphere_radius,
    )
    subregion_ar = sitk.GetArrayFromImage(subregion_im)
    total_subregions = int(subregion_ar.max()) if subregion_ar.size else 0
    bboxes = ndimage.find_objects(subregion_ar, max_label=total_subregions)

    rows: list[dict[str, Any]] = []
    for i in range(total_subregions):
        bbox = bboxes[i] if i < len(bboxes) else None
        if bbox is None:
            continue
        subregion_bool = subregion_ar[bbox] == (i + 1)
        total_voxels = int(subregion_bool.sum())
        if total_voxels > 0:
            total_volume = float(total_voxels * voxel_volume)
            suv_max = float(pt_ar[bbox][subregion_bool].max())
            suv_mean = float(pt_ar[bbox][subregion_bool].mean())
            ct_hu_mean = float(ct_ar[bbox][subregion_bool].mean())
            pred_ttb_overlap = float(
                np.logical_and(subregion_bool, pred_ttb_ar[bbox] > 0.5).sum() / total_voxels
            )
            pred_norm_overlap = float(
                np.logical_and(subregion_bool, pred_norm_ar[bbox] > 0.5).sum() / total_voxels
            )
            rows.append(
                {
                    "subregion_index": i + 1,
                    "total_voxels": total_voxels,
                    "total_volume": total_volume,
                    "suv_max": suv_max,
                    "suv_mean": suv_mean,
                    "ct_hu_mean": ct_hu_mean,
                    "pred_ttb_overlap": pred_ttb_overlap,
                    "pred_norm_overlap": pred_norm_overlap,
                }
            )

    df = pd.DataFrame(rows)
    output_ar = np.zeros(pt_ar.shape, dtype="int8")
    subregion_records: list[dict[str, Any]] = []
    accepted_count = 0

    for _, row in df.iterrows():
        label_value = int(row["subregion_index"])
        x = np.expand_dims(row[TRAINING_COLUMNS].values.astype(np.float32), 0)
        fold_scores: list[float] = []
        votes = 0
        for model in models:
            raw_out = model(x)
            score = float(raw_out.numpy()[0][0] if hasattr(raw_out, "numpy") else np.asarray(raw_out)[0][0])
            fold_scores.append(score)
            if score > 0.5:
                votes += 1
        accepted = votes > (float(len(models)) / 2.0)
        if accepted:
            bbox = bboxes[label_value - 1]
            output_ar[bbox][subregion_ar[bbox] == label_value] = 1
            accepted_count += 1
        subregion_records.append(
            {
                "subregion_id": label_value,
                "voxels": int(row["total_voxels"]),
                "volume_ml": float(row["total_volume"]),
                "normalized_suv_max": float(row["suv_max"]),
                "normalized_suv_mean": float(row["suv_mean"]),
                "raw_suv_max": float(row["suv_max"] * suv_threshold),
                "raw_suv_mean": float(row["suv_mean"] * suv_threshold),
                "ct_hu_mean": float(row["ct_hu_mean"]),
                "pred_ttb_overlap": float(row["pred_ttb_overlap"]),
                "pred_norm_overlap": float(row["pred_norm_overlap"]),
                "fold_scores": [round(s, 6) for s in fold_scores],
                "votes_positive": int(votes),
                "models_count": len(models),
                "accepted": bool(accepted),
            }
        )

    consensus_voxels = int((output_ar > 0).sum())
    # Re-paint any above-threshold voxel predicted as TTB by nnU-Net
    output_ar[np.logical_and(pred_ttb_ar > 0, tar > 0)] = 1
    final_voxels = int((output_ar > 0).sum())

    output_im = sitk.GetImageFromArray(output_ar)
    output_im.CopyInformation(pt_rs)
    counts = {
        "above_threshold_voxels": int(tar.sum()),
        "nnunet_ttb_voxels": int(pred_ttb_ar.sum()),
        "nnunet_norm_voxels": int(pred_norm_ar.sum()),
        "subregion_label_max": total_subregions,
        "nonempty_subregion_count": len(subregion_records),
        "accepted_subregion_count": accepted_count,
        "rejected_subregion_count": len(subregion_records) - accepted_count,
        "consensus_accepted_voxels": consensus_voxels,
        "inpainted_ttb_additional_voxels": final_voxels - consensus_voxels,
        "final_ttb_voxels": final_voxels,
    }
    return output_im, subregion_im, subregion_records, counts


def compute_ttb_summary_and_components(
    pt_suv: sitk.Image,
    ct_rs: sitk.Image,
    ttb_mask: sitk.Image,
) -> dict[str, Any]:
    """Compute whole-burden SUV/volume biomarkers and 6-connected component statistics."""
    spacing = pt_suv.GetSpacing()
    voxel_vol_ml = float(np.prod(np.array(spacing)) / 1000.0)
    suv_ar = sitk.GetArrayFromImage(pt_suv).astype(np.float64)
    ct_ar = sitk.GetArrayFromImage(ct_rs).astype(np.float64)
    mask_bool = sitk.GetArrayFromImage(ttb_mask) > 0

    total_voxels = int(mask_bool.sum())
    if total_voxels == 0:
        return {
            "empty_prediction": True,
            "ttb_voxel_count": 0,
            "ttb_volume_ml": 0.0,
            "suv_mean": None,
            "suv_max": None,
            "suv_min": None,
            "suv_median": None,
            "suv_std": None,
            "volume_intensity_product_suv_ml": 0.0,
            "ct_hu_mean": None,
            "connected_component_connectivity": "3D_6_connected",
            "connected_component_count": 0,
            "connected_components": [],
        }

    suv_vals = suv_ar[mask_bool]
    ct_vals = ct_ar[mask_bool]
    vol_ml = float(total_voxels * voxel_vol_ml)
    suv_mean = float(suv_vals.mean())
    suv_max = float(suv_vals.max())
    suv_min = float(suv_vals.min())
    suv_median = float(np.median(suv_vals))
    suv_std = float(suv_vals.std())

    comp_ar, n_comp = ndimage.label(mask_bool)
    comp_bboxes = ndimage.find_objects(comp_ar, max_label=int(n_comp))
    components: list[dict[str, Any]] = []
    for comp_idx in range(1, int(n_comp) + 1):
        bbox = comp_bboxes[comp_idx - 1]
        if bbox is None:
            continue
        c_bool = comp_ar[bbox] == comp_idx
        c_vox = int(c_bool.sum())
        c_suv = suv_ar[bbox][c_bool]
        c_ct = ct_ar[bbox][c_bool]
        components.append(
            {
                "component_id": comp_idx,
                "voxels": c_vox,
                "volume_ml": round(float(c_vox * voxel_vol_ml), 6),
                "suv_max": round(float(c_suv.max()), 6),
                "suv_mean": round(float(c_suv.mean()), 6),
                "ct_hu_mean": round(float(c_ct.mean()), 4),
            }
        )

    return {
        "empty_prediction": False,
        "ttb_voxel_count": total_voxels,
        "ttb_volume_ml": round(vol_ml, 6),
        "suv_mean": round(suv_mean, 6),
        "suv_max": round(suv_max, 6),
        "suv_min": round(suv_min, 6),
        "suv_median": round(suv_median, 6),
        "suv_std": round(suv_std, 6),
        "volume_intensity_product_suv_ml": round(vol_ml * suv_mean, 6),
        "ct_hu_mean": round(float(ct_vals.mean()), 4),
        "connected_component_connectivity": "3D_6_connected",
        "connected_component_count": int(n_comp),
        "connected_components": components,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(GTRC_TASKS))
    parser.add_argument("--pt-suv", required=True, type=Path)
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--suv-threshold", type=float, default=None)
    parser.add_argument("--suv-threshold-source", type=str, default=None)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-mask", required=True, type=Path)
    parser.add_argument("--output-nnunet-labels", type=Path, default=None)
    parser.add_argument("--output-subregions", type=Path, default=None)
    parser.add_argument("--output-lesions-json", type=Path, default=None)
    parser.add_argument("--output-provenance-json", type=Path, default=None)
    parser.add_argument("--job-context-json", type=Path, default=None)
    parser.add_argument(
        "--gtrc-acceptance-bootstrap",
        action="store_true",
        help="Explicit acceptance harness flag; bypasses existing marker requirement during fresh acceptance runs.",
    )
    args = parser.parse_args(argv)

    spec = GTRC_TASKS[args.task]
    model_root = Path(os.environ.get("RTSEG_MODEL_ROOT", "/models"))
    source_dir = model_root / spec["source_dir"]
    weights_root = model_root / spec["weights_root"]
    nnunet_results_dir = weights_root / "data/nnUNet_data/results"
    consensus_dir = weights_root / spec["consensus_subdir"]

    # Capture GPU runtime provenance before TensorFlow CPU-isolation sets CUDA_VISIBLE_DEVICES=-1
    rt_prov = runtime_provenance()

    commit = installed_source_commit(source_dir)
    if commit != spec["source_commit"]:
        raise SystemExit(
            f"GTRC source commit verification failed for {args.task}: "
            f"expected {spec['source_commit']}, got {commit!r}"
        )

    ok_w, msg_w, weight_records = verify_weight_inventory(
        args.task, model_root, verify_hashes=True
    )
    if not ok_w:
        raise SystemExit(f"GTRC weight verification failed for {args.task}: {msg_w}")

    try:
        runtime_lock_sha256 = expected_runtime_lock_sha256(model_root)
    except ValueError as exc:
        raise SystemExit(f"GTRC runtime lock verification failed for {args.task}: {exc}") from exc

    # Production inference and acceptance-bootstrap both enforce live GPU/runtime compatibility.
    if not rt_prov.get("cuda_available"):
        raise SystemExit(
            f"GTRC GPU runtime gate refused {args.task}: torch.cuda.is_available() is False"
        )
    if not rt_prov.get("cuda_arch_compatible"):
        raise SystemExit(
            f"GTRC GPU runtime gate refused {args.task}: device {rt_prov.get('gpu_name')!r} "
            f"capability {rt_prov.get('device_capability')} incompatible with compiled archs "
            f"{rt_prov.get('cuda_arch_flags')}"
        )
    try:
        live_device_key = device_key(rt_prov)
    except ValueError as exc:
        raise SystemExit(f"GTRC GPU device gate refused {args.task}: {exc}") from exc
    rt_prov["device_key"] = live_device_key

    if not args.gtrc_acceptance_bootstrap:
        try:
            validate_completion(model_root, gpu=rt_prov)
        except ValueError as exc:
            raise SystemExit(f"GTRC completion gate refused {args.task}: {exc}") from exc
        try:
            validate_acceptance(args.task, model_root, gpu=rt_prov)
        except ValueError as exc:
            raise SystemExit(f"GTRC acceptance gate refused {args.task}: {exc}") from exc
        try:
            validate_orientation(args.task, model_root, gpu=rt_prov)
        except ValueError as exc:
            raise SystemExit(f"GTRC orientation gate refused {args.task}: {exc}") from exc

    job_context = (
        json.loads(args.job_context_json.read_text())
        if (args.job_context_json and args.job_context_json.is_file())
        else {}
    )
    try:
        threshold, threshold_source = resolve_job_suv_threshold(
            args.task,
            explicit_threshold=args.suv_threshold,
            explicit_source=args.suv_threshold_source,
        )
    except ValueError as exc:
        raise SystemExit(f"GTRC SUV threshold validation failed for {args.task}: {exc}") from exc

    args.work_dir.mkdir(parents=True, exist_ok=True)
    nn_input_dir = args.work_dir / "nn_input"
    nn_output_dir = args.work_dir / "nn_output"
    nn_input_dir.mkdir(parents=True, exist_ok=True)
    nn_output_dir.mkdir(parents=True, exist_ok=True)

    pt_suv = sitk.ReadImage(str(args.pt_suv))
    ct = sitk.ReadImage(str(args.ct))
    pt_rs, ct_rs = preprocess_pet_ct(pt_suv, ct, threshold)

    sitk.WriteImage(pt_rs, str(nn_input_dir / "gtrc_0000.nii.gz"))
    sitk.WriteImage(ct_rs, str(nn_input_dir / "gtrc_0001.nii.gz"))

    pred_path = run_nnunet_predict(
        nn_input_dir,
        nn_output_dir,
        dataset_id=int(spec["dataset_id"]),
        nnunet_results_dir=nnunet_results_dir,
        work_dir=args.work_dir,
        fold=args.fold,
    )
    flags_file = nn_output_dir / "nnunet_runtime_flags.json"
    nnunet_flags = (
        json.loads(flags_file.read_text(encoding="utf-8"))
        if flags_file.is_file()
        else None
    )

    pred_im = sitk.ReadImage(str(pred_path))
    subregion_fn = load_upstream_subregion_fn(source_dir)
    models = load_keras_consensus_models(consensus_dir, fold=args.fold)

    output_im, subregion_im, subregion_records, counts = evaluate_consensus_and_paint(
        pt_rs,
        ct_rs,
        pred_im,
        models,
        suv_threshold=threshold,
        subregion_fn=subregion_fn,
    )

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_im, str(args.output_mask), True)
    if args.output_nnunet_labels is not None:
        args.output_nnunet_labels.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(pred_im, str(args.output_nnunet_labels), True)
    if args.output_subregions is not None:
        args.output_subregions.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(subregion_im, str(args.output_subregions), True)

    biomarkers = compute_ttb_summary_and_components(pt_suv, ct_rs, output_im)

    report = {
        "schema_version": 1,
        "task": args.task,
        "family": "gtrc",
        "tracer": spec["tracer"],
        "dataset_id": spec["dataset_id"],
        "dataset_name": spec["dataset_name"],
        "roi_name": spec["roi_name"],
        "inference_mode": "production",
        "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
        "likelihood_semantics": (
            "No calibrated per-lesion probability is defined by upstream GTRC-Net; "
            "each candidate watershed subregion receives five fold-level Keras MLP "
            "sigmoid scores (positive vote > 0.5, accepted by strict majority > 2.5)."
        ),
        "suv_threshold": threshold,
        "suv_threshold_default": spec["default_suv_threshold"],
        "suv_threshold_source": threshold_source,
        "subregion_parameters": {
            "local_maxima_threshold_normalized_suv": LOCAL_MAXIMA_THRESHOLD,
            "local_maxima_threshold_raw_suv": round(LOCAL_MAXIMA_THRESHOLD * threshold, 6),
            "sphere_radius_mm": SPHERE_RADIUS_MM,
            "vote_threshold": 0.5,
            "majority_rule": "votes_positive > models_count / 2",
            "inpainting_rule": "(nnunet_ttb_label == 1) and (suv >= suv_threshold)",
        },
        **biomarkers,
        "counts": counts,
        "subregions": subregion_records,
        "job_context": job_context,
        "provenance": {
            "source_url": spec["source_url"],
            "source_commit": commit,
            "paper_doi": spec["paper_doi"],
            "inference_mode": "production",
            "nnunet_launch_path": "DETERMINISTIC_NNUNET_ENTRYPOINT",
            "runtime_lock_sha256": runtime_lock_sha256,
            "runtime": rt_prov,
            "nnunet_runtime_flags": nnunet_flags,
            "checkpoints": weight_records,
        },
    }

    if args.output_lesions_json is not None:
        args.output_lesions_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_lesions_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.output_provenance_json is not None:
        args.output_provenance_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_provenance_json.write_text(
            json.dumps(report["provenance"], indent=2) + "\n"
        )

    print(
        f"GTRC {args.task}: ttb_voxels={biomarkers['ttb_voxel_count']} "
        f"vol_ml={biomarkers['ttb_volume_ml']:.2f} "
        f"subregions={counts['accepted_subregion_count']}/{counts['nonempty_subregion_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

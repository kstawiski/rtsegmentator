"""Small label contracts shared by the runners and catalogue builder; no ML imports."""
from __future__ import annotations

import json
import re
from pathlib import Path


def roi_name(name) -> str:
    """ROI name as DICOM stores it (LO: no padding, at most 64 characters)."""
    text = str(name).strip()
    if not text:
        raise SystemExit("empty ROI name in the label map")
    if len(text) > 64:
        raise SystemExit(f"ROI name longer than 64 characters: {text!r}")
    return text


def unique_roi_names(names):
    """Preserve the writer's normalisation and duplicate rejection."""
    normalised = [roi_name(name) for name in names]
    seen = set()
    for name in normalised:
        if name in seen:
            raise SystemExit(f"duplicate ROI name after normalisation: {name!r}")
        seen.add(name)
    return normalised


def final_roi_names(labels):
    """Return foreground names in numeric label-index order, as written by the converter."""
    return unique_roi_names(v for k, v in sorted(labels.items(), key=lambda kv: int(kv[0])) if int(k) > 0)


def dataset_json(model_folder: Path) -> Path:
    for candidate in (model_folder / "dataset.json", *(p / "dataset.json" for p in model_folder.parents)):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No dataset.json found for {model_folder}")


def dataset_labels(raw):
    labels = {}
    for name, value in raw.items():
        if str(name).lower() == "background":
            continue
        if isinstance(value, list):
            raise SystemExit(f"Overlapping region label {name!r} requires a dedicated exporter")
        labels[str(int(value))] = str(name)
    if not labels:
        raise SystemExit("dataset.json contains no scalar foreground labels")
    return labels


def kept_label(keep_label, keep_name=None):
    return {"1": keep_name or f"label_{keep_label}"}


def spec_labels(spec, app_root: Path, model_root: Path):
    if spec.get("labels_file"):
        return json.loads((app_root / spec["labels_file"]).read_text())
    if spec.get("labels_from_metadata"):
        metadata = json.loads((model_root / "weights/monai-bundles" / spec["bundle"] / "configs/metadata.json").read_text())
        channel_def = metadata["network_data_format"]["outputs"]["pred"]["channel_def"]
        return {str(k): v for k, v in channel_def.items() if str(k) != "0"}
    return spec["labels"]


def lut_labels(path: Path, *, freesurfer=False):
    labels = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^\s*(\d+)\s+([^\s#]+)" if freesurfer else
                         r"^\s*(\d+)\s+([^#]+?)(?:\s+\d+\s+\d+\s+\d+\s+\d+)?\s*$", line)
        if match and int(match[1]):
            labels[str(int(match[1]))] = match[2].strip().replace(" ", "_")
    if not labels:
        raise ValueError("LUT contains no foreground names")
    return labels


def sct_output_key(filename):
    key = re.sub(r"\.nii(?:\.gz)?$", "", Path(filename).name)
    return re.sub(r"^prediction_?", "", key).strip("_")


def sct_matched_name(filename, names):
    key = sct_output_key(filename)
    return next((v for k, v in names.items() if k.lower() in key.lower()), None)


def sct_output_name(filename, names, *, single=False, index=1):
    label = sct_matched_name(filename, names)
    if label is None and single and len(set(names.values())) == 1:
        label = next(iter(names.values()))
    return label or sct_output_key(filename) or f"SCT_Label_{index}"


def sct_labels(spec, app_root):
    """All supported SCT exports, including aliases and independent mask ordering."""
    if spec.get("labels_file"):
        return json.loads((app_root / spec["labels_file"]).read_text())
    if spec.get("multiclass_labels"):
        return spec["multiclass_labels"]
    names = list(dict.fromkeys(sct_output_name("prediction_" + token + ".nii.gz", spec["output_names"])
                               for token in spec["output_names"]))
    # The overlap-preserving runner serializes its ROI-name-to-mask map with sort_keys=True.
    if spec.get("preserve_overlaps"):
        names.sort()
    return dict(enumerate(names, 1))


INLINE_LABELS = {
    "rano2_assist_t1c_mr": {"1": "Nonenhancing_Tumor_Core", "2": "Peritumoral_Edema",
                            "3": "Enhancing_Tumor_Core", "4": "Resection_Cavity"},
    "dotatate_pet_lesions": {"1": "DOTATATE_Avid_Lesion"},
}

# These are exporter inputs, not presentation overrides. Every runtime use of these
# files goes through repo_label_path; indices and names remain in the label files.
REPO_LABEL_FILES = {
    "hippunfold_adni_t1_subfields_mr": "labels/hippunfold-bigbrain.json",
    "hippunfold_maguire_t2_subfields_mr": "labels/hippunfold-bigbrain.json",
    "siam_v3_head_mr_ct": "labels/siam-v3.json",
    "lungtumormask_ct": "labels/lung-tumor.json",
    "synthseg_v1_brain": "labels/synthseg-v1.json",
    "universal_model_unet_ct": "labels/universal-model.json",
    "universal_model_swinunetr_ct": "labels/universal-model.json",
    "lungmask_r231": "labels/lungmask-lungs.json",
    "lungmask_ltrc_lobes": "labels/lungmask-lobes.json",
    "lungmask_ltrc_lobes_r231": "labels/lungmask-lobes.json",
    "lnctvseg_nnunet_v1": "labels/lnctvseg.json",
    "hd_bet_mr": "labels/hd-bet.json", "hd_ctbet_ct": "labels/hd-bet.json",
    "totalspineseg": "labels/totalspineseg.json",
    "mrisegmentator": "labels/mrisegmentator.json",
    "mrsegmentator": "labels/mrsegmentator.json",
    "mrsegmentator_body_comp": "labels/mrsegmentator-body-comp.json",
    "medlsam_structseg_han_oars": "labels/medlsam-structseg-han.json",
    "lnq_atlas": "labels/lnq-atlas.json", "hnlnl_2d3d": "hnlnl-labels.json",
    "dbdmp_lnq": "mediastinal-node-label.json", "compai_lnq": "mediastinal-node-label.json",
    "lnq_inguinal_v1": "lnq-inguinal-label.json",
    "lnq_abdominopelvic_v1": "lnq-abdominopelvic-label.json",
    "lnq_axillary_v1": "lnq-axillary-label.json",
    "lnq_mediastinal_v1": "lnq-mediastinal-label.json",
    "raidionics_mediastinal_lymphnodes": "labels/raidionics-mediastinal-lymphnodes.json",
    "pediatric_thoracic_lymphoma": "labels/pediatric-thoracic-lymphoma.json",
    "uwlair_hntsmrg_pre_t2": "labels/hntsmrg-gtv.json",
}
for _task in ("isrt_ct_resunet", "isrt_ct_segresnet", "isrt_ct_swinunetr",
              "isrt_pet1_early_deform", "isrt_pet1_early_rigid", "isrt_pet1_late_deform", "isrt_pet1_late_rigid"):
    REPO_LABEL_FILES[_task] = "labels/isrt-ctv.json"
PROMPTED_TASKS = (
    "pam_prompted", "sam_med2d_prompted", "sam_med3d_turbo_prompted", "medsam_vit_b_box_2d",
    "medical_sam3_3d_prompted", "medsam2_ct_lesion_prompted", "medical_sam2_prompted_ct",
    "sat3d_prompted", "nninteractive_v1_prompted", "vista3d_prompted_ct",
    "voxtell_v1_1_text_prompted", "segvol_prompted_ct",
)
for _task in PROMPTED_TASKS:
    REPO_LABEL_FILES[_task] = "prompted-target-label.json"


def repo_label_path(task: str, app_root: Path) -> Path:
    return app_root / REPO_LABEL_FILES[task]

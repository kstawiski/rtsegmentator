#!/usr/bin/env python3
"""Run the CERR SMIT-family checkpoints with a uniform NIfTI interface."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "CT_HN_SMIT": dict(a_min=-750, a_max=1750, spacing=(1.5, 1.5, 1.5),
                       roi=(128, 128, 128), out_channels=17, use_smit=0,
                       model_feature=128),
    "CT_HeartSubstruct_SMIT": dict(a_min=-200, a_max=300, spacing=(1, 1, 3),
                                    roi=(128, 128, 128), out_channels=10,
                                    use_smit=1),
    "CT_Lung_SMIT": dict(a_min=-500, a_max=500, spacing=(1.5, 1.5, 2),
                         roi=(128, 128, 128), out_channels=2, use_smit=1),
    "CT_WHOLEBODY_SMITplus": dict(a_min=-500, a_max=500,
                                   spacing=(1.5, 1.5, 2), roi=(128, 128, 128),
                                   out_channels=47, use_smit=0,
                                   model_feature=128),
    "MR_HN_Nodule_SMIT": dict(spacing=(1.5, 1.5, 1.5),
                               roi=(128, 128, 128), out_channels=2,
                               use_smit=1, num_workers=0),
}


def patched_standard_driver(args, work: Path) -> Path:
    """Avoid importing MONAI's optional handler stack under modern Python.

    Upstream uses only the tiny ``from_engine`` accessor but imports it through
    ``monai.handlers``, whose MONAI 0.8 package initializer eagerly imports old
    TensorBoard integrations.  The equivalent local accessor keeps inference
    behavior unchanged and permits an H100-capable modern PyTorch runtime.
    """
    source = (args.source / "run_segmentation.py").read_text()
    needle = "from monai.handlers.utils import from_engine"
    replacement = (
        "def from_engine(keys):\n"
        "    def extract(data):\n"
        "        if isinstance(data, dict):\n"
        "            values = [data[key] for key in keys]\n"
        "        else:\n"
        "            values = [[item[key] for item in data] for key in keys]\n"
        "        return tuple(values) if len(values) > 1 else values[0]\n"
        "    return extract\n"
    )
    if source.count(needle) != 1:
        raise SystemExit(f"Unexpected from_engine import in {args.source}")
    driver = work / "run_segmentation_compat.py"
    driver.write_text(source.replace(needle, replacement))
    return driver


def command_for_standard_model(args, case_dir: Path, result_dir: Path, work: Path):
    config = CONFIGS[args.model]
    sx, sy, sz = config["spacing"]
    rx, ry, rz = config["roi"]
    command = [
        sys.executable, str(patched_standard_driver(args, work)),
        "--roi_x", str(rx), "--roi_y", str(ry), "--roi_z", str(rz),
        "--space_x", str(sx), "--space_y", str(sy), "--space_z", str(sz),
        "--data_dir", str(case_dir), "--out_channels", str(config["out_channels"]),
        "--load_weight_name", str(args.checkpoint),
        "--save_folder", str(result_dir), "--use_smit", str(config["use_smit"]),
    ]
    if "a_min" in config:
        command += ["--a_min", str(config["a_min"]), "--a_max", str(config["a_max"])]
    if "model_feature" in config:
        command += ["--model_feature", str(config["model_feature"])]
    if "num_workers" in config:
        command += ["--num_workers", str(config["num_workers"])]
    return command


def command_for_rectum(args, case_dir: Path, result_dir: Path, work: Path):
    source = (args.source / "run_segmentation_rectum.py").read_text()
    override = (
        "args.pretrained_model_path = os.path.join("
        "os.path.dirname(os.path.abspath(__file__)), 'model_final.pt')"
    )
    if source.count(override) != 1:
        raise SystemExit(f"Unexpected checkpoint override in {args.source}")
    driver = work / "run_segmentation_rectum_compat.py"
    driver.write_text(source.replace(
        override,
        "if args.pretrained_model_path is None:\n        " + override,
    ))
    return [
        sys.executable, str(driver),
        "--data_dir", str(case_dir), "--json_list", "data.json", "--datasets", "val",
        "--model_name", "smit", "--pretrained_model_path", str(args.checkpoint),
        "--results_dir", str(result_dir), "--in_channels", "1", "--out_channels", "2",
        "--norm_name", "instance", "--a_min", "0", "--a_max", "800",
        "--b_min", "0", "--b_max", "1", "--space_x", "1", "--space_y", "1",
        "--space_z", "1", "--roi_x", "128", "--roi_y", "128", "--roi_z", "64",
        "--infer_overlap", "0.5", "--sw_batch_size", "8", "--use_tta",
        "--postproc", "standard",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=sorted([*CONFIGS, "MR_Rectum_GTV_SMIT"]))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()

    for required in (args.source, args.checkpoint, args.input):
        if not required.exists():
            raise SystemExit(f"Required CERR resource is missing: {required}")
    case_dir, result_dir = args.work / "input", args.work / "result"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_input = case_dir / "case.nii.gz"
    shutil.copyfile(args.input, model_input)
    (case_dir / "data.json").write_text(
        json.dumps({"val": [{"image": model_input.name}]}, indent=2) + "\n"
    )

    if args.model == "MR_Rectum_GTV_SMIT":
        command = command_for_rectum(args, case_dir, result_dir, args.work)
    else:
        command = command_for_standard_model(args, case_dir, result_dir, args.work)
    environment = dict(os.environ)
    # The official CERR checkpoints predate PyTorch 2.6 and contain trusted
    # NumPy scalar metadata in addition to tensors.
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    environment["PYTHONPATH"] = ":".join(
        value for value in (str(args.source), environment.get("PYTHONPATH", "")) if value
    )
    subprocess.run(command, cwd=args.source, env=environment, check=True)

    predictions = sorted(result_dir.glob("*.nii*"))
    if len(predictions) != 1:
        raise SystemExit(
            f"{args.model} produced {len(predictions)} NIfTI predictions in {result_dir}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(predictions[0], args.output)


if __name__ == "__main__":
    main()

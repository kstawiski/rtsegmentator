#!/usr/bin/env python3
"""Portable wrapper around the publisher's legacy MultiTalent V1 inference."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--masks-json", required=True)
    parser.add_argument("--fold", default="all")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # Training-only path variables are read at import time by legacy nnU-Net.
    # Inference uses the embedded plans and does not access these directories.
    for key in ("nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER"):
        env.setdefault(key, str(output / f"_{key}"))

    command = [
        args.python, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=1", str(source / "nnunet/inference/predict_MultiTalent.py"),
        "-i", args.input_dir, "-o", str(output), "-m", args.model_folder,
        "-f", args.fold, "--tta", "0", "--num_threads_preprocessing", "1",
        "--num_threads_nifti_save", "4",
    ]
    subprocess.run(command, check=True, env=env)

    names = json.loads(Path(args.labels).read_text())
    masks = {}
    for region, roi_name in names.items():
        filename = output / "individual" / f"case_{region}.nii.gz"
        if not filename.is_file():
            raise SystemExit(f"MultiTalent did not create {filename}")
        masks[roi_name] = str(filename)
    Path(args.masks_json).write_text(json.dumps(masks, indent=2) + "\n")


if __name__ == "__main__":
    main()

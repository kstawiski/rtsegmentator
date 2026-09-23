#!/usr/bin/env python3
"""Run the released single-T1c RANO2.0-assist DynUNet checkpoint."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original T1c NIfTI")
    parser.add_argument("--work", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--hd-bet", required=True)
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    brain = work / "brain.nii.gz"
    subprocess.run([args.hd_bet, "-i", args.input, "-o", str(brain)], check=True)
    # In image-output mode HD-BET writes the extracted image to the exact
    # requested path; the ``*_bet`` suffix is used for its optional mask.
    brain_extracted = brain
    if not brain_extracted.is_file():
        raise SystemExit(f"HD-BET did not create {brain_extracted}")

    datalist = work / "inference_files.json"
    datalist.write_text(json.dumps([{
        "images": {"t1c": str(brain_extracted)},
        "save_path": "case.nii.gz",
    }], indent=2) + "\n")
    results = work / "results"
    source = Path(args.source)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source / "src")
    subprocess.run([
        sys.executable, str(source / "src/inference.py"),
        "--task_dir", args.task_dir,
        "--args_file", "config/infer_args.json",
        "--inference_files_path", str(datalist),
        "--out_dir", str(results),
        "--no-reg", "--no-bet", "--input_is_bet",
    ], check=True, env=env)
    prediction = results / "unmerged_model/fold0/case.nii.gz"
    if not prediction.is_file():
        raise SystemExit(f"RANO2.0-assist did not create {prediction}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    os.replace(prediction, args.output)


if __name__ == "__main__":
    main()

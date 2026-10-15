#!/usr/bin/env python3
"""Compatibility launcher for downloadable legacy CERR inference packages."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_lung_mrrn(source: Path, weights: Path, input_dir: Path, output_dir: Path, work: Path) -> None:
    pipeline = work / "pipeline"
    shutil.copytree(source, pipeline, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    model_dir = pipeline / "model"
    model_dir.mkdir(exist_ok=True)
    shutil.copy2(weights, model_dir / "seg_best_net_Seg_A.pth")

    inference = pipeline / "model_wrapper" / "run_inference_nii.py"
    text = inference.read_text()
    text = text.replace(
        "model.load_state_dict(torch.load(modelWeightsPath))",
        "model.load_state_dict(torch.load(modelWeightsPath, map_location=device, weights_only=False))\n"
        "    model = model.to(device)\n    model.eval()",
    )
    # The upstream code detects a device but then unconditionally calls CUDA.
    text = text.replace("inputslice = torch.from_numpy(inputslice).cuda().float()",
                        "inputslice = torch.from_numpy(inputslice).to(device).float()")
    # Device must exist before checkpoint loading after the compatibility patch.
    text = text.replace("model = get_Incre_MRRN_Pytorch()", "device = get_safe_device()\n    model = get_Incre_MRRN_Pytorch()")
    text = text.replace("    device = get_safe_device()\n\n    with torch.no_grad():", "    with torch.no_grad():")
    inference.write_text(text)

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [os.environ.get("RTSEG_CERR_MODERN_PYTHON", str(Path(os.environ.get("RTSEG_RUNTIME_ROOT", "/runtimes")) / "envs/segmentation-modern/bin/python")),
         str(pipeline / "run_inference_lung_oars.py"), str(input_dir), str(output_dir)],
        cwd=pipeline, check=True,
    )


def run_cardiac_deeplab(source: Path, weights: Path, input_dir: Path, output_dir: Path, work: Path) -> None:
    pipeline = work / "pipeline"
    shutil.copytree(source, pipeline, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    model_dir = pipeline / "model"
    shutil.copytree(weights, model_dir, dirs_exist_ok=True)
    resnet_source = model_dir / "resnet" / "resnet101-5d3b4d8f.pth"
    if resnet_source.exists():
        resnet_cache = model_dir / "resnet" / "hub" / "checkpoints"
        resnet_cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resnet_source, resnet_cache / resnet_source.name)
    driver = pipeline / "model_wrapper" / "runSegmentation.py"
    text = driver.read_text()
    text = text.replace("torch.load(self.validate_model)",
                        "torch.load(self.validate_model, weights_only=False)")
    text = text.replace(
        "torch.load(self.validate_model, map_location=lambda storage, loc: storage)",
        "torch.load(self.validate_model, map_location=lambda storage, loc: storage, weights_only=False)",
    )
    driver.write_text(text)
    # NumPy removed these aliases after the package was published.
    for path in pipeline.rglob("*.py"):
        source_text = path.read_text()
        source_text = source_text.replace("np.float)", "float)").replace("np.int)", "int)")
        path.write_text(source_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [os.environ.get("RTSEG_CERR_MODERN_PYTHON", str(Path(os.environ.get("RTSEG_RUNTIME_ROOT", "/runtimes")) / "envs/segmentation-modern/bin/python")),
         str(driver), str(input_dir), str(output_dir)],
        cwd=driver.parent, check=True,
    )


def run_prostate_deeplab(source: Path, weights: Path, input_dir: Path,
                         output_dir: Path, work: Path) -> None:
    """Run the published TensorFlow 1.15 prostate graph in its archived environment."""
    pipeline = work / "pipeline"
    shutil.copytree(source, pipeline, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__"))
    model_dir = pipeline / "model"
    model_dir.mkdir(exist_ok=True)
    weight_file = weights if weights.is_file() else weights / "frozen_inference_graph.pb"
    shutil.copy2(weight_file, model_dir / "frozen_inference_graph.pb")
    output_dir.mkdir(parents=True, exist_ok=True)
    python = os.environ.get(
        "RTSEG_CERR_PROSTATE_PYTHON",
        str(Path(os.environ.get("RTSEG_MODEL_ROOT", "/models")) / "runtime-homes/cerr/env6-prostate/bin/python"),
    )
    subprocess.run(
        [python, str(pipeline / "model_wrapper" / "run_inference_nii.py"),
         str(input_dir), str(output_dir)],
        cwd=pipeline / "model_wrapper", check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["CT_LungOAR_incrMRRN", "CT_cardiac_structures_deeplab",
                                 "MR_Prostate_Deeplab"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.model == "CT_LungOAR_incrMRRN":
        run_lung_mrrn(args.source, args.weights, args.input, args.output, args.work)
    elif args.model == "CT_cardiac_structures_deeplab":
        run_cardiac_deeplab(args.source, args.weights, args.input, args.output, args.work)
    else:
        run_prostate_deeplab(args.source, args.weights, args.input, args.output, args.work)


if __name__ == "__main__":
    main()

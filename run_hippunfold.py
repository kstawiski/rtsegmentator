"""Run a released HippUnfold checkpoint and emit one source-grid label map."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MODELS = {
    "ADNI_T1w_v1": "T1w",
    "maguire_T2w": "T2w",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--overlay", help="Writable Apptainer overlay used by the v2 workflow")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--cpu", action="store_true",
                        help="Run nnU-Net on CPU for legacy images without H100 kernels")
    parser.add_argument("--apptainer", default=os.environ.get("RTSEG_APPTAINER"))
    parser.add_argument("--app-root", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    apptainer = args.apptainer or shutil.which("apptainer")
    if not apptainer:
        raise SystemExit("Apptainer was not found; set RTSEG_APPTAINER")
    for required in (args.input, args.image, args.cache):
        if not Path(required).exists():
            raise SystemExit(f"required HippUnfold path does not exist: {required}")

    modality = MODELS[args.model]
    atlas = "multihist7" if args.model == "maguire_T2w" else "bigbrain"
    work = Path(args.work_dir).resolve()
    bids, derivatives = work / "bids", work / "derivatives"
    anat = bids / "sub-rtseg" / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    derivatives.mkdir(parents=True, exist_ok=True)
    (bids / "dataset_description.json").write_text(json.dumps({
        "Name": "RTsegmentator transient HippUnfold input",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
    }, indent=2))
    shutil.copy2(args.input, anat / f"sub-rtseg_{modality}.nii.gz")
    (anat / f"sub-rtseg_{modality}.json").write_text("{}\n")

    binds = f"{bids}:/bids:ro,{derivatives}:/output,{Path(args.cache).resolve()}:/cache"
    command = [apptainer, "run"]
    if args.model == "maguire_T2w":
        # The published v2 development image omits the Snakemake conda
        # prefixes declared below /src. Supply the prebuilt NFS-hosted runtime
        # overlay without modifying the immutable SIF.
        overlay = Path(args.overlay).resolve() if args.overlay else Path(args.cache).resolve().parent / "hippunfold-v2-overlay-dir"
        if not overlay.exists():
            raise SystemExit(f"required HippUnfold v2 overlay does not exist: {overlay}")
        command += ["--overlay", str(overlay)]
    if not args.cpu:
        command.append("--nv")
    command += ["--cleanenv", "--bind", binds,
                "--env", "HIPPUNFOLD_CACHE_DIR=/cache", str(Path(args.image).resolve())]
    # The published v2 development image is a generic Pixi container whose
    # OCI entrypoint executes the first argument.  Unlike the v1.5.x BIDS App
    # image, it therefore needs the workflow executable supplied explicitly.
    if args.model == "maguire_T2w":
        command += ["/src/hippunfold/run.py"]
    command += [
        "/bids", "/output", "participant", "--participant-label", "rtseg",
        "--modality", modality, "--force-nnunet-model", args.model,
        "--atlas", atlas, "--cores", "4",
    ]
    if not args.cpu:
        command.append("--use-gpu")
    subprocess.run(command, check=True)

    native_space = modality
    pattern = f"sub-rtseg_hemi-{{hemi}}_space-{native_space}_label-hipp_desc-subfields_atlas-{atlas}_dseg.nii.gz"
    native: dict[str, Path] = {}
    for hemi in ("L", "R"):
        matches = list(derivatives.rglob(pattern.format(hemi=hemi)))
        if len(matches) != 1:
            raise SystemExit(f"HippUnfold produced {len(matches)} native {hemi} subfield maps")
        native[hemi] = matches[0]

    app_root = Path(args.app_root).resolve()
    resampled = {}
    for hemi in ("L", "R"):
        target = work / f"subfields_{hemi}_source_grid.nii.gz"
        subprocess.run([
            sys.executable, str(app_root / "resample_label_to_reference.py"),
            str(native[hemi]), args.input, str(target),
        ], check=True)
        resampled[hemi] = target
    subprocess.run([
        sys.executable, str(app_root / "combine_hippunfold_subfields.py"),
        "--left", str(resampled["L"]), "--right", str(resampled["R"]),
        "--reference", args.input, "--output", args.output,
    ], check=True)


if __name__ == "__main__":
    main()

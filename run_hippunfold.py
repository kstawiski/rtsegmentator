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
MAGUIRE_ARCHIVE = "Dataset001_HippUnfoldMag.nnUNetTrainer__nnUNetPlans__3d_fullres.tar.gz"
OFFLINE_WORKFLOW = "/src/hippunfold/workflow/rtseg-offline.smk"


def discover_subfields(derivatives: Path, modality: str, atlas: str) -> dict[str, Path]:
    """Find final native-coordinate labels, preferring the published crop grids.

    v1.5.3 omits label-hipp; v2 includes it. Neither crop is an atlas-space
    image: its affine places it in the original modality's physical space.
    Full native grids are a fallback for workflows retaining those outputs.
    """
    native = {}
    for hemi in ("L", "R"):
        searched = []
        for space in (f"crop{modality}", modality):
            matches = []
            for label in ("", "_label-hipp"):
                pattern = (
                    f"sub-rtseg_hemi-{hemi}_space-{space}{label}"
                    f"_desc-subfields_atlas-{atlas}_dseg.nii.gz"
                )
                searched.append(pattern)
                matches.extend(path for path in derivatives.rglob(pattern) if path.is_file())
            if len(matches) > 1:
                raise SystemExit(
                    f"HippUnfold produced {len(matches)} native {hemi} subfield maps "
                    f"in space-{space}; ambiguous outputs: {', '.join(map(str, sorted(matches)))}"
                )
            if matches:
                native[hemi] = matches[0]
                break
        else:
            raise SystemExit(
                f"HippUnfold produced 0 native {hemi} subfield maps; "
                f"searched {derivatives} for: {', '.join(searched)}"
            )
    return native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--overlay", help="Prebuilt v2 Apptainer overlay, mounted read-only")
    parser.add_argument("--model-archive", help=(
        "Pre-staged Maguire archive; defaults to --cache/model/" + MAGUIRE_ARCHIVE
    ))
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
    app_root = Path(args.app_root).resolve()
    cache = Path(args.cache).resolve()
    if args.model_archive and args.model != "maguire_T2w":
        parser.error("--model-archive applies only to maguire_T2w")
    if args.model == "maguire_T2w":
        archive = Path(args.model_archive).resolve() if args.model_archive else cache / "model" / MAGUIRE_ARCHIVE
        if not archive.is_file() or archive.stat().st_size == 0:
            raise SystemExit(
                f"required offline HippUnfold Maguire model archive is missing or empty: {archive}; "
                "stage the archive from Zenodo record 18432751 before running inference"
            )
        for resource in (cache / "template/CITI168", cache / "atlases/tpl-multihist7"):
            if not resource.is_dir() or not any(resource.iterdir()):
                raise SystemExit(f"required offline HippUnfold resource is missing or empty: {resource}")
        offline_workflow = app_root / "hippunfold_offline.smk"
        if not offline_workflow.is_file():
            raise SystemExit(f"required offline HippUnfold workflow does not exist: {offline_workflow}")
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

    binds = f"{bids}:/bids:ro,{derivatives}:/output,{cache}:/cache"
    command = [apptainer, "run"]
    if args.model == "maguire_T2w":
        # The published v2 development image omits the Snakemake conda
        # prefixes declared below /src. Supply the prebuilt runtime
        # overlay without modifying the immutable SIF.
        overlay = Path(args.overlay).resolve() if args.overlay else cache.parent / "hippunfold-v2-overlay-dir"
        if not overlay.exists():
            raise SystemExit(f"required HippUnfold v2 overlay does not exist: {overlay}")
        command += ["--overlay", f"{overlay}:ro"]
        # Imported v2 rules consume pre-staged resources as external inputs.
        # Excluding their producer rules prevents HTTP existence checks and
        # prevents Snakemake from deleting the model archive as a temp output.
        binds += (
            f":ro,{archive}:/cache/model/{MAGUIRE_ARCHIVE}:ro,"
            f"{offline_workflow}:{OFFLINE_WORKFLOW}:ro"
        )
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
    if args.model == "maguire_T2w":
        command += ["--snakefile", OFFLINE_WORKFLOW]
    subprocess.run(command, check=True)

    native = discover_subfields(derivatives, modality, atlas)

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

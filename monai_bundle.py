"""Job-local MONAI configuration repairs; never modify the installed bundle."""
import json
from pathlib import Path


def compatible_config(bundle: Path, spec: dict, work: Path) -> Path:
    source = bundle / spec["config"]
    if not (spec.get("remove_network_img_size") or spec.get("postprocess_on_cpu")):
        return source
    if source.suffix in {".yaml", ".yml"}:
        import yaml
        config = yaml.safe_load(source.read_text())
    else:
        config = json.loads(source.read_text())
    if spec.get("remove_network_img_size"):
        # MONAI 1.6 infers SwinUNETR's image size dynamically.
        config["network_def"].pop("img_size", None)
    if spec.get("postprocess_on_cpu"):
        transforms = config["postprocessing"]["transforms"]
        # The catalog opts in after inspection of each official pipeline. Fail
        # closed if its inversion/resampling step has changed instead of silently
        # running the full-resolution multi-channel allocation on the GPU.
        resamplers = {"Invertd", "Spacingd", "ResampleToMatchd", "SpatialResampled", "Resized"}
        index = next((i for i, transform in enumerate(transforms)
                      if isinstance(transform, dict)
                      and transform.get("_target_", "").split(".")[-1] in resamplers
                      and "pred" in ([transform.get("keys")] if isinstance(transform.get("keys"), str)
                                     else transform.get("keys", []))), None)
        if index is None:
            raise ValueError("CPU postprocessing requires an explicit pred inversion/resampling transform")
        transforms.insert(index, {"_target_": "EnsureTyped", "keys": "pred", "device": "cpu"})
        # Inversion itself and the returned tensor must stay on CPU, including
        # configs that explicitly request another output device on Invertd.
        for transform in transforms[index + 1:]:
            if isinstance(transform, dict) and transform.get("_target_", "").split(".")[-1] == "Invertd":
                transform["device"] = "cpu"
    target = bundle_overlay(bundle, work) / "inference-compatible.json"
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target


def bundle_overlay(bundle: Path, work: Path) -> Path:
    """Return a job-local ``configs`` directory inside a symlink mirror of the bundle.

    MONAI puts a config's bundle root (the parent of a directory named ``configs``) on
    ``sys.path`` and reads ``metadata.json``/``logging.conf`` beside the config. A modified
    config outside the bundle would lose bundle-local code such as ``scripts.networks``;
    the mirror keeps every official file in place and read-only.
    """
    overlay = work / "bundle-overlay"
    configs = overlay / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for entry in sorted(bundle.iterdir()):
        if entry.name == "configs" or entry == work or entry in work.parents:
            continue
        link = overlay / entry.name
        if not link.is_symlink():
            link.symlink_to(entry)
    if (bundle / "configs").is_dir():
        for entry in sorted((bundle / "configs").iterdir()):
            link = configs / entry.name
            if not link.is_symlink():
                link.symlink_to(entry)
    return configs


def bundle_command(runtime_python, bundle, spec, config, nifti, inference_output):
    """Share the production invocation with the off/on GPU comparison harness."""
    datalist = [{"image": str(nifti)}] if spec.get("datalist_records") else [str(nifti)]
    command = [str(runtime_python), "-m", "monai.bundle", "run",
               "--config_file", str(config), "--bundle_root", str(bundle),
               "--datalist", repr(datalist), "--output_dir", str(inference_output),
               "--dataloader#num_workers", "0"]
    if spec.get("trusted_architecture_pickle"):
        # Only the pinned official DiNTS architecture artifact uses this override.
        command += ["--arch_ckpt", "$torch.load(@arch_ckpt_path, map_location=torch.device('cuda'), weights_only=False)"]
    if spec.get("run_id"):
        command += ["--run_id", spec["run_id"]]
    for key, value in spec.get("overrides", {}).items():
        encoded = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        command += [f"--{key}", encoded]
    return command

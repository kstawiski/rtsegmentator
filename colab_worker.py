"""Bootstrap and execute one uploaded portal job inside a Colab runtime."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import traceback
import zipfile
from pathlib import Path


def extract(source: Path, destination: Path, *, trusted_links: bool = False) -> None:
    with tarfile.open(source) as archive:
        destination_root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination, filter="fully_trusted" if trusted_links else "data")


root = Path("/content/rtsegmentator")
root.mkdir(exist_ok=True)
extract(Path("/content/rtseg-app.tar.gz"), root)
extract(Path("/content/rtseg-job.tar.gz"), root)
status_path = Path("/content/rtseg-status.json")
status_path.write_text(json.dumps({"status": "starting"}))
assets = Path("/content/rtseg-assets.tar.gz")
asset_parts = sorted(Path("/content").glob("assets.part-*"))
if asset_parts:
    with assets.open("wb") as destination:
        for part in asset_parts:
            with part.open("rb") as source:
                while chunk := source.read(8 * 1024**2):
                    destination.write(chunk)
            part.unlink()
if assets.is_file():
    extract(assets, Path("/"), trusted_links=True)
app = root / "app"
job = root / "job"
(job / "output").mkdir(parents=True, exist_ok=True)

# Colab already provides a CUDA-enabled PyTorch build; preserve it instead of
# replacing it with the container-specific pinned wheel.
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "-r", str(app / "requirements-worker.txt")], check=True)

    environment = os.environ.copy()
    config_path = app / "colab-paths.json"
    asset_paths = json.loads(config_path.read_text()) if config_path.is_file() else {}
    def asset_path(key, default):
        return os.environ.get(key, asset_paths.get(key, default))
    environment.update(
        RTSEG_APP_ROOT=str(app),
        RTSEG_MODEL_ROOT=asset_path("RTSEG_MODEL_ROOT", "/models") if assets.is_file() else "/content/models",
        RTSEG_RUNTIME_ROOT=asset_path("RTSEG_RUNTIME_ROOT", "/runtimes") if assets.is_file() else "/content/models",
        RTSEG_SCRATCH_ROOT="/content/rtseg-tmp",
    )
    if assets.is_file():
        uv_python = Path(asset_path("RTSEG_COLAB_UV_PYTHON", "/runtimes/uv-python"))
        versions = sorted(uv_python.glob("cpython-3.11.*-linux-x86_64-gnu"))
        compatibility = uv_python / "cpython-3.11-linux-x86_64-gnu"
        if versions and not compatibility.exists():
            compatibility.symlink_to(versions[-1], target_is_directory=True)
        Path(asset_path("RTSEG_COLAB_MODERN_LINK", "/runtimes/modern-env")).symlink_to(Path(environment["RTSEG_RUNTIME_ROOT"]) / "envs/segmentation-modern", target_is_directory=True)
        Path(asset_path("RTSEG_COLAB_MODEL_LINK", "/runtimes/model-bundle")).symlink_to(environment["RTSEG_MODEL_ROOT"], target_is_directory=True)
    data = json.loads((job / "job.json").read_text())
    for task in data["models"]:
        completed = subprocess.run([sys.executable, str(app / "run_task.py"), str(job), task],
                                   env=environment, capture_output=True, text=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode:
            raise RuntimeError(f"{task} exited with status {completed.returncode}\n{completed.stderr[-6000:]}")

    results = sorted((job / "output").glob("*.dcm"))
    if not results:
        raise RuntimeError("Worker completed without producing an RTSTRUCT file")
    with zipfile.ZipFile("/content/rtseg-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            archive.write(result, result.name)
            orientation = Path(str(result) + ".orientation.json")
            if orientation.is_file():
                archive.write(orientation, orientation.name)
    status_path.write_text(json.dumps({"status": "complete", "results": [p.name for p in results]}))
except Exception:
    error = traceback.format_exc()
    print(error, file=sys.stderr)
    status_path.write_text(json.dumps({"status": "failed", "error": error[-12000:]}))

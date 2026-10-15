"""Run one staged job from the NFS worker in an ephemeral Colab session.

Large model/runtime assets are archived on worker NFS and uploaded directly to
Colab. They never pass through the space-constrained portal host.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("job", type=Path)
p.add_argument("--session", required=True)
p.add_argument("--gpu", default="T4")
p.add_argument("--app-root", type=Path, required=True)
p.add_argument("--model-root", type=Path, required=True)
p.add_argument("--runtime-root", type=Path, required=True)
p.add_argument("--colab", default=str(Path.home() / ".local/bin/colab"))
a = p.parse_args()

work = a.job / "colab-stage"
work.mkdir(parents=True, exist_ok=True)
app_archive, job_archive = work / "app.tar.gz", work / "job.tar.gz"
data = json.loads((a.job / "job.json").read_text())
cache_root = a.model_root / "runtime-homes/colab-bundles"
cache_root.mkdir(parents=True, exist_ok=True)
status_file, results_file = work / "status.json", work / "results.zip"
progress_file = a.job / "colab-progress.json"


def refresh_colab_session() -> None:
    """Replace the expiring runtime-proxy token with the current assignment token."""
    cli_python = Path(a.colab).resolve().parent / "python3"
    code = """
import sys
from colab_cli.common import state
name = sys.argv[1]
sessions, assignments = state.sync_sessions()
session = state.store.get(name)
if session is None:
    raise SystemExit(f"session not found: {name}")
assignment = next((item for item in assignments if item.endpoint == session.endpoint), None)
if assignment is None:
    raise SystemExit(f"active assignment not found: {session.endpoint}")
session.token = assignment.runtime_proxy_info.token
session.url = assignment.runtime_proxy_info.url
state.store.add(session)
"""
    subprocess.run([str(cli_python), "-c", code, a.session], check=True, timeout=60)


def colab(*args: str, timeout: int = 3600, output=None, attempts: int = 1,
          refresh_session: bool = False) -> None:
    for attempt in range(1, attempts + 1):
        try:
            if refresh_session:
                refresh_colab_session()
            subprocess.run([a.colab, "--auth=oauth2", *args], check=True, timeout=timeout,
                           stdout=output, stderr=subprocess.STDOUT if output else None, text=True)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == attempts:
                raise
            delay = min(30, 2 ** attempt)
            print(f"[colab] Transient command failure; retrying in {delay}s ({attempt}/{attempts})",
                  flush=True)
            time.sleep(delay)


def publish_progress(detail: str, progress: int | None) -> None:
    incoming = progress_file.with_suffix(".json.incoming")
    incoming.write_text(json.dumps({"detail": detail, "progress": progress}))
    incoming.replace(progress_file)


def retrieve_over_kernel() -> set[str]:
    """Retrieve artifacts through kernel stdout when Contents API is unreliable."""
    stream = work / "retrieval.stream"
    refresh_colab_session()
    with stream.open("w") as output:
        colab("exec", "-s", a.session, "-f", str(a.app_root / "colab_retrieve.py"),
              "--timeout", "600", timeout=660, output=output, attempts=3,
              refresh_session=True)
    targets = {"status": status_file, "results": results_file}
    handles = {}
    expected = {}
    try:
        for line in stream.read_text().splitlines():
            fields = line.split("\t", 3)
            if fields[0] == "RTSEG_FILE" and len(fields) == 4 and fields[1] in targets:
                expected[fields[1]] = (int(fields[2]), fields[3])
                handles[fields[1]] = targets[fields[1]].open("wb")
            elif fields[0] == "RTSEG_CHUNK" and len(fields) == 3 and fields[1] in handles:
                handles[fields[1]].write(base64.b64decode(fields[2], validate=True))
            elif fields[0] == "RTSEG_END" and len(fields) >= 2 and fields[1] in handles:
                handles.pop(fields[1]).close()
    finally:
        for handle in handles.values():
            handle.close()
        stream.unlink(missing_ok=True)
    if "status" not in expected or not status_file.is_file():
        raise RuntimeError("Colab kernel stream did not return status")
    for name in expected:
        target = targets[name]
        if not target.is_file():
            raise RuntimeError(f"Colab kernel stream did not return {name}")
        size, digest = expected[name]
        if target.stat().st_size != size or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Colab kernel stream checksum failed for {name}")
    return set(expected)


def asset_roots(task: str) -> list[Path]:
    os.environ.update(RTSEG_APP_ROOT=str(a.app_root), RTSEG_MODEL_ROOT=str(a.model_root),
                      RTSEG_RUNTIME_ROOT=str(a.runtime_root))
    import model_preflight  # noqa: PLC0415

    roots: list[Path] = []
    for requirement in model_preflight.REQUIREMENTS.get(task, []):
        if requirement.startswith("@") or requirement.startswith("validation/"):
            continue
        if requirement.startswith("envs/"):
            path = a.runtime_root / requirement
            roots.append(a.runtime_root / "envs" / Path(requirement).parts[1])
            try:
                interpreter = (path if path.name == "python" else path.parent / "python").resolve()
                marker = Path("/.local/share/uv/python/")
                if marker.as_posix() in interpreter.as_posix():
                    parts = interpreter.parts
                    index = parts.index("python", parts.index("uv") + 1)
                    roots.append(Path(*parts[:index + 2]))
            except Exception:
                pass
        else:
            path = a.model_root / requirement
            if not path.exists():
                continue
            if requirement.startswith("sources/"):
                roots.append(a.model_root.joinpath(*Path(requirement).parts[:2]))
            elif path.is_dir():
                roots.append(path)
            else:
                roots.append(path.parent)
    unique = []
    for path in roots:
        path = path.resolve()
        if path.exists() and not any(path == old or old in path.parents for old in unique):
            unique = [old for old in unique if path not in old.parents]
            unique.append(path)
    return unique


try:
    with tarfile.open(app_archive, "w:gz") as archive:
        for path in a.app_root.iterdir():
            if path.name in {".git", ".venv", "data", "__pycache__"}:
                continue
            if path.is_file() or path.name in {"catalog", "labels", "models"}:
                archive.add(path, arcname=f"app/{path.name}")
        paths = {"RTSEG_MODEL_ROOT": str(a.model_root), "RTSEG_RUNTIME_ROOT": str(a.runtime_root)}
        for key in ("RTSEG_COLAB_UV_PYTHON", "RTSEG_COLAB_MODERN_LINK", "RTSEG_COLAB_MODEL_LINK"):
            if key in os.environ:
                paths[key] = os.environ[key]
        payload = json.dumps(paths).encode()
        info = tarfile.TarInfo("app/colab-paths.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(job_archive, "w:gz") as archive:
        for name in ("input", "input_ct", "prompt.json", "job.json"):
            path = a.job / name
            if path.exists(): archive.add(path, arcname=f"job/{name}")

    roots = []
    asset_tasks = []
    for task in data["models"]:
        task_roots = asset_roots(task)
        roots.extend(task_roots)
        if task_roots:
            asset_tasks.append(task)
    cache_key = hashlib.sha256("\0".join(sorted(asset_tasks)).encode()).hexdigest()[:16]
    asset_archive = cache_root / f"{cache_key}.tar.gz"
    if roots and not asset_archive.is_file():
        incoming = asset_archive.with_suffix(".tar.gz.incoming")
        with tarfile.open(incoming, "w:gz") as archive:
            for path in sorted(set(roots)):
                archive.add(path, arcname=str(path).lstrip("/"), recursive=True)
        incoming.replace(asset_archive)

    publish_progress("Provisioning Colab GPU", 1)
    colab("new", "-s", a.session, "--gpu", a.gpu, timeout=300)
    colab("upload", "-s", a.session, str(app_archive), "/content/rtseg-app.tar.gz", attempts=5)
    colab("upload", "-s", a.session, str(job_archive), "/content/rtseg-job.tar.gz", attempts=5)
    if asset_archive.is_file():
        parts = []
        with asset_archive.open("rb") as source:
            index = 0
            while chunk := source.read(64 * 1024**2):
                part = work / f"assets.part-{index:04d}"
                part.write_bytes(chunk)
                parts.append(part)
                index += 1
        try:
            completed = [0]
            completed_lock = threading.Lock()
            def upload_part(part: Path) -> None:
                colab("upload", "-s", a.session, str(part), f"/content/{part.name}",
                      timeout=300, attempts=5)
                with completed_lock:
                    completed[0] += 1
                    percent = 5 + int(35 * completed[0] / len(parts))
                    publish_progress(f"Uploading model runtime to Colab ({completed[0]}/{len(parts)} chunks)", percent)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(upload_part, part) for part in parts]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        finally:
            for part in parts:
                part.unlink(missing_ok=True)
    publish_progress("Colab GPU inference in progress", None)
    with (a.job / "colab.log").open("w") as log:
        colab("exec", "-s", a.session, "-f", str(a.app_root / "colab_worker.py"),
              "--timeout", str(12 * 3600), timeout=12 * 3600 + 300, output=log)
    publish_progress("Retrieving and verifying Colab results", 90)
    retrieved = retrieve_over_kernel()
    status = json.loads(status_file.read_text())
    if status.get("status") != "complete":
        raise RuntimeError(status.get("error") or "Colab worker failed")
    if "results" not in retrieved or not results_file.is_file():
        raise RuntimeError("Colab worker reported completion without a results archive")
    with zipfile.ZipFile(results_file) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).name != info.filename or not info.filename.endswith(".dcm"):
                raise RuntimeError("Unsafe Colab result archive")
        archive.extractall(a.job / "output")
finally:
    subprocess.run([a.colab, "--auth=oauth2", "stop", "-s", a.session], timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(work, ignore_errors=True)
    progress_file.unlink(missing_ok=True)

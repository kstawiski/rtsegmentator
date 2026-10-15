"""Stream Colab result artifacts over the kernel channel, bypassing Contents API."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path


for name, path in (("status", Path("/content/rtseg-status.json")),
                   ("results", Path("/content/rtseg-results.zip"))):
    if not path.is_file():
        if name == "status":
            raise FileNotFoundError(path)
        continue
    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"RTSEG_FILE\t{name}\t{size}\t{digest}", flush=True)
    with path.open("rb") as source:
        while chunk := source.read(256 * 1024):
            print(f"RTSEG_CHUNK\t{name}\t{base64.b64encode(chunk).decode('ascii')}", flush=True)
    print(f"RTSEG_END\t{name}", flush=True)

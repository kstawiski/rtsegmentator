"""Emit a deterministic size/hash manifest for an offline payload."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
records = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    records.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
print(json.dumps({"schema_version": 1, "files": records}, indent=2))

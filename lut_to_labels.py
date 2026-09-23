"""Convert GOUHFI's documented FreeSurfer-style LUT to RTSTRUCT label JSON."""
import argparse, json, re
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("lut"); p.add_argument("output"); a=p.parse_args()
labels={}
for line in Path(a.lut).read_text().splitlines():
    match=re.match(r"^\s*(\d+)\s+([^#]+?)(?:\s+\d+\s+\d+\s+\d+\s+\d+)?\s*$",line)
    if match and int(match.group(1)):
        labels[match.group(1)]=match.group(2).strip().replace(" ","_")
if not labels: raise SystemExit(f"No labels parsed from {a.lut}")
Path(a.output).write_text(json.dumps(labels,indent=2,sort_keys=True)+"\n")

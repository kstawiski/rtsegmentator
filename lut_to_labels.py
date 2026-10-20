"""Convert GOUHFI's documented FreeSurfer-style LUT to RTSTRUCT label JSON."""
import argparse, json
from structure_labels import EmptyLookupTable, lut_labels
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("lut"); p.add_argument("output"); a=p.parse_args()
try: labels=lut_labels(Path(a.lut))
except EmptyLookupTable: raise SystemExit(f"No labels parsed from {a.lut}")
Path(a.output).write_text(json.dumps(labels,indent=2,sort_keys=True)+"\n")

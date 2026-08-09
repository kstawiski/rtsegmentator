#!/usr/bin/env python3
"""Create an inference-only ISRT PET1 ablation config from an upstream template."""
import argparse
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("template")
p.add_argument("data_root")
p.add_argument("datalist")
p.add_argument("bundle_root")
p.add_argument("output")
a = p.parse_args()

text = Path(a.template).read_text()
replacements = {
    r"(?m)^extra_modalities:.*$": "extra_modalities: {image1: CT, image0: CT}",
    r"(?m)^input_channels:.*$": "input_channels: 3",
    r"(?m)^dataroot:.*$": f"dataroot: {a.data_root}",
    r"(?m)^datalist:.*$": f"datalist: {a.datalist}",
    r"(?m)^bundle_root:.*$": f"bundle_root: {a.bundle_root}",
    r"(?m)^ckpt_path:.*$": "ckpt_path: $@bundle_root + '/ckpt_f1'",
    r"(?m)^infer:.*?(?=\n\S|\Z)": ("infer: {enabled: true, ckpt_name: $@ckpt_path + '/model.pt', "
                                         "output_path: $@bundle_root + '/prediction_f1', data_list_key: testing}"),
    r"(?m)^resample:.*$": "resample: true",
    r"(?m)^num_workers:.*$": "num_workers: 0",
    r"(?m)^multigpu:.*$": "multigpu: false",
}
for pattern, replacement in replacements.items():
    text, count = re.subn(pattern, replacement, text, flags=re.S if "infer:" in pattern else 0)
    if count != 1:
        raise SystemExit(f"Expected one match for {pattern!r}, found {count}")
Path(a.output).write_text(text)
print(a.output)

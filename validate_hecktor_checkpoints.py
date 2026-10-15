#!/usr/bin/env python3
"""Strictly validate every published HECKTOR 2020 ensemble checkpoint."""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.source) / "src"))
    from models import FastSmoothSENormDeepUNet_supervision_skip_no_drop

    records = []
    for checkpoint in sorted(Path(args.weights).glob("*_best_model_weights.pt")):
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        state = OrderedDict((key.removeprefix("module."), value)
                            for key, value in state.items())
        model = FastSmoothSENormDeepUNet_supervision_skip_no_drop(2, 2, 24, 2)
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise SystemExit(f"state mismatch in {checkpoint.name}: {result}")
        records.append({"checkpoint": checkpoint.name, "tensors": len(state),
                        "bytes": checkpoint.stat().st_size})
    if len(records) != 8:
        raise SystemExit(f"expected eight checkpoints, found {len(records)}")
    report = {"result": "passed", "architecture":
              "FastSmoothSENormDeepUNet_supervision_skip_no_drop",
              "input_channels": 2, "classes": 2, "checkpoints": records}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

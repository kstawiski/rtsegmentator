#!/usr/bin/env python3
"""Run one MOOSE model and export its checkpoint-owned label mapping."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()

    from moosez import models
    from moosez.moosez import moose

    # MOOSE binds its package-local model directory as a default argument at
    # import time. Override that bound default so checkpoints remain in the
    # separately mounted private model bundle.
    models.Model.__init__.__defaults__ = (str(Path(args.model_root)),)
    outputs, used_models = moose(
        args.input, args.model, output_dir=str(Path(args.output).parent), accelerator="cuda"
    )
    if len(outputs) != 1 or len(used_models) != 1:
        raise SystemExit(f"MOOSE returned {len(outputs)} outputs for one requested model")

    produced = Path(outputs[0])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    produced.replace(destination)

    raw = used_models[0].dataset.get("labels", {})
    labels = {}
    for name, value in raw.items():
        if name.lower() == "background":
            continue
        if isinstance(value, list):
            raise SystemExit(f"MOOSE region label {name!r} is not a scalar class: {value!r}")
        labels[str(int(value))] = str(name)
    if not labels:
        raise SystemExit("MOOSE checkpoint contains no foreground label mapping")
    Path(args.labels).write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

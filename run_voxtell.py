"""Run the official VoxTell v1.1 model for one free-text 3-D prompt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()

    prompt_data = json.loads(Path(args.prompt).read_text())
    text_prompt = prompt_data.get("text")
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        raise SystemExit("VoxTell requires a non-empty string in prompt.text")
    text_prompt = " ".join(text_prompt.split())
    if len(text_prompt) > 512 or any(ord(char) < 32 for char in text_prompt):
        raise SystemExit("VoxTell prompt.text must be at most 512 printable characters")

    for item in (args.input, args.model_dir, args.embeddings, args.source,
                 args.text_model, args.site_packages):
        if not Path(item).exists():
            raise SystemExit(f"required VoxTell path does not exist: {item}")

    sys.path.insert(0, str(Path(args.site_packages).resolve()))
    sys.path.insert(0, str(Path(args.source).resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch  # noqa: PLC0415
    from voxtell.inference.predictor import VoxTellPredictor  # noqa: PLC0415

    predictor = VoxTellPredictor(
        model_dir=str(Path(args.model_dir).resolve()),
        device=torch.device("cuda:0"),
        text_encoding_model=str(Path(args.text_model).resolve()),
        embedding_bank=str(Path(args.embeddings).resolve()),
    )
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    written = predictor.predict_from_files(
        args.input, work, text_prompt, save_combined=False, verbose=True
    )
    if len(written) != 1 or not Path(written[0]).is_file():
        raise SystemExit(f"VoxTell produced {len(written)} outputs; expected one")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(written[0], output)


if __name__ == "__main__":
    main()

"""prepare_isrt_pet_config.py: inference-only config and the optional sliding-window override."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = """extra_modalities: {image: PET}
input_channels: 1
dataroot: /x
datalist: /x.json
bundle_root: /x
ckpt_path: /x
roi_size: [192, 192, 96]
output_classes: 2
infer: {enabled: false, ckpt_name: model.pt,
  output_path: /x}
resample: false
num_workers: 4
multigpu: true
"""


def run(tmp_path, *extra, template=TEMPLATE):
    source = tmp_path / "hyper_parameters.yaml"; source.write_text(template)
    output = tmp_path / "config.yaml"
    result = subprocess.run([sys.executable, str(ROOT / "prepare_isrt_pet_config.py"), str(source), "/data",
                             "/data/datalist.json", "/bundle", str(output), *extra], capture_output=True, text=True)
    return result, output


def test_default_keeps_upstream_inferer(tmp_path):
    result, output = run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "sliding_inferrer" not in output.read_text()


def test_late_fusion_override_uses_one_window(tmp_path):
    result, output = run(tmp_path, "--sw-batch-size", "1")
    assert result.returncode == 0, result.stderr
    line = [l for l in output.read_text().splitlines() if l.startswith("sliding_inferrer:")][0]
    assert "sw_batch_size: 1" in line and "overlap: 0.625" in line and "mode: gaussian" in line
    assert "roi_size: '@roi_size'" in line and "@output_classes" in line


def test_override_refuses_existing_inferer(tmp_path):
    result, _ = run(tmp_path, "--sw-batch-size", "1", template=TEMPLATE + "sliding_inferrer: {}\n")
    assert result.returncode != 0

"""Synthetic CPU regressions; only the Apptainer inference call is replaced."""
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

import run_hippunfold as runner


ROOT = Path(__file__).resolve().parents[1]


def result_name(hemi, modality="T1w", atlas="bigbrain", *, label="", crop=True):
    space = f"crop{modality}" if crop else modality
    return f"sub-rtseg_hemi-{hemi}_space-{space}{label}_desc-subfields_atlas-{atlas}_dseg.nii.gz"


def setup_runner(tmp_path, monkeypatch, model="ADNI_T1w_v1", *, cpu=True):
    reference = tmp_path / "input.nii.gz"
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([1.2, 1.6, 2.0])
    affine[:3, 3] = [-21, 18, 35]
    nib.save(nib.Nifti1Image(np.zeros((14, 9, 7), np.float32), affine), reference)
    image = tmp_path / "image.sif"
    image.touch()
    cache = tmp_path / "cache"
    cache.mkdir()
    work = tmp_path / "pipeline"
    output = tmp_path / "prediction.nii.gz"
    argv = ["run_hippunfold.py", "--input", str(reference), "--output", str(output),
            "--work-dir", str(work), "--image", str(image), "--cache", str(cache),
            "--model", model, "--app-root", str(ROOT), "--apptainer", "fake-apptainer"]
    if cpu:
        argv.append("--cpu")
    if model == "maguire_T2w":
        (cache / "model").mkdir()
        (cache / "model" / runner.MAGUIRE_ARCHIVE).write_bytes(b"synthetic model stand-in")
        for directory in ("template/CITI168", "atlases/tpl-multihist7"):
            resource = cache / directory
            resource.mkdir(parents=True)
            (resource / "synthetic.txt").write_text("synthetic cached resource")
        (cache.parent / "hippunfold-v2-overlay-dir").mkdir()
    monkeypatch.setattr(sys, "argv", argv)
    return reference, output, work, cache


@pytest.mark.parametrize("model,modality,atlas,label", [
    ("ADNI_T1w_v1", "T1w", "bigbrain", ""),
    ("maguire_T2w", "T2w", "multihist7", "_label-hipp"),
])
def test_cropped_outputs_return_to_exact_input_grid(tmp_path, monkeypatch, model, modality, atlas, label):
    reference, output, work, _ = setup_runner(tmp_path, monkeypatch, model)
    original_run = subprocess.run
    commands = []
    expected = np.zeros((14, 9, 7), dtype=np.uint16)

    def run(command, **kwargs):
        commands.append(command)
        if command[0] != "fake-apptainer":
            return original_run(command, **kwargs)
        # Both documented versions write final cropped native-coordinate maps.
        anat = work / "derivatives/hippunfold/sub-rtseg/anat"
        anat.mkdir(parents=True)
        ref = nib.load(reference)
        for hemi, start, values, offset in (("L", 2, (1, 7), 0), ("R", 8, (6, 8), 8)):
            affine = ref.affine.copy()
            affine[:3, 3] = (ref.affine @ [start, 3, 2, 1])[:3]
            affine[:3, :3] /= 2  # twice the resolution, translated and oblique
            data = np.full((4, 4, 4), values[0], np.uint16)
            data[2:] = values[1]
            path = anat / result_name(hemi, modality, atlas, label=label)
            nib.save(nib.Nifti1Image(data, affine), path)
            expected[start, 3:5, 2:4] = values[0] + offset
            expected[start + 1, 3:5, 2:4] = values[1] + offset
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.main()
    result = nib.load(output)
    assert result.shape == nib.load(reference).shape
    np.testing.assert_array_equal(result.affine, nib.load(reference).affine)
    np.testing.assert_array_equal(np.asanyarray(result.dataobj), expected)
    assert result.get_data_dtype() == np.uint16
    assert [Path(command[1]).name for command in commands[1:]] == [
        "resample_label_to_reference.py", "resample_label_to_reference.py",
        "combine_hippunfold_subfields.py",
    ]
    # Preserve the existing 1..8 left / 9..16 right bigbrain label convention.
    labels = json.loads((ROOT / "labels/hippunfold-bigbrain.json").read_text())
    assert labels["1"] == "Hipp_Subiculum_L"
    assert labels["14"] == "Hipp_DentateGyrus_R"
    assert labels["16"] == "Hipp_Cyst_R"


@pytest.mark.parametrize("missing", ["L", "R"])
def test_missing_hemisphere_stops_before_resampling(tmp_path, monkeypatch, missing):
    _, output, work, _ = setup_runner(tmp_path, monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        anat = work / "derivatives/hippunfold/sub-rtseg/anat"
        anat.mkdir(parents=True)
        (anat / result_name("R" if missing == "L" else "L")).touch()
        # A registered-space result or a different atlas cannot fill the gap.
        (anat / result_name(missing, "corobl", crop=False)).touch()
        (anat / result_name(missing, atlas="other")).touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(SystemExit, match=f"0 native {missing} subfield maps; searched"):
        runner.main()
    assert len(calls) == 1
    assert not output.exists()


def test_discovery_prefers_crop_and_rejects_ambiguous_crops(tmp_path):
    for hemi in ("L", "R"):
        (tmp_path / result_name(hemi)).touch()
        (tmp_path / result_name(hemi, crop=False)).touch()
        (tmp_path / result_name(hemi, label="_label-dentate")).touch()
    found = runner.discover_subfields(tmp_path, "T1w", "bigbrain")
    assert all("space-cropT1w" in path.name for path in found.values())
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / result_name("L")).touch()
    with pytest.raises(SystemExit, match="2 native L subfield maps.*ambiguous"):
        runner.discover_subfields(tmp_path, "T1w", "bigbrain")


@pytest.mark.parametrize("label", ["", "_label-hipp"])
def test_discovery_native_grid_fallback(tmp_path, label):
    for hemi in ("L", "R"):
        (tmp_path / result_name(hemi, crop=False, label=label)).touch()
    assert set(runner.discover_subfields(tmp_path, "T1w", "bigbrain")) == {"L", "R"}


@pytest.mark.parametrize("cpu", [True, False])
@pytest.mark.parametrize("custom_archive", [True, False])
def test_v2_offline_command(tmp_path, monkeypatch, cpu, custom_archive):
    _, _, work, cache = setup_runner(tmp_path, monkeypatch, "maguire_T2w", cpu=cpu)
    archive = cache / "model" / runner.MAGUIRE_ARCHIVE
    if custom_archive:
        alternative = tmp_path / "local model.tar.gz"
        archive.rename(alternative)
        archive = alternative
        sys.argv.extend(["--model-archive", str(archive)])
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd))
    # Stop at collection: this test does not pretend to execute inference.
    with pytest.raises(SystemExit, match="0 native L subfield maps"):
        runner.main()
    assert len(commands) == 1
    command = commands[0]
    binds = command[command.index("--bind") + 1].split(",")
    assert f"{cache}:/cache:ro" in binds
    assert f"{archive}:/cache/model/{runner.MAGUIRE_ARCHIVE}:ro" in binds
    wrapper = "/src/hippunfold/workflow/rtseg-offline.smk"
    assert f"{ROOT / 'hippunfold_offline.smk'}:{wrapper}:ro" in binds
    assert f"{work / 'bids'}:/bids:ro" in binds
    assert command[command.index("--env") + 1] == "HIPPUNFOLD_CACHE_DIR=/cache"
    assert command[command.index("--overlay") + 1].endswith("hippunfold-v2-overlay-dir:ro")
    assert "--cleanenv" in command
    assert "/src/hippunfold/run.py" in command
    assert command[-2:] == ["--snakefile", wrapper]
    assert ("--nv" in command) is (not cpu)
    assert ("--use-gpu" in command) is (not cpu)


@pytest.mark.parametrize("condition", ["missing", "empty", "directory"])
def test_v2_requires_local_model_before_starting(tmp_path, monkeypatch, condition):
    _, _, work, cache = setup_runner(tmp_path, monkeypatch, "maguire_T2w")
    archive = cache / "model" / runner.MAGUIRE_ARCHIVE
    archive.unlink()
    if condition == "empty":
        archive.touch()
    elif condition == "directory":
        archive.mkdir()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("inference started"))
    with pytest.raises(SystemExit, match="required offline HippUnfold Maguire model archive.*") as error:
        runner.main()
    assert str(archive) in str(error.value)
    assert "18432751" in str(error.value)
    assert not work.exists()


@pytest.mark.parametrize("resource", ["template/CITI168", "atlases/tpl-multihist7"])
def test_v2_requires_cached_resources(tmp_path, monkeypatch, resource):
    _, _, _, cache = setup_runner(tmp_path, monkeypatch, "maguire_T2w")
    (cache / resource / "synthetic.txt").unlink()
    with pytest.raises(SystemExit, match="required offline HippUnfold resource is missing or empty"):
        runner.main()


def test_v1_command_keeps_legacy_entrypoint(tmp_path, monkeypatch):
    setup_runner(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd))
    with pytest.raises(SystemExit, match="0 native L subfield maps"):
        runner.main()
    command, = commands
    assert command[command.index("--atlas") + 1] == "bigbrain"
    assert "--snakefile" not in command
    assert "--overlay" not in command
    assert "/src/hippunfold/run.py" not in command
    assert "--nv" not in command

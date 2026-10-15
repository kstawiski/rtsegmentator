import ast
import shutil
from pathlib import Path

import pytest


def mirror_function():
    # Dispatcher has top-level inference; load the unchanged helper without dispatching.
    tree = ast.parse((Path(__file__).parents[1] / "run_task.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "writable_mirror")
    ns = {"Path": Path, "shutil": shutil}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "run_task.py", "exec"), ns)
    return ns["writable_mirror"]


def test_mirror_keeps_files_linked_but_directories_writable(tmp_path):
    source, target = tmp_path / "bundle", tmp_path / "job"
    (source / "weights").mkdir(parents=True)
    (source / "weights/model").write_bytes(b"immutable checkpoint")
    (source / "alias").symlink_to(source / "weights", target_is_directory=True)
    mirror_function()(source, target)
    assert not (target / "alias").is_symlink()
    assert (target / "alias/model").is_symlink()
    (target / "alias/generated.json").write_text("{}")
    assert not (source / "weights/generated.json").exists()
    assert (source / "weights/model").read_bytes() == b"immutable checkpoint"


def test_mirror_rejects_cycles_and_overlap(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    mirror = mirror_function()
    with pytest.raises(ValueError):
        mirror(source, source / "nested")
    (source / "loop").symlink_to(source, target_is_directory=True)
    target = tmp_path / "target"
    with pytest.raises(ValueError):
        mirror(source, target)
    assert not target.exists()

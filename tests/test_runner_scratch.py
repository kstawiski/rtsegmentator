"""Real child processes and multiprocessing sockets on synthetic project storage."""
import ast
import atexit
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def short_temp(tmp_path, monkeypatch):
    # A short descriptor alias keeps all test files under pytest's authorized storage.
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    alias = Path(f'/proc/self/fd/{fd}')
    monkeypatch.setattr(tempfile, 'tempdir', str(alias))
    try:
        yield alias, fd
    finally:
        os.close(fd)


def runner_helpers(monkeypatch, tmp_path, short_temp, configured=None):
    alias, fd = short_temp
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(tmp_path / 'uninstalled-readonly-bundle'))
    if configured is None:
        monkeypatch.delenv('RTSEG_SCRATCH_ROOT', raising=False)
    else:
        monkeypatch.setenv('RTSEG_SCRATCH_ROOT', str(configured))
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(tmp_path / 'job'), 'total'])
    # Execute the real initialization and run helper, excluding inference dispatch.
    tree = ast.parse((ROOT / 'run_task.py').read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.Try): break
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'signal':
            continue
        body.append(node)
    callbacks = []
    monkeypatch.setattr(atexit, 'register', lambda fn, *args, **kwargs: callbacks.append((fn, args, kwargs)))
    original_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: original_popen(*args, pass_fds=(fd,), **kwargs))
    ns = {'__file__': str(ROOT / 'run_task.py')}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'run_task.py', 'exec'), ns)
    return ns, callbacks


@pytest.mark.parametrize('configured_kind', ['default', 'short', 'long', 'unwritable'])
def test_runner_temp_supports_manager_and_cleans_owned_paths(monkeypatch, tmp_path, short_temp, configured_kind, caplog):
    alias, _ = short_temp
    configured = None
    if configured_kind == 'short':
        configured = alias / 'scratch'
    elif configured_kind == 'long':
        configured = tmp_path / ('long-directory-' * 8)
    elif configured_kind == 'unwritable':
        configured = alias / 'not-a-directory'
        configured.write_text('retained file')
    ns, callbacks = runner_helpers(monkeypatch, tmp_path, short_temp, configured)
    result = tmp_path / 'child.txt'
    code = '''import multiprocessing, os, tempfile
from pathlib import Path
assert os.environ['TMPDIR'] == os.environ['TEMP'] == os.environ['TMP']
assert len(os.fsencode(os.environ['TMPDIR'])) <= 60
with multiprocessing.Manager() as manager:
    data = manager.dict(answer=42)
    assert data['answer'] == 42
Path(os.environ['RESULT']).write_text(tempfile.gettempdir())
'''
    try:
        ns['run']([sys.executable, '-c', code], env={**os.environ, 'RESULT': str(result)})
        child_temp = Path(result.read_text())
        assert child_temp.is_dir()
        assert not (tmp_path / 'uninstalled-readonly-bundle').exists()
        if configured_kind == 'short':
            assert child_temp == configured
        else:
            assert child_temp.name.startswith('rtseg-')
        if configured_kind in {'long', 'unwritable'}:
            assert 'RTSEG_SCRATCH_ROOT' in caplog.text
    finally:
        for fn, args, kwargs in callbacks: fn(*args, **kwargs)
    if configured_kind == 'short':
        assert configured.is_dir()  # Operator-owned scratch is never removed.
    else:
        assert not child_temp.exists()
    assert result.is_file()  # Outputs are not scratch.


def test_scratch_cleanup_registered_even_when_child_fails(monkeypatch, tmp_path, short_temp):
    ns, callbacks = runner_helpers(monkeypatch, tmp_path, short_temp)
    root = ns['SCRATCH_ROOT']
    try:
        with pytest.raises(SystemExit) as exc:
            ns['run']([sys.executable, '-c', 'raise SystemExit(7)'])
        assert exc.value.code == 7
        assert callbacks and root.is_dir()
    finally:
        for fn, args, kwargs in callbacks: fn(*args, **kwargs)
    assert not root.exists()


@pytest.mark.parametrize('exit_code', [0, 7])
def test_real_process_exit_removes_default_scratch(tmp_path, short_temp, exit_code):
    alias, fd = short_temp
    code = '''import ast, os, sys
from pathlib import Path
source = Path(sys.argv[1])
sys.argv = [str(source), sys.argv[2], 'total']
tree = ast.parse(source.read_text())
body = []
for node in tree.body:
    if isinstance(node, ast.Try): break
    body.append(node)
ns = {'__file__': str(source)}
exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), ns)
print(ns['SCRATCH_ROOT'], flush=True)
ns['run']([sys.executable, '-c', 'raise SystemExit(' + os.environ['EXIT_CODE'] + ')'])
'''
    environment = {**os.environ, 'TMPDIR': str(alias), 'RTSEG_MODEL_ROOT': str(tmp_path / 'models'),
                   'EXIT_CODE': str(exit_code)}
    environment.pop('RTSEG_SCRATCH_ROOT', None)
    result = subprocess.run([sys.executable, '-c', code, str(ROOT / 'run_task.py'), str(tmp_path / 'job')],
                            env=environment, pass_fds=(fd,), capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    path = Path(result.stdout.strip())
    assert path.name.startswith('rtseg-')
    assert not path.exists()

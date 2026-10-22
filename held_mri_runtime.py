"""Installed-file and import-policy evidence; no CUDA or model execution."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PYTHON_POLICY = {'flags': ['-E', '-s', '-B', '-X', 'pycache_prefix=<empty-private-directory>'],
                 'cache_directory': 'new per process; mode 0700; removed after exit',
                 'sourceless_bytecode': 'inventoried outside __pycache__'}


@contextmanager
def python_command(python, *, directory=None, isolated=False):
    """Redirect cache reads before Python startup; -B alone only stops writes."""
    with tempfile.TemporaryDirectory(prefix='held-pycache-', dir=directory) as cache:
        yield [str(python), *(['-I'] if isolated else ['-E', '-s']), '-B',
               '-X', 'pycache_prefix=' + cache]


def check_python_policy():
    prefix = sys.pycache_prefix
    if (not sys.flags.ignore_environment or not sys.flags.no_user_site
            or not sys.dont_write_bytecode or not prefix
            or sys.flags.optimize or sys.flags.isolated or sys.flags.safe_path
            or sys._xoptions != {'pycache_prefix': prefix}):
        raise ValueError('Held MRI requires -E -s -B -X pycache_prefix=<empty-private-directory> at startup')
    path = Path(prefix)
    if (path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.getuid()
            or path.stat().st_mode & 0o077 or any(path.iterdir())):
        raise ValueError('Held MRI bytecode cache directory must be private and empty')


def runtime_identity(python):
    script = ('import sys,sysconfig,json,importlib.metadata as m; '
              'print(json.dumps(dict(version=sys.version,prefix=sys.prefix,stdlib=sysconfig.get_path("stdlib"),'
              'import_path=sys.path,user_site=False,'
              'libpython=str(__import__("pathlib").Path(sysconfig.get_config_var("LIBDIR"))/sysconfig.get_config_var("LDLIBRARY")),'
              'packages=sorted((d.metadata["Name"],d.version) for d in m.distributions()))))')
    try:
        with python_command(python, isolated=True) as command:
            result = subprocess.run([*command, '-c', script],
                                    check=True, capture_output=True, text=True, timeout=120)
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError('Cannot verify installed MRI runtime identity: ' + str(exc)) from exc


def runtime_manifest(python, *, uncached=False):
    """Inventory the whole environment, stdlib and every additional import root.

    Redirected source caches cannot execute. Legacy/sourceless .pyc files outside
    __pycache__ remain executable and are hashed. Site .pth files, sitecustomize,
    zip imports and editable-package roots are included, not inferred from locks.
    """
    from prostate_models import file_sha256
    python = Path(python).absolute()
    identity = runtime_identity(python)
    env = Path(identity.pop('prefix', python.parent.parent))
    stdlib = Path(identity.pop('stdlib'))
    paths = {'interpreter': python}
    libpython = identity.pop('libpython', None)
    if libpython and Path(libpython).is_file():
        paths['libpython'] = Path(libpython)
    roots = [('environment', env), ('stdlib', stdlib)]
    for index, entry in enumerate(identity.get('import_path', [])):
        path = Path(entry)
        if not path.is_absolute():
            raise ValueError(f'Uncontrolled relative runtime import path: {entry}')
        if path.is_file():
            paths[f'import-file/{index}'] = path
        elif path.is_dir() and not any(path.is_relative_to(base) for _, base in roots):
            roots.append((f'import-root/{index}', path))
    for prefix, directory in roots:
        visited = set()
        for current, directories, names in os.walk(directory, followlinks=True):
            current = Path(current)
            resolved = current.resolve()
            if resolved in visited:
                directories[:] = []
                continue
            visited.add(resolved)
            directories[:] = sorted(name for name in directories if name != '__pycache__'
                                    and not (prefix == 'stdlib' and name == 'site-packages'))
            for name in names:
                path = current / name
                if path.is_file():
                    paths[f'{prefix}/{path.relative_to(directory).as_posix()}'] = path
    files = {key: {'size': path.stat().st_size, 'sha256': file_sha256(path, uncached=uncached)}
             for key, path in sorted(paths.items())}
    return {'schema_version': 2, 'identity': identity, 'python_policy': PYTHON_POLICY, 'files': files}


def worker_runtime_manifest(python, *, uncached=False):
    manifest = runtime_manifest(python, uncached=uncached)
    return {'executable': str(Path(python).absolute()), 'runtime': manifest,
            'import_environment': {'python_policy': PYTHON_POLICY,
                'script_directory': 'bound application root',
                'PYTHONPATH': 'ignored', 'PYTHONHOME': 'ignored', 'user_site': False,
                'site_import_path': manifest['identity'].get('import_path', [])},
            'coverage': 'whole worker environment, standard library and additional site import roots'}


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

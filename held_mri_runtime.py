"""Installed-file and import-policy evidence; no CUDA or model execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat as stat_module
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
            or sys.flags.optimize or sys.flags.isolated or getattr(sys.flags, 'safe_path', 0)
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


def _collect_root_files(roots):
    def scan_dir(item):
        prefix, real_cur = item
        rows = []
        try:
            with os.scandir(real_cur) as entries:
                for entry in entries:
                    if entry.name == '__pycache__' or (prefix == 'stdlib' and entry.name == 'site-packages'):
                        continue
                    try:
                        lst = entry.stat(follow_symlinks=False)
                        is_link = stat_module.S_ISLNK(lst.st_mode)
                        st = entry.stat(follow_symlinks=True) if is_link else lst
                    except OSError:
                        continue
                    real = str(Path(entry.path).resolve()) if is_link else os.path.join(real_cur, entry.name)
                    if stat_module.S_ISDIR(st.st_mode):
                        rows.append((entry.name, 'dir', real, st))
                    elif stat_module.S_ISREG(st.st_mode):
                        rows.append((entry.name, 'file', real, st))
        except OSError:
            return (prefix, real_cur), []
        rows.sort(key=lambda row: row[0])
        return (prefix, real_cur), rows

    dir_children = {}
    frontier = []
    seen = set()
    resolved_roots = []
    for prefix, directory in roots:
        try:
            real_root = str(directory.resolve())
            if not directory.is_dir():
                continue
        except OSError:
            continue
        resolved_roots.append((prefix, directory, real_root))
        if (prefix, real_root) not in seen:
            seen.add((prefix, real_root))
            frontier.append((prefix, real_root))
    with ThreadPoolExecutor(max_workers=32) as executor:
        while frontier:
            next_frontier = []
            for key, rows in executor.map(scan_dir, frontier):
                dir_children[key] = rows
                prefix = key[0]
                for _, kind, real_child, _ in rows:
                    if kind == 'dir' and (prefix, real_child) not in seen:
                        seen.add((prefix, real_child))
                        next_frontier.append((prefix, real_child))
            frontier = next_frontier
    collected = {}
    for prefix, directory, real_root in resolved_roots:
        visited = set()
        def dfs(cur: Path, real_cur: str, rel_prefix: str):
            if real_cur in visited:
                return
            visited.add(real_cur)
            for name, kind, real_child, st in dir_children.get((prefix, real_cur), ()):
                child = cur / name
                rel = f'{rel_prefix}/{name}' if rel_prefix else name
                if kind == 'dir':
                    dfs(child, real_child, rel)
                elif kind == 'file':
                    collected[f'{prefix}/{rel}'] = (child, real_child, st)
        dfs(directory, real_root, '')
    return collected


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
    paths = {'interpreter': (python, str(python.resolve()), python.stat())}
    libpython = identity.pop('libpython', None)
    if libpython and Path(libpython).is_file():
        lp = Path(libpython)
        paths['libpython'] = (lp, str(lp.resolve()), lp.stat())
    roots = [('environment', env), ('stdlib', stdlib)]
    for index, entry in enumerate(identity.get('import_path', [])):
        path = Path(entry)
        if not path.is_absolute():
            raise ValueError(f'Uncontrolled relative runtime import path: {entry}')
        if path.is_file():
            paths[f'import-file/{index}'] = (path, str(path.resolve()), path.stat())
        elif path.is_dir() and not any(path.is_relative_to(base) for _, base in roots):
            roots.append((f'import-root/{index}', path))
    paths.update(_collect_root_files(roots))
    files = {key: {'size': st.st_size,
                   'sha256': file_sha256(path, uncached=uncached, _stat=st, _resolved=real)}
             for key, (path, real, st) in sorted(paths.items())}
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

"""Prostate runner contracts, staged DICOM validation and CPU orchestration."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat as stat_module
import subprocess
import tempfile
import threading
import time

from held_mri import TASKS as HELD_MRI_TASKS, COMMITS as HELD_MRI_COMMITS

CHANNEL_ROLES = ('t2w', 'adc', 'hbv')
TRAINER = 'nnUNetTrainerV2_Loss_FL_and_CE_checkpoints'
UMU_FOLDER = 'Dataset077_ProstateZones/nnUNetTrainer__nnUNetPlans__3d_fullres'
TASKS = {
    'picai_nnunet_supervised_bpmri': ('picai_supervised', 'picai', 'run_picai.py'),
    'picai_nnunet_semisupervised_bpmri': ('picai_semi_supervised', 'picai', 'run_picai.py'),
    'umu_prostate_zones_urethra_t2': ('umu_prostate_zones', 'umu-nnunetv2', 'run_umu_prostate.py'),
    'prostate158_tumor_bpmri': ('prostate158_tumor', 'prostate158', 'run_prostate158.py'),
}
SOURCE_COMMITS = {
    'picai_supervised': 'fbea30efc5f57adab0445e554f3de3c0a2dca64a',
    'picai_semi_supervised': '6129f7e0fe3fc3994064672aa2c7f629584004a3',
    'umu_prostate_zones': 'eacfe6436635f43b74dadf4d31e8424a856014e5',
    'prostate158_tumor': '8187638887409c95ba0b091f249a40c0822c89e0',
}
TASKS.update(HELD_MRI_TASKS)
SOURCE_COMMITS.update(HELD_MRI_COMMITS)

UMU_LABELS = {'background': 0, 'PZ': 1, 'CZ': 2, 'TZ': 3, 'AFS': 4, 'Urethra': 5}


def artifact_names(task):
    """Finite, task-owned download allowlist; inference work files are excluded."""
    if task in HELD_MRI_TASKS:
        suffixes = ['.lesions.json', '.probability.nii.gz', '.detection.nii.gz']
        if task == 'nci_monai_prostate_lesion_bpmri':
            suffixes.append('.gland.nii.gz')
        return [f'{task}_RTSTRUCT.dcm{s}' for s in suffixes]
    if task not in TASKS:
        return []
    suffixes = ('.lesions.json', '.probability.nii.gz', '.detection.nii.gz') if task.startswith('picai_') else (
        '.provenance.json', '.segmentation.nii.gz')
    if task == 'umu_prostate_zones_urethra_t2':
        suffixes += ('.probabilities.npz',)
    return [f'{task}_RTSTRUCT.dcm{suffix}' for suffix in suffixes]


def model_folder(task):
    model, _, _ = TASKS[task]
    if model.startswith('picai_'):
        number = 2201 if model == 'picai_supervised' else 2203
        return Path('weights') / model / 'results/nnUNet/3d_fullres' / f'Task{number}_picai_baseline' / f'{TRAINER}__nnUNetPlansv2.1'
    if model == 'umu_prostate_zones':
        return Path('weights') / model / UMU_FOLDER
    return Path('weights') / model


PREFLIGHT_CACHE_ENV = 'RTSEG_HELD_MRI_PREFLIGHT_CACHE'
PREFLIGHT_CACHE_FILENAME = 'digests.json'
PREFLIGHT_CACHE_SCHEMA = 1
PREFLIGHT_FLUSH_INTERVAL_SECONDS = 30.0
_PREFLIGHT_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
_SHA256_CACHE = {}
_PREFLIGHT_STATE = threading.local()


def _default_protected_roots(model_root=None, app_root=None, runtime_root=None):
    return (
        app_root or os.environ.get('RTSEG_APP_ROOT') or Path(__file__).resolve().parent,
        model_root or os.environ.get('RTSEG_MODEL_ROOT'),
        runtime_root or os.environ.get('RTSEG_RUNTIME_ROOT'),
        Path(__file__).resolve().parent,
    )


def _scratch_cache_directory() -> Path:
    scratch = os.environ.get('RTSEG_SCRATCH_ROOT', '').strip() or '/scratch'
    return Path(scratch) / 'held-mri-preflight-cache'


def _home_cache_candidate() -> tuple[Path, Path] | None:
    xdg = os.environ.get('XDG_CACHE_HOME', '').strip()
    if xdg:
        base = Path(xdg)
        if not base.is_absolute():
            return None
        return base / 'rtsegmentator' / 'held-mri-preflight-cache', base
    if 'HOME' in os.environ:
        home = os.environ['HOME'].strip()
        if not home:
            return None
        base = Path(home)
    else:
        try:
            base = Path.home()
        except (RuntimeError, KeyError, OSError):
            return None
    if not base.is_absolute():
        return None
    return base / '.cache' / 'rtsegmentator' / 'held-mri-preflight-cache', base


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _valid_cache_stat(st, *, directory: bool) -> bool:
    kind_ok = stat_module.S_ISDIR(st.st_mode) if directory else stat_module.S_ISREG(st.st_mode)
    return bool(kind_ok and st.st_uid == os.getuid() and (st.st_mode & 0o022) == 0)


def _validated_cache_directory(cache_dir: Path, protected_roots, *, create: bool = False,
                               base_dir: Path | None = None) -> Path | None:
    cache_dir = Path(cache_dir)
    base_dir = Path(base_dir) if base_dir is not None else cache_dir
    if not cache_dir.is_absolute() or not base_dir.is_absolute():
        return None
    try:
        if any(p.is_symlink() for p in (cache_dir, *cache_dir.parents)):
            return None
    except OSError:
        return None
    if cache_dir == base_dir or not cache_dir.is_relative_to(base_dir):
        check_dirs = [cache_dir]
    else:
        check_dirs = []
        cur = cache_dir
        while True:
            check_dirs.append(cur)
            if cur == base_dir:
                break
            cur = cur.parent
        check_dirs.reverse()
    for d in check_dirs:
        try:
            lst = os.lstat(d)
        except FileNotFoundError:
            if not create:
                return None
            continue
        except OSError:
            return None
        if not _valid_cache_stat(lst, directory=True):
            return None
    if not cache_dir.exists():
        if not create:
            return None
        missing = []
        cur = cache_dir
        while not cur.exists():
            if cur.is_symlink():
                return None
            missing.append(cur)
            if cur.parent == cur:
                return None
            cur = cur.parent
        try:
            anc_st = os.lstat(cur)
            prospective = cur.resolve() / cache_dir.relative_to(cur)
        except OSError:
            return None
        if not stat_module.S_ISDIR(anc_st.st_mode) or (anc_st.st_mode & 0o200) == 0:
            return None
        for root in protected_roots:
            if root is None:
                continue
            try:
                resolved_root = Path(root).resolve()
            except OSError:
                continue
            if _paths_overlap(prospective, resolved_root):
                return None
        try:
            for d in reversed(missing):
                d.mkdir(mode=0o700, exist_ok=True)
                os.chmod(d, 0o700)
        except OSError:
            return None
    for d in check_dirs:
        try:
            lst = os.lstat(d)
        except OSError:
            return None
        if not _valid_cache_stat(lst, directory=True):
            return None
    try:
        lst = os.lstat(cache_dir)
        resolved_dir = cache_dir.resolve()
        st = os.stat(resolved_dir)
    except OSError:
        return None
    if (Path(os.path.abspath(cache_dir)) != resolved_dir
            or (st.st_dev, st.st_ino) != (lst.st_dev, lst.st_ino)
            or not _valid_cache_stat(st, directory=True)):
        return None
    for root in protected_roots:
        if root is None:
            continue
        try:
            resolved_root = Path(root).resolve()
        except OSError:
            continue
        if _paths_overlap(resolved_dir, resolved_root):
            return None
    # An existing directory may be safe but not writable by this process; a cache it cannot write
    # would leave every later preflight cold, so treat it as unusable (default locations fall back).
    if not os.access(resolved_dir, os.W_OK | os.X_OK):
        return None
    return resolved_dir


def preflight_cache_directory(protected_roots=None, *, create: bool = True) -> Path:
    explicit = os.environ.get(PREFLIGHT_CACHE_ENV, '').strip()
    if explicit:
        return Path(explicit)
    roots = _default_protected_roots() if protected_roots is None else protected_roots
    candidate = _home_cache_candidate()
    if candidate is not None:
        home_dir, home_base = candidate
        if _validated_cache_directory(home_dir, roots, create=create, base_dir=home_base) is not None:
            return home_dir
    return _scratch_cache_directory()


def _resolve_preflight_cache_directory(protected_roots) -> Path | None:
    explicit = os.environ.get(PREFLIGHT_CACHE_ENV, '').strip()
    if explicit:
        return _validated_cache_directory(Path(explicit), protected_roots, create=True)
    candidate = _home_cache_candidate()
    if candidate is not None:
        home_dir, home_base = candidate
        validated = _validated_cache_directory(home_dir, protected_roots, create=True, base_dir=home_base)
        if validated is not None:
            return validated
    if not os.environ.get('RTSEG_SCRATCH_ROOT', '').strip() and not Path('/scratch').exists():
        return None
    return _validated_cache_directory(_scratch_cache_directory(), protected_roots, create=True)


def _load_preflight_cache_file(cache_dir: Path) -> tuple[dict, bool, bool]:
    cache_file = cache_dir / PREFLIGHT_CACHE_FILENAME
    try:
        lst = os.lstat(cache_file)
    except FileNotFoundError:
        return {}, True, False
    except OSError:
        return {}, False, False
    if not _valid_cache_stat(lst, directory=False):
        return {}, False, False
    fd = None
    try:
        fd = os.open(cache_file, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        fst = os.fstat(fd)
        if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino) or not _valid_cache_stat(fst, directory=False):
            return {}, False, False
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            raw = stream.read()
    except OSError:
        return {}, True, True
    finally:
        if fd is not None:
            os.close(fd)
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return {}, True, True
    if not isinstance(payload, dict) or payload.get('schema_version') != PREFLIGHT_CACHE_SCHEMA or not isinstance(payload.get('entries'), dict):
        return {}, True, True
    entries = {}
    for resolved, row in payload['entries'].items():
        if not isinstance(resolved, str) or not resolved or not isinstance(row, list) or len(row) != 6:
            return {}, True, True
        dev, ino, size, mtime_ns, ctime_ns, digest = row
        if (not all(type(v) is int for v in (dev, ino, size, mtime_ns, ctime_ns))
                or size < 0
                or not isinstance(digest, str) or len(digest) != 64
                or not all(c in '0123456789abcdef' for c in digest)):
            return {}, True, True
        entries[(resolved, dev, ino, size, mtime_ns, ctime_ns)] = digest
    return entries, True, False


def _save_preflight_cache_file(cache_dir: Path, protected_roots, updates: dict) -> bool:
    validated_dir = _validated_cache_directory(cache_dir, protected_roots, create=True)
    if validated_dir is None:
        return False
    lock_path = validated_dir / 'digests.lock'
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        if not _valid_cache_stat(os.fstat(lock_fd), directory=False):
            return False
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            for stale in validated_dir.glob('.digests-*.tmp'):
                stale.unlink(missing_ok=True)
            existing, writable, _ = _load_preflight_cache_file(validated_dir)
            if not writable:
                return False
            by_path = {key[0]: [*key[1:], digest] for key, digest in existing.items()}
            for key, digest in updates.items():
                by_path[key[0]] = [*key[1:], digest]
            payload = (json.dumps({'schema_version': PREFLIGHT_CACHE_SCHEMA, 'entries': dict(sorted(by_path.items()))}, separators=(',', ':')) + '\n').encode('utf-8')
            fd, tmp_name = tempfile.mkstemp(prefix='.digests-', suffix='.tmp', dir=validated_dir)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    fd = None
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp_name, validated_dir / PREFLIGHT_CACHE_FILENAME)
            finally:
                if fd is not None:
                    os.close(fd)
                Path(tmp_name).unlink(missing_ok=True)
            return True
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError:
        return False
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def _flush_preflight_session(session, *, now: float | None = None) -> bool:
    if session is None or not session.get('writable') or session.get('flushing'):
        return False
    if not session.get('dirty') and not session.get('corrupt'):
        return False
    cache_dir = session.get('cache_dir')
    if cache_dir is None:
        return False
    session['flushing'] = True
    previous_mask = None
    try:
        if hasattr(signal, 'pthread_sigmask') and threading.current_thread() is threading.main_thread():
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _PREFLIGHT_SIGNALS)
        saved = _save_preflight_cache_file(cache_dir, session['protected_roots'], session['updates'])
        if saved:
            session['dirty'] = False
            session['corrupt'] = False
            session['last_flush'] = time.monotonic() if now is None else now
        return saved
    finally:
        session['flushing'] = False
        if previous_mask is not None:
            # Restore the caller's exact mask: signals it had blocked stay blocked.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _preflight_session():
    return getattr(_PREFLIGHT_STATE, 'session', None)


@contextmanager
def preflight_digest_cache(model_root=None, app_root=None, runtime_root=None):
    """Persistent digest cache for held-MRI preflight only; never used at execution or publication."""
    current = getattr(_PREFLIGHT_STATE, 'session', None)
    if current is not None:
        yield current
        return
    protected = _default_protected_roots(model_root=model_root, app_root=app_root, runtime_root=runtime_root)
    validated_dir = _resolve_preflight_cache_directory(protected)
    entries, writable, corrupt = ({}, False, False)
    if validated_dir is not None:
        entries, writable, corrupt = _load_preflight_cache_file(validated_dir)
    session = {
        'cache_dir': validated_dir,
        'protected_roots': protected,
        'digests': entries,
        'updates': {},
        'dirty': False,
        'flushing': False,
        'last_flush': time.monotonic(),
        'writable': bool(validated_dir is not None and writable),
        'corrupt': corrupt,
        'bindings': {},
        'gpus': {},
    }
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        def _restore_handlers():
            for sig, prev in list(previous_handlers.items()):
                try:
                    signal.signal(sig, prev)
                except (OSError, ValueError):
                    pass
            previous_handlers.clear()

        def _handle_signal(signum, frame):
            prev = previous_handlers.get(signum, signal.SIG_DFL)
            _flush_preflight_session(session)
            _restore_handlers()
            if callable(prev):
                prev(signum, frame)
            raise SystemExit(128 + signum)

        for sig in _PREFLIGHT_SIGNALS:
            try:
                previous_handlers[sig] = signal.signal(sig, _handle_signal)
            except (OSError, ValueError):
                pass
    _PREFLIGHT_STATE.session = session
    try:
        yield session
    finally:
        _PREFLIGHT_STATE.session = None
        try:
            _flush_preflight_session(session)
        finally:
            if previous_handlers:
                for sig, prev in list(previous_handlers.items()):
                    try:
                        signal.signal(sig, prev)
                    except (OSError, ValueError):
                        pass
                previous_handlers.clear()


def _hash_file_bytes(path: Path, stat=None) -> str:
    path = Path(path)
    stat = stat if stat is not None else path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'File changed while hashing: {path}')
    return digest.hexdigest()


def file_sha256(path, *, uncached=False, _stat=None, _resolved=None):
    """Hash current bytes; ctime and inode detect timestamp-preserving changes."""
    path = Path(path)
    stat = _stat if _stat is not None else path.stat()
    resolved = _resolved if _resolved is not None else str(path.resolve())
    key = (resolved, stat.st_dev, stat.st_ino, stat.st_size,
           stat.st_mtime_ns, stat.st_ctime_ns)
    session = None if uncached else _preflight_session()
    if session is not None:
        if key in session['digests']:
            return session['digests'][key]
        value = _hash_file_bytes(path, stat)
        session['digests'][key] = value
        session['updates'][key] = value
        session['dirty'] = True
        _SHA256_CACHE[key] = value
        now = time.monotonic()
        if now - session['last_flush'] >= PREFLIGHT_FLUSH_INTERVAL_SECONDS:
            _flush_preflight_session(session, now=now)
        return value
    if uncached or key not in _SHA256_CACHE:
        _SHA256_CACHE[key] = _hash_file_bytes(path, stat)
    return _SHA256_CACHE[key]


def weight_files(task):
    if task in HELD_MRI_TASKS:
        from held_mri import weights
        return weights(task)
    folder = model_folder(task)
    if task.startswith('picai_'):
        return [folder / 'plans.pkl', *[folder / f'fold_{fold}' / name for fold in range(5)
                for name in ('model_best.model', 'model_best.model.pkl')]]
    if task == 'umu_prostate_zones_urethra_t2':
        return [folder / 'dataset.json', folder / 'plans.json',
                *[folder / f'fold_{fold}/checkpoint_best.pth' for fold in range(5)]]
    return [folder / 'tumor.pt']


def requirements(task):
    if task in HELD_MRI_TASKS:
        from held_mri import requirements as held_requirements
        return held_requirements(task)
    model, env, runner = TASKS[task]
    files = [f'@{runner}', '@prostate_models.py', '@prostate_postprocess.py',
             f'runtime-envs/{env}/bin/python', f'sources/{model}/LICENSE',
             *map(str, weight_files(task)), f'validation/{task}.json']
    if task.startswith('picai_'):
        files += ['@convert_masks_validate_rtstruct.py', '@picai_predict.py', f'runtime-envs/{env}/bin/nnUNet_predict']
        files += [f'sources/{model}/{name}' for name in (
            'nnUNetTrainerV2_focalLoss.py', 'nnUNetTrainerV2_Loss_CE_checkpoints.py',
            'nnUNetTrainerV2_Loss_FL_and_CE.py')]
    else:
        files += [f'@labels/{model}.json']
        if model == 'umu_prostate_zones':
            files += [f'runtime-envs/{env}/bin/nnUNetv2_predict_from_modelfolder',
                      '@third_party/umu_prostate_zones/utils_postprocess_nnUNet.py',
                      '@third_party/umu_prostate_zones/LICENSE']
    return files


def marker_has_failure(value):
    """A success label cannot conceal failed checks or a waived failure."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'passed', 'predeclared_passed'} and item is not True:
                return True
            if key in {'result', 'status'} and item not in {'passed', 'nonempty'}:
                return True
            if key in {'error', 'errors', 'fatal_error', 'failure', 'failures', 'failed_checks',
                       'unresolved_failures', 'deviations'} and item:
                return True
            if marker_has_failure(item):
                return True
    elif isinstance(value, list):
        return any(marker_has_failure(item) for item in value)
    return False


def installed_source_commit(source):
    """Commit of a pinned upstream checkout, read from .git without running git.

    The portal runs as root in a container while the bundle belongs to another user,
    so `git rev-parse` refuses the repository ("dubious ownership"). Reading HEAD and
    the ref it names (loose or packed) needs no git binary. Without .git, the install
    manifest records the commit.
    """
    git = Path(source) / '.git'
    if not git.is_dir():
        return json.loads((Path(source) / 'RTSEG_MANIFEST.json').read_text())['commit']
    try:
        head = (git / 'HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = head[5:].strip()
            loose = git / ref
            if loose.is_file():
                head = loose.read_text().strip()
            else:
                packed = (git / 'packed-refs').read_text().splitlines()
                head = next(line.split()[0] for line in packed
                            if not line.startswith(('#', '^')) and line.split()[-1] == ref)
    except (OSError, StopIteration, IndexError) as exc:
        raise ValueError('Cannot verify installed source commit') from exc
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('Cannot verify installed source commit')
    return head


def validate_acceptance(record, task, root):
    """Validate the prostate harness marker against this installed release."""
    if task in HELD_MRI_TASKS:
        from held_mri import validate_marker
        return validate_marker(record, task, root)
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(isinstance(record, dict) and record.get('task') == task and record.get('result') == 'passed',
            'Acceptance task/result mismatch')
    require(not marker_has_failure(record), 'Acceptance contains a failure or deviation')
    modes = record.get('acceptance_modes')
    require(isinstance(modes, list) and len(modes) == 2 and 'reference' in modes and 'dicom' in modes,
            'Both acceptance modes are required')
    proof = record.get('reference_mode')
    require(isinstance(proof, dict) and proof.get('passed') is True and
            isinstance(proof.get('result_sha256'), str) and re.fullmatch('[0-9a-f]{64}', proof['result_sha256']) and
            type(proof.get('runs')) is int and proof['runs'] == (2 if task.startswith('picai_') else 1),
            'Missing bound reference evidence')
    checks = record.get('checks')
    required = {'gpu_inference', 'exact_input_grid', 'rtstruct_roundtrip', 'orientation', 'both_acceptance_modes',
                'author_reference_and_repeatability' if task.startswith('picai_') else 'model_output_contract'}
    require(isinstance(checks, list) and all(isinstance(c, str) for c in checks) and required <= set(checks),
            'Incomplete acceptance checks')
    model = TASKS[task][0]
    source = Path(root) / 'sources' / model
    installed_commit = installed_source_commit(source)
    require(record.get('source_commit') == installed_commit == SOURCE_COMMITS[model], 'Source commit mismatch')
    hashes = {relative.as_posix(): file_sha256(Path(root) / relative) for relative in weight_files(task)}
    require(record.get('checkpoint_hashes') == hashes, 'Checkpoint hashes mismatch')
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(record.get('checkpoint_sha256') == digest, 'Checkpoint digest mismatch')
    revision = (installed_commit if task.startswith('picai_') else
                'Dataset077_ProstateZones/checkpoint_best.pth' if model == 'umu_prostate_zones' else
                'zenodo:7040585/tumor.pt')
    require(record.get('checkpoint_revision') == revision, 'Checkpoint revision mismatch')


def validate_orientation(record, task, root=None):
    if task in HELD_MRI_TASKS:
        from held_mri import validate_marker
        if root is None:
            raise ValueError('Installed root is required for MRI orientation binding')
        return validate_marker(record, task, root)
    if (not isinstance(record, dict) or record.get('task') != task or record.get('result') != 'passed'
            or marker_has_failure(record)):
        raise ValueError('Orientation task/result mismatch or failed checks')


def validate_umu_dataset(data):
    if data.get('labels') != UMU_LABELS or data.get('channel_names') != {'0': 'T2'}:
        raise ValueError('Umeå Dataset077 channel/label contract differs from the pinned export')
    if data.get('file_ending') != '.nrrd':
        raise ValueError('Umeå Dataset077 requires its native .nrrd input contract')


def role_directories(job, *, multisequence=True):
    """Reject incomplete or redirected role mappings before any conversion."""
    job = Path(job)
    data = json.loads((job / 'job.json').read_text())
    if data.get('input_preparation_interrupted'):
        raise ValueError('Input preparation was interrupted')
    roles = CHANNEL_ROLES if multisequence else ('t2w',)
    records = data.get('input_roles', {})
    manifest = data.get('input_manifest')
    if multisequence and not isinstance(manifest, dict):
        raise ValueError('Multi-sequence input requires a complete input_manifest')
    result = {}
    for role in roles:
        dirname = 'input' if role == 't2w' else f'input_{role}'
        if multisequence and (not isinstance(records.get(role), dict) or records[role].get('dir') != dirname):
            raise ValueError(f'Missing or invalid input role {role}: expected {dirname}')
        directory = job / dirname
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f'Missing or redirected input directory: {dirname}')
        files = sorted(directory.iterdir())
        if not files or any(p.is_symlink() or not p.is_file() for p in files):
            raise ValueError(f'{dirname} must contain DICOM files only')
        actual = [{'name': p.name, 'size': p.stat().st_size} for p in files]
        if isinstance(manifest, dict) and manifest.get(dirname) != actual:
            raise ValueError(f'Incomplete input manifest for {role}')
        result[role] = directory
    return result


def validate_dicom_roles(directories, records):
    """Check assessed SOP identities, physical geometry and documented diffusion tags.

    Intensity values and rescale tags are left to the existing DICOM converter.
    Enhanced multi-frame input needs the input adapter's frame extraction first.
    """
    import numpy as np
    import pydicom
    summaries = {}
    for role, directory in directories.items():
        headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in sorted(directory.iterdir())]
        if any(str(getattr(d, 'Modality', '')) != 'MR' or int(getattr(d, 'NumberOfFrames', 1)) != 1 for d in headers):
            raise ValueError(f'{role} requires single-frame MR slices forming one 3-D volume')
        identities = {}
        for tag in ('StudyInstanceUID', 'SeriesInstanceUID', 'FrameOfReferenceUID'):
            values = {str(getattr(d, tag, '')) for d in headers}
            if len(values) != 1 or '' in values:
                raise ValueError(f'{role}: inconsistent or missing {tag}')
            identities[tag] = values.pop()
        if records.get(role, {}).get('series_uid', identities['SeriesInstanceUID']) != identities['SeriesInstanceUID']:
            raise ValueError(f'{role}: selected SeriesInstanceUID does not match the staged series')
        if len({str(d.SOPInstanceUID) for d in headers}) != len(headers):
            raise ValueError(f'{role}: duplicate SOPInstanceUID')
        first = headers[0]
        direction = np.asarray(first.ImageOrientationPatient, float)
        normal = np.cross(direction[:3], direction[3:])
        if (direction.shape != (6,) or not np.isfinite(direction).all()
                or not np.allclose([np.linalg.norm(direction[:3]), np.linalg.norm(direction[3:])], 1, atol=1e-4)
                or not np.isclose(direction[:3] @ direction[3:], 0, atol=1e-4)
                or not np.isclose(np.linalg.norm(normal), 1, atol=1e-4)):
            raise ValueError(f'{role}: invalid slice orientation')
        spacing = np.asarray(first.PixelSpacing, float)
        if (spacing.shape != (2,) or not np.isfinite(spacing).all()
                or np.any(spacing <= 0) or first.Rows < 1 or first.Columns < 1):
            raise ValueError(f'{role}: invalid pixel grid')
        positions = []
        for header in headers:
            if (header.Rows != first.Rows or header.Columns != first.Columns
                    or not np.allclose(header.PixelSpacing, first.PixelSpacing, atol=1e-5, rtol=0)
                    or not np.allclose(header.ImageOrientationPatient, direction, atol=1e-5, rtol=0)):
                raise ValueError(f'{role}: inconsistent slice geometry')
            position = np.asarray(header.ImagePositionPatient, float)
            if not np.isfinite(position).all():
                raise ValueError(f'{role}: nonfinite slice position')
            positions.append(float(position @ normal))
        gaps = np.diff(sorted(positions))
        if len(gaps) and (np.any(gaps < 1e-4) or not np.allclose(gaps, np.median(gaps), atol=1e-3, rtol=1e-3)):
            raise ValueError(f'{role}: duplicate or irregular slice positions; expected one 3-D volume')
        expected_slices = records.get(role, {}).get('expected_slices')
        if expected_slices is not None and len(headers) != int(expected_slices):
            raise ValueError(f'{role}: missing DICOM slices (expected {expected_slices}, found {len(headers)})')
        expected = records.get(role, {}).get('expected_sop_instances')
        if expected is not None:
            actual = {str(d.SOPInstanceUID): [[float(v) for v in d.ImagePositionPatient]] for d in headers}
            if set(actual) != set(expected) or any(
                    not np.allclose(actual[uid], expected[uid], atol=1e-5, rtol=0) for uid in actual):
                raise ValueError(f'{role}: assessed SOP instances or physical positions differ from staged slices')
        if role == 'hbv':
            selected = records.get(role, {})
            value = selected.get('b_value')
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0 or not selected.get('b_value_source')):
                raise ValueError('HBV requires a recorded b_value and documented b_value_source')
            from mri_inputs import diffusion_b_value
            for header in headers:
                observed, source = diffusion_b_value(header)
                if observed is None or source is None:
                    raise ValueError('HBV requires a documented b-value on every selected instance (including private creator)')
                if not math.isclose(observed, value, abs_tol=1e-3):
                    raise ValueError('HBV contains an unselected b-value')
        summaries[role] = identities
    if len({s['StudyInstanceUID'] for s in summaries.values()}) != 1:
        raise ValueError('All prostate input roles must share StudyInstanceUID')
    reference = summaries['t2w']['FrameOfReferenceUID']
    warnings = [f'{role}: FrameOfReferenceUID differs from T2; no registration is applied'
                for role, summary in summaries.items() if summary['FrameOfReferenceUID'] != reference]
    return warnings


def native_input(converted, destination):
    """Undo the shared converter's lossless storage permutation before the model."""
    import nibabel as nib
    from orientation_contract import read_input_record, restore_image
    image = nib.load(converted)
    record = read_input_record(converted)
    if record:
        image, _ = restore_image(image, record['source']['shape'], record['source']['affine'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, destination)


def runtime_provenance(task, root):
    from importlib.metadata import PackageNotFoundError, version
    model, _, _ = TASKS[task]
    packages = {}
    for package in ('torch', 'numpy', 'scipy', 'SimpleITK', 'nnunet', 'nnunetv2', 'monai',
                    'picai-prep', 'picai-baseline', 'picai-eval', 'report-guided-annotation',
                    'mamba-ssm', 'causal-conv1d', 'nibabel', 'scikit-image', 'monai-deploy-app-sdk'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    weights = []
    for relative in weight_files(task):
        path = Path(root) / relative
        weights.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': file_sha256(path)})
    return {'task': task, 'source_commit': SOURCE_COMMITS[model], 'packages': packages, 'weights': weights}


def run_task(task, job, app_root, model_root, worker_python, run, *, acceptance_audit=False):
    """Use model-root runtime-envs assets; write only to the job workspace/results."""
    model, environment, script = TASKS[task]
    if task in HELD_MRI_TASKS and not acceptance_audit:
        from held_mri import binding, gpu_preflight, validate_marker
        records = [json.loads((model_root / folder / f'{task}.json').read_text())
                   for folder in ('validation', 'orientation-validation')]
        current = binding(task, model_root, worker_python=worker_python, uncached=True)
        gpu = gpu_preflight(task, model_root)
        for record in records:
            validate_marker(record, task, model_root, worker_python=worker_python, gpu=gpu, expected=current)
    job = Path(job).absolute()
    work = job / 'external' / task
    output = job / 'output' / f'{task}_RTSTRUCT.dcm'
    output.parent.mkdir(parents=True, exist_ok=True)
    # A retry must not retain a previous positive result after a negative run.
    for artifact in output.parent.glob(output.name + '*'):
        if artifact.is_file():
            artifact.unlink()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    metadata = json.loads((job / 'job.json').read_text())
    dirs = role_directories(job, multisequence=model != 'umu_prostate_zones')
    warnings = validate_dicom_roles(dirs, metadata.get('input_roles', {}))
    def execute(command):
        if task in HELD_MRI_TASKS:
            from held_mri_runtime import python_command
            with python_command(command[0], directory=work) as prefix:
                return run([*prefix, *command[1:]])
        return run(command)

    for role, directory in dirs.items():
        if model == 'nci_monai_lesion':
            continue  # Model runtime uses the SDK's DICOM operators, without generic conversion.
        converted = work / 'converted' / f'{role}.nii.gz'
        execute([worker_python, str(app_root / 'dicom_series_to_nifti.py'), str(directory), str(converted)])
        native_input(converted, work / 'native' / f'{role}.nii.gz')
    context = {'input_roles': metadata.get('input_roles', {}), 'warnings': warnings,
               'channel_order': list(dirs), 'reference_role': 't2w'}
    if model == 'nci_monai_lesion':
        context['dicom_roles'] = {role: str(path) for role, path in dirs.items()}
    (work / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    execute([str(model_root / 'runtime-envs' / environment / 'bin/python'), str(app_root / script),
         '--task', task, '--work', str(work), '--model-root', str(model_root), '--output', str(output),
         *(['--worker-python', str(worker_python)] if task in HELD_MRI_TASKS else []),
         *(['--held-mri-acceptance-audit'] if acceptance_audit else [])])
    result = json.loads((work / 'result.json').read_text())
    if result['empty']:
        # Existing portal/SSH handling recognises this completed-empty marker.
        if acceptance_audit:
            return
        shutil.rmtree(work)
        raise SystemExit('empty prediction: no RTSTRUCT created')
    args = ['--model-input', str(work / 'native/t2w.nii.gz'), '--dicom', str(job / 'input'), '--output', str(output)]
    if model.startswith('picai_') or task in HELD_MRI_TASKS:
        execute([worker_python, str(app_root / 'convert_masks_validate_rtstruct.py'),
             '--masks-json', str(work / 'masks.json'), *args])
    else:
        run([worker_python, str(app_root / 'convert_validate_rtstruct.py'),
             '--prediction', str(work / 'restored.nii.gz'), '--labels', str(app_root / 'labels' / f'{model}.json'), *args])
    if not acceptance_audit:
        shutil.rmtree(work)

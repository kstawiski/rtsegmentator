"""CPU command/fallback tests; no worker or network access."""
import io
import subprocess
from types import SimpleNamespace

import pytest


def test_command_construction(portal):
    command = portal.worker_rsync_command('/tmp/synthetic series.tar', '/jobs/test/series.tar')
    assert command == ['rsync', '-a', '--info=progress2', '--no-inc-recursive',
                       '--compress', '--compress-choice=zstd', '--compress-level=3',
                       '/tmp/synthetic series.tar', f'{portal.WORKER}:/jobs/test/series.tar']
    plain = portal.worker_rsync_command('source', 'dest', compressed=False)
    assert plain == ['rsync', '-a', '--info=progress2', '--no-inc-recursive', 'source', f'{portal.WORKER}:dest']


def setup_transfer(portal, tmp_path, monkeypatch, outcomes):
    source = tmp_path / 'synthetic.tar'; source.write_bytes(b'x' * 10000)
    calls, progress = [], []
    def popen(command, **kwargs):
        calls.append(command)
        output, returncode = outcomes.pop(0)
        return SimpleNamespace(stdout=io.StringIO(output), wait=lambda: returncode)
    monkeypatch.setattr(portal.subprocess, 'Popen', popen)
    monkeypatch.setattr(portal, 'save_metadata', lambda job, **changes: progress.append(changes))
    return source, calls, progress


def test_compressed_progress_counts_uncompressed_bytes(portal, tmp_path, monkeypatch):
    source, calls, progress = setup_transfer(portal, tmp_path, monkeypatch,
        [('  5,000  50%  1.0MB/s\r 10,000 100% 1.0MB/s\n', 0)])
    portal.transfer_to_worker(tmp_path, source, '/jobs/test.tar')
    assert len(calls) == 1 and '--compress-choice=zstd' in calls[0]
    assert any(p.get('transferred_bytes') == 5000 and p.get('progress') == 50 for p in progress)
    assert progress[-1]['transferred_bytes'] == 10000 and progress[-1]['progress'] == 100


@pytest.mark.parametrize('error', ['rsync: --compress-choice=zstd: unknown option\n',
    'rsync: unknown compress name: zstd\n', 'remote rsync does not support --compress-level\n',
    'Failed to negotiate a compression choice\n',
    'Your --compress-choice value (zstd) was rejected by the remote server\n',
    'rsync was not compiled with zstd support\n'])
def test_fallback_and_cached_plain_command(portal, tmp_path, monkeypatch, error):
    source, calls, progress = setup_transfer(portal, tmp_path, monkeypatch,
                                           [(error, 4), ('10,000 100%\n', 0), ('10,000 100%\n', 0)])
    portal.transfer_to_worker(tmp_path, source, '/jobs/test.tar')
    portal.transfer_to_worker(tmp_path, source, '/jobs/test.tar')
    assert len(calls) == 3
    assert '--compress' in calls[0]
    assert all('--compress' not in c for c in calls[1:])
    assert portal._rsync_zstd_supported[portal.WORKER] is False
    assert progress[-1]['transferred_bytes'] == 10000


def test_network_error_does_not_retry_or_cache_unsupported(portal, tmp_path, monkeypatch):
    source, calls, _ = setup_transfer(portal, tmp_path, monkeypatch, [('ssh: connection lost\n', 255)])
    with pytest.raises(subprocess.CalledProcessError) as exc:
        portal.transfer_to_worker(tmp_path, source, '/jobs/test.tar')
    assert len(calls) == 1
    assert exc.value.output == 'ssh: connection lost\n'
    assert portal.WORKER not in portal._rsync_zstd_supported


def test_plain_fallback_failure_is_propagated(portal, tmp_path, monkeypatch):
    source, calls, _ = setup_transfer(portal, tmp_path, monkeypatch,
        [('rsync: --compress-choice=zstd: unknown option\n', 4), ('Permission denied\n', 23)])
    with pytest.raises(subprocess.CalledProcessError) as exc:
        portal.transfer_to_worker(tmp_path, source, '/jobs/test.tar')
    assert len(calls) == 2 and '--compress' not in exc.value.cmd
    assert exc.value.returncode == 23
    assert portal._rsync_zstd_supported[portal.WORKER] is False

"""Bounded log parsing and activity accounting; independent of the web server."""
from __future__ import annotations

import json
import re
from pathlib import Path

STAGES = {
    'staging': 'Preparing model input', 'conversion': 'Converting DICOM to NIfTI',
    'inference': 'Running model inference', 'postprocessing': 'Post-processing labels',
    'rtstruct_contours': 'Creating RTSTRUCT contours', 'rtstruct_writing': 'Writing RTSTRUCT',
    'rtstruct_validation': 'Validating RTSTRUCT geometry', 'export': 'Exporting results',
    'resampling': 'Resampling images', 'saving': 'Saving results',
}
LOG_BYTE_CAP = 256 * 1024
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def log_lines(text):
    """Keep the latest rendition of each carriage-return progress line."""
    return [line for raw in text.split('\n')
            if (line := next((part.strip() for part in reversed(ANSI.sub('', raw).split('\r')) if part.strip()), ''))]


def parse_progress(text):
    # Inspect carriage-return updates too: a later tqdm message must not hide
    # a structured stage emitted before it in the same read.
    for raw in reversed(ANSI.sub('', text).replace('\r', '\n').splitlines()):
        if 'RTSEG_PROGRESS ' in raw:
            try:
                value = json.loads(raw.split('RTSEG_PROGRESS ', 1)[1])
                stage = value.get('stage')
                if stage not in STAGES: continue
                result = {'stage_key': stage, 'stage': STAGES[stage]}
                done, total = value.get('done'), value.get('total')
                if (type(done) is int and type(total) is int and 0 <= done <= total and total > 0):
                    result.update(stage_done=done, stage_total=total)
                return result
            except (ValueError, TypeError, AttributeError):
                continue
        lowered = raw.lower()
        for marker, stage in [('predicting', 'inference'), ('resampling', 'resampling'), ('saving', 'saving')]:
            if marker in lowered:
                return {'stage_key': stage, 'stage': STAGES[stage]}
    return {}


def tail_log(path, lines=40):
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        start = max(0, size - LOG_BYTE_CAP)
        stream.seek(start)
        data = stream.read(LOG_BYTE_CAP)
    if start: data = data.partition(b'\n')[2]
    return log_lines(data.decode('utf-8', errors='replace'))[-min(200, max(1, lines)):]


class TaskActivity:
    def __init__(self, now):
        self.position = 0
        self.pending = ''
        self.last_activity = now
        self.last_cpu_activity = now
        self.cpu = {}
        self.stage = {'stage_key': 'staging', 'stage': STAGES['staging']}
        self.stage_started = now

    def sample(self, log: Path, now, cpu=None):
        size = log.stat().st_size
        if size != self.position:
            self.last_activity = now
            with log.open('rb') as stream:
                stream.seek(max(0, size - LOG_BYTE_CAP, self.position if size >= self.position else 0))
                content = stream.read(LOG_BYTE_CAP).decode('utf-8', errors='replace')
            self.position = size
            content = self.pending + content
            progress = parse_progress(content)
            self.pending = content.rsplit('\n', 1)[-1][-4096:] if not content.endswith('\n') else ''
            if progress:
                if progress['stage_key'] != self.stage['stage_key']:
                    self.stage_started = now
                self.stage = progress
        if cpu is not None:
            if any(ticks > self.cpu.get(pid, 0) for pid, ticks in cpu.items()):
                self.last_cpu_activity = now
            self.cpu = cpu
        return {**self.stage, 'stage_done': self.stage.get('stage_done'),
                'stage_total': self.stage.get('stage_total'), 'stage_started': self.stage_started,
                'last_activity': self.last_activity, 'last_cpu_activity': self.last_cpu_activity}

    def stalled(self, now, threshold, *, remote=False):
        last = self.last_activity if remote else max(self.last_activity, self.last_cpu_activity)
        return now - last >= threshold


def stage_phrase(label):
    """Lowercase a sentence's first letter while preserving RTSTRUCT/NIfTI."""
    return label[:1].lower() + label[1:]


def task_detail(task, index, total, fields):
    counts = (f" · {fields['stage_done']}/{fields['stage_total']} structures"
              if fields.get('stage_total') and fields['stage_key'] in
              {'rtstruct_contours', 'rtstruct_validation'} else '')
    return f"{task} ({index}/{total}): {stage_phrase(fields['stage'])}{counts}"

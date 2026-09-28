"""Shared settings/I-O for Claude's C04 experiment. Importing this module starts no job.

Reads Codex B01 artifacts strictly read-only; every C04 output stays under matcher/."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
CLAUDE = ROOT / 'matcher'
CODEX_SCRIPTS = ROOT / 'retrieval' / 'scripts'
if str(CODEX_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CODEX_SCRIPTS))   # side-effect-free imports only: b01_pair_features, b01_match_metrics
CONFIG = CLAUDE / 'configs' / 'c04.json'


def environment():
    """Refuse to run unless matcher/env.sh was sourced (keeps temp files and caches inside the repo)."""
    tmp = os.environ.get('TMPDIR', '')
    if not tmp.startswith(str(CLAUDE)):
        raise RuntimeError('Run through bash matcher/run_c04.sh (it sources matcher/env.sh). TMPDIR must be inside matcher/.')


def settings(smoke=False):
    cfg = json.loads(CONFIG.read_text())
    if smoke:
        cfg.update(cfg['smoke'])
    cfg['smoke_mode'] = bool(smoke)
    cfg.setdefault('limit_parts', 0)
    out = (ROOT / cfg['run_dir']).resolve()
    if not out.is_relative_to(CLAUDE.resolve()):
        raise ValueError('run_dir must stay inside matcher/')
    for key in ('codex_candidates', 'codex_features', 'codex_prepared'):
        cfg[key] = ROOT / cfg[key]
    return cfg, out


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    partial.replace(path)


def table_write(path, table):
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    pq.write_table(table, partial, compression='zstd')
    partial.replace(path)


def identity(path):
    s = Path(path).stat()
    return {'path': str(Path(path).relative_to(ROOT)), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def code_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def stable_hash(text, seed):
    """Identical to Codex's b01_match_common.stable_hash (same deterministic query sampling)."""
    return int.from_bytes(hashlib.blake2b(f'{seed}:{text}'.encode(), digest_size=8).digest(), 'little')


def stage_lock(folder):
    """Keep the returned handle alive for the whole stage to prevent duplicate writers."""
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / 'run.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'Another process is already writing this stage: {folder}') from None
    return handle


def require_manifest(folder, signature):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'manifest.json'
    text = json.loads(json.dumps(signature, sort_keys=True, default=str))
    if path.exists():
        if read_json(path) != text:
            raise RuntimeError(f'Inputs/settings/code changed since this stage started. Keep the old checkpoints and use a fresh run_dir: {folder}')
    else:
        atomic_json(path, text)


def codex_parts(folder, limit=0):
    """Numbered Codex candidate parts; refuses an incomplete Codex stage."""
    if not (folder / 'complete.json').is_file():
        raise RuntimeError(f'Codex stage incomplete: {folder}')
    n = read_json(folder / 'complete.json')['chunks']
    if len(list(folder.glob('part-*.complete.json'))) != n:
        raise RuntimeError(f'Missing Codex chunk checkpoints: {folder}')
    return list(range(n if not limit else min(limit, n)))


def append_progress(cfg, event, message):
    """Append one CLAUDE entry to the repo-root progress.txt (never in smoke mode)."""
    if cfg.get('smoke_mode'):
        return
    path = ROOT / 'progress.txt'
    marker = f'[CLAUDE {cfg["experiment"]} {event}]'
    with path.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        if marker not in stream.read():
            stream.write(f'\n{marker}\nUTC: {datetime.now(timezone.utc).isoformat()}\n{message.rstrip()}\n')
            stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


def smoke_flag(argv):
    return '--smoke' in argv

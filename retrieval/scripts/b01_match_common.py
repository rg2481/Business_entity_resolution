"""Shared I/O for the user-run B01 matcher; imports do not start jobs."""
from pathlib import Path
import hashlib
import json
from b01_common import ROOT, CODEX, atomic_json, read_config

DEFAULT_CONFIG = 'retrieval/configs/b01_matcher.json'
COUNTRIES = ('India', 'US')


def settings(path=None, base_config=None):
    path = Path(path or ROOT / DEFAULT_CONFIG)
    cfg = json.loads(path.read_text())
    if base_config is not None:
        cfg['base_config'] = str(base_config)
    base, run = read_config(ROOT / cfg['base_config'])
    out = (ROOT / cfg['experiment_dir']).resolve()
    if not out.is_relative_to(CODEX.resolve()):
        raise ValueError('experiment_dir must stay inside retrieval/')
    for key in ('query_batch', 'scan_batch_rows', 'top_k', 'cpu_threads', 'training_query_modulus',
                'trees', 'checkpoint_trees', 'num_leaves', 'min_data_in_leaf'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    n = cfg['validation_entities_per_country']
    if n < 4 or n % 2:
        raise ValueError('validation_entities_per_country must be even and at least 4')
    if not 0 < cfg['minimum_fold0_recall'] <= 1 or cfg['distractor_fp_weight'] < 1:
        raise ValueError('Invalid quality gate or distractor FP weight')
    return cfg, base, run, out


def stage_lock(folder):
    """Keep the returned handle alive for the whole stage to prevent duplicate writers."""
    import fcntl
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / 'run.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'Another process is already writing this stage: {folder}') from None
    return handle


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_manifest(folder, signature):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise RuntimeError(f'Inputs/settings/code changed. Keep these checkpoints and use a fresh experiment_dir: {folder}')
    else:
        atomic_json(path, signature)


def read_json(path):
    return json.loads(Path(path).read_text())


def table_write(path, table):
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    pq.write_table(table, partial, compression='zstd')
    partial.replace(path)


def stable_hash(text, seed):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{text}'.encode(), digest_size=8).digest(), 'little')


def chunk_files(folder):
    paths = sorted(folder.glob('part-*.complete.json'))
    if not (folder / 'complete.json').is_file():
        raise RuntimeError(f'Stage incomplete: {folder}')
    if len(paths) != read_json(folder / 'complete.json')['chunks']:
        raise RuntimeError(f'Missing chunk checkpoints: {folder}')
    return paths


def append_progress(event, message):
    """Called only by actual training/evaluation, never by setup or tiny checks."""
    import fcntl
    from datetime import datetime, timezone
    path = ROOT / 'progress.txt'
    with path.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        marker = f'[CODEX B01 {event}]'
        if marker not in stream.read():
            stream.write(f'\n{marker}\nUTC: {datetime.now(timezone.utc).isoformat()}\n{message.rstrip()}\n')
            stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)

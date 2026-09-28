"""C04 test stage 2 (CPU, user-run, parallel): score EVERY Codex test candidate with the C04 model.

Per candidate part: Codex's 61 B01 features (retrieval/scripts/b01_pair_features.matrix, read-only) + the 31 C04 features
-> C04 LightGBM probability -> one global winner per target (Codex's winners(), all competitors) -> accepted when the
winner's score >= the C04 threshold chosen on TUNE references (matcher/runs/c04_v1/evaluation/report.json).
Features are not stored (242M pairs); one winners file + checkpoint per part. Parallel 'spawn' workers; resumable."""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from c04_common import (ROOT, settings, environment, stage_lock, require_manifest, atomic_json, table_write, read_json,
                        identity, codex_parts, code_hash, CLAUDE, CODEX_SCRIPTS)
from tqdm.auto import tqdm

G = {}


def _reference_text(tc, country, src):
    import polars as pl
    refs = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id'])
    folder = ROOT / tc['codex_test_prepared'] / 'test_source1'
    s1 = (pl.scan_parquet(sorted(str(p) for p in folder.glob('part-*.parquet'))).filter(pl.col('country') == country)
          .select('entity_id', 'name_core', 'name_full', 'address_clean', 'raw_address').collect())
    T = refs.join(s1, on='entity_id', how='left', maintain_order='left')
    if T['name_core'].null_count():
        raise RuntimeError('Test reference text alignment failed.')
    return {k: T[k].to_list() for k in ('name_core', 'name_full', 'address_clean', 'raw_address')}


def _init(job):
    """Runs once in each worker: load the model, S1 stats/text and vocabulary (kept for the worker's lifetime)."""
    import lightgbm as lgb
    import polars as pl
    from pathlib import Path
    G.update(job)
    G['model'] = lgb.Booster(model_file=job['model'])
    prep = Path(job['prep'])
    G['rstats'] = pl.read_parquet(prep / 'refs.parquet')
    G['vocab'] = pl.read_parquet(prep / 'vocab.parquet')
    G['refs_text'] = _reference_text(job['tc'], job['country'], Path(job['src']))


def _score_part(p):
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    from b01_pair_features import FEATURES, matrix
    from b01_match_metrics import winners
    from c04_features import FEATURES_V2, QUERY_COLS, REF_COLS, pair_features
    src, folder = Path(G['src']), Path(G['folder'])
    qpath, ppath = src / f'part-{p:05d}.queries.parquet', src / f'part-{p:05d}.pairs.parquet'
    queries = pq.read_table(qpath).to_pydict()
    pt = pq.read_table(ppath)
    pairs = {k: pt[k].to_numpy() for k in pt.column_names}
    count = len(queries['entity_id'])
    lo = int(queries['query_index'][0])
    if not np.array_equal(queries['query_index'], np.arange(lo, lo + count)):
        raise RuntimeError(f'Part {p}: noncontiguous query ordinals.')
    qi, ri = pairs['query_index'], pairs['ref_index']
    if len(qi):
        Xb = matrix(pairs, queries, G['refs_text'], G['name_weight'])
        qs = pl.from_arrow(pq.read_table(Path(G['prep']) / 'queries.parquet',
                                         filters=[('query_index', '>=', lo), ('query_index', '<', lo + count)]))
        if qs.height != count or qs['query_index'][0] != lo:
            raise RuntimeError(f'Part {p}: prepared query stats misaligned.')
        Xv = pair_features(qs[(qi - lo).astype(np.int64)].select(QUERY_COLS), G['rstats'][ri.astype(np.int64)].select(REF_COLS), G['vocab'], workers=1)
        X = np.hstack([Xb, Xv]).astype(np.float32)
        s = np.asarray(G['model'].predict(X, num_threads=G['threads']), np.float64)
        if not np.isfinite(s).all():
            raise RuntimeError(f'Part {p}: invalid probabilities.')
        w = winners(qi, ri, np.full(len(s), -1, np.int32), s)
    else:
        w = {'query_index': np.array([], np.int64)}
    res = {'query_index': np.arange(lo, lo + count, dtype=np.int64), 'ref_index': np.full(count, -1, np.int32),
           'score': np.zeros(count), 'runnerup_score': np.full(count, -1.0)}
    local = w['query_index'] - lo
    for k in ('ref_index', 'score', 'runnerup_score'):
        if len(local):
            res[k][local] = w[k]
    res['accepted'] = (res['ref_index'] >= 0) & (res['score'] >= G['threshold'])
    out = folder / f'part-{p:05d}.winners.parquet'
    table_write(out, pa.table(res))
    stats = {'queries': count, 'pairs': int(len(qi)), 'accepted': int(res['accepted'].sum()),
             'no_candidates': int(count - len(local))}
    atomic_json(folder / f'part-{p:05d}.complete.json', {'inputs': {'queries': identity(qpath), 'pairs': identity(ppath)},
                                                         'counts': stats})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--country', choices=['France', 'India', 'US'], required=True)
    parser.add_argument('--workers', type=int, default=0, help='parallel worker processes (default from config)')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    tc = cfg['test']
    workers = args.workers or tc['workers']
    src = ROOT / tc['codex_test_candidates'] / f'test_{args.country}'
    parts = codex_parts(src, cfg.get('test_limit_parts', 0) if cfg['smoke_mode'] else 0)
    prep = out / 'test' / 'prepare' / args.country
    if not (prep / 'complete.json').is_file():
        raise RuntimeError(f'Run first: bash matcher/run_c04.sh test-prepare --country {args.country}')
    model_done = read_json(out / 'model' / 'complete.json')
    evaluation = read_json(out / 'evaluation' / 'report.json')
    threshold = float(evaluation['threshold'])
    folder = out / 'test' / 'scores' / args.country
    _lock = stage_lock(folder)
    signature = {'stage': 'test-score', 'country': args.country, 'parts': len(parts), 'threshold': threshold,
                 'model_sha256': model_done['model_sha256'], 'prepare': read_json(prep / 'complete.json'),
                 'candidates': read_json(src / 'complete.json'),
                 'code': code_hash(CLAUDE / 'scripts' / 'c04t_score.py', CLAUDE / 'scripts' / 'c04_features.py',
                                   CODEX_SCRIPTS / 'b01_pair_features.py', CODEX_SCRIPTS / 'b01_match_metrics.py')}
    require_manifest(folder, signature)
    if (folder / 'complete.json').is_file():
        print('Reusing completed test scores:\n' + (folder / 'complete.json').read_text())
        return
    done_parts = {p for p in parts if (folder / f'part-{p:05d}.complete.json').is_file()}
    todo = [p for p in parts if p not in done_parts]
    total = dict(queries=0, pairs=0, accepted=0, no_candidates=0)
    for p in done_parts:
        for k, v in read_json(folder / f'part-{p:05d}.complete.json')['counts'].items():
            total[k] += v
    print(f'C04 TEST-SCORE {args.country}: {len(parts)} parts ({len(done_parts)} already done), {workers} workers x '
          f'{tc["threads_per_worker"]} threads, threshold {threshold:.3f}. Each worker loads the model and S1 text first '
          f'(~20-60 s before the bar moves). CPU only; no submission.', flush=True)
    os.environ['POLARS_MAX_THREADS'] = str(tc['threads_per_worker'])
    os.environ['OMP_NUM_THREADS'] = str(tc['threads_per_worker'])
    job = {'model': str(out / 'model' / 'model.txt'), 'prep': str(prep), 'src': str(src), 'folder': str(folder),
           'country': args.country, 'tc': tc, 'name_weight': cfg['name_weight'], 'threshold': threshold,
           'threads': tc['threads_per_worker']}
    t0 = time.time()
    with tqdm(total=len(parts), initial=len(done_parts), desc=f'Score test/{args.country}', unit='parts', dynamic_ncols=True) as bar:
        if todo:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=_init, initargs=(job,)) as pool:
                futures = {pool.submit(_score_part, p): p for p in todo}
                for f in as_completed(futures):
                    for k, v in f.result().items():
                        total[k] += v
                    bar.update(1)
                    bar.set_postfix(pairs=f"{total['pairs']:,}", accepted=f"{total['accepted']:,}")
    report = dict(total, country=args.country, parts=len(parts), threshold=threshold, workers=workers,
                  accepted_share=total['accepted'] / max(total['queries'], 1), seconds_this_run=round(time.time() - t0, 1),
                  model_sha256=model_done['model_sha256'], smoke=cfg['smoke_mode'])
    atomic_json(folder / 'complete.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

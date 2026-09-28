"""C04 stage 1 (CPU, user-run): per-country record statistics for the V2 features.

Reads Codex B01 candidates and prepared S1 text read-only. Writes matcher/runs/c04_v1/prepare/<country>/:
  refs.parquet     one row per S1 reference in Codex's retrieval-index order (row == ref_index)
  queries.parquet  one row per S2/S3 target query (row == query_index)
  vocab.parquet    S1 core-name token -> number of S1 core names containing it
  priors.parquet   token -> smoothed P(matched | token), fitted ONLY on training folds (owner_fold != validation fold)
No labels enter any feature except the fold-safe word priors. Twins and claims are label-free pool counts."""
import argparse
import json
import time
from c04_common import (settings, environment, stage_lock, require_manifest, atomic_json, table_write, read_json,
                        identity, code_hash, codex_parts, smoke_flag, CLAUDE)
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from c04_features import text_stats, QUERY_COLS, REF_COLS

t0 = time.time()


def step(msg):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--country', choices=['India', 'US'], required=True)
    parser.add_argument('--smoke', action='store_true', help='first few candidate parts only; writes to the smoke run_dir')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    src = cfg['codex_candidates'] / f'train_{args.country}'
    parts = codex_parts(src, cfg['limit_parts'])
    folder = out / 'prepare' / args.country
    _lock = stage_lock(folder)
    s1_folder = cfg['codex_prepared'] / 'train_source1'
    signature = {'stage': 'prepare', 'country': args.country, 'parts': len(parts),
                 'settings': {k: cfg[k] for k in ('seed', 'validation_fold', 'prior_smoothing', 'prior_min_count', 'limit_parts')},
                 'inputs': {'references': identity(src / 'references.parquet'), 'candidates': read_json(src / 'complete.json'),
                            's1_text': read_json(s1_folder / 'complete.json')},
                 'code': code_hash(*(CLAUDE / 'scripts' / f for f in ('c04_prepare.py', 'c04_features.py', 'c04_common.py')))}
    require_manifest(folder, signature)
    if (folder / 'complete.json').is_file():
        print('Reusing completed preparation:\n' + (folder / 'complete.json').read_text())
        return
    print(f'C04 PREPARE {args.country}: {len(parts)} candidate parts. CPU only; no GPU, training or submission.', flush=True)

    # 1) S1 references, aligned to Codex's ref_index order
    refs = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id'])
    if not np.array_equal(refs['ref_index'].to_numpy(), np.arange(refs.height)):
        raise RuntimeError('Codex references are not in ref_index order.')
    s1 = (pl.scan_parquet(sorted(str(p) for p in s1_folder.glob('part-*.parquet')))
          .filter(pl.col('country') == args.country).select('entity_id', 'name_core', 'raw_address').collect())
    R = refs.join(s1, on='entity_id', how='left', maintain_order='left')
    if R['name_core'].null_count():
        raise RuntimeError('Reference text alignment failed.')
    step(f'S1 references: {R.height:,}')
    R = text_stats(R, 'name_core', 'raw_address', 'r_').with_columns(pl.len().over('name_core').alias('r_name_mult'))

    # 2) S1 claims: how many target queries rank this reference first by combo similarity (label-free)
    claims = np.zeros(R.height, np.int64)
    for p in tqdm(parts, desc=f'{args.country} S1 claims (combo rank 1)', unit='parts', dynamic_ncols=True):
        t = pq.read_table(src / f'part-{p:05d}.pairs.parquet', columns=['ref_index', 'combo_rank'])
        claims += np.bincount(t['ref_index'].to_numpy()[t['combo_rank'].to_numpy() == 1], minlength=R.height)
    R = R.with_columns(pl.Series('r_claims', claims))
    vocab = R.select(pl.col('r_tok').alias('token')).explode('token').drop_nulls('token').group_by('token').len('freq')
    step(f'S1 vocabulary: {vocab.height:,} tokens')

    # 3) Target queries (row == query_index)
    cols = ['query_index', 'name_core', 'address_clean', 'raw_address', 'owner_index', 'owner_fold']
    frames = [pl.read_parquet(src / f'part-{p:05d}.queries.parquet', columns=cols)
              for p in tqdm(parts, desc=f'{args.country} read target queries', unit='parts', dynamic_ncols=True)]
    Q = pl.concat(frames)
    del frames
    if not np.array_equal(Q['query_index'].to_numpy(), np.arange(Q.height)):
        raise RuntimeError('Codex queries are not contiguous in query_index order.')
    step(f'target queries: {Q.height:,}')
    Q = text_stats(Q, 'name_core', 'raw_address', 'q_')
    Q = Q.with_columns((pl.col('raw_address').fill_null('').str.strip_chars() == '').alias('q_addr_empty'),
                       pl.col('q_numset').list.sort().list.join(',').alias('_nums'))
    Q = Q.with_columns((pl.len().over('name_core', 'address_clean') - 1).alias('tw_full'),
                       (pl.len().over('name_core', '_nums') - 1).alias('tw_num'),
                       (pl.len().over('name_core') - 1).alias('tw_name'),
                       pl.when(pl.col('q_addr_empty')).then(0).otherwise(pl.len().over('address_clean') - 1).alias('tw_addr'))
    step('twin counts done')

    # 4) Word priors on training folds only (validation-fold queries never contribute)
    eligible = Q.filter(pl.col('owner_fold') != cfg['validation_fold']).select(pl.col('q_tok').alias('token'), (pl.col('owner_index') >= 0).alias('m'))
    p0 = float(eligible['m'].mean())
    a, mc = cfg['prior_smoothing'], cfg['prior_min_count']
    priors = (eligible.explode('token').drop_nulls('token').group_by('token').agg(pl.len().alias('n'), pl.col('m').sum().alias('matched'))
              .with_columns(pl.when(pl.col('n') < mc).then(pl.lit(p0))
                            .otherwise((pl.col('matched') + a * p0) / (pl.col('n') + a)).alias('prior')))
    del eligible
    toks = Q.select('query_index', pl.col('q_tok').alias('token')).explode('token').drop_nulls('token')
    wp = (toks.join(priors.select('token', 'prior'), on='token', how='left').with_columns(pl.col('prior').fill_null(p0))
          .group_by('query_index').agg(pl.col('prior').min().alias('wp_min'), pl.col('prior').mean().alias('wp_mean'),
                                       (pl.col('prior') < 0.5).sum().alias('wp_low')))
    vm = (toks.join(vocab, on='token', how='left').with_columns(pl.col('freq').fill_null(0))
          .group_by('query_index').agg(pl.col('freq').min().log1p().alias('q_vocab_min')))
    del toks
    Q = (Q.join(wp, on='query_index', how='left').join(vm, on='query_index', how='left')
         .with_columns(pl.col('wp_min').fill_null(p0), pl.col('wp_mean').fill_null(p0), pl.col('wp_low').fill_null(0),
                       pl.col('q_vocab_min').fill_null(-1.0)).sort('query_index'))
    step(f'word priors: {priors.height:,} tokens (base rate {p0:.4f}); query aggregates done')

    table_write(folder / 'refs.parquet', R.select('ref_index', *REF_COLS).to_arrow())
    table_write(folder / 'queries.parquet', Q.select('query_index', *QUERY_COLS).to_arrow())
    table_write(folder / 'vocab.parquet', vocab.to_arrow())
    table_write(folder / 'priors.parquet', priors.to_arrow())
    report = {'country': args.country, 'references': R.height, 'queries': Q.height, 'vocab_tokens': vocab.height,
              'prior_tokens': priors.height, 'prior_base_rate': p0, 'parts': len(parts), 'seconds': round(time.time() - t0, 1),
              'twin_rate_num': float((Q['tw_num'] >= 1).mean()), 'smoke': cfg['smoke_mode']}
    atomic_json(folder / 'complete.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

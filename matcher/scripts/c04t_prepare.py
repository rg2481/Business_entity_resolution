"""C04 test stage 1 (CPU, user-run): per-country record statistics for the TEST pool.

Same statistics as c04_prepare.py, computed on Codex's test candidates (read-only), with two differences:
  * no labels exist, so word priors are taken from the TRAINING priors (matcher/runs/c04_v1/prepare/<country>);
    a country without training data (France) uses the pooled India+US priors;
  * query stats are written with query_index-ordered row groups, so parallel scoring workers read only their rows.
Outputs: matcher/runs/c04_v1/test/prepare/<country>/{refs,queries,vocab,priors}.parquet"""
import argparse
import json
import time
from c04_common import (ROOT, settings, environment, stage_lock, require_manifest, atomic_json, table_write, read_json,
                        identity, codex_parts, code_hash, CLAUDE)
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from c04_features import text_stats, QUERY_COLS, REF_COLS

t0 = time.time()


def step(msg):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


def training_priors(cfg, out, country):
    """Train-fold priors for this country, or pooled priors for a country unseen in training."""
    tc = cfg['test']
    names = [country] if country in cfg['countries'] else tc['prior_countries_for_unseen']
    frames, rates = [], []
    for c in names:
        folder = out / 'prepare' / c
        done = read_json(folder / 'complete.json')
        frames.append(pl.read_parquet(folder / 'priors.parquet').select('token', 'n', 'matched'))
        rates.append(done['prior_base_rate'])
    p0 = float(np.mean(rates))
    a, mc = cfg['prior_smoothing'], cfg['prior_min_count']
    pri = (pl.concat(frames).group_by('token').agg(pl.col('n').sum(), pl.col('matched').sum())
           .with_columns(pl.when(pl.col('n') < mc).then(pl.lit(p0)).otherwise((pl.col('matched') + a * p0) / (pl.col('n') + a)).alias('prior')))
    return pri, p0, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--country', choices=['France', 'India', 'US'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    tc = cfg['test']
    src = ROOT / tc['codex_test_candidates'] / f'test_{args.country}'
    parts = codex_parts(src, cfg.get('test_limit_parts', 0) if cfg['smoke_mode'] else 0)
    folder = out / 'test' / 'prepare' / args.country
    _lock = stage_lock(folder)
    s1_folder = ROOT / tc['codex_test_prepared'] / 'test_source1'
    signature = {'stage': 'test-prepare', 'country': args.country, 'parts': len(parts),
                 'settings': {k: cfg[k] for k in ('prior_smoothing', 'prior_min_count')},
                 'inputs': {'references': identity(src / 'references.parquet'), 'candidates': read_json(src / 'complete.json'),
                            's1_text': read_json(s1_folder / 'complete.json'),
                            'train_priors': {c: read_json(out / 'prepare' / c / 'complete.json') for c in cfg['countries']}},
                 'code': code_hash(*(CLAUDE / 'scripts' / f for f in ('c04t_prepare.py', 'c04_features.py', 'c04_common.py')))}
    require_manifest(folder, signature)
    if (folder / 'complete.json').is_file():
        print('Reusing completed test preparation:\n' + (folder / 'complete.json').read_text())
        return
    print(f'C04 TEST-PREPARE {args.country}: {len(parts)} candidate parts. CPU only; no labels; no submission.', flush=True)
    refs = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id'])
    if not np.array_equal(refs['ref_index'].to_numpy(), np.arange(refs.height)):
        raise RuntimeError('Codex test references are not in ref_index order.')
    s1 = (pl.scan_parquet(sorted(str(p) for p in s1_folder.glob('part-*.parquet')))
          .filter(pl.col('country') == args.country).select('entity_id', 'name_core', 'raw_address').collect())
    R = refs.join(s1, on='entity_id', how='left', maintain_order='left')
    if R['name_core'].null_count():
        raise RuntimeError('Test reference text alignment failed.')
    R = text_stats(R, 'name_core', 'raw_address', 'r_').with_columns(pl.len().over('name_core').alias('r_name_mult'))
    claims = np.zeros(R.height, np.int64)
    for p in tqdm(parts, desc=f'{args.country} test S1 claims', unit='parts', dynamic_ncols=True):
        t = pq.read_table(src / f'part-{p:05d}.pairs.parquet', columns=['ref_index', 'combo_rank'])
        claims += np.bincount(t['ref_index'].to_numpy()[t['combo_rank'].to_numpy() == 1], minlength=R.height)
    R = R.with_columns(pl.Series('r_claims', claims))
    vocab = R.select(pl.col('r_tok').alias('token')).explode('token').drop_nulls('token').group_by('token').len('freq')
    step(f'test S1 references {R.height:,}; vocabulary {vocab.height:,} tokens')
    frames = [pl.read_parquet(src / f'part-{p:05d}.queries.parquet', columns=['query_index', 'name_core', 'address_clean', 'raw_address'])
              for p in tqdm(parts, desc=f'{args.country} read test queries', unit='parts', dynamic_ncols=True)]
    Q = pl.concat(frames)
    del frames
    if not np.array_equal(Q['query_index'].to_numpy(), np.arange(Q.height)):
        raise RuntimeError('Codex test queries are not contiguous in query_index order.')
    Q = text_stats(Q, 'name_core', 'raw_address', 'q_')
    Q = Q.with_columns((pl.col('raw_address').fill_null('').str.strip_chars() == '').alias('q_addr_empty'),
                       pl.col('q_numset').list.sort().list.join(',').alias('_nums'))
    Q = Q.with_columns((pl.len().over('name_core', 'address_clean') - 1).alias('tw_full'),
                       (pl.len().over('name_core', '_nums') - 1).alias('tw_num'),
                       (pl.len().over('name_core') - 1).alias('tw_name'),
                       pl.when(pl.col('q_addr_empty')).then(0).otherwise(pl.len().over('address_clean') - 1).alias('tw_addr'))
    priors, p0, used = training_priors(cfg, out, args.country)
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
    step(f'test queries {Q.height:,}; twins and priors (from {"+".join(used)}, base rate {p0:.4f}) done')
    table_write(folder / 'refs.parquet', R.select('ref_index', *REF_COLS).to_arrow())
    qpath = folder / 'queries.parquet'
    part = qpath.with_suffix('.parquet.part')
    pq.write_table(Q.select('query_index', *QUERY_COLS).to_arrow(), part, compression='zstd', row_group_size=tc['query_row_group'])
    part.replace(qpath)
    table_write(folder / 'vocab.parquet', vocab.to_arrow())
    report = {'country': args.country, 'references': R.height, 'queries': Q.height, 'vocab_tokens': vocab.height,
              'priors_from': used, 'prior_base_rate': p0, 'parts': len(parts), 'seconds': round(time.time() - t0, 1),
              'twin_rate_num': float((Q['tw_num'] >= 1).mean()), 'smoke': cfg['smoke_mode']}
    atomic_json(folder / 'complete.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

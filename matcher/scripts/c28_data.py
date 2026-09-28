"""C28 step 1: text pairs for the cross-encoder.
  train : 25% of C04's training queries (hash), all their kept pairs (positives + hard negatives), with raw text.
  val   : C04 held-out winners (India/US) with owner labels and C04 score.
  test  : C04 test winners with score >= MIN_TEST (all countries), with raw text.
Text of a record = "name | address". Output: matcher/runs/c28/data/{train,val,test}.parquet
Run: source matcher/env.sh && python3 matcher/scripts/c28_data.py
"""
import glob
import os
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
C04 = ROOT / 'matcher/runs/c04_v1'
TRAIN_C = ROOT / 'retrieval/runs/b01_v1/matcher_v1/candidates'
TEST_C = ROOT / 'retrieval/runs/b01_v1/test_stable_v1/candidates'
PQ = ROOT / 'matcher/eda/pq'
OUT = ROOT / 'matcher/runs/c28/data'
MIN_TEST = 0.02
BUCKET = int(os.environ.get('C28_BUCKET', '0'))  # 0 = original run; 1..3 = extra training shards only
TXT = lambda n, a: (pl.col(n).fill_null('') + pl.lit(' | ') + pl.col(a).fill_null('')).str.slice(0, 300)


def queries(src):
    return pl.concat([pl.read_parquet(f, columns=['query_index', 'raw_name', 'raw_address']) for f in
                      tqdm(sorted(glob.glob(str(src / 'part-*.queries.parquet'))), desc=f'queries {src.name}', unit='parts')]
                     ).select('query_index', TXT('raw_name', 'raw_address').alias('b'))


def refs(src, split):
    r = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id'])
    s1 = pl.read_parquet(PQ / f'{split}_source1.parquet', columns=['entity_id', 'business_name', 'business_address'])
    return r.join(s1, on='entity_id', how='left').select('ref_index', TXT('business_name', 'business_address').alias('a'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = [], []
    for c in ['India', 'US']:
        src = TRAIN_C / f'train_{c}'
        q, r = queries(src), refs(src, 'train')
        p = pl.concat([pl.read_parquet(f, columns=['query_index', 'ref_index', 'label']) for f in
                       tqdm(sorted(glob.glob(str(C04 / 'features' / c / 'part-*.train.parquet'))), desc=f'train pairs {c}', unit='parts')])
        p = p.filter((pl.col('query_index').hash(7) % 4) == BUCKET)
        tr.append(p.join(q, on='query_index').join(r, on='ref_index').select('a', 'b', pl.col('label').cast(pl.Float32), pl.lit(c).alias('country')))
        w = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(C04 / 'evaluation' / c / 'part-*.winners.parquet')))])
        va.append(w.join(q, on='query_index').join(r, on='ref_index').select(
            'query_index', 'ref_index', 'owner_index', 'score', 'a', 'b', pl.lit(c).alias('country')))
    tr = pl.concat(tr).sample(fraction=1.0, shuffle=True, seed=1)
    tr.write_parquet(OUT / ('train.parquet' if BUCKET == 0 else f'train_b{BUCKET}.parquet'))
    if BUCKET:
        print('train shard', BUCKET, tr.height); return
    pl.concat(va).write_parquet(OUT / 'val.parquet')
    print('train', tr.height, 'positives', int(tr['label'].sum()), '| val', sum(v.height for v in va), flush=True)
    te = []
    for c in ['France', 'India', 'US']:
        src = TEST_C / f'test_{c}'
        q, r = queries(src), refs(src, 'test')
        w = pl.concat([pl.read_parquet(f) for f in tqdm(sorted(glob.glob(str(C04 / 'test/scores' / c / 'part-*.winners.parquet'))), desc=f'test winners {c}', unit='parts')])
        w = w.filter(pl.col('score') >= MIN_TEST)
        qe = pl.concat([pl.read_parquet(f, columns=['query_index', 'entity_id']) for f in sorted(glob.glob(str(src / 'part-*.queries.parquet')))]).rename({'entity_id': 't'})
        re = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id']).rename({'entity_id': 's1'})
        te.append(w.join(q, on='query_index').join(r, on='ref_index').join(qe, on='query_index').join(re, on='ref_index')
                  .select('s1', 't', 'score', 'a', 'b', pl.lit(c).alias('country')))
    te = pl.concat(te)
    te.write_parquet(OUT / 'test.parquet')
    print('test winners scored >=', MIN_TEST, ':', te.height, flush=True)


if __name__ == '__main__':
    main()

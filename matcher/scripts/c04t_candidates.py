"""C04 test stage 4 (CPU, user-run): write candidate_pairs.tsv, the exact candidate union the C04 model scored
(Codex's test candidates), with one row per test S1, next to the chosen matching_results.tsv variant.

Memory-bounded: pass 1 routes (S1 ordinal, target ID) pairs into on-disk parquet buckets; pass 2 groups one bucket at a
time and writes every S1 of the block (empty list when it has no candidates). It also checks that every accepted match
of the chosen matching file is among its S1's candidates (the organizers' subset rule)."""
import argparse
import json
import math
import shutil
import time
from c04_common import ROOT, settings, environment, stage_lock, atomic_json, read_json, codex_parts
import polars as pl
import pyarrow.parquet as pq
from tqdm.auto import tqdm

BUCKETS = 32
FOLDERS = {'c04': 'output', 'france-b01': 'output_france_b01', 'us-offset-reject': 'output_us_offset_reject'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=list(FOLDERS), default='us-offset-reject')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    tc = cfg['test']
    folder = out / 'test' / FOLDERS[args.variant]
    matching = folder / 'matching_results.tsv'
    if not matching.is_file():
        raise RuntimeError(f'Run first: bash matcher/run_c04.sh test-export --variant {args.variant}')
    _lock = stage_lock(folder)
    t0 = time.time()
    tmp = folder / 'candidate_buckets'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    M = pl.read_csv(matching, separator='\t', quote_char=None, missing_utf8_is_empty_string=True,
                    schema={'source1_entity_id': pl.String, 'matched_entity_ids': pl.String})
    M = (M.filter(pl.col('matched_entity_ids') != '').select('source1_entity_id', pl.col('matched_entity_ids').str.split(',').alias('t'))
         .explode('t'))
    dest = folder / 'candidate_pairs.tsv'
    part = dest.with_suffix('.tsv.part')
    rows = pairs_total = violations = 0
    with part.open('w', encoding='utf-8', newline='\n') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for country in tc['countries']:
            src = ROOT / tc['codex_test_candidates'] / f'test_{country}'
            parts = codex_parts(src, cfg.get('test_limit_parts', 0) if cfg['smoke_mode'] else 0)
            refs = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id'])
            n = refs.height
            size = math.ceil(n / BUCKETS)
            writers = {}
            for p in tqdm(parts, desc=f'Route candidates {country}', unit='parts', dynamic_ncols=True):
                pr = pl.read_parquet(src / f'part-{p:05d}.pairs.parquet', columns=['query_index', 'ref_index'])
                q = pl.read_parquet(src / f'part-{p:05d}.queries.parquet', columns=['query_index', 'entity_id'])
                d = pr.join(q, on='query_index', how='left').select('ref_index', pl.col('entity_id').alias('target'))
                if d['target'].null_count():
                    raise RuntimeError(f'{country} part {p}: candidate/query alignment failed.')
                pairs_total += d.height
                for (b,), g in d.with_columns((pl.col('ref_index') // size).alias('b')).group_by('b'):
                    if b not in writers:
                        writers[b] = pq.ParquetWriter(tmp / f'{country}-{b:03d}.parquet', g.drop('b').to_arrow().schema, compression='zstd')
                    writers[b].write_table(g.drop('b').to_arrow())
            for w in writers.values():
                w.close()
            mref = (M.join(refs.rename({'entity_id': 'source1_entity_id'}), on='source1_entity_id', how='inner'))
            for b in tqdm(range(BUCKETS), desc=f'Write candidates {country}', unit='buckets', dynamic_ncols=True):
                lo, hi = b * size, min(n, (b + 1) * size)
                if lo >= hi:
                    continue
                path = tmp / f'{country}-{b:03d}.parquet'
                block = refs.filter((pl.col('ref_index') >= lo) & (pl.col('ref_index') < hi))
                if path.is_file():
                    c = pl.read_parquet(path)
                    grouped = c.group_by('ref_index').agg(pl.col('target').unique().sort().str.join(',').alias('ids'))
                    mb = mref.filter((pl.col('ref_index') >= lo) & (pl.col('ref_index') < hi)).select('ref_index', pl.col('t').alias('target'))
                    violations += mb.join(c, on=['ref_index', 'target'], how='anti').height
                    path.unlink()
                else:
                    grouped = pl.DataFrame({'ref_index': [], 'ids': []}, schema={'ref_index': refs['ref_index'].dtype, 'ids': pl.String})
                    violations += mref.filter((pl.col('ref_index') >= lo) & (pl.col('ref_index') < hi)).height
                lines = (block.join(grouped, on='ref_index', how='left', maintain_order='left').with_columns(pl.col('ids').fill_null(''))
                         .select(pl.col('entity_id') + '\t' + pl.col('ids')))
                f.write('\n'.join(lines.to_series().to_list()) + '\n')
                rows += block.height
    shutil.rmtree(tmp, ignore_errors=True)
    if violations:
        raise RuntimeError(f'{violations} accepted matches are not among their S1 candidates; candidate file left as {part}')
    part.replace(dest)
    expected = pl.read_parquet(ROOT / tc['raw_test_source1'], columns=['entity_id']).height
    report = {'variant': args.variant, 'file': str(dest.relative_to(ROOT)), 'bytes': dest.stat().st_size, 's1_rows': rows,
              'expected_s1_rows': expected, 'candidate_pairs': pairs_total, 'matches_outside_candidates': violations,
              'seconds': round(time.time() - t0, 1), 'smoke': cfg['smoke_mode']}
    atomic_json(folder / 'candidate_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if rows != expected:
        raise SystemExit(f'Row count {rows:,} differs from the {expected:,} test S1 entities.')
    print(f'\nFinal-package files ready in {folder.relative_to(ROOT)}/: matching_results.tsv + candidate_pairs.tsv', flush=True)


if __name__ == '__main__':
    main()

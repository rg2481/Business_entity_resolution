"""C04 test stage 3 (CPU, user-run): write matching_results.tsv from the C04 test winners, validate it with the
OFFICIAL validator (--check-ids), and compare per-country statistics with the submitted B01 file.

  --variant c04         (default) every country from C04          -> test/output/matching_results.tsv
  --variant france-b01  US/India from C04, France rows from B01   -> test/output_france_b01/matching_results.tsv
                        (a France-only probe: its leaderboard difference to the c04 file isolates France)
  --variant us-offset-reject  C04 everywhere, but in the US drop accepts whose core name matches and whose house
                        numbers (same digit count) differ by 1-9 -> test/output_us_offset_reject/ (probe of the US shift)
candidate_pairs.tsv: C04 scores exactly Codex's test candidate union; test-candidates (c04t_candidates.py) writes it
next to the chosen matching_results.tsv and checks that every accepted match is among its S1's candidates."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from c04_common import ROOT, settings, environment, stage_lock, atomic_json, read_json, codex_parts
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm.auto import tqdm

HEADER = 'source1_entity_id\tmatched_entity_ids'


def offset_mask(out, country, acc):
    """True where the record's core-name tokens are all in its S1's and the (role-aware) house numbers have the same
    digit count and differ by 1-9: the US test-shift group (claude_analysys.txt E5)."""
    prep = out / 'test' / 'prepare' / country
    qs = pl.read_parquet(prep / 'queries.parquet', columns=['query_index', 'q_house', 'q_tok'])
    rs = pl.read_parquet(prep / 'refs.parquet', columns=['ref_index', 'r_house', 'r_tok'])
    X = acc.join(qs, on='query_index', how='left').join(rs, on='ref_index', how='left')
    d = (pl.col('q_house').cast(pl.Int64, strict=False) - pl.col('r_house').cast(pl.Int64, strict=False)).abs()
    m = ((pl.col('q_tok').list.set_difference('r_tok').list.len() == 0) & (pl.col('q_house').str.len_chars() == pl.col('r_house').str.len_chars())
         & (d >= 1) & (d <= 9)).fill_null(False)
    return X.select(m.alias('m'))['m'].to_numpy()


def country_pairs(cfg, out, country, reject_offsets=False):
    tc = cfg['test']
    src = ROOT / tc['codex_test_candidates'] / f'test_{country}'
    folder = out / 'test' / 'scores' / country
    if not (folder / 'complete.json').is_file():
        raise RuntimeError(f'Run first: bash matcher/run_c04.sh test-score --country {country}')
    parts = codex_parts(src, cfg.get('test_limit_parts', 0) if cfg['smoke_mode'] else 0)
    ref_ids = pq.read_table(src / 'references.parquet', columns=['entity_id'])['entity_id'].to_numpy(zero_copy_only=False)
    frames = []
    for p in tqdm(parts, desc=f'Collect {country}', unit='parts', dynamic_ncols=True):
        w = pl.read_parquet(folder / f'part-{p:05d}.winners.parquet', columns=['query_index', 'ref_index', 'accepted'])
        q = pl.read_parquet(src / f'part-{p:05d}.queries.parquet', columns=['query_index', 'entity_id'])
        a = w.filter(pl.col('accepted')).join(q, on='query_index', how='left')
        if a['entity_id'].null_count():
            raise RuntimeError(f'{country} part {p}: winner/query alignment failed.')
        frames.append(a.select('query_index', pl.col('ref_index'), pl.col('entity_id').alias('target')))
    P = pl.concat(frames)
    if reject_offsets:
        drop = offset_mask(out, country, P)
        print(f'{country}: rejecting {int(drop.sum()):,} same-name accepts whose house numbers differ by 1-9 (of {P.height:,} accepts).', flush=True)
        P = P.filter(pl.Series(~drop))
    P = P.select('ref_index', 'target')
    return P.with_columns(pl.Series('s1', ref_ids[P['ref_index'].to_numpy()])).select('s1', 'target'), set(ref_ids.tolist())


def read_tsv(path):
    return pl.read_csv(path, separator='\t', quote_char=None, schema={'source1_entity_id': pl.String, 'matched_entity_ids': pl.String},
                       missing_utf8_is_empty_string=True)


def stats(df, s1_country):
    d = df.join(s1_country, on='source1_entity_id', how='left').with_columns(
        pl.when(pl.col('matched_entity_ids') == '').then(0).otherwise(pl.col('matched_entity_ids').str.count_matches(',') + 1).alias('k'))
    return {c: {'s1': int(g.height), 'matched_targets': int(g['k'].sum()), 'empty_s1_share': round(float((g['k'] == 0).mean()), 5),
                'mean_matches': round(float(g['k'].mean()), 4)} for (c,), g in d.group_by('country')}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['c04', 'france-b01', 'us-offset-reject'], default='c04')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    tc = cfg['test']
    folder = out / 'test' / {'c04': 'output', 'france-b01': 'output_france_b01', 'us-offset-reject': 'output_us_offset_reject'}[args.variant]
    _lock = stage_lock(folder)
    t0 = time.time()
    threshold = read_json(out / 'evaluation' / 'report.json')['threshold']
    pairs, s1_sets = [], {}
    for c in tc['countries']:
        P, ids = country_pairs(cfg, out, c, reject_offsets=(args.variant == 'us-offset-reject' and c == 'US'))
        pairs.append(P)
        s1_sets[c] = ids
    M = pl.concat(pairs)
    if M['target'].n_unique() != M.height:
        raise RuntimeError('A target was assigned to more than one S1.')
    grouped = M.group_by('s1').agg(pl.col('target').sort().str.join(',').alias('matched_entity_ids'))
    raw = pl.read_parquet(ROOT / tc['raw_test_source1'], columns=['entity_id', 'country'])
    s1_country = raw.rename({'entity_id': 'source1_entity_id'})
    result = (raw.select(pl.col('entity_id').alias('source1_entity_id'))
              .join(grouped.rename({'s1': 'source1_entity_id'}), on='source1_entity_id', how='left', maintain_order='left')
              .with_columns(pl.col('matched_entity_ids').fill_null('')))
    b01_path = ROOT / tc['codex_b01_matching']
    b01 = read_tsv(b01_path) if b01_path.is_file() else None
    if args.variant == 'france-b01' and b01 is None:
        raise RuntimeError(f'The france-b01 variant needs the B01 file: {b01_path}')
    if args.variant == 'france-b01':
        fr = set(raw.filter(pl.col('country') == 'France')['entity_id'].to_list())
        b01_fr = b01.filter(pl.col('source1_entity_id').is_in(list(fr)))
        result = (result.join(b01_fr.rename({'matched_entity_ids': 'b01'}), on='source1_entity_id', how='left', maintain_order='left')
                  .with_columns(pl.when(pl.col('b01').is_not_null()).then(pl.col('b01')).otherwise(pl.col('matched_entity_ids'))
                                .alias('matched_entity_ids')).drop('b01'))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'matching_results.tsv'
    part = path.with_suffix('.tsv.part')
    result.write_csv(part, separator='\t', quote_style='never', include_header=True, line_terminator='\n')
    part.replace(path)
    print(f'Wrote {path.relative_to(ROOT)} ({result.height:,} S1 rows). Running the OFFICIAL validator with --check-ids (1-3 min)...', flush=True)
    v = subprocess.run([sys.executable, str(ROOT / tc['official_validator']), '--matching', str(path),
                        '--candidate', str(folder / 'no_candidate_file_for_leaderboard.tsv'),
                        '--test-dir', str(ROOT / tc['test_dir']), '--check-ids'], capture_output=True, text=True)
    print(v.stdout[-3000:], v.stderr[-2000:], flush=True)
    new = stats(result, s1_country)
    old = stats(b01, s1_country) if b01 is not None else None
    changed = (result.join(b01.rename({'matched_entity_ids': 'b01'}), on='source1_entity_id').filter(pl.col('matched_entity_ids') != pl.col('b01')).height
               if b01 is not None else None)
    report = {'variant': args.variant, 'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha256(path),
              'official_validator_exit_code': v.returncode, 'official_validator_pass': v.returncode == 0 and 'PASS' in v.stdout,
              'threshold': threshold, 'rows': result.height, 's1_rows_changed_vs_b01': changed,
              'per_country_c04_or_variant': new, 'per_country_b01_submitted': old, 'seconds': round(time.time() - t0, 1),
              'smoke': cfg['smoke_mode'], 'leaderboard': 'Not submitted by this script.'}
    atomic_json(folder / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['official_validator_pass']:
        raise SystemExit('Official validation FAILED. Do not upload this file.')
    if cfg['smoke_mode']:
        print('\nSMOKE file (first parts only). Do NOT upload it.', flush=True)
    else:
        print(f'\nREADY TO UPLOAD: {path}', flush=True)
        parts = path.parts
        if len(parts) > 3 and parts[1] == 'mnt' and len(parts[2]) == 1:
            print(f"Windows path: {parts[2].upper()}:\\{chr(92).join(parts[3:])}", flush=True)


if __name__ == '__main__':
    main()

"""C28 step 4: test export. Accept a record's C04 winner when the C04+TF stack >= threshold (matcher/runs/c28/stack.json),
reject US pairs caught by the existing US near-number rule (c04t_export.offset_mask), then validate.
  python3 matcher/scripts/c28_export.py pure            -> matcher/runs/c28/export_pure/matching_results.tsv
Output keeps the official row order; official validator with --check-ids.
"""
import glob, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'matcher/scripts'))
RUN = ROOT / 'matcher/runs/c28'
C04 = ROOT / 'matcher/runs/c04_v1'
SUF = os.environ.get('C28_SUF', '')   # '' = model 1, '_model2' = model 2


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def stacked(d):
    st = json.loads((RUN / f'stack{SUF}.json').read_text())
    w, b0 = st['stack_coef']['coef'], st['stack_coef']['intercept']
    a, b = lg(d['score'].to_numpy()), lg(d['tf'].to_numpy())
    z = b0 + w[0] * a + w[1] * b + w[2] * a * b / 10
    return 1 / (1 + np.exp(-z)), st['stack']['threshold']


def us_offset(P):
    """same predicate as c04t_export.offset_mask: record core-name tokens within the S1's, same-width house |diff| 1-9."""
    src = ROOT / 'retrieval/runs/b01_v1/test_stable_v1/candidates/test_US'
    refs = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id']).rename({'entity_id': 's1'})
    q = pl.concat([pl.read_parquet(f, columns=['query_index', 'entity_id']) for f in sorted(glob.glob(str(src / 'part-*.queries.parquet')))]).rename({'entity_id': 't'})
    qs = pl.read_parquet(C04 / 'test/prepare/US/queries.parquet', columns=['query_index', 'q_house', 'q_tok'])
    rs = pl.read_parquet(C04 / 'test/prepare/US/refs.parquet', columns=['ref_index', 'r_house', 'r_tok'])
    X = P.join(refs, on='s1').join(q, on='t').join(qs, on='query_index', how='left').join(rs, on='ref_index', how='left')
    d = (pl.col('q_house').cast(pl.Int64, strict=False) - pl.col('r_house').cast(pl.Int64, strict=False)).abs()
    m = ((pl.col('q_tok').list.set_difference('r_tok').list.len() == 0) & (pl.col('q_house').str.len_chars() == pl.col('r_house').str.len_chars())
         & (d >= 1) & (d <= 9)).fill_null(False)
    return X.filter(m).select('s1', 't')


def write(P, name):
    base = pl.read_csv(ROOT / 'matcher/runs/c04_v1/test/output_us_offset_reject/matching_results.tsv', separator='\t', quote_char=None,
                       missing_utf8_is_empty_string=True, schema={'source1_entity_id': pl.String, 'matched_entity_ids': pl.String})
    assert P['t'].n_unique() == P.height
    g = P.group_by('s1').agg(pl.col('t').sort().str.join(',').alias('m'))
    out = base.select('source1_entity_id').join(g.rename({'s1': 'source1_entity_id'}), on='source1_entity_id', how='left',
                                                maintain_order='left').with_columns(pl.col('m').fill_null('')).rename({'m': 'matched_entity_ids'})
    d = RUN / f'export_{name}{SUF}'; d.mkdir(parents=True, exist_ok=True); path = d / 'matching_results.tsv'
    out.write_csv(path, separator='\t', quote_style='never', include_header=True, line_terminator='\n')
    v = subprocess.run([sys.executable, str(ROOT / 'student_resource/utils/validate_submission.py'), '--matching', str(path), '--candidate',
                        str(d / 'none.tsv'), '--test-dir', str(ROOT / 'student_resource/dataset/test'), '--check-ids'], capture_output=True, text=True)
    rep = {'file': str(path.relative_to(ROOT)), 'pairs': P.height, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'official_validator': 'PASS' if (v.returncode == 0 and 'PASS' in v.stdout) else 'FAIL ' + v.stdout[-500:]}
    (d / 'report.json').write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


def pairs(p):
    d = pl.read_csv(p, separator='\t', quote_char=None, missing_utf8_is_empty_string=True,
                    schema={'source1_entity_id': pl.String, 'matched_entity_ids': pl.String})
    return d.filter(pl.col('matched_entity_ids') != '').select(pl.col('source1_entity_id').alias('s1'),
                                                               pl.col('matched_entity_ids').str.split(',').alias('t')).explode('t')


def main():
    """Final decision: accept C04's winner when the C04+transformer stack >= its tune-selected threshold, then the US rule."""
    d = pl.read_parquet(RUN / f'scores_test{SUF}.parquet')
    p, thr = stacked(d)
    d = d.with_columns(pl.Series('stack', p))
    d.select('s1', 't', 'country', 'score', 'tf', 'stack').write_parquet(RUN / f'test_stack{SUF}.parquet')
    P = d.filter(pl.col('stack') >= thr).select('s1', 't')
    rm = us_offset(P)
    print('stack accepts', P.height, '| US rule removes', rm.height, flush=True)
    write(P.join(rm, on=['s1', 't'], how='anti'), 'pure')



if __name__ == '__main__':
    main()

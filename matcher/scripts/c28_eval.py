"""C28 step 3: does the cross-encoder help? Held-out check with the official per-S1 F0.5.
Stack = logistic regression on [logit C04, logit TF, product], fitted on winners NOT pointing at CHECK references;
threshold chosen on TUNE (India/US equal weight, 1.9x unmatched-FP stress as in C04); CHECK is reported only.
Writes matcher/runs/c28/stack.json (coefficients + threshold) used by the test export.
"""
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'matcher/runs/c28'
SUF = os.environ.get('C28_SUF', '')   # '' = model 1; '_model2' = model 2
GRID = np.r_[np.linspace(0.3, 0.98, 69), 0.985, 0.99, 0.995]


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def feats(d):
    a, b = lg(d['score'].to_numpy()), lg(d['tf'].to_numpy())
    return np.column_stack([a, b, a * b / 10])


def metrics(d, refs, accept, stress=1.0):
    a = d.filter(pl.Series(accept))
    per = a.group_by('ref_index').agg((pl.col('owner_index') == pl.col('ref_index')).sum().alias('tp'),
                                      ((pl.col('owner_index') != pl.col('ref_index')) & (pl.col('owner_index') >= 0)).sum().alias('fw'),
                                      (pl.col('owner_index') < 0).sum().alias('fu'))
    r = refs.join(per, on='ref_index', how='left').fill_null(0)
    tp, fp, tr = r['tp'].to_numpy(), r['fw'].to_numpy() + stress * r['fu'].to_numpy(), r['truth_count'].to_numpy()
    den = 4 * tp + 4 * fp + tr
    f = np.where(den == 0, 1.0, 5 * tp / np.maximum(den, 1e-9))
    role = r['role'].to_numpy()
    return f[role == 1].mean(), f[role == 2].mean()


def main():
    d = pl.read_parquet(RUN / f'scores_val{SUF}.parquet')
    refs = {c: pl.read_parquet(ROOT / f'retrieval/runs/b01_v1/matcher_v1/features/{c}/references.parquet',
                               columns=['ref_index', 'truth_count', 'role']).filter(pl.col('role').is_in([1, 2])) for c in ['India', 'US']}
    role = pl.concat([d.filter(pl.col('country') == c).join(refs[c], on='ref_index', how='left').select(pl.col('role').fill_null(0)) for c in ['India', 'US']])
    d = pl.concat([d.filter(pl.col('country') == c) for c in ['India', 'US']]).with_columns(role['role'])
    y = (d['ref_index'] == d['owner_index']).to_numpy()
    fit = d['role'].to_numpy() != 2
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(d.filter(pl.Series(fit))), y[fit])
    d = d.with_columns(pl.Series('stack', lr.predict_proba(feats(d))[:, 1]))
    res = {}
    for name, col, grid in (('c04', 'score', [0.76]), ('tf', 'tf', GRID), ('stack', 'stack', GRID)):
        best = None
        for t in grid:
            tune_s, chk, tune_o = [], [], []
            for c in ['India', 'US']:
                g = d.filter(pl.col('country') == c)
                acc = (g[col] >= t).to_numpy()
                ts, _ = metrics(g, refs[c], acc, 1.9)
                to, ck = metrics(g, refs[c], acc)
                tune_s.append(ts); chk.append(ck); tune_o.append(to)
            row = {'threshold': float(t), 'tune_stress': float(np.mean(tune_s)), 'tune': float(np.mean(tune_o)),
                   'check': float(np.mean(chk)), 'check_India': float(chk[0]), 'check_US': float(chk[1])}
            if best is None or row['tune_stress'] > best['tune_stress']:
                best = row
        res[name] = best
        print(name, {k: round(v, 6) for k, v in best.items()}, flush=True)
    res['stack_coef'] = {'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0])}
    (RUN / f'stack{SUF}.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()

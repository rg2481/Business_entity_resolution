"""C04 stage 4 (CPU, user-run): held-out entity evaluation with Codex B01's exact protocol and metric code.

Winner per held-out query over ALL its candidates -> threshold chosen on TUNE references only (country-reweighted,
1.9x unmatched-FP stress objective, as B01) -> official per-S1 macro-F0.5 on CHECK references. Compared with B01."""
import argparse
import csv
import json
import time
from c04_common import (settings, environment, stage_lock, require_manifest, atomic_json, table_write, read_json,
                        code_hash, append_progress, CLAUDE, CODEX_SCRIPTS)
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from b01_pair_features import FEATURES
from b01_match_metrics import winners, threshold_grid, accumulate_hist, metric_curves, choose_threshold, at_threshold
from c04_features import FEATURES_V2
from c04_train import ALL_FEATURES, sha


def evaluate_country(country, cfg, out, model, trees, grid):
    feat_src = cfg['codex_features'] / country
    refs = pq.read_table(feat_src / 'references.parquet').to_pandas()
    roles = refs['role'].to_numpy()
    selected = np.flatnonzero((roles == 1) | (roles == 2))
    selected_map = np.full(len(roles), -1, np.int32)
    selected_map[selected] = np.arange(len(selected), dtype=np.int32)
    truth = refs['truth_count'].to_numpy()[selected]
    covered = refs['retrieved_true_count'].to_numpy()[selected]
    hist = np.zeros((3, len(selected), len(grid)), np.int32)
    mine = out / 'features' / country
    parts = sorted(mine.glob('part-*.eval.parquet'))
    wdir = out / 'evaluation' / country
    pairs = queries = 0
    with tqdm(total=len(parts), desc=f'Evaluate {country}', unit='parts', dynamic_ncols=True) as bar:
        for path in parts:
            base = pq.read_table(feat_src / path.name, columns=['query_index', 'ref_index', 'owner_index', *FEATURES])
            v2 = pq.read_table(path)
            if base.num_rows == 0:
                bar.update(1)
                continue
            q, r = base['query_index'].to_numpy(), base['ref_index'].to_numpy()
            if not (np.array_equal(q, v2['query_index'].to_numpy()) and np.array_equal(r, v2['ref_index'].to_numpy())):
                raise RuntimeError(f'Row alignment failed: {path.name}')
            X = np.column_stack([base[f].to_numpy() for f in FEATURES] + [v2[f].to_numpy() for f in FEATURES_V2]).astype(np.float32)
            s = model.predict(X, num_iteration=trees, num_threads=cfg['cpu_threads'])
            pred = winners(q, r, base['owner_index'].to_numpy(), s)
            accumulate_hist(hist, selected_map, pred['ref_index'], pred['owner_index'], pred['score'], grid)
            table_write(wdir / path.name.replace('eval', 'winners'), pa.table(pred))
            pairs += base.num_rows
            queries += len(pred['query_index'])
            bar.update(1)
    counts, official, stress = metric_curves(hist, truth, cfg['distractor_fp_weight'])
    d = 4.0 * covered + truth
    oracle = np.divide(5.0 * covered, d, out=np.ones_like(d, dtype=float), where=d != 0)
    return dict(country=country, roles=roles[selected], truth=truth, counts=counts, official=official, stress=stress,
                oracle=oracle, population=len(roles), pairs=pairs, queries=queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--trees', type=int, default=0, help='evaluate the model truncated to this many trees (0 = all)')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    model_dir = out / 'model'
    done = read_json(model_dir / 'complete.json')
    if sha(model_dir / 'model.txt') != done['model_sha256']:
        raise RuntimeError('Model checksum failed.')
    trees = args.trees or done['trees']
    directory = out / 'evaluation' if trees == done['trees'] else out / f'evaluation_trees{trees:05d}'
    _lock = stage_lock(directory)
    signature = {'stage': 'evaluate', 'model_sha256': done['model_sha256'], 'trees': trees, 'features': ALL_FEATURES,
                 'distractor_fp_weight': cfg['distractor_fp_weight'],
                 'code': code_hash(CLAUDE / 'scripts' / 'c04_evaluate.py', CODEX_SCRIPTS / 'b01_match_metrics.py')}
    require_manifest(directory, signature)
    if (directory / 'report.json').is_file():
        print('Completed evaluation:\n' + (directory / 'report.json').read_text())
        return
    model = lgb.Booster(model_file=str(model_dir / 'model.txt'))
    if model.feature_name() != ALL_FEATURES:
        raise RuntimeError('Booster feature order differs from the code.')
    t0 = time.time()
    grid = threshold_grid()
    values = []
    for c in cfg['countries']:
        v = evaluate_country(c, cfg, out, model, trees, grid)
        # keep per-country outputs next to the report
        values.append(v)
    population = sum(v['population'] for v in values)
    tune_official, tune_stress = np.zeros(len(grid)), np.zeros(len(grid))
    for v in values:
        tune = v['roles'] == 1
        w = v['population'] / population
        tune_official += w * v['official'][tune].mean(axis=0)
        tune_stress += w * v['stress'][tune].mean(axis=0)
    index = choose_threshold(tune_stress, grid)
    threshold = float(grid[index])
    report = {'experiment': cfg['experiment'], 'threshold': threshold, 'trees': trees, 'model_sha256': done['model_sha256'],
              'countries': {}, 'b01_check': cfg['b01_check'], 'seconds': round(time.time() - t0, 1), 'smoke': cfg['smoke_mode'],
              'protocol': 'Codex B01 held-out entities/query isolation; winner over all candidates; threshold from TUNE only.'}
    for v in values:
        groups = {}
        for g, role in (('tune', 1), ('check', 2)):
            keep = v['roles'] == role
            groups[g] = at_threshold(v['counts'], v['official'], v['stress'], v['truth'], keep, index)
            groups[g]['candidate_oracle_macro_f05'] = float(v['oracle'][keep].mean())
        groups['check_minus_b01'] = groups['check']['macro_f05'] - cfg['b01_check'][v['country']]
        report['countries'][v['country']] = dict(groups, evaluated_pairs=v['pairs'], winner_queries=v['queries'])
    checks = [report['countries'][c]['check']['macro_f05'] for c in cfg['countries']]
    report['check_mean'] = float(np.mean(checks))
    report['b01_check_mean'] = float(np.mean([cfg['b01_check'][c] for c in cfg['countries']]))
    with (directory / 'tuning_thresholds.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'country_reweighted_macro_f05', 'country_reweighted_stress_macro_f05'])
        w.writerows(zip(grid, tune_official, tune_stress))
    atomic_json(directory / 'report.json', report)
    lines = [f"C04 local held-out evaluation ({trees} trees, threshold {threshold:.3f} chosen on TUNE):"]
    for c in cfg['countries']:
        k = report['countries'][c]
        lines.append(f"  {c}: check macro-F0.5 {k['check']['macro_f05']:.6f} (B01 {cfg['b01_check'][c]:.6f}, "
                     f"{k['check_minus_b01']:+.6f}); TP {k['check']['tp']:,} FP {k['check']['fp_unmatched'] + k['check']['fp_wrong_owner']:,} "
                     f"FN {k['check']['fn']:,}; zero-truth false-merge rate {k['check']['zero_truth_false_merge_rate']}")
    lines.append(f"  mean check {report['check_mean']:.6f} vs B01 {report['b01_check_mean']:.6f} "
                 f"({report['check_mean'] - report['b01_check_mean']:+.6f}). Leaderboard: not submitted.")
    append_progress(cfg, f'evaluate:trees{trees}', '\n'.join(lines))
    print(json.dumps(report, indent=2), flush=True)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()

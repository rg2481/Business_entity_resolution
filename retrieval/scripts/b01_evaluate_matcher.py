"""Stream held-out inference, tune a threshold, and check entity macro-F0.5."""
import argparse
import csv
import heapq
import json
import time
from b01_common import ROOT, atomic_json, code_hash, environment, identity
environment()
from b01_match_common import (COUNTRIES, settings, require_manifest, read_json, file_digest, digest,
                              append_progress, chunk_files, table_write, stage_lock)
from b01_pair_features import FEATURES
from b01_match_metrics import winners, threshold_grid, accumulate_hist, metric_curves, choose_threshold, at_threshold
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def evaluate_country(country, cfg, base, out, model, model_sha):
    source = out / 'features' / country
    chunks = chunk_files(source)
    refs = pq.read_table(source / 'references.parquet')
    roles = refs['role'].to_numpy()
    selected = np.flatnonzero((roles == 1) | (roles == 2))
    selected_map = np.full(len(roles), -1, np.int32)
    selected_map[selected] = np.arange(len(selected), dtype=np.int32)
    truth = refs['truth_count'].to_numpy()[selected]
    covered = refs['retrieved_true_count'].to_numpy()[selected]
    directory = out / 'evaluation' / country
    signature = {'features': read_json(source / 'manifest.json'), 'model_sha256': model_sha,
                 'code': code_hash('b01_evaluate_matcher.py', 'b01_match_metrics.py', 'b01_match_common.py'),
                 'feature_files': [identity(source / p.name.replace('complete.json', 'eval.parquet')) for p in chunks]}
    require_manifest(directory, signature)
    grid = threshold_grid()
    hist = np.zeros((3, len(selected), len(grid)), np.int32)
    errors = []
    winner_count = 0
    total = read_json(source / 'complete.json')['eval_pairs']
    start = time.perf_counter()
    with tqdm(total=total, desc=f'Predict held-out {country}', unit='pairs', dynamic_ncols=True) as bar:
        for number, marker in enumerate(chunks):
            features_path = source / f'part-{number:05d}.eval.parquet'
            prediction_path = directory / f'part-{number:05d}.winners.parquet'
            stamp = directory / f'part-{number:05d}.complete.json'
            feature_rows = pq.ParquetFile(features_path).metadata.num_rows
            expected = {'feature_file': identity(features_path), 'pairs': feature_rows}
            if stamp.exists():
                saved = read_json(stamp)
                if saved['inputs'] != expected or not prediction_path.exists():
                    raise RuntimeError('Evaluation checkpoint does not match input features.')
                table = pq.read_table(prediction_path)
                if table.num_rows != saved['winners']:
                    raise RuntimeError('Evaluation checkpoint row count failed.')
                pred = {key: table[key].to_numpy() for key in table.column_names}
            else:
                table = pq.read_table(features_path)
                X = np.empty((feature_rows, len(FEATURES)), np.float32)
                for i, key in enumerate(FEATURES):
                    X[:, i] = table[key].to_numpy()
                scores = np.asarray(model.predict(X, num_threads=cfg['cpu_threads']), np.float64) if feature_rows else np.array([], np.float64)
                if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
                    raise RuntimeError('Invalid model probabilities.')
                pred = winners(table['query_index'].to_numpy(), table['ref_index'].to_numpy(),
                               table['owner_index'].to_numpy(), scores)
                table_write(prediction_path, pa.table(pred))
                atomic_json(stamp, {'inputs': expected, 'winners': len(pred['query_index'])})
            # The winner was chosen before applying the selected-reference mask.
            accumulate_hist(hist, selected_map, pred['ref_index'], pred['owner_index'], pred['score'], grid)
            chosen = selected_map[pred['ref_index']] >= 0
            bad = np.flatnonzero(chosen & (pred['owner_index'] != pred['ref_index']))
            for j in bad:
                item = (float(pred['score'][j]), int(pred['query_index'][j]), int(pred['ref_index'][j]),
                        int(pred['owner_index'][j]), float(pred['runnerup_score'][j]), number)
                if len(errors) < 100:
                    heapq.heappush(errors, item)
                elif item > errors[0]:
                    heapq.heapreplace(errors, item)
            winner_count += len(pred['query_index'])
            bar.update(feature_rows)
    counts, official, stress = metric_curves(hist, truth, cfg['distractor_fp_weight'])
    oracle = np.ones(len(selected), np.float64)
    positive = truth > 0
    oracle[positive] = 5.0 * covered[positive] / (4.0 * covered[positive] + truth[positive])
    with (directory / 'highest_scoring_wrong_assignments.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['probability', 'query_index', 'predicted_ref_id', 'true_ref_id_or_unmatched',
                         'runnerup_probability', 'query_checkpoint'])
        for score, qi, ri, oi, runnerup, number in sorted(errors, reverse=True):
            writer.writerow([score, qi, refs['entity_id'][ri].as_py(), refs['entity_id'][oi].as_py() if oi >= 0 else '',
                             runnerup, str((out / 'candidates' / f'train_{country}' / f'part-{number:05d}.queries.parquet').relative_to(ROOT))])
    atomic_json(directory / 'complete.json', {'chunks': len(chunks), 'eval_pairs': total, 'winner_queries': winner_count,
                'seconds_this_invocation': time.perf_counter() - start, 'selected_entities': len(selected)})
    return dict(country=country, reference_population=len(roles), roles=roles[selected], truth=truth,
                counts=counts, official=official, stress=stress, oracle=oracle,
                eval_pairs=total, winner_queries=winner_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--base-config')
    args = parser.parse_args()
    cfg, base, run, out = settings(args.config, args.base_config)
    directory = out / 'evaluation'
    _lock = stage_lock(directory)
    model_dir = out / 'model'
    model_done = read_json(model_dir / 'complete.json')
    model_sha = file_digest(model_dir / 'model.txt')
    if model_sha != model_done['model_sha256']:
        raise RuntimeError('Model checksum failed.')
    model_manifest = read_json(model_dir / 'manifest.json')
    if model_manifest['features'] != FEATURES:
        raise RuntimeError('Model feature schema does not match current code.')
    for country, saved in zip(COUNTRIES, model_manifest['sources']):
        folder = out / 'features' / country
        if saved['manifest'] != read_json(folder / 'manifest.json') or saved['complete'] != read_json(folder / 'complete.json'):
            raise RuntimeError('Features changed after model training.')
    signature = {'model_sha256': model_sha, 'distractor_fp_weight': cfg['distractor_fp_weight'],
                 'code': code_hash('b01_evaluate_matcher.py', 'b01_match_metrics.py', 'b01_match_common.py')}
    require_manifest(directory, signature)
    if (directory / 'report.json').is_file():
        append_progress(model_done['event'] + ':evaluation:' + digest(signature)[:12],
                        'B01 local held-out entity evaluation completed.\n' + (directory / 'report.json').read_text() +
                        '\nReview wrong assignments and the check score before test inference. Leaderboard: not submitted.')
        print('Completed evaluation.\n' + (directory / 'report.json').read_text())
        return
    model = lgb.Booster(model_file=str(model_dir / 'model.txt'))
    if model.feature_name() != FEATURES:
        raise RuntimeError('Booster feature order changed.')
    values = [evaluate_country(country, cfg, base, out, model, model_sha) for country in COUNTRIES]
    grid = threshold_grid()
    population = sum(v['reference_population'] for v in values)
    tune_official = np.zeros(len(grid), np.float64)
    tune_stress = np.zeros(len(grid), np.float64)
    for v in values:
        tune = v['roles'] == 1
        weight = v['reference_population'] / population
        tune_official += weight * v['official'][tune].mean(axis=0)
        tune_stress += weight * v['stress'][tune].mean(axis=0)
    index = choose_threshold(tune_stress, grid)
    threshold = float(grid[index])
    # Threshold choice depends exclusively on tuning-reference metrics.
    atomic_json(directory / 'threshold.json', {'threshold': threshold, 'index': index,
                'selection': 'Maximize country-reweighted tuning macro-F0.5 with separately weighted unmatched FPs; stricter threshold breaks ties.',
                'distractor_fp_weight': cfg['distractor_fp_weight'], 'model_sha256': model_sha})
    report = {'threshold': threshold, 'model_sha256': model_sha, 'countries': {}, 'combined': {},
              'validation': {'entities_per_country': cfg['validation_entities_per_country'],
                             'tune_fraction': 0.5, 'fold': base['validation_fold'],
                             'query_isolation': 'All owner/candidate-touching queries excluded from training, with all candidates retained for validation.',
                             'reference_pool': 'Full country pools, with all targets able to assign to each selected reference.',
                             'sampling': 'Reference-level sample. This is not full-population macro-F0.5 or five-fold OOF.',
                             'threshold_selection': 'Tuning subset only; fixed-tree classifier, no held-out early stopping.'},
              'stress_caveat': f"The {cfg['distractor_fp_weight']}x unmatched-FP weighting is a cost sensitivity analysis; it does not synthesize extra distractors or their incidence on empty references.",
              'france': 'No labeled France validation is available; transfer remains unmeasured.',
              'leaderboard': 'Not submitted; no leaderboard score.'}
    for v in values:
        groups = {}
        for group, role in (('tune', 1), ('check', 2)):
            keep = v['roles'] == role
            groups[group] = at_threshold(v['counts'], v['official'], v['stress'], v['truth'], keep, index)
            groups[group]['candidate_oracle_macro_f05'] = float(v['oracle'][keep].mean())
        report['countries'][v['country']] = dict(groups, full_reference_population=v['reference_population'],
                                                evaluated_pairs=v['eval_pairs'], winner_queries=v['winner_queries'])
    for group in ('tune', 'check'):
        result = {}
        for key in ('macro_f05', 'distractor_fp_stress_macro_f05', 'candidate_oracle_macro_f05'):
            metrics = [report['countries'][v['country']][group][key] for v in values]
            sizes = [report['countries'][v['country']][group]['entities'] for v in values]
            result[key] = float(np.average(metrics, weights=sizes))
            result['country_reweighted_' + key] = float(np.average(metrics, weights=[v['reference_population'] for v in values]))
        for key in ('entities', 'tp', 'fp_unmatched', 'fp_wrong_owner', 'fn', 'zero_truth_entities'):
            result[key] = sum(report['countries'][v['country']][group][key] for v in values)
        report['combined'][group] = result
    with (directory / 'tuning_thresholds.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['threshold', 'country_reweighted_macro_f05', 'country_reweighted_distractor_fp_stress_macro_f05'])
        writer.writerows(zip(grid, tune_official, tune_stress))
    atomic_json(directory / 'report.json', report)
    append_progress(model_done['event'] + ':evaluation:' + digest(signature)[:12],
                    'B01 local held-out entity evaluation completed.\n' + json.dumps(report, indent=2) +
                    '\nReview wrong assignments and the check score before test inference. Leaderboard: not submitted.')
    print(json.dumps(report, indent=2), flush=True)
    print('Validation complete. No test predictions or submission upload were started.', flush=True)


if __name__ == '__main__':
    main()

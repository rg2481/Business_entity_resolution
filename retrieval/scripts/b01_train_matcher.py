"""User-run CPU LightGBM matcher; fixed tree budget and resumable checkpoints."""
import argparse
import gc
import json
import time
from pathlib import Path
from b01_common import ROOT, atomic_json, code_hash, environment, identity
environment()
from b01_match_common import (COUNTRIES, settings, require_manifest, read_json, file_digest, digest,
                              append_progress, chunk_files, stage_lock)
from b01_pair_features import FEATURES
import lightgbm as lgb
import numpy as np
import psutil
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def model_parameters(cfg):
    return dict(objective='binary', metric='binary_logloss', learning_rate=cfg['learning_rate'],
                num_leaves=cfg['num_leaves'], min_data_in_leaf=cfg['min_data_in_leaf'],
                max_bin=127, num_threads=cfg['cpu_threads'], seed=cfg['seed'],
                deterministic=True, force_col_wise=True, verbosity=-1, feature_pre_filter=False)


def fit_model(X, y, cfg, directory, progress):
    """Synthetic checks can call this without modifying the shared experiment log."""
    checkpoint = directory / 'checkpoint.json'
    initial, initial_trees = None, 0
    if checkpoint.is_file():
        saved = read_json(checkpoint)
        initial = directory / saved['file']
        if file_digest(initial) != saved['sha256']:
            raise RuntimeError('Training checkpoint checksum failed.')
        initial_trees = lgb.Booster(model_file=str(initial)).current_iteration()
        if initial_trees != saved['trees']:
            raise RuntimeError('Training checkpoint tree count changed.')
    if initial_trees > cfg['trees']:
        raise RuntimeError('Cannot resume with fewer requested trees; use a fresh experiment_dir.')
    progress.update(initial_trees)
    if initial_trees == cfg['trees']:
        return lgb.Booster(model_file=str(initial))
    train = lgb.Dataset(X, label=y, feature_name=FEATURES, free_raw_data=True)
    print('LightGBM will construct feature bins before the first tree update. This setup call has no progress callback.', flush=True)

    def checkpoint_callback(env):
        progress.update(1)
        trees = env.model.current_iteration()
        if trees % cfg['checkpoint_trees'] == 0 or env.iteration + 1 == env.end_iteration:
            path = directory / f'checkpoint-{trees:05d}.txt'
            partial = path.with_suffix('.txt.part')
            env.model.save_model(str(partial))
            partial.replace(path)
            atomic_json(checkpoint, {'file': path.name, 'trees': trees, 'sha256': file_digest(path)})
    checkpoint_callback.order = 40
    checkpoint_callback.before_iteration = False
    return lgb.train(model_parameters(cfg), train, num_boost_round=cfg['trees'] - initial_trees,
                     init_model=str(initial) if initial else None, callbacks=[checkpoint_callback])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--base-config')
    args = parser.parse_args()
    cfg, base, run, out = settings(args.config, args.base_config)
    folders = [out / 'features' / country for country in COUNTRIES]
    train_paths, sources = [], []
    for folder in folders:
        chunks = chunk_files(folder)
        sources.append({'manifest': read_json(folder / 'manifest.json'), 'complete': read_json(folder / 'complete.json')})
        train_paths.extend(folder / p.name.replace('complete.json', 'train.parquet') for p in chunks)
    directory = out / 'model'
    _lock = stage_lock(directory)
    signature = {'sources': sources, 'files': [identity(p) for p in train_paths],
                 'parameters': model_parameters(cfg), 'trees': cfg['trees'],
                 'features': FEATURES, 'code': code_hash('b01_train_matcher.py', 'b01_pair_features.py', 'b01_match_common.py')}
    require_manifest(directory, signature)
    event = digest(signature)[:16]
    if (directory / 'complete.json').is_file():
        done = read_json(directory / 'complete.json')
        if file_digest(directory / 'model.txt') != done['model_sha256']:
            raise RuntimeError('Completed model checksum failed.')
        append_progress(event + ':fit', json.dumps(done, indent=2) + '\nNext: threshold tuning and held-out entity evaluation. Leaderboard: not submitted.')
        print('Model already complete.\n' + json.dumps(done, indent=2))
        return
    counts = [pq.ParquetFile(path).metadata.num_rows for path in train_paths]
    rows = sum(counts)
    if not rows:
        raise RuntimeError('No classifier training pairs.')
    estimate = rows * (len(FEATURES) * 4 + 1)
    available = psutil.virtual_memory().available
    print(f'Training matrix: {rows:,} pairs x {len(FEATURES)} features; {estimate / 2**30:.2f} GiB for X/y. '
          f'Available RAM: {available / 2**30:.2f} GiB; LightGBM also needs bins and working memory.', flush=True)
    if estimate * 3 + 2**30 > available:
        raise RuntimeError('Insufficient RAM for the matrix plus conservative training headroom. Close other jobs and retry.')
    append_progress(event + ':start',
        f'Experiment B01 matcher_v1 started by the user. Output: {out.relative_to(ROOT)}\n'
        f'Settings: {json.dumps(cfg, sort_keys=True)}\n'
        f'Features: {", ".join(FEATURES)}\n'
        'Retrieval: optimized CUDA, union of name/address/combo top-10; full country reference and target pools.\n'
        'Train: folds 1-4 references, deterministic query sampling; no query touching a selected held-out reference enters training.\n'
        f'Validation: {cfg["validation_entities_per_country"]} uniformly sampled fold0 references per country; '
        'half threshold tuning, half threshold check, with all candidate competitors. Dictionary excludes fold0.\n'
        'No forced true candidates, no reference-pool deletion. Official per-entity F0.5 includes empty truth.\n'
        f'Threshold objective: country-reweighted tuning macro-F0.5 with {cfg["distractor_fp_weight"]}x genuine-distractor FP cost (sensitivity analysis).\n'
        'Local validation results: pending. Leaderboard: not submitted; no score claimed.')
    X = np.empty((rows, len(FEATURES)), np.float32)
    y = np.empty(rows, np.uint8)
    offset = 0
    with tqdm(total=rows, desc='Load training pairs', unit='pairs', dynamic_ncols=True) as bar:
        for path, count in zip(train_paths, counts):
            if count:
                table = pq.read_table(path, columns=['label', *FEATURES])
                for i, key in enumerate(FEATURES):
                    X[offset:offset + count, i] = table[key].to_numpy()
                y[offset:offset + count] = table['label'].to_numpy()
                offset += count
                bar.update(count)
    if len(np.unique(y)) != 2:
        raise RuntimeError('Training requires both positive and negative pairs.')
    positives = int(y.sum())
    gc.collect()
    start = time.perf_counter()
    with tqdm(total=cfg['trees'], desc='Train matcher', unit='trees', dynamic_ncols=True) as bar:
        model = fit_model(X, y, cfg, directory, bar)
    partial = directory / 'model.txt.part'
    model.save_model(str(partial))
    partial.replace(directory / 'model.txt')
    importance = directory / 'feature_importance.csv'
    import csv
    with importance.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['feature', 'gain', 'splits'])
        writer.writerows(zip(FEATURES, model.feature_importance('gain'), model.feature_importance('split')))
    done = {'pairs': rows, 'positive_pairs': positives, 'negative_pairs': rows - positives,
            'trees': model.current_iteration(), 'requested_trees': cfg['trees'],
            'fit_seconds_this_invocation': time.perf_counter() - start,
            'model_sha256': file_digest(directory / 'model.txt'), 'event': event,
            'scope': 'Classifier fit complete. No validation or leaderboard score yet.'}
    atomic_json(directory / 'complete.json', done)
    append_progress(event + ':fit', json.dumps(done, indent=2) + '\nNext: threshold tuning and held-out entity evaluation. Leaderboard: not submitted.')
    print(json.dumps(done, indent=2), flush=True)


if __name__ == '__main__':
    main()

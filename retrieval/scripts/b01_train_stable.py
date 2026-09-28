"""User-run regularized B01 refit, reusing completed candidate/feature caches."""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import time
from b01_common import ROOT, CODEX, atomic_json, code_hash, environment, identity
environment()
from b01_match_common import (COUNTRIES, settings, require_manifest, read_json, file_digest, digest,
                              append_progress, chunk_files, stage_lock)
from b01_pair_features import FEATURES
from b01_train_matcher import model_parameters
import lightgbm as lgb
import numpy as np
import psutil
import pyarrow.parquet as pq
from tqdm.auto import tqdm

DEFAULT_CONFIG = 'retrieval/configs/b01_stable.json'


def stable_parameters(cfg):
    params = model_parameters(cfg)
    for key in ('lambda_l2', 'min_sum_hessian_in_leaf', 'max_delta_step'):
        value = float(cfg[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive for the stable refit')
        params[key] = value
    return params


def attach_cached_data(cfg, base, out):
    """Create directory links inside codex; training/evaluation only read these caches."""
    original, original_base, _, source = settings(ROOT / cfg['cached_matcher_config'])
    if source == out:
        raise ValueError('The stable refit needs its own experiment_dir; original model is preserved.')
    for key in ('seed', 'validation_entities_per_country', 'training_query_modulus', 'top_k', 'query_batch', 'scan_batch_rows'):
        if cfg[key] != original[key]:
            raise ValueError(f'Cached data requires the original {key}; this run changes model regularization only.')
    if base != original_base:
        raise ValueError('Base preparation/index settings differ from the cached experiment.')
    for country in COUNTRIES:
        folder = source / 'features' / country
        chunk_files(folder)
        manifest = read_json(folder / 'manifest.json')
        if manifest['features'] != FEATURES:
            raise ValueError('Cached feature schema differs from the current model schema.')
        for key in ('seed', 'validation_entities_per_country', 'training_query_modulus'):
            if manifest['settings'][key] != cfg[key]:
                raise ValueError(f'Cached feature selection differs: {key}')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('features', 'candidates'):
        target = (source / name).resolve()
        if not target.is_relative_to(CODEX.resolve()) or not target.is_dir():
            raise ValueError('Cached data must exist inside retrieval/')
        link = out / name
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f'Existing data link points elsewhere: {link}')
        elif link.exists():
            raise ValueError(f'Unexpected directory at the cache-link path: {link}')
        else:
            link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
    return source


def leaf_audit(path, cfg, initial_bias):
    """Check actual serialized tree contributions, allowing the first-tree base score."""
    cap = cfg['learning_rate'] * cfg['max_delta_step']
    trees, leaves, max_update, largest_raw = 0, 0, 0.0, 0.0
    for line in Path(path).read_text().splitlines():
        if line.startswith('Tree='):
            trees += 1
        elif line.startswith('leaf_value='):
            values = np.fromstring(line.split('=', 1)[1], sep=' ')
            if not len(values) or not np.isfinite(values).all():
                raise RuntimeError('Invalid serialized leaf values.')
            largest_raw = max(largest_raw, float(np.max(np.abs(values))))
            updates = values - initial_bias if trees == 1 else values
            value = float(np.max(np.abs(updates)))
            max_update = max(max_update, value)
            leaves += len(values)
            if value > cap + 1e-6 * (1 + abs(initial_bias)):
                raise RuntimeError(f'Tree {trees} has update {value:g}, above the configured {cap:g} cap.')
    if not trees or not leaves:
        raise RuntimeError('Empty model in stability audit.')
    return {'passed': True, 'trees': trees, 'leaves': leaves, 'configured_max_tree_update': cap,
            'observed_max_tree_update': max_update, 'initial_log_odds': initial_bias,
            'largest_raw_leaf_including_initial_log_odds': largest_raw,
            'scope': 'Serialized-tree bound check, not a validation accuracy measurement.'}


def fit_stable(X, y, cfg, directory, progress):
    checkpoint = directory / 'checkpoint.json'
    initial, initial_trees = None, 0
    positive = int(y.sum())
    if not 0 < positive < len(y):
        raise ValueError('Both training classes are required.')
    bias = math.log(positive / (len(y) - positive))
    if checkpoint.exists():
        saved = read_json(checkpoint)
        initial = directory / saved['file']
        if file_digest(initial) != saved['sha256']:
            raise RuntimeError('Stable checkpoint checksum failed.')
        leaf_audit(initial, cfg, bias)
        initial_trees = lgb.Booster(model_file=str(initial)).current_iteration()
        if initial_trees != saved['trees'] or initial_trees > cfg['trees']:
            raise RuntimeError('Stable checkpoint has incompatible tree count.')
    progress.update(initial_trees)
    if initial_trees == cfg['trees']:
        return lgb.Booster(model_file=str(initial)), bias
    train = lgb.Dataset(X, label=y, feature_name=FEATURES, free_raw_data=True)
    print('LightGBM constructs bins before the first tree update. Existing features are reused; no retrieval runs.', flush=True)

    def callback(env):
        progress.update(1)
        count = env.model.current_iteration()
        if count % cfg['checkpoint_trees'] == 0 or env.iteration + 1 == env.end_iteration:
            final = directory / f'checkpoint-{count:05d}.txt'
            partial = final.with_suffix('.txt.part')
            env.model.save_model(str(partial))
            audit = leaf_audit(partial, cfg, bias)
            partial.replace(final)
            atomic_json(checkpoint, {'file': final.name, 'trees': count, 'sha256': file_digest(final), 'stability_audit': audit})
            progress.set_postfix(max_update=f"{audit['observed_max_tree_update']:.3f}")
    callback.order = 40
    callback.before_iteration = False
    model = lgb.train(stable_parameters(cfg), train, num_boost_round=cfg['trees'] - initial_trees,
                      init_model=str(initial) if initial else None, callbacks=[callback])
    return model, bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--base-config')
    args = parser.parse_args()
    cfg, base, _, out = settings(args.config, args.base_config)
    params = stable_parameters(cfg)
    directory = out / 'model'
    _lock = stage_lock(directory)
    source = attach_cached_data(cfg, base, out)
    print(f'Reuse candidates and features from {source.relative_to(ROOT)}. Write only the new model/evaluation under {out.relative_to(ROOT)}.', flush=True)
    paths, sources = [], []
    for country in COUNTRIES:
        folder = out / 'features' / country
        chunks = chunk_files(folder)
        sources.append({'manifest': read_json(folder / 'manifest.json'), 'complete': read_json(folder / 'complete.json')})
        paths.extend(folder / p.name.replace('complete.json', 'train.parquet') for p in chunks)
    signature = {'sources': sources, 'files': [identity(p) for p in paths], 'parameters': params, 'trees': cfg['trees'],
                 'features': FEATURES, 'cached_source': str(source.relative_to(ROOT)),
                 'code': code_hash('b01_train_stable.py', 'b01_train_matcher.py', 'b01_pair_features.py', 'b01_match_common.py')}
    require_manifest(directory, signature)
    event = 'stable:' + digest(signature)[:16]
    if (directory / 'complete.json').is_file():
        done = read_json(directory / 'complete.json')
        if file_digest(directory / 'model.txt') != done['model_sha256']:
            raise RuntimeError('Stable model checksum failed.')
        leaf_audit(directory / 'model.txt', cfg, done['stability_audit']['initial_log_odds'])
        append_progress(event + ':fit', json.dumps(done, indent=2) + '\nLocal validation pending; leaderboard not submitted.')
        print('Stable model already complete.\n' + json.dumps(done, indent=2))
        return
    counts = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    rows = sum(counts)
    memory = rows * (4 * len(FEATURES) + 1)
    if rows == 0 or memory * 3 + 2**30 > psutil.virtual_memory().available:
        raise RuntimeError('No training rows or insufficient RAM for conservative fitting headroom.')
    append_progress(event + ':start',
        f'B01 stable_v2 regularization experiment started by the user. Output: {out.relative_to(ROOT)}\n'
        f'Cached data: {source.relative_to(ROOT)}; identical pairs, labels, features and held-out split.\n'
        f'Parameters: {json.dumps(params, sort_keys=True)}; fixed {cfg["trees"]} trees.\n'
        'Reason: baseline serialized tree updates reached 338,396.17, with probability-1 wrong assignments.\n'
        'Add L2 regularization, a minimum Hessian sum, and a strict per-tree update cap. No new training data or retrieval.\n'
        'Tune the decision threshold on the original tuning entities only. The original check result has already been reviewed; '
        'this is a diagnostic comparison on reused held-out entities, not a fresh confirmatory holdout.\n'
        'France quality is still unmeasured. Local validation: pending. Leaderboard: not submitted.')
    print(f'Load {rows:,} cached training pairs; X/y require {memory / 2**30:.2f} GiB.', flush=True)
    X, y = np.empty((rows, len(FEATURES)), np.float32), np.empty(rows, np.uint8)
    offset = 0
    with tqdm(total=rows, desc='Load cached training pairs', unit='pairs', dynamic_ncols=True) as bar:
        for path, count in zip(paths, counts):
            if count:
                table = pq.read_table(path, columns=['label', *FEATURES])
                for i, key in enumerate(FEATURES):
                    X[offset:offset + count, i] = table[key].to_numpy()
                y[offset:offset + count] = table['label'].to_numpy()
                offset += count
                bar.update(count)
    if offset != rows or not np.isfinite(X).all():
        raise RuntimeError('Invalid cached training matrix.')
    positive = int(y.sum())
    gc.collect()
    start = time.perf_counter()
    with tqdm(total=cfg['trees'], desc='Train stable matcher', unit='trees', dynamic_ncols=True) as bar:
        model, bias = fit_stable(X, y, cfg, directory, bar)
    path = directory / 'model.txt'
    partial = path.with_suffix('.txt.part')
    model.save_model(str(partial))
    audit = leaf_audit(partial, cfg, bias)
    partial.replace(path)
    with (directory / 'feature_importance.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['feature', 'gain', 'splits'])
        writer.writerows(zip(FEATURES, model.feature_importance('gain'), model.feature_importance('split')))
    done = {'experiment': 'B01 stable_v2', 'pairs': rows, 'positive_pairs': positive, 'negative_pairs': rows - positive,
            'trees': model.current_iteration(), 'requested_trees': cfg['trees'],
            'fit_seconds_this_invocation': time.perf_counter() - start, 'model_sha256': file_digest(path),
            'event': event, 'stability_audit': audit, 'scope': 'Refit complete. Validation pending; no leaderboard score.'}
    atomic_json(directory / 'complete.json', done)
    append_progress(event + ':fit', json.dumps(done, indent=2) + '\nLocal validation pending; leaderboard not submitted.')
    print(json.dumps(done, indent=2), flush=True)


if __name__ == '__main__':
    main()

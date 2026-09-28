"""C04 stage 3 (CPU, user-run): one shared LightGBM for India+US on Codex's 61 B01 features + 31 C04 features.

Fixed tree budget, tqdm bar per tree, a checkpoint every `checkpoint_trees` trees; an interrupted run resumes."""
import argparse
import csv
import json
import time
from c04_common import (settings, environment, stage_lock, require_manifest, atomic_json, read_json, code_hash,
                        append_progress, CLAUDE)
import lightgbm as lgb
import numpy as np
import psutil
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from b01_pair_features import FEATURES
from c04_features import FEATURES_V2
from c04_common import CODEX_SCRIPTS
import hashlib

ALL_FEATURES = FEATURES + FEATURES_V2


def model_parameters(cfg):
    return dict(objective='binary', metric='binary_logloss', learning_rate=cfg['learning_rate'], num_leaves=cfg['num_leaves'],
                min_data_in_leaf=cfg['min_data_in_leaf'], lambda_l2=cfg['lambda_l2'], min_sum_hessian_in_leaf=cfg['min_sum_hessian_in_leaf'],
                max_delta_step=cfg['max_delta_step'], feature_fraction=cfg['feature_fraction'], max_bin=cfg['max_bin'],
                num_threads=cfg['cpu_threads'], seed=cfg['seed'], deterministic=True, force_col_wise=True, verbosity=-1,
                feature_pre_filter=False)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    folders = [out / 'features' / c for c in cfg['countries']]
    for f in folders:
        if not (f / 'complete.json').is_file():
            raise RuntimeError(f'Run the build stage first: {f}')
    paths = [p for f in folders for p in sorted(f.glob('part-*.train.parquet'))]
    directory = out / 'model'
    _lock = stage_lock(directory)
    signature = {'stage': 'train', 'sources': {f.name: read_json(f / 'complete.json') for f in folders}, 'files': len(paths),
                 'parameters': model_parameters(cfg), 'trees': cfg['trees'], 'features': ALL_FEATURES,
                 'code': code_hash(CLAUDE / 'scripts' / 'c04_train.py', CLAUDE / 'scripts' / 'c04_features.py', CODEX_SCRIPTS / 'b01_pair_features.py')}
    require_manifest(directory, signature)
    if (directory / 'complete.json').is_file():
        print('Model already complete:\n' + (directory / 'complete.json').read_text())
        return
    counts = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    rows = int(sum(counts))
    need = rows * (len(ALL_FEATURES) * 4 + 1)
    avail = psutil.virtual_memory().available
    print(f'C04 TRAIN: {rows:,} pairs x {len(ALL_FEATURES)} features = {need / 2**30:.2f} GiB for X/y; '
          f'available RAM {avail / 2**30:.2f} GiB (LightGBM also needs its bins, ~{rows * len(ALL_FEATURES) / 2**30:.2f} GiB).', flush=True)
    if need * 2.5 + 2**30 > avail:
        raise RuntimeError('Not enough free RAM. Close other programs, or raise the WSL memory limit (see the runbook), then retry.')
    t0 = time.time()
    X = np.empty((rows, len(ALL_FEATURES)), np.float32)
    y = np.empty(rows, np.uint8)
    off = 0
    with tqdm(total=rows, desc='Load training pairs', unit='pairs', unit_scale=True, dynamic_ncols=True) as bar:
        for p, n in zip(paths, counts):
            if n:
                t = pq.read_table(p, columns=['label', *ALL_FEATURES])
                for i, f in enumerate(ALL_FEATURES):
                    X[off:off + n, i] = t[f].to_numpy()
                y[off:off + n] = t['label'].to_numpy()
                off += n
                bar.update(n)
    positives = int(y.sum())
    append_progress(cfg, 'train:start',
                    f"Experiment C04 training started by the user. Output: {out.relative_to(CLAUDE.parent)}\n"
                    f"Pairs {rows:,} ({positives:,} positives); features: Codex B01's 61 + C04's 31 ({', '.join(FEATURES_V2)}).\n"
                    f"Sampling: 1/{cfg['training_query_modulus']} eligible training queries (B01 used 1/64); positives + top-"
                    f"{cfg['train_negative_top_k']} negatives + {cfg['train_extra_negative_permille']/10:.0f}% of the rest.\n"
                    f"Model: LightGBM {json.dumps(model_parameters(cfg), sort_keys=True)}, {cfg['trees']} trees.\n"
                    'Validation: Codex B01 held-out tune/check entities and query isolation (unchanged). Leaderboard: not submitted.')
    ckpt = directory / 'checkpoint.json'
    init, done = None, 0
    if ckpt.is_file():
        saved = read_json(ckpt)
        init = directory / saved['file']
        if sha(init) != saved['sha256']:
            raise RuntimeError('Checkpoint checksum failed.')
        done = saved['trees']
        print(f'Resuming from checkpoint with {done} trees.', flush=True)
    bar = tqdm(total=cfg['trees'], initial=done, desc='LightGBM trees', unit='tree', dynamic_ncols=True)

    def callback(env):
        bar.update(1)
        trees = env.model.current_iteration()
        if trees % cfg['checkpoint_trees'] == 0 or env.iteration + 1 == env.end_iteration:
            path = directory / f'checkpoint-{trees:05d}.txt'
            part = path.with_suffix('.txt.part')
            env.model.save_model(str(part))
            part.replace(path)
            atomic_json(ckpt, {'file': path.name, 'trees': trees, 'sha256': sha(path)})
    callback.order, callback.before_iteration = 40, False
    print('LightGBM is binning the features (no progress bar for this step; ~1-3 minutes)...', flush=True)
    data = lgb.Dataset(X, label=y, feature_name=ALL_FEATURES, free_raw_data=True)
    model = lgb.train(model_parameters(cfg), data, num_boost_round=cfg['trees'] - done,
                      init_model=str(init) if init else None, callbacks=[callback])
    bar.close()
    model.save_model(str(directory / 'model.txt'))
    gain, split = model.feature_importance('gain'), model.feature_importance('split')
    with (directory / 'feature_importance.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'gain_share', 'splits'])
        for name, g, s in sorted(zip(ALL_FEATURES, gain / gain.sum(), split), key=lambda t: -t[1]):
            w.writerow([name, f'{g:.6f}', int(s)])
    top = sorted(zip(ALL_FEATURES, gain / gain.sum()), key=lambda t: -t[1])[:12]
    report = {'rows': rows, 'positives': positives, 'trees': model.current_iteration(), 'features': len(ALL_FEATURES),
              'model_sha256': sha(directory / 'model.txt'), 'seconds': round(time.time() - t0, 1), 'smoke': cfg['smoke_mode'],
              'top_gain': {k: round(float(v), 4) for k, v in top}}
    atomic_json(directory / 'complete.json', report)
    append_progress(cfg, 'train:fit', json.dumps(report, indent=2) + '\nNext: bash matcher/run_c04.sh evaluate. Leaderboard: not submitted.')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""C04 stage 2 (CPU, user-run): per-part training and evaluation feature files.

For every Codex candidate part of the country:
  train  Codex's deterministic query split with a larger sample (1/training_query_modulus of eligible queries).
         Kept pairs: all positives + negatives ranked <= train_negative_top_k by combo similarity + a deterministic
         extra-negative sample. Features: Codex's 61 B01 features (recomputed with retrieval/scripts/b01_pair_features.matrix,
         read-only import) + the 31 C04 features.
  eval   Exactly Codex's cached held-out rows (same query isolation and competitors). Only the 31 C04 features are
         written here; the 61 B01 features are read row-aligned from Codex's cached eval parts at evaluation time.
Resumable: one checkpoint per part; a changed input/setting/code refuses to resume."""
import argparse
import json
import time
from c04_common import (settings, environment, stage_lock, require_manifest, atomic_json, table_write, read_json,
                        identity, code_hash, codex_parts, stable_hash, CLAUDE, CODEX_SCRIPTS)
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from b01_pair_features import FEATURES, matrix
from c04_features import FEATURES_V2, gather_pairs


def split_queries(queries, pairs, roles, heldout_fold, modulus, seed):
    """Identical logic to Codex b01_match_data.split_queries (same evaluation set; larger training sample)."""
    owners = np.asarray(queries['owner_index'], np.int32)
    valid_owner = owners >= 0
    selected = (roles == 1) | (roles == 2)
    evaluate = np.zeros(len(owners), bool)
    evaluate[valid_owner] = selected[owners[valid_owner]]
    qlocal = np.asarray(pairs['query_index'], np.int64) - queries['query_index'][0]
    evaluate[qlocal[selected[np.asarray(pairs['ref_index'], np.int32)]]] = True
    eligible = (~evaluate) & (np.asarray(queries['owner_fold']) != heldout_fold)
    train = np.zeros(len(owners), bool)
    for i in np.flatnonzero(eligible):
        train[i] = stable_hash(queries['entity_id'][i], seed) % modulus == 0
    return train, evaluate


def reference_text(cfg, country, n_refs, src):
    """Codex-normalized S1 text as python lists aligned to ref_index (what Codex's matrix() expects)."""
    refs = pl.read_parquet(src / 'references.parquet', columns=['ref_index', 'entity_id'])
    folder = cfg['codex_prepared'] / 'train_source1'
    s1 = (pl.scan_parquet(sorted(str(p) for p in folder.glob('part-*.parquet'))).filter(pl.col('country') == country)
          .select('entity_id', 'name_core', 'name_full', 'address_clean', 'raw_address').collect())
    T = refs.join(s1, on='entity_id', how='left', maintain_order='left')
    if T.height != n_refs or T['name_core'].null_count():
        raise RuntimeError('Reference text alignment failed.')
    return {k: T[k].to_list() for k in ('name_core', 'name_full', 'address_clean', 'raw_address')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--country', choices=['India', 'US'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    environment()
    cfg, out = settings(args.smoke)
    src = cfg['codex_candidates'] / f'train_{args.country}'
    feat_src = cfg['codex_features'] / args.country
    parts = codex_parts(src, cfg['limit_parts'])
    prep = out / 'prepare' / args.country
    if not (prep / 'complete.json').is_file():
        raise RuntimeError(f'Run the prepare stage first: bash matcher/run_c04.sh prepare --country {args.country}')
    folder = out / 'features' / args.country
    _lock = stage_lock(folder)
    signature = {'stage': 'build', 'country': args.country, 'parts': len(parts),
                 'settings': {k: cfg[k] for k in ('seed', 'validation_fold', 'training_query_modulus', 'train_negative_top_k',
                                                  'train_extra_negative_permille', 'name_weight', 'limit_parts')},
                 'inputs': {'prepare': read_json(prep / 'complete.json'), 'codex_candidates': read_json(src / 'complete.json'),
                            'codex_features': read_json(feat_src / 'complete.json'), 'roles': identity(feat_src / 'references.parquet')},
                 'features': FEATURES + FEATURES_V2,
                 'code': code_hash(*(CLAUDE / 'scripts' / f for f in ('c04_build.py', 'c04_features.py', 'c04_common.py')),
                                   CODEX_SCRIPTS / 'b01_pair_features.py')}
    require_manifest(folder, signature)
    if (folder / 'complete.json').is_file():
        print('Reusing completed C04 features:\n' + (folder / 'complete.json').read_text())
        return
    t0 = time.time()
    roles = pl.read_parquet(feat_src / 'references.parquet', columns=['role'])['role'].to_numpy()
    qstats = pl.read_parquet(prep / 'queries.parquet')
    rstats = pl.read_parquet(prep / 'refs.parquet')
    vocab = pl.read_parquet(prep / 'vocab.parquet')
    if not np.array_equal(rstats['ref_index'].to_numpy(), np.arange(len(roles))):
        raise RuntimeError('Prepared reference stats are not aligned with Codex roles.')
    refs_text = reference_text(cfg, args.country, len(roles), src)
    print(f'C04 BUILD {args.country}: {len(parts)} parts; loaded stats in {time.time() - t0:.0f}s. CPU only.', flush=True)
    counts = dict(parts=0, train_queries=0, train_pairs=0, train_positive_pairs=0, eval_pairs=0)
    with tqdm(total=len(parts), desc=f'C04 features {args.country}', unit='parts', dynamic_ncols=True) as bar:
        for p in parts:
            qpath, ppath = src / f'part-{p:05d}.queries.parquet', src / f'part-{p:05d}.pairs.parquet'
            epath = feat_src / f'part-{p:05d}.eval.parquet'
            stamp = folder / f'part-{p:05d}.complete.json'
            expected = {'queries': identity(qpath), 'pairs': identity(ppath), 'codex_eval': identity(epath)}
            if stamp.is_file():
                saved = read_json(stamp)
                if saved['inputs'] != expected:
                    raise RuntimeError(f'Inputs changed after part {p} was built; use a fresh run_dir.')
                for kind in ('train', 'eval'):
                    path = folder / f'part-{p:05d}.{kind}.parquet'
                    if not path.is_file() or pq.ParquetFile(path).metadata.num_rows != saved['counts'][kind + '_pairs']:
                        raise RuntimeError(f'Incomplete checkpoint: {path}')
            else:
                queries = pq.read_table(qpath).to_pydict()
                pt = pq.read_table(ppath)
                pairs = {k: pt[k].to_numpy() for k in pt.column_names}
                train, evaluate = split_queries(queries, pairs, roles, cfg['validation_fold'], cfg['training_query_modulus'], cfg['seed'])
                local = pairs['query_index'] - queries['query_index'][0]
                owner = np.asarray(queries['owner_index'], np.int32)[local]
                label = (owner == pairs['ref_index']) & (owner >= 0)
                extra = ((pairs['query_index'].astype(np.uint64) * np.uint64(2654435761) + pairs['ref_index'].astype(np.uint64) * np.uint64(40503))
                         % np.uint64(1000)) < cfg['train_extra_negative_permille']
                keep = (train[local] & (roles[pairs['ref_index']] == 0)
                        & (label | (pairs['combo_rank'] <= cfg['train_negative_top_k']) | extra))
                sel = {k: v[keep] for k, v in pairs.items()}
                Xb = matrix(sel, queries, refs_text, cfg['name_weight'])
                Xv = gather_pairs(qstats, rstats, sel['query_index'], sel['ref_index'], vocab)
                cols = {'query_index': sel['query_index'], 'ref_index': sel['ref_index'], 'owner_index': owner[keep],
                        'label': label[keep].astype(np.uint8)}
                cols.update({f: Xb[:, i] for i, f in enumerate(FEATURES)})
                cols.update({f: Xv[:, i] for i, f in enumerate(FEATURES_V2)})
                table_write(folder / f'part-{p:05d}.train.parquet', pa.table(cols))
                ev = pq.read_table(epath, columns=['query_index', 'ref_index'])
                if ev.num_rows != int(evaluate[local].sum()):
                    raise RuntimeError(f'Part {p}: Codex eval rows ({ev.num_rows}) differ from the reproduced split ({int(evaluate[local].sum())}).')
                Xe = gather_pairs(qstats, rstats, ev['query_index'].to_numpy(), ev['ref_index'].to_numpy(), vocab)
                ecols = {'query_index': ev['query_index'].to_numpy(), 'ref_index': ev['ref_index'].to_numpy()}
                ecols.update({f: Xe[:, i] for i, f in enumerate(FEATURES_V2)})
                table_write(folder / f'part-{p:05d}.eval.parquet', pa.table(ecols))
                saved = {'inputs': expected, 'counts': {'train_queries': int(train.sum()), 'train_pairs': int(keep.sum()),
                                                        'train_positive_pairs': int(label[keep].sum()), 'eval_pairs': ev.num_rows}}
                atomic_json(stamp, saved)
            counts['parts'] += 1
            for k, v in saved['counts'].items():
                counts[k] += v
            bar.update(1)
            bar.set_postfix(train=f"{counts['train_pairs']:,}", pos=f"{counts['train_positive_pairs']:,}", eval=f"{counts['eval_pairs']:,}")
    report = dict(counts, country=args.country, features=len(FEATURES) + len(FEATURES_V2), seconds=round(time.time() - t0, 1),
                  smoke=cfg['smoke_mode'], sampling=f"1/{cfg['training_query_modulus']} eligible queries; positives + top-{cfg['train_negative_top_k']} "
                  f"negatives + {cfg['train_extra_negative_permille']/10:.0f}% of the rest")
    atomic_json(folder / 'complete.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

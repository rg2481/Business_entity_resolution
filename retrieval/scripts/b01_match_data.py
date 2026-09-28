"""User-run feature preparation with held-out reference entities and query isolation."""
import argparse
import json
from b01_common import atomic_json, code_hash, environment, identity
environment()
from b01_match_common import settings, require_manifest, read_json, table_write, stable_hash, chunk_files, stage_lock
from b01_pair_features import FEATURES, matrix
from b01_retrieval_pilot import prepared, dataset
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def select_roles(folds, heldout_fold, count, seed):
    eligible = np.flatnonzero(folds == heldout_fold)
    if count > len(eligible):
        raise ValueError('Requested validation population exceeds the held-out fold.')
    selected = np.random.default_rng(seed).permutation(eligible)[:count]
    roles = np.where(folds == heldout_fold, 3, 0).astype(np.int8)
    roles[selected[:count // 2]] = 1  # threshold tuning
    roles[selected[count // 2:]] = 2  # threshold check
    return roles


def split_queries(queries, pairs, roles, heldout_fold, modulus, seed):
    owners = np.asarray(queries['owner_index'], np.int32)
    valid_owner = owners >= 0
    selected = (roles == 1) | (roles == 2)
    evaluate = np.zeros(len(owners), bool)
    evaluate[valid_owner] = selected[owners[valid_owner]]
    qlocal = np.asarray(pairs['query_index'], np.int64) - queries['query_index'][0]
    candidate_held = selected[np.asarray(pairs['ref_index'], np.int32)]
    evaluate[qlocal[candidate_held]] = True
    eligible = (~evaluate) & (np.asarray(queries['owner_fold']) != heldout_fold)
    train = np.zeros(len(owners), bool)
    for i in np.flatnonzero(eligible):
        train[i] = stable_hash(queries['entity_id'][i], seed) % modulus == 0
    return train, evaluate


def feature_table(pairs, queries, refs, select, name_weight, progress):
    selected = {key: np.asarray(value)[select] for key, value in pairs.items()}
    X = matrix(selected, queries, refs, name_weight, progress)
    local = selected['query_index'] - queries['query_index'][0]
    owner = np.asarray(queries['owner_index'], np.int32)[local]
    label = ((owner == selected['ref_index']) & (owner >= 0)).astype(np.uint8)
    columns = {'query_index': selected['query_index'], 'ref_index': selected['ref_index'],
               'owner_index': owner, 'label': label}
    columns.update({key: X[:, i] for i, key in enumerate(FEATURES)})
    return pa.table(columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--base-config')
    parser.add_argument('--country', choices=['India', 'US'], required=True)
    args = parser.parse_args()
    cfg, base, run, out = settings(args.config, args.base_config)
    source = out / 'candidates' / f'train_{args.country}'
    chunks = chunk_files(source)
    candidate_report = read_json(source / 'report.json')
    if candidate_report['fold0_recall'] < cfg['minimum_fold0_recall']:
        raise RuntimeError('Candidate recall gate failed. Inspect retrieval misses first.')
    directory = out / 'features' / args.country
    _lock = stage_lock(directory)
    signature = {'candidate_manifest': read_json(source / 'manifest.json'),
                 'references': identity(source / 'references.parquet'),
                 'candidate_chunks': [identity(p) for p in chunks],
                 'settings': {k: cfg[k] for k in ('seed', 'validation_entities_per_country', 'training_query_modulus')},
                 'code': code_hash('b01_match_data.py', 'b01_pair_features.py', 'b01_match_common.py'), 'features': FEATURES}
    require_manifest(directory, signature)
    if (directory / 'complete.json').is_file():
        if not (directory / 'references.parquet').is_file():
            raise RuntimeError('Missing validation reference checkpoint.')
        for marker in chunk_files(directory):
            saved = read_json(marker)
            for suffix in ('train.parquet', 'eval.parquet'):
                path = marker.with_name(marker.name.replace('complete.json', suffix))
                kind = suffix.split('.')[0]
                if not path.is_file() or pq.ParquetFile(path).metadata.num_rows != saved['counts'][kind + '_pairs']:
                    raise RuntimeError(f'Missing feature checkpoint: {path}')
        print('Reusing completed features.\n' + (directory / 'complete.json').read_text())
        return
    reference = pq.read_table(source / 'references.parquet')
    roles = select_roles(reference['fold'].to_numpy(), base['validation_fold'],
                         cfg['validation_entities_per_country'], cfg['seed'])
    role_path = directory / 'references.parquet'
    if role_path.exists():
        stored = pq.read_table(role_path, columns=['role'])['role'].to_numpy()
        if not np.array_equal(stored, roles):
            raise RuntimeError('Validation selection changed.')
    else:
        table_write(role_path, reference.append_column('role', pa.array(roles)))
    # Arrow holds the full country references, not Python dictionaries per row.
    # Convert only the four columns required by the bounded feature cache.
    print('Load prepared reference text, preserving the cached retrieval-index row order.', flush=True)
    columns = ['entity_id', 'name_core', 'name_full', 'address_clean', 'raw_address']
    scanner = dataset(prepared(run, 'train', 1)).scanner(columns=columns, filter=ds.field('country') == args.country,
                                                       batch_size=base['batch_rows'], use_threads=False)
    pieces = []
    with tqdm(total=reference.num_rows, desc='Reference text', unit='refs', dynamic_ncols=True) as bar:
        for batch in scanner.to_batches():
            if batch.num_rows:
                pieces.append(pa.Table.from_batches([batch]))
                bar.update(batch.num_rows)
    texts = pa.concat_tables(pieces)
    if not texts['entity_id'].combine_chunks().equals(reference['entity_id'].combine_chunks()):
        raise RuntimeError('Reference text/index row alignment failed.')
    refs = {key: texts[key].to_pylist() for key in columns if key != 'entity_id'}
    del texts, pieces, reference
    counts = dict(chunks=0, train_queries=0, eval_queries=0, train_pairs=0, eval_pairs=0, train_positive_pairs=0, eval_positive_pairs=0)
    with tqdm(total=candidate_report['queries'], desc=f'Feature queries {args.country}', unit='queries', dynamic_ncols=True) as bar:
        for number, marker in enumerate(chunks):
            qpath = source / f'part-{number:05d}.queries.parquet'
            ppath = source / f'part-{number:05d}.pairs.parquet'
            stamp = directory / f'part-{number:05d}.complete.json'
            expected = {'queries_input': identity(qpath), 'pairs_input': identity(ppath)}
            if stamp.exists():
                saved = read_json(stamp)
                if saved['inputs'] != expected:
                    raise RuntimeError('Candidate files changed after feature preparation.')
                for kind in ('train', 'eval'):
                    path = directory / f'part-{number:05d}.{kind}.parquet'
                    if not path.is_file() or pq.ParquetFile(path).metadata.num_rows != saved['counts'][kind + '_pairs']:
                        raise RuntimeError(f'Incomplete feature checkpoint: {path}')
            else:
                queries = pq.read_table(qpath).to_pydict()
                pt = pq.read_table(ppath)
                pairs = {key: pt[key].to_numpy() for key in pt.column_names}
                train, evaluate = split_queries(queries, pairs, roles, base['validation_fold'],
                                                cfg['training_query_modulus'], cfg['seed'])
                local = pairs['query_index'] - queries['query_index'][0]
                train_pairs = train[local] & (roles[pairs['ref_index']] == 0)
                eval_pairs = evaluate[local]
                if np.any(train & evaluate) or np.any(train_pairs & eval_pairs):
                    raise RuntimeError('Training/validation query overlap.')
                stat = dict(chunks=1, train_queries=int(train.sum()), eval_queries=int(evaluate.sum()))
                with tqdm(total=int(train_pairs.sum() + eval_pairs.sum()), desc=f'Chunk {number + 1} pair features',
                          unit='pairs', leave=False, dynamic_ncols=True) as pairbar:
                    for kind, keep in (('train', train_pairs), ('eval', eval_pairs)):
                        table = feature_table(pairs, queries, refs, keep, base['name_weight'], pairbar)
                        table_write(directory / f'part-{number:05d}.{kind}.parquet', table)
                        stat[kind + '_pairs'] = table.num_rows
                        stat[kind + '_positive_pairs'] = int(table['label'].to_numpy().sum())
                saved = {'inputs': expected, 'queries': len(queries['entity_id']), 'counts': stat}
                atomic_json(stamp, saved)
            for key, value in saved['counts'].items():
                counts[key] += value
            bar.update(saved['queries'])
            bar.set_postfix(train_pairs=counts['train_pairs'], eval_pairs=counts['eval_pairs'])
    report = dict(counts, country=args.country, feature_count=len(FEATURES),
                  heldout_entities=cfg['validation_entities_per_country'],
                  train_sampling=f"Deterministic 1/{cfg['training_query_modulus']} eligible target queries; then all non-fold0 candidates.",
                  validation='Uniform fold0 references, half tune/half check. Every owner/candidate-touching query is isolated from training; all its competing candidates are retained.',
                  scope='Feature preparation only; no model fitted. progress.txt remains untouched.')
    atomic_json(directory / 'complete.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

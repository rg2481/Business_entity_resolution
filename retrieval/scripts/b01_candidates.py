"""User-run, resumable full-country candidate retrieval. No classifier or upload."""
import argparse
import gc
import hashlib
import json
import time
from b01_common import allowed_states, atomic_json, code_hash, environment, fold_of, identity
environment()
from b01_match_common import settings, require_manifest, table_write, read_json, stage_lock
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm
from b01_retrieval_pilot import prepared, dataset, index_for, vectorizer, binary_vectors, weighted_vectors, gpu_csr
from b01_gpu_search import retrieve_batch

MODES = ('name', 'address', 'combo')


def union_candidates(result, references, offset):
    """Deduplicate finite top-k candidates, preserving all component scores/ranks."""
    first = result['name'][0]
    count, k = first.shape
    key_parts, names, addresses, modes, ranks = [], [], [], [], []
    context = {}
    for mode_no, mode in enumerate(MODES):
        ids, scores, ns, ads = result[mode]
        finite = np.isfinite(scores)
        q = np.broadcast_to(np.arange(count)[:, None], ids.shape)
        key_parts.append((q[finite] * references + ids[finite]).astype(np.int64))
        names.append(ns[finite]); addresses.append(ads[finite])
        modes.append(np.full(finite.sum(), mode_no, np.int8))
        ranks.append(np.broadcast_to(np.arange(1, k + 1), ids.shape)[finite])
        context[mode + '_best'] = np.where(finite[:, 0], scores[:, 0], -1).astype(np.float32)
        context[mode + '_second'] = (np.where(finite[:, 1], scores[:, 1], -1).astype(np.float32)
                                     if k > 1 else np.full(count, -1, np.float32))
    keys, take, inverse = np.unique(np.concatenate(key_parts), return_index=True, return_inverse=True)
    all_modes, all_ranks = np.concatenate(modes), np.concatenate(ranks)
    data = {'query_index': keys // references + offset, 'ref_index': (keys % references).astype(np.int32),
            'name_score': np.concatenate(names)[take].astype(np.float32),
            'address_score': np.concatenate(addresses)[take].astype(np.float32)}
    for mode_no, mode in enumerate(MODES):
        rank = np.full(len(keys), k + 1, np.int16)
        select = all_modes == mode_no
        np.minimum.at(rank, inverse[select], all_ranks[select])
        data[mode + '_rank'] = rank
    return data, context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--base-config')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--country', choices=['India', 'US', 'France'], required=True)
    args = parser.parse_args()
    cfg, base, run, out = settings(args.config, args.base_config)
    if args.split == 'train' and args.country == 'France':
        parser.error('There are no labeled France training references.')
    base = dict(base, query_batch=cfg['query_batch'], top_k=cfg['top_k'])
    paths = prepared(run, args.split, 2) + prepared(run, args.split, 3)
    directory = out / 'candidates' / f'{args.split}_{args.country}'
    _lock = stage_lock(directory)
    index = index_for(run, base, args.split, args.country)
    signature = {'inputs': [identity(p) for p in paths], 'index': read_json(index / 'manifest.json'),
                 'split': args.split, 'country': args.country,
                 'config': {k: cfg[k] for k in ('query_batch', 'scan_batch_rows', 'top_k')},
                 'scoring': {k: base[k] for k in ('hash_features', 'name_weight', 'state_filter', 'folds', 'validation_fold')},
                 'code': code_hash('b01_candidates.py', 'b01_match_common.py', 'b01_gpu_search.py')}
    require_manifest(directory, signature)
    if (directory / 'complete.json').exists():
        done = read_json(directory / 'complete.json')
        markers = sorted(directory.glob('part-*.complete.json'))
        if len(markers) != done['chunks'] or not (directory / 'references.parquet').is_file():
            raise RuntimeError('Completed candidate directory is missing files.')
        for marker in markers:
            saved = read_json(marker)
            for suffix, key in (('queries.parquet', 'queries'), ('pairs.parquet', 'pairs')):
                path = marker.with_name(marker.name.replace('complete.json', suffix))
                if not path.is_file() or pq.ParquetFile(path).metadata.num_rows != saved[key]:
                    raise RuntimeError(f'Missing checkpoint file beside {marker}')
        report = read_json(directory / 'report.json')
        print('Reusing completed candidates; no GPU search started.\n' + json.dumps(report, indent=2))
        if report['fold0_recall'] is not None and report['fold0_recall'] < cfg['minimum_fold0_recall']:
            raise RuntimeError('Candidate recall gate failed; inspect report before training.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run in your normal WSL terminal; no CPU brute-force fallback.')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.cuda.set_per_process_memory_fraction(base['gpu_memory_fraction'])
    print('Load country reference IDs and cached CSR matrices. No full-source dense matrix is allocated.', flush=True)
    metadata = pq.read_table(index / 'metadata.parquet').to_pydict()
    ref_ids = metadata['entity_id']
    nref = len(ref_ids)
    ref_lookup = {v: i for i, v in enumerate(ref_ids)} if args.split == 'train' else {}
    folds = np.fromiter((fold_of(v, base['folds']) for v in tqdm(ref_ids, desc='Reference folds', unit='refs')),
                        dtype=np.int8, count=nref)
    with tqdm(total=4, desc='Load GPU index', unit='parts') as bar:
        pn = gpu_csr(sp.load_npz(index / 'name.npz')); bar.update(1)
        pad = gpu_csr(sp.load_npz(index / 'address.npz')); bar.update(1)
        with np.load(index / 'idf.npz') as values:
            idf_name, idf_address = values['name'].copy(), values['address'].copy()
        bar.update(1)
        states = torch.tensor(metadata['reference_state_mask'], device='cuda', dtype=torch.int64)
        ambiguous = (states == 0) | ((states & (states - 1)) != 0)
        bar.update(1)
    del metadata
    gc.collect()
    hv = vectorizer(base)
    data, filt = dataset(paths), ds.field('country') == args.country
    print('Count target rows (country-filter pass); then retrieve every target, including distractors.', flush=True)
    total = data.count_rows(filter=filt)
    columns = ['entity_id', 'country', 'source', 'raw_name', 'raw_address', 'name_full', 'name_core', 'address_clean',
               'owner_id', 'owner_fold', 'name_script', 'unknown_native_tokens']
    scanner = data.scanner(columns=columns, filter=filt, batch_size=cfg['scan_batch_rows'], use_threads=False)
    truth = np.zeros(nref, np.int32)
    recovered = np.zeros(nref, np.int32)
    offset = pair_count = chunk_count = cached_queries = 0
    new_seconds = 0.0
    with tqdm(total=total, desc=f'Candidates {args.split}/{args.country}', unit='queries', dynamic_ncols=True) as bar:
        for batch in scanner.to_batches():
            if not batch.num_rows:
                continue
            stamp = directory / f'part-{chunk_count:05d}.complete.json'
            queries_path = directory / f'part-{chunk_count:05d}.queries.parquet'
            pairs_path = directory / f'part-{chunk_count:05d}.pairs.parquet'
            d = batch.to_pydict()
            count = batch.num_rows
            expected = {'offset': offset, 'queries': count,
                        'id_hash': hashlib.sha256('\n'.join(d['entity_id']).encode()).hexdigest()}
            if args.split == 'train':
                owners = np.array([ref_lookup[v] if v else -1 for v in d['owner_id']], dtype=np.int32)
            else:
                owners = np.full(count, -1, np.int32)
            np.add.at(truth, owners[owners >= 0], 1)
            if stamp.exists():
                saved = read_json(stamp)
                if any(saved[k] != v for k, v in expected.items()):
                    raise RuntimeError('Query order changed; cannot resume these checkpoints.')
                for path, rows in ((queries_path, count), (pairs_path, saved['pairs'])):
                    if not path.exists() or pq.ParquetFile(path).metadata.num_rows != rows:
                        raise RuntimeError(f'Incomplete or damaged checkpoint: {path}')
                pair = pq.read_table(pairs_path, columns=['query_index', 'ref_index']).to_pydict()
                qidx = np.asarray(pair['query_index'], np.int64) - offset
                ridx = np.asarray(pair['ref_index'], np.int32)
                cached_queries += count
                bar.update(count)
            else:
                start = time.perf_counter()
                pieces, contexts = [], {f'{mode}_{key}': [] for mode in MODES for key in ('best', 'second')}
                for a in range(0, count, cfg['query_batch']):
                    b = min(count, a + cfg['query_batch'])
                    qn = weighted_vectors(binary_vectors(hv, d['name_core'][a:b]), idf_name)
                    qa = weighted_vectors(binary_vectors(hv, d['address_clean'][a:b]), idf_address)
                    results = retrieve_batch(pn, pad, qn, qa, states, ambiguous,
                                             [allowed_states(v) for v in d['name_script'][a:b]], base)
                    pairs, context = union_candidates(results, nref, offset + a)
                    pieces.append(pa.table(pairs))
                    for key, value in context.items():
                        contexts[key].append(value)
                    bar.update(b - a)
                table = pa.concat_tables(pieces)
                qidx = table['query_index'].to_numpy() - offset
                ridx = table['ref_index'].to_numpy()
                query_table = pa.Table.from_batches([batch])
                query_table = query_table.append_column('query_index', pa.array(np.arange(offset, offset + count, dtype=np.int64)))
                query_table = query_table.append_column('owner_index', pa.array(owners))
                for key, value in contexts.items():
                    query_table = query_table.append_column(key, pa.array(np.concatenate(value)))
                table_write(queries_path, query_table)
                table_write(pairs_path, table)
                elapsed = time.perf_counter() - start
                new_seconds += elapsed
                atomic_json(stamp, dict(expected, pairs=table.num_rows, seconds=elapsed))
            hit = (owners[qidx] == ridx) & (owners[qidx] >= 0)
            # Each query/reference occurs once in the deduplicated union.
            np.add.at(recovered, ridx[hit], 1)
            pair_count += len(ridx)
            offset += count
            chunk_count += 1
            bar.set_postfix(cached=cached_queries, chunks=chunk_count)
    if offset != total or np.any(recovered > truth):
        raise RuntimeError('Candidate coverage accounting failed.')
    selected = folds == base['validation_fold']
    matched = int(truth.sum())
    val_matched = int(truth[selected].sum())
    oracle = np.ones(nref, np.float64)
    nonempty = truth > 0
    oracle[nonempty] = 5 * recovered[nonempty] / (4.0 * recovered[nonempty] + truth[nonempty])
    report = {'split': args.split, 'country': args.country, 'queries': offset, 'references': nref,
              'candidate_pairs': pair_count, 'average_candidates': pair_count / max(1, offset),
              'matched_targets': matched if args.split == 'train' else None,
              'union_recall': float(recovered.sum() / matched) if matched else None,
              'fold0_recall': float(recovered[selected].sum() / val_matched) if val_matched else None,
              'fold0_candidate_oracle_macro_f05': float(oracle[selected].mean()) if val_matched else None,
              'cached_queries': cached_queries, 'new_search_and_checkpoint_seconds': new_seconds,
              'scope': 'Full-country retrieval union. Oracle assumes perfect matching decisions; it is not a model score.'}
    table_write(directory / 'references.parquet', pa.table({'ref_index': np.arange(nref, dtype=np.int32), 'entity_id': ref_ids,
                'fold': folds, 'truth_count': truth, 'retrieved_true_count': recovered}))
    atomic_json(directory / 'report.json', report)
    atomic_json(directory / 'complete.json', {'chunks': chunk_count, 'queries': offset, 'pairs': pair_count})
    print(json.dumps(report, indent=2), flush=True)
    print('Candidates saved. No training, inference or submission started. progress.txt is untouched.', flush=True)
    if val_matched and report['fold0_recall'] < cfg['minimum_fold0_recall']:
        raise RuntimeError('Candidate recall gate failed; inspect misses before training.')


if __name__ == '__main__':
    main()

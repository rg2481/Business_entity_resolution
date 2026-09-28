"""Frozen-model test-run settings and bounded ID handling; imports start no job."""
from pathlib import Path
import numpy as np
from b01_common import ROOT, CODEX, code_hash, identity, environment
environment()
from b01_match_common import settings, read_json, file_digest, require_manifest
from b01_pair_features import FEATURES
from b01_train_stable import leaf_audit, stable_parameters
from b01_retrieval_pilot import prepared

DEFAULT_CONFIG = 'retrieval/configs/b01_test.json'
MATCH_HEADER = b'source1_entity_id\tmatched_entity_ids\n'
CANDIDATE_HEADER = b'source1_entity_id\tcandidate_entity_ids\n'
PAIR_DTYPE = np.dtype([('ref', '<i4'), ('query', '<i8'), ('matched', 'u1')])


def packed_ids(values, width, prefixes):
    encoded = [v.encode('ascii') if isinstance(v, str) else bytes(v) for v in values]
    for value in encoded:
        if (not value.startswith(prefixes) or len(value) > width or value.strip() != value
                or any(c in value for c in (b',', b'\t', b'\r', b'\n', b'"', b'\x00'))):
            raise RuntimeError(f'Invalid or oversized ID: {value!r}; no ID will be truncated.')
    return np.asarray(encoded, dtype=f'S{width}')


def load_test_settings(path=None):
    cfg, base, run, out = settings(path or ROOT / DEFAULT_CONFIG)
    model_cfg, model_base, model_run, model_out = settings(ROOT / cfg['model_config'])
    cached = ROOT / read_json(ROOT / model_cfg['cached_matcher_config'])['experiment_dir']
    if (base != model_base or run != model_run or out == model_out or out == cached.resolve()
            or out.is_relative_to(model_out) or out.is_relative_to(cached.resolve())):
        raise ValueError('Test run needs the frozen model preparation and a separate output directory.')
    if cfg['top_k'] != model_cfg['top_k']:
        raise ValueError('Test candidate top-k must match the validated matcher.')
    if not cfg['countries'] or len(set(cfg['countries'])) != len(cfg['countries']):
        raise ValueError('Unique nonempty country list required.')
    if set(cfg['countries']) - {'France', 'India', 'US'}:
        raise ValueError('An additional country requires extending the frozen retrieval driver first.')
    for key in ('export_reference_block', 'export_memory_mib', 'id_bytes', 'validation_batch_rows'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    model_path = model_out / 'model/model.txt'
    done = read_json(model_out / 'model/complete.json')
    manifest = read_json(model_out / 'model/manifest.json')
    report = read_json(model_out / 'evaluation/report.json')
    model_sha = file_digest(model_path)
    if model_sha != done['model_sha256'] or model_sha != report['model_sha256']:
        raise RuntimeError('Shared model does not match its validation report.')
    expected_code = code_hash('b01_train_stable.py', 'b01_train_matcher.py', 'b01_pair_features.py', 'b01_match_common.py')
    if manifest['code'] != expected_code or manifest['features'] != FEATURES or manifest['parameters'] != stable_parameters(model_cfg):
        raise RuntimeError('Frozen stable-model code, parameters or feature definitions changed.')
    leaf_audit(model_path, model_cfg, done['stability_audit']['initial_log_odds'])
    threshold = float(report['threshold'])
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., 2.):
        raise RuntimeError('Invalid validated threshold.')
    plan = {'config': cfg, 'base': base, 'model_sha256': model_sha, 'threshold': threshold,
            'model_report': identity(model_out / 'evaluation/report.json'),
            'prepared': {str(s): [identity(p) for p in prepared(run, 'test', s)] for s in (1, 2, 3)},
            'raw_test_files': [identity(ROOT / cfg['test_dir'] / f'test_source{s}.tsv') for s in (1, 2, 3)],
            'common_code': code_hash('b01_test_common.py')}
    require_manifest(out, plan)
    return cfg, base, run, out, model_path, model_sha, threshold


def candidate_signature(folder, cfg, base, run, country):
    saved = read_json(folder / 'manifest.json')
    expected = {'inputs': [identity(p) for s in (2, 3) for p in prepared(run, 'test', s)],
                'index': read_json(run / f'indexes/test_{country}/manifest.json'), 'split': 'test', 'country': country,
                'config': {k: cfg[k] for k in ('query_batch', 'scan_batch_rows', 'top_k')},
                'scoring': {k: base[k] for k in ('hash_features', 'name_weight', 'state_filter', 'folds', 'validation_fold')},
                'code': code_hash('b01_candidates.py', 'b01_match_common.py', 'b01_gpu_search.py')}
    if saved != expected or saved['index']['prepared'] != [identity(p) for p in prepared(run, 'test', 1)]:
        raise RuntimeError('Test candidates differ from frozen retrieval/preparation settings.')
    return saved


def check_country(cfg, country):
    if country not in cfg['countries']:
        raise ValueError(f'Country {country!r} not configured for this test run.')


def validate_pair_order(pairs, queries, reference_count):
    count = len(queries['entity_id'])
    if not count:
        raise RuntimeError('Empty candidate query checkpoint.')
    offset = int(queries['query_index'][0])
    if not np.array_equal(queries['query_index'], np.arange(offset, offset + count)):
        raise RuntimeError('Noncontiguous query ordinals.')
    qi, ri = pairs['query_index'], pairs['ref_index']
    if np.any(qi < offset) or np.any(qi >= offset + count) or np.any(ri < 0) or np.any(ri >= reference_count):
        raise RuntimeError('Candidate query/reference ordinal out of bounds.')
    keys = (qi - offset) * reference_count + ri
    if np.any(keys[1:] <= keys[:-1]):
        raise RuntimeError('Candidate union must be ordered and deduplicated by query/reference.')
    return offset, keys

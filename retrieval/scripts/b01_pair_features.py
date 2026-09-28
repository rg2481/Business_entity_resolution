"""Name/address/number features; no IDs, ownership or labels enter the model."""
from functools import lru_cache
import math
import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

FEATURES = [
    'name_cosine', 'address_cosine', 'combo_cosine', 'name_rank', 'address_rank', 'combo_rank',
    'name_best', 'name_second', 'name_to_best', 'name_top_margin',
    'address_best', 'address_second', 'address_to_best', 'address_top_margin',
    'combo_best', 'combo_second', 'combo_to_best', 'combo_top_margin',
    'name_ratio', 'name_sort_ratio', 'name_set_ratio', 'name_compact_ratio', 'name_full_ratio',
    'name_exact', 'name_jaccard', 'name_query_coverage', 'name_ref_coverage', 'name_initials_equal',
    'name_length_ratio', 'name_query_tokens', 'name_ref_tokens',
    'address_ratio', 'address_sort_ratio', 'address_set_ratio', 'address_compact_ratio', 'address_exact',
    'address_jaccard', 'address_query_coverage', 'address_ref_coverage',
    'address_query_missing', 'address_ref_missing', 'address_length_ratio',
    'address_query_tokens', 'address_ref_tokens',
    'number_query_count', 'number_ref_count', 'number_intersection', 'number_jaccard',
    'number_query_coverage', 'number_ref_coverage', 'number_sets_equal',
    'number_first_equal', 'number_first_edit', 'number_best_edit', 'number_first_log_difference',
    'number_first_same_length', 'number_first_prefix', 'number_first_suffix',
    'query_has_native_name', 'query_unknown_native_tokens', 'query_source',
]
DIGITS = re.compile(r'\d+')


@lru_cache(maxsize=50000)
def parsed(name, full, address, raw_address):
    nt, at = frozenset(name.split()), frozenset(address.split())
    # Python int also handles Unicode decimal digits. Leading zeros do not
    # distinguish house numbers, but raw text remains available in checkpoints.
    nums = tuple(str(int(v)) for v in DIGITS.findall(raw_address) if len(v) < 20)
    return (name, full, address, nt, at, name.replace(' ', ''), address.replace(' ', ''),
            ''.join(v[0] for v in name.split() if v), nums, frozenset(nums))


def overlap(a, b):
    both = len(a & b)
    return both / max(1, len(a | b)), both / max(1, len(a)), both / max(1, len(b))


def length_ratio(a, b):
    return min(len(a), len(b)) / max(1, len(a), len(b))


def text_features(q, r, native, unknown, source):
    qn, qfull, qa, qnt, qat, qnc, qac, qi, qnums, qns = q
    rn, rfull, ra, rnt, rat, rnc, rac, ri, rnums, rns = r
    qfirst, rfirst = (qnums[0] if qnums else ''), (rnums[0] if rnums else '')
    has_first = bool(qfirst and rfirst)
    first_edit = Levenshtein.normalized_similarity(qfirst, rfirst) if has_first else 0.0
    best_edit = max((Levenshtein.normalized_similarity(a, b) for a in qns for b in rns), default=0.0)
    values = [
        fuzz.ratio(qn, rn) / 100, fuzz.token_sort_ratio(qn, rn) / 100,
        fuzz.token_set_ratio(qn, rn) / 100, fuzz.ratio(qnc, rnc) / 100, fuzz.ratio(qfull, rfull) / 100,
        float(bool(qn) and qn == rn), *overlap(qnt, rnt), float(bool(qi) and qi == ri),
        length_ratio(qn, rn), len(qnt), len(rnt),
        fuzz.ratio(qa, ra) / 100, fuzz.token_sort_ratio(qa, ra) / 100,
        fuzz.token_set_ratio(qa, ra) / 100, fuzz.ratio(qac, rac) / 100,
        float(bool(qa) and qa == ra), *overlap(qat, rat), float(not qa), float(not ra),
        length_ratio(qa, ra), len(qat), len(rat),
        len(qns), len(rns), len(qns & rns), *overlap(qns, rns), float(bool(qns) and qns == rns),
        float(has_first and qfirst == rfirst), first_edit, best_edit,
        math.log1p(abs(int(qfirst) - int(rfirst))) if has_first else -1.0,
        float(has_first and len(qfirst) == len(rfirst)),
        float(has_first and (qfirst.startswith(rfirst) or rfirst.startswith(qfirst))),
        float(has_first and (qfirst.endswith(rfirst) or rfirst.endswith(qfirst))),
        float(bool(native)), unknown, source,
    ]
    return values


def matrix(pairs, queries, refs, name_weight, progress=None):
    n = len(pairs['ref_index'])
    result = np.empty((n, len(FEATURES)), dtype=np.float32)
    qlocal = np.asarray(pairs['query_index'], np.int64) - queries['query_index'][0]
    ridx = np.asarray(pairs['ref_index'], np.int32)
    ns, ads = np.asarray(pairs['name_score']), np.asarray(pairs['address_score'])
    cs = name_weight * ns + (1 - name_weight) * ads
    result[:, :6] = np.column_stack([ns, ads, cs, pairs['name_rank'], pairs['address_rank'], pairs['combo_rank']])
    for i, (mode, scores) in enumerate(zip(('name', 'address', 'combo'), (ns, ads, cs))):
        best = np.asarray(queries[mode + '_best'])[qlocal]
        second = np.asarray(queries[mode + '_second'])[qlocal]
        result[:, 6 + 4 * i:10 + 4 * i] = np.column_stack([best, second, scores - best, best - second])
    qcache = {}
    for i, (qj, rj) in enumerate(zip(qlocal, ridx)):
        if qj not in qcache:
            qcache[qj] = parsed(*(queries[key][qj] for key in ('name_core', 'name_full', 'address_clean', 'raw_address')))
        ref = parsed(*(refs[key][rj] for key in ('name_core', 'name_full', 'address_clean', 'raw_address')))
        result[i, 18:] = text_features(qcache[qj], ref, queries['name_script'][qj],
                                      queries['unknown_native_tokens'][qj], queries['source'][qj])
        if progress is not None and (i + 1) % 2048 == 0:
            progress.update(2048)
    if progress is not None:
        progress.update(n % 2048)
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite matcher feature.')
    return result

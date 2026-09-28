"""Official per-reference F0.5 and a separately labeled distractor-FP stress curve."""
import numpy as np


def winners(query_index, ref_index, owner_index, scores):
    """One winner per target over ALL competitors; stable reference-ordinal ties."""
    if not len(scores):
        return {key: np.array([], dtype=dtype) for key, dtype in (
            ('query_index', np.int64), ('ref_index', np.int32), ('owner_index', np.int32),
            ('score', np.float64), ('runnerup_score', np.float64))}
    order = np.lexsort((ref_index, -scores, query_index))
    qi, ri, oi, sc = query_index[order], ref_index[order], owner_index[order], scores[order]
    starts = np.r_[0, np.flatnonzero(qi[1:] != qi[:-1]) + 1]
    ends = np.r_[starts[1:], len(qi)]
    runnerup = np.full(len(starts), -1.0)
    multiple = ends - starts > 1
    runnerup[multiple] = sc[starts[multiple] + 1]
    return {'query_index': qi[starts], 'ref_index': ri[starts], 'owner_index': oi[starts],
            'score': sc[starts], 'runnerup_score': runnerup}


def threshold_grid():
    # Includes an all-reject decision even if a model returns probability 1.
    return np.unique(np.r_[np.linspace(0, 1, 201), .9975, .999, .9995, .9999, np.nextafter(1.0, 2.0)])


def accumulate_hist(hist, selected_map, predicted, owners, scores, thresholds):
    local = selected_map[predicted]
    keep = local >= 0
    local, predicted, owners, scores = local[keep], predicted[keep], owners[keep], scores[keep]
    bins = np.searchsorted(thresholds, scores, side='right') - 1
    if np.any(bins < 0):
        raise ValueError('Invalid predicted probability')
    kind = np.where(owners == predicted, 0, np.where(owners < 0, 1, 2))  # TP, unmatched FP, wrong-owner FP
    np.add.at(hist, (kind, local, bins), 1)


def metric_curves(hist, truth, stress_weight):
    counts = np.cumsum(hist[:, :, ::-1], axis=2, dtype=np.int64)[:, :, ::-1]
    tp, unmatched, wrong = counts
    if np.any(tp > truth[:, None]):
        raise RuntimeError('Duplicate positive predictions or incorrect ground-truth counts.')
    fp = unmatched + wrong
    denominator = 4.0 * tp + 4 * fp + truth[:, None]
    official = np.divide(5.0 * tp, denominator, out=np.ones_like(denominator), where=denominator != 0)
    weighted = 4.0 * tp + 4 * (stress_weight * unmatched + wrong) + truth[:, None]
    stress = np.divide(5.0 * tp, weighted, out=np.ones_like(weighted), where=weighted != 0)
    return counts, official, stress


def choose_threshold(curve, thresholds):
    best = np.flatnonzero(np.isclose(curve, np.max(curve), rtol=0, atol=1e-12))
    return int(best[-1])  # prefer the stricter threshold only when scores tie


def at_threshold(counts, official, stress, truth, keep, index):
    tp, unmatched, wrong = counts[:, keep, index].sum(axis=1)
    nonempty = truth[keep] > 0
    pred = counts[:, keep, index].sum(axis=0)
    singleton = ~nonempty
    return {'entities': int(keep.sum()), 'macro_f05': float(official[keep, index].mean()),
            'distractor_fp_stress_macro_f05': float(stress[keep, index].mean()),
            'tp': int(tp), 'fp_unmatched': int(unmatched), 'fp_wrong_owner': int(wrong),
            'fn': int(truth[keep].sum() - tp), 'zero_truth_entities': int(singleton.sum()),
            'zero_truth_false_merge_rate': float((pred[singleton] > 0).mean()) if singleton.any() else None}

"""C04 V2 pair features, vectorized with polars + rapidfuzz. No IDs, owners or labels enter the features.

Families (each aimed at a measured B01 error bucket, see claude_analysys.txt E1.5-E1.6):
  name relation  - how the record's core-name words relate to the S1 core-name words
  word priors    - P(matched | word), learned on training folds only (prepare stage)
  number relation- role-aware house/unit/floor numbers, one-digit substitutions, alphanumeric premises
  cross-record   - twins inside the S2/S3 pool, S1 name multiplicity, S1 claims
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

FEATURES_V2 = [
    'nr_shared', 'nr_q_only', 'nr_r_only', 'nr_rel', 'nr_swap_ratio', 'nr_q_only_filler',
    'nr_q_only_vocab_min', 'nr_r_only_vocab_min', 'q_vocab_min', 'nr_acronym', 'nr_compact_prefix', 'nr_alias',
    'wp_min', 'wp_mean', 'wp_low',
    'hn_equal', 'hn_sub1', 'hn_absdiff_log', 'unit_equal', 'unit_conflict', 'alnum_shared', 'alnum_conflict',
    'num_q_extra', 'num_r_missing',
    'tw_full', 'tw_num', 'tw_name', 'tw_addr', 'r_name_mult', 'r_claims', 'q_addr_empty',
]
# nr_rel codes: 0 same words, 1 record drops words, 2 record adds words, 3 one word swapped,
#               4 all words replaced, 5 several words differ, 6 an empty core name
FILLER = sorted(set(
    'inc incorporated llc corp corporation co company ltd limited pvt private llp plc lp pllc pc '
    'sarl sas sasu eurl sa sci snc ets etablissements ei cie groupe group the and of services service '
    'center centre partners ms m s sri shri smt mr dr mrs'.split()))
UNIT_RE = (r'\b(?:unit|apt|apartment|suite|ste|flat|floor|flr|fl|room|rm|office|shop|pmb|box)\b'
           r'[^0-9a-z]{0,3}(?:no[^0-9a-z]{0,2})?(?:apartment[^0-9a-z]{0,2}|apt[^0-9a-z]{0,2})?[a-z]?\d+')
FLOOR_RE = r'\b\d+\s*(?:st|nd|rd|th)?\s*(?:floor|flr)\b'
ALNUM_RE = r'[a-z0-9]+(?:[-/][a-z0-9]+)+|[a-z]+\d+[a-z0-9]*|\d+[a-z]+[a-z0-9]*'
EMPTY = pl.lit([], dtype=pl.List(pl.String))


def _strip_zeros(e):
    s = e.str.strip_chars_start('0')
    return pl.when(e.is_null()).then(None).when(s == '').then(pl.lit('0')).otherwise(s)


def text_stats(df, name_col, raw_address_col, prefix):
    """Per-record lists/scalars used by the pair features. Returns df plus prefixed columns."""
    low = pl.col(raw_address_col).fill_null('').str.to_lowercase()
    words = pl.col(name_col).fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element() != ''))
    segs = pl.concat_list(low.str.extract_all(UNIT_RE), low.str.extract_all(FLOOR_RE))
    rest = low.str.replace_all(UNIT_RE, ' ').str.replace_all(FLOOR_RE, ' ')
    return df.with_columns(
        words.list.unique().alias(prefix + 'tok'),
        pl.col(name_col).fill_null('').str.replace_all(' ', '').alias(prefix + 'compact'),
        words.list.eval(pl.element().str.slice(0, 1)).list.join('').alias(prefix + 'initials'),
        _strip_zeros(rest.str.extract(r'(\d+)', 1)).alias(prefix + 'house'),
        segs.list.eval(pl.element().str.extract(r'(\d+)\D*$', 1).str.strip_chars_start('0')).list.unique().fill_null(EMPTY).alias(prefix + 'units'),
        low.str.extract_all(ALNUM_RE).list.eval(pl.element().str.replace_all(r'[-/]', '')
                                                .filter(~pl.element().str.contains(r'^\d+(?:st|nd|rd|th)$'))).list.unique()
        .fill_null(EMPTY).alias(prefix + 'alnum'),
        low.str.extract_all(r'\d+').list.eval(pl.when(pl.element().str.strip_chars_start('0') == '').then(pl.lit('0'))
                                              .otherwise(pl.element().str.strip_chars_start('0'))).list.unique()
        .fill_null(EMPTY).alias(prefix + 'numset'),
    )


def _only_min(P, col, vocab):
    """log1p(min S1-vocabulary frequency) over the tokens in list column `col`; -1 when the list is empty."""
    e = P.select('_row', pl.col(col)).explode(col).drop_nulls(col)
    if e.height == 0:
        return np.full(P.height, -1.0, np.float32)
    e = e.join(vocab, left_on=col, right_on='token', how='left').with_columns(pl.col('freq').fill_null(0))
    m = e.group_by('_row').agg(pl.col('freq').min().alias('m'))
    out = P.select('_row').join(m, on='_row', how='left')['m']
    return np.where(out.is_null().to_numpy(), -1.0, np.log1p(out.fill_null(0).to_numpy())).astype(np.float32)


def pair_features(Q, R, vocab, workers=4):
    """Q, R: row-aligned per-pair query/ref stat frames (q_* / r_* columns). Returns float32 [n, len(FEATURES_V2)]."""
    P = pl.concat([Q, R], how='horizontal').with_row_index('_row')
    n = P.height
    if n == 0:
        return np.empty((0, len(FEATURES_V2)), np.float32)
    P = P.with_columns(
        pl.col('q_tok').list.set_intersection('r_tok').list.len().alias('nr_shared'),
        pl.col('q_tok').list.set_difference('r_tok').alias('_qo'),
        pl.col('r_tok').list.set_difference('q_tok').alias('_ro'),
        pl.col('q_tok').list.len().alias('_ql'), pl.col('r_tok').list.len().alias('_rl'))
    P = P.with_columns(pl.col('_qo').list.len().alias('nr_q_only'), pl.col('_ro').list.len().alias('nr_r_only'))
    P = P.with_columns(
        pl.when((pl.col('_ql') == 0) | (pl.col('_rl') == 0)).then(6)
        .when((pl.col('nr_q_only') == 0) & (pl.col('nr_r_only') == 0)).then(0)
        .when(pl.col('nr_q_only') == 0).then(1).when(pl.col('nr_r_only') == 0).then(2)
        .when(pl.col('nr_shared') == 0).then(4)
        .when((pl.col('nr_q_only') == 1) & (pl.col('nr_r_only') == 1)).then(3).otherwise(5).alias('nr_rel'),
        pl.col('_qo').list.eval(pl.element().is_in(FILLER)).list.sum().fill_null(0).alias('nr_q_only_filler'),
        (((pl.col('q_compact').str.len_chars() >= 2) & (pl.col('q_compact') == pl.col('r_initials')) & (pl.col('_rl') >= 2)) |
         ((pl.col('r_compact').str.len_chars() >= 2) & (pl.col('r_compact') == pl.col('q_initials')) & (pl.col('_ql') >= 2))).alias('nr_acronym'),
        ((pl.col('q_compact') != pl.col('r_compact')) &
         (((pl.col('q_compact').str.len_chars() >= 4) & pl.col('r_compact').str.starts_with(pl.col('q_compact'))) |
          ((pl.col('r_compact').str.len_chars() >= 4) & pl.col('q_compact').str.starts_with(pl.col('r_compact'))))).alias('nr_compact_prefix'),
        pl.col('q_units').list.set_intersection('r_units').list.len().alias('_ushared'),
        pl.col('q_alnum').list.set_intersection('r_alnum').list.len().alias('alnum_shared'),
        pl.col('q_numset').list.set_difference('r_numset').list.len().alias('num_q_extra'),
        pl.col('r_numset').list.set_difference('q_numset').list.len().alias('num_r_missing'))
    qo, ro = P['_qo'].list.join(' ').to_list(), P['_ro'].list.join(' ').to_list()
    swap = process.cpdist(qo, ro, scorer=fuzz.ratio, workers=workers).astype(np.float32) / 100
    swap[(P['nr_q_only'].to_numpy() == 0) | (P['nr_r_only'].to_numpy() == 0)] = 0
    qh, rh = P['q_house'], P['r_house']
    both = (qh.is_not_null() & rh.is_not_null()).to_numpy()
    lev = process.cpdist(qh.fill_null('').to_list(), rh.fill_null('').to_list(), scorer=Levenshtein.distance, workers=workers)
    same_len = (qh.str.len_chars() == rh.str.len_chars()).fill_null(False).to_numpy()
    qi, ri = qh.cast(pl.Int64, strict=False).to_numpy(), rh.cast(pl.Int64, strict=False).to_numpy()
    with np.errstate(invalid='ignore'):
        diff = np.where(both & ~np.isnan(qi.astype(float)) & ~np.isnan(ri.astype(float)),
                        np.log1p(np.abs(np.nan_to_num(qi.astype(float)) - np.nan_to_num(ri.astype(float)))), -1.0)
    hn_equal = np.where(both, (qh == rh).fill_null(False).to_numpy().astype(np.float32), -1.0)
    qu, ru = P['q_units'].list.len().to_numpy(), P['r_units'].list.len().to_numpy()
    qa, ra = P['q_alnum'].list.len().to_numpy(), P['r_alnum'].list.len().to_numpy()
    q_only_vocab = _only_min(P, '_qo', vocab)
    alias = (P['_ql'].to_numpy() == 1) & (P['q_vocab_min'].to_numpy() <= 0)
    cols = {
        'nr_shared': P['nr_shared'].to_numpy(), 'nr_q_only': P['nr_q_only'].to_numpy(), 'nr_r_only': P['nr_r_only'].to_numpy(),
        'nr_rel': P['nr_rel'].to_numpy(), 'nr_swap_ratio': swap, 'nr_q_only_filler': P['nr_q_only_filler'].to_numpy(),
        'nr_q_only_vocab_min': q_only_vocab, 'nr_r_only_vocab_min': _only_min(P, '_ro', vocab),
        'q_vocab_min': P['q_vocab_min'].to_numpy(), 'nr_acronym': P['nr_acronym'].to_numpy(),
        'nr_compact_prefix': P['nr_compact_prefix'].to_numpy(), 'nr_alias': alias,
        'wp_min': P['wp_min'].to_numpy(), 'wp_mean': P['wp_mean'].to_numpy(), 'wp_low': P['wp_low'].to_numpy(),
        'hn_equal': hn_equal, 'hn_sub1': both & same_len & (lev == 1), 'hn_absdiff_log': diff,
        'unit_equal': (qu > 0) & (ru > 0) & (P['_ushared'].to_numpy() > 0),
        'unit_conflict': (qu > 0) & (ru > 0) & (P['_ushared'].to_numpy() == 0),
        'alnum_shared': P['alnum_shared'].to_numpy(), 'alnum_conflict': (qa > 0) & (ra > 0) & (P['alnum_shared'].to_numpy() == 0),
        'num_q_extra': P['num_q_extra'].to_numpy(), 'num_r_missing': P['num_r_missing'].to_numpy(),
        'tw_full': P['tw_full'].to_numpy(), 'tw_num': P['tw_num'].to_numpy(), 'tw_name': P['tw_name'].to_numpy(),
        'tw_addr': P['tw_addr'].to_numpy(), 'r_name_mult': P['r_name_mult'].to_numpy(), 'r_claims': P['r_claims'].to_numpy(),
        'q_addr_empty': P['q_addr_empty'].to_numpy(),
    }
    X = np.column_stack([np.asarray(cols[f], dtype=np.float32) for f in FEATURES_V2])
    if not np.isfinite(X).all():
        raise RuntimeError('Non-finite C04 feature.')
    return X


QUERY_COLS = ['q_tok', 'q_compact', 'q_initials', 'q_house', 'q_units', 'q_alnum', 'q_numset',
              'q_addr_empty', 'tw_full', 'tw_num', 'tw_name', 'tw_addr', 'wp_min', 'wp_mean', 'wp_low', 'q_vocab_min']
REF_COLS = ['r_tok', 'r_compact', 'r_initials', 'r_house', 'r_units', 'r_alnum', 'r_numset', 'r_name_mult', 'r_claims']


def gather_pairs(qstats, rstats, query_index, ref_index, vocab, workers=4):
    """Row-gather the prepared per-record stats for aligned (query_index, ref_index) arrays and build features."""
    Q = qstats[np.asarray(query_index, np.int64)].select(QUERY_COLS)
    R = rstats[np.asarray(ref_index, np.int64)].select(REF_COLS)
    return pair_features(Q, R, vocab, workers)

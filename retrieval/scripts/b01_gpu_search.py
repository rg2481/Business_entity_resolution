"""B01 scoring with sparse query transfer and contiguous, row-wise top-k.

The frozen preparation/index/pilot code stays unchanged. This module changes
execution only: same float32 vectors, cosine scores, weights and state mask.
"""
import numpy as np
import torch


def device_dense_queries(matrix, device='cuda'):
    """Transfer CSR nonzeros, then materialize features x queries on the device."""
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    queries, features = matrix.shape
    dense = torch.zeros((features, queries), dtype=torch.float32, device=device)
    if matrix.nnz:
        rows = np.repeat(np.arange(queries, dtype=np.int64), np.diff(matrix.indptr))
        positions = matrix.indices.astype(np.int64) * queries + rows
        values = np.ascontiguousarray(matrix.data, dtype=np.float32)
        dense.view(-1).index_copy_(
            0, torch.from_numpy(positions).to(device), torch.from_numpy(values).to(device))
    return dense


def retrieve_batch(pn, pa, qn, qa, ref_states, ref_ambiguous, query_masks, cfg):
    """Return the same result layout as the original pilot, on CPU or CUDA."""
    device = pn.device
    with torch.inference_mode():
        sn = torch.sparse.mm(pn, device_dense_queries(qn, device)).T.contiguous()
        sa = torch.sparse.mm(pa, device_dense_queries(qa, device)).T.contiguous()
        keep = None
        if cfg['state_filter'] and any(query_masks):
            allowed = torch.tensor(query_masks, dtype=torch.int64, device=device)
            keep = ((allowed[:, None] & ref_states[None, :]) != 0)
            keep |= (allowed[:, None] == 0) | ref_ambiguous[None, :]
        k = min(cfg['top_k'], sn.shape[1])
        result = {}
        for mode in ('name', 'address', 'combo'):
            scores = sn if mode == 'name' else sa if mode == 'address' else (
                sn * cfg['name_weight'] + sa * (1 - cfg['name_weight']))
            ranked = scores.masked_fill(~keep, float('-inf')) if keep is not None else scores
            values, indices = torch.topk(ranked, k, dim=1)
            result[mode] = (indices.cpu().numpy(), values.cpu().numpy(),
                            sn.gather(1, indices).cpu().numpy(), sa.gather(1, indices).cpu().numpy())
            del ranked, scores, values, indices
        return result

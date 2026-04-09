import math

import torch


BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

VARIANT_MANIFEST = [
    {
        "name": "default",
    }
]


def setup(suite_specs, device, variants):
    return None


def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    batch_size, num_heads, t_max, head_dim = q.shape
    assert head_dim == HEAD_DIM
    assert t_max % BLOCK_SIZE == 0

    device = q.device
    batch_heads = batch_size * num_heads
    num_q_blocks = row_ptr.shape[-1] - 1
    num_k_blocks = t_max // BLOCK_SIZE

    q_f = q.to(torch.float32).reshape(batch_heads, t_max, head_dim)
    k_blocks = k.to(torch.float32).reshape(batch_heads, num_k_blocks, BLOCK_SIZE, head_dim)
    v_blocks = v.to(torch.float32).reshape(batch_heads, num_k_blocks, BLOCK_SIZE, head_dim)

    flat_k_blocks = k_blocks.reshape(batch_heads * num_k_blocks, BLOCK_SIZE, head_dim)
    flat_v_blocks = v_blocks.reshape(batch_heads * num_k_blocks, BLOCK_SIZE, head_dim)

    row_ptr_2d = row_ptr.reshape(batch_heads, num_q_blocks + 1).to(torch.int64)
    col_idx_2d = col_idx.reshape(batch_heads, -1).to(torch.int64)
    seq_lens_2d = seq_lens[:, None].expand(batch_size, num_heads).reshape(batch_heads).to(torch.int64)

    block_token_offsets = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)

    # Compute the maximum degree across all (bh, q_block) once. Pad every
    # q_block to this constant. One single host sync per call.
    all_degrees = row_ptr_2d[:, 1:] - row_ptr_2d[:, :-1]  # (bh, nq)
    max_degree = int(all_degrees.max().item())
    if max_degree <= 0:
        empty_out = torch.zeros((batch_heads, t_max, head_dim), device=device, dtype=torch.float32)
        empty_lse = torch.full((batch_heads, t_max), -torch.inf, device=device, dtype=torch.float32)
        return empty_out.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16), empty_lse.reshape(
            batch_size, num_heads, t_max
        )

    nq = num_q_blocks
    md = max_degree

    # Slot offsets: (1, 1, md)
    slot_offsets = torch.arange(md, device=device, dtype=torch.int64)[None, None, :]
    # row_starts: (bh, nq, 1)
    row_starts = row_ptr_2d[:, :nq, None]
    # gather positions in the (bh, max_nnz) col_idx_2d table.
    max_nnz = col_idx_2d.shape[1]
    gather_positions = torch.clamp(row_starts + slot_offsets, max=max_nnz - 1)  # (bh, nq, md)
    # 3-way gather across the col_idx table.
    col_idx_expanded = col_idx_2d[:, None, :].expand(batch_heads, nq, max_nnz)
    gathered_block_indices = torch.gather(col_idx_expanded, 2, gather_positions)  # (bh, nq, md)

    slot_valid = slot_offsets < all_degrees[:, :, None]  # (bh, nq, md)
    gathered_block_indices = torch.where(
        slot_valid,
        gathered_block_indices,
        torch.zeros_like(gathered_block_indices),
    )

    # Resolve to flat (bh*num_k_blocks) indices for K/V gather.
    bh_block_base = (
        torch.arange(batch_heads, device=device, dtype=torch.int64) * num_k_blocks
    )[:, None, None]  # (bh, 1, 1)
    flat_block_indices = (bh_block_base + gathered_block_indices).reshape(-1)  # (bh*nq*md,)

    # Gather K/V tiles: (bh, nq, md, BLOCK_SIZE, head_dim) -> (bh, nq, md*BLOCK_SIZE, head_dim)
    gathered_k = flat_k_blocks.index_select(0, flat_block_indices).reshape(
        batch_heads, nq, md * BLOCK_SIZE, head_dim
    )
    gathered_v = flat_v_blocks.index_select(0, flat_block_indices).reshape(
        batch_heads, nq, md * BLOCK_SIZE, head_dim
    )

    # Q chunks: (bh, nq, BLOCK_SIZE, head_dim)
    q_chunks = q_f.reshape(batch_heads, nq, BLOCK_SIZE, head_dim)

    # Scores: (bh, nq, BLOCK_SIZE, md*BLOCK_SIZE)
    scores = torch.matmul(q_chunks, gathered_k.transpose(-1, -2)) * SCALE

    # --- Build the boolean mask -----------------------------------------------
    # Key absolute positions: (bh, nq, md*BLOCK_SIZE)
    key_positions = (
        gathered_block_indices[:, :, :, None] * BLOCK_SIZE + block_token_offsets[None, None, None, :]
    ).reshape(batch_heads, nq, md * BLOCK_SIZE)
    key_padding_ok = key_positions < seq_lens_2d[:, None, None]
    slot_valid_tokens = (
        slot_valid[:, :, :, None]
        .expand(batch_heads, nq, md, BLOCK_SIZE)
        .reshape(batch_heads, nq, md * BLOCK_SIZE)
    )
    key_valid = slot_valid_tokens & key_padding_ok  # (bh, nq, md*BLOCK_SIZE)

    # Q positions per (nq, BLOCK_SIZE)
    q_positions_per_block = (
        torch.arange(nq, device=device, dtype=torch.int64)[:, None] * BLOCK_SIZE
        + block_token_offsets[None, :]
    )  # (nq, BLOCK_SIZE)
    query_valid = q_positions_per_block[None, :, :] < seq_lens_2d[:, None, None]  # (bh, nq, BLOCK_SIZE)

    # cases.py guarantees col_idx[q_block] only contains blocks <= q_block, so
    # an unconditional `key_pos <= q_pos` check is a no-op for non-diagonal
    # blocks and correctly enforces causal masking inside the diagonal block.
    # We fold all conditions into a single (bh, nq, BLOCK_SIZE_q, md*BLOCK_SIZE_k)
    # boolean built in one fused expression to minimize intermediate allocations.
    mask = (
        key_valid[:, :, None, :]
        & query_valid[:, :, :, None]
        & (key_positions[:, :, None, :] <= q_positions_per_block[None, :, :, None])
    )

    scores = scores.masked_fill(~mask, -torch.inf)
    row_max = torch.max(scores, dim=-1).values  # (bh, nq, BLOCK_SIZE)
    valid_rows = query_valid & torch.isfinite(row_max)
    row_max_safe = torch.where(valid_rows, row_max, torch.zeros_like(row_max))

    exp_scores = torch.exp(scores - row_max_safe[:, :, :, None]) * mask.to(torch.float32)
    denom = exp_scores.sum(dim=-1)  # (bh, nq, BLOCK_SIZE)
    denom_safe = torch.where(valid_rows, denom, torch.ones_like(denom))

    out_blocks = torch.matmul(exp_scores, gathered_v) / denom_safe[:, :, :, None]  # (bh, nq, BLOCK_SIZE, head_dim)
    lse_blocks = torch.where(
        valid_rows,
        row_max_safe + torch.log(denom_safe),
        torch.full_like(row_max_safe, -torch.inf),
    )

    # Mask out invalid query rows in the output.
    out_blocks = torch.where(
        valid_rows[:, :, :, None],
        out_blocks,
        torch.zeros_like(out_blocks),
    )

    output = out_blocks.reshape(batch_heads, t_max, head_dim)
    lse = lse_blocks.reshape(batch_heads, t_max)

    return output.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16), lse.reshape(
        batch_size, num_heads, t_max
    )

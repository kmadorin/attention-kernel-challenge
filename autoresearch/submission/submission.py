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

    # --- Build mask components (shared between paths) ------------------------
    key_positions = (
        gathered_block_indices[:, :, :, None] * BLOCK_SIZE + block_token_offsets[None, None, None, :]
    ).reshape(batch_heads, nq, md * BLOCK_SIZE)
    key_padding_ok = key_positions < seq_lens_2d[:, None, None]
    slot_valid_tokens = (
        slot_valid[:, :, :, None]
        .expand(batch_heads, nq, md, BLOCK_SIZE)
        .reshape(batch_heads, nq, md * BLOCK_SIZE)
    )
    key_valid = slot_valid_tokens & key_padding_ok
    key_invalid = ~key_valid

    q_positions_per_block = (
        torch.arange(nq, device=device, dtype=torch.int64)[:, None] * BLOCK_SIZE
        + block_token_offsets[None, :]
    )
    query_valid = q_positions_per_block[None, :, :] < seq_lens_2d[:, None, None]

    invalid_4d = key_invalid[:, :, None, :] | (
        key_positions[:, :, None, :] > q_positions_per_block[None, :, :, None]
    )
    neg_inf = float("-inf")

    if q.is_cuda:
        # CUDA path: use the fused efficient attention kernel. Saves the
        # ~2 GB scores materialization and delivers output + LSE in one
        # call.
        attn_bias = torch.zeros(
            (batch_heads, nq, BLOCK_SIZE, md * BLOCK_SIZE),
            device=device,
            dtype=torch.float32,
        )
        attn_bias.masked_fill_(invalid_4d, neg_inf)
        B = batch_heads * nq
        q_for_sdpa = q_chunks.reshape(B, 1, BLOCK_SIZE, head_dim)
        k_for_sdpa = gathered_k.reshape(B, 1, md * BLOCK_SIZE, head_dim)
        v_for_sdpa = gathered_v.reshape(B, 1, md * BLOCK_SIZE, head_dim)
        bias_for_sdpa = attn_bias.reshape(B, 1, BLOCK_SIZE, md * BLOCK_SIZE)
        out_sdpa, lse_sdpa, _seed, _offset = torch.ops.aten._scaled_dot_product_efficient_attention(
            q_for_sdpa,
            k_for_sdpa,
            v_for_sdpa,
            bias_for_sdpa,
            True,   # compute_log_sumexp
            0.0,    # dropout_p
            False,  # is_causal
            scale=SCALE,
        )
        out_blocks = out_sdpa.reshape(batch_heads, nq, BLOCK_SIZE, head_dim)
        lse_blocks = lse_sdpa.reshape(batch_heads, nq, BLOCK_SIZE)
        # Zero / -inf invalid query rows (SDPA may leak garbage into them).
        out_blocks = torch.where(
            query_valid[:, :, :, None],
            out_blocks,
            torch.zeros_like(out_blocks),
        )
        lse_blocks = torch.where(
            query_valid,
            lse_blocks,
            torch.full_like(lse_blocks, -torch.inf),
        )
    else:
        # CPU fallback: manual fp32 path (same as prior best).
        scores = torch.matmul(q_chunks, gathered_k.transpose(-1, -2))
        scores.mul_(SCALE)
        scores.masked_fill_(invalid_4d, neg_inf)
        row_max = torch.max(scores, dim=-1).values
        valid_rows = query_valid & torch.isfinite(row_max)
        row_max_safe = torch.where(valid_rows, row_max, torch.zeros_like(row_max))
        scores.sub_(row_max_safe[:, :, :, None])
        scores.exp_()
        exp_scores = scores
        denom = exp_scores.sum(dim=-1)
        denom_safe = torch.where(valid_rows, denom, torch.ones_like(denom))
        out_blocks = torch.matmul(exp_scores, gathered_v) / denom_safe[:, :, :, None]
        lse_blocks = torch.where(
            valid_rows,
            row_max_safe + torch.log(denom_safe),
            torch.full_like(row_max_safe, -torch.inf),
        )
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

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

    output = torch.zeros((batch_heads, t_max, head_dim), device=device, dtype=torch.float32)
    lse = torch.full((batch_heads, t_max), -torch.inf, device=device, dtype=torch.float32)

    batch_head_block_base = (
        torch.arange(batch_heads, device=device, dtype=torch.int64)[:, None] * num_k_blocks
    )
    block_token_offsets = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)
    slot_offsets_cache = {}

    for q_block in range(num_q_blocks):
        q_start = q_block * BLOCK_SIZE
        q_end = q_start + BLOCK_SIZE
        q_chunk = q_f[:, q_start:q_end]

        row_start = row_ptr_2d[:, q_block]
        row_end = row_ptr_2d[:, q_block + 1]
        degrees = row_end - row_start
        max_degree = int(degrees.max().item())
        if max_degree <= 0:
            continue

        slot_offsets = slot_offsets_cache.get(max_degree)
        if slot_offsets is None:
            slot_offsets = torch.arange(max_degree, device=device, dtype=torch.int64)[None, :]
            slot_offsets_cache[max_degree] = slot_offsets

        slot_valid = slot_offsets < degrees[:, None]
        gather_positions = torch.clamp(row_start[:, None] + slot_offsets, max=col_idx_2d.shape[1] - 1)
        gathered_block_indices = torch.gather(col_idx_2d, 1, gather_positions)
        gathered_block_indices = torch.where(
            slot_valid,
            gathered_block_indices,
            torch.zeros_like(gathered_block_indices),
        )

        flat_block_indices = batch_head_block_base + gathered_block_indices
        gathered_k_blocks = flat_k_blocks.index_select(0, flat_block_indices.reshape(-1)).reshape(
            batch_heads, max_degree, BLOCK_SIZE, head_dim
        )
        gathered_v_blocks = flat_v_blocks.index_select(0, flat_block_indices.reshape(-1)).reshape(
            batch_heads, max_degree, BLOCK_SIZE, head_dim
        )

        key_positions = (
            gathered_block_indices[:, :, None] * BLOCK_SIZE + block_token_offsets[None, None, :]
        ).reshape(batch_heads, max_degree * BLOCK_SIZE)
        key_valid = (
            slot_valid[:, :, None] & (key_positions.reshape(batch_heads, max_degree, BLOCK_SIZE) < seq_lens_2d[:, None, None])
        ).reshape(batch_heads, max_degree * BLOCK_SIZE)
        diag_key = (
            (gathered_block_indices == q_block)[:, :, None]
            .expand(batch_heads, max_degree, BLOCK_SIZE)
            .reshape(batch_heads, max_degree * BLOCK_SIZE)
        )

        q_positions = q_start + block_token_offsets[None, :]
        query_valid = q_positions < seq_lens_2d[:, None]

        k_tokens = gathered_k_blocks.reshape(batch_heads, max_degree * BLOCK_SIZE, head_dim)
        v_tokens = gathered_v_blocks.reshape(batch_heads, max_degree * BLOCK_SIZE, head_dim)

        scores = torch.matmul(q_chunk, k_tokens.transpose(1, 2)) * SCALE

        mask = key_valid[:, None, :] & query_valid[:, :, None]
        causal_ok = key_positions[:, None, :] <= q_positions[:, :, None]
        mask = mask & ((~diag_key)[:, None, :] | causal_ok)

        scores = scores.masked_fill(~mask, -torch.inf)
        row_max = torch.max(scores, dim=-1).values
        valid_rows = query_valid & torch.isfinite(row_max)
        row_max_safe = torch.where(valid_rows, row_max, torch.zeros_like(row_max))

        exp_scores = torch.exp(scores - row_max_safe[:, :, None]) * mask.to(torch.float32)
        denom = exp_scores.sum(dim=-1)
        denom_safe = torch.where(valid_rows, denom, torch.ones_like(denom))

        out_block = torch.matmul(exp_scores, v_tokens) / denom_safe[:, :, None]
        lse_block = torch.where(
            valid_rows,
            row_max_safe + torch.log(denom_safe),
            torch.full_like(row_max_safe, -torch.inf),
        )

        output[:, q_start:q_end] = torch.where(
            valid_rows[:, :, None],
            out_block,
            output[:, q_start:q_end],
        )
        lse[:, q_start:q_end] = torch.where(
            valid_rows,
            lse_block,
            lse[:, q_start:q_end],
        )

    return output.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16), lse.reshape(
        batch_size, num_heads, t_max
    )

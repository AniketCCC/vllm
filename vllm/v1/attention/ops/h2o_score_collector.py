# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental decode-time H2O block attention-mass collector.

Production FlashAttention does not expose attention probabilities. This module
provides a **slow reference** path that materializes QKᵀ/softmax for the current
query tokens against retained KV pages and sums mass per logical block.

TODO: replace with an in-kernel block-mass proxy that avoids full attention
matrix materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)


@dataclass
class H2OScoreAccumulator:
    """Per-step, per-request cumulative block scores (CPU floats).

    Scores are summed across layers/heads that call ``add_layer_scores``.
    """

    enabled: bool = False
    # req_id -> {logical_block_id -> score}
    scores: dict[str, dict[int, float]] = field(default_factory=dict)
    # Parallel lists describing the current decode batch (set by model runner).
    req_ids: list[str] = field(default_factory=list)
    # Per-request number of query tokens in this step (usually 1 for decode).
    query_lens: list[int] = field(default_factory=list)
    block_size: int = 16
    # Logical block tables with nulls: [num_reqs, max_blocks]
    block_tables: torch.Tensor | None = None
    # Original sequence lengths (token count), used to know valid tokens.
    seq_lens: torch.Tensor | None = None

    def reset(self) -> None:
        self.scores = {}

    def configure_batch(
        self,
        *,
        req_ids: list[str],
        query_lens: list[int],
        block_size: int,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        self.req_ids = list(req_ids)
        self.query_lens = list(query_lens)
        self.block_size = block_size
        self.block_tables = block_tables
        self.seq_lens = seq_lens
        self.reset()

    def add_layer_scores(self, layer_scores: dict[str, dict[int, float]]) -> None:
        for req_id, blk_scores in layer_scores.items():
            dst = self.scores.setdefault(req_id, {})
            for lid, val in blk_scores.items():
                dst[lid] = dst.get(lid, 0.0) + float(val)

    def snapshot(self) -> dict[str, dict[int, float]]:
        return {rid: dict(sc) for rid, sc in self.scores.items()}


_ACCUMULATOR = H2OScoreAccumulator()


def get_h2o_score_accumulator() -> H2OScoreAccumulator:
    return _ACCUMULATOR


def compute_block_attention_mass(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    scale: float,
    block_size: int,
    null_block_id: int = NULL_BLOCK_ID,
) -> list[dict[int, float]]:
    """Compute per-request logical-block attention mass for the current queries.

    Args:
        query: [num_tokens, num_heads, head_size] (or flat [num_tokens, H*D])
        key_cache: paged KV cache K half. Expected layout
            [num_blocks, block_size, num_kv_heads, head_size] or
            [num_blocks, 2, block_size, num_kv_heads, head_size] (K=0).
        block_tables: [num_reqs, max_blocks] int32 logical→physical
        seq_lens: [num_reqs] original token counts
        query_start_loc: [num_reqs + 1]
        scale: attention scale (1/sqrt(d))

    Returns:
        List of length num_reqs; each is {logical_block_id: mass}.
    """
    if query.dim() == 2:
        # [T, H*D] — unknown head split; treat as single head.
        query = query.unsqueeze(1)

    num_reqs = seq_lens.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]

    if key_cache.dim() == 5:
        # Either [2, num_blocks, block_size, kv_heads, D] (FA)
        # or [num_blocks, 2, block_size, kv_heads, D] (Triton).
        if key_cache.shape[0] == 2:
            k_cache = key_cache[0]
        else:
            k_cache = key_cache[:, 0]
    else:
        k_cache = key_cache

    num_kv_heads = k_cache.shape[2]
    assert num_heads % num_kv_heads == 0
    q_per_kv = num_heads // num_kv_heads

    results: list[dict[int, float]] = []
    for req_idx in range(num_reqs):
        q0 = int(query_start_loc[req_idx].item())
        q1 = int(query_start_loc[req_idx + 1].item())
        if q1 <= q0:
            results.append({})
            continue
        # Use the last query token in the chunk (decode / chunk end).
        q = query[q1 - 1].float()  # [H, D]
        seq_len = int(seq_lens[req_idx].item())
        if seq_len <= 0:
            results.append({})
            continue

        num_logical_blocks = (seq_len + block_size - 1) // block_size
        masses: dict[int, float] = {}
        # Gather keys for all non-null logical blocks.
        keys_list: list[torch.Tensor] = []
        logical_ids: list[int] = []
        token_counts: list[int] = []
        for lid in range(num_logical_blocks):
            phys = int(block_tables[req_idx, lid].item())
            if phys == null_block_id:
                continue
            start = lid * block_size
            end = min(start + block_size, seq_len)
            n_tok = end - start
            # k_cache[phys]: [block_size, kv_heads, D]
            keys_list.append(k_cache[phys, :n_tok].float())
            logical_ids.append(lid)
            token_counts.append(n_tok)

        if not keys_list:
            results.append({})
            continue

        keys = torch.cat(keys_list, dim=0)  # [S, kv_heads, D]
        # Expand KV heads to query heads.
        if q_per_kv > 1:
            keys = keys.repeat_interleave(q_per_kv, dim=1)  # [S, H, D]
        # scores: [H, S]
        scores = torch.einsum("hd,shd->hs", q, keys) * scale
        probs = torch.softmax(scores, dim=-1)  # [H, S]
        # Sum over heads, then over tokens inside each logical block.
        token_mass = probs.sum(dim=0)  # [S]
        offset = 0
        for lid, n_tok in zip(logical_ids, token_counts):
            masses[lid] = float(token_mass[offset : offset + n_tok].sum().item())
            offset += n_tok
        results.append(masses)
    return results


def maybe_accumulate_h2o_scores_from_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    *,
    scale: float,
) -> None:
    """Best-effort hook after attention; no-op unless accumulator is enabled."""
    acc = _ACCUMULATOR
    if not acc.enabled or not acc.req_ids:
        return
    if kv_cache is None or not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
        return

    # Prefer logical tables stored on the accumulator (pre-compaction). Attention
    # metadata may already contain compacted pages for the FA kernel.
    block_tables = acc.block_tables
    seq_lens = acc.seq_lens
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if block_tables is None or seq_lens is None or query_start_loc is None:
        return

    try:
        per_req = compute_block_attention_mass(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            query_start_loc,
            scale=scale,
            block_size=acc.block_size,
        )
    except Exception:
        logger.exception("H2O experimental score collection failed; skipping layer")
        return

    layer_scores: dict[str, dict[int, float]] = {}
    for req_id, masses in zip(acc.req_ids, per_req):
        if masses:
            layer_scores[req_id] = masses
    if layer_scores:
        acc.add_layer_scores(layer_scores)


def compact_block_table_for_attention(
    block_table_row: list[int],
    seq_len: int,
    block_size: int,
    *,
    null_block_id: int = NULL_BLOCK_ID,
) -> tuple[list[int], int]:
    """Compact non-null logical blocks for attention while preserving RoPE.

    Returns (compacted_physical_block_ids, retained_token_count).
    Query positions must remain the original sequence positions.
    """
    if seq_len <= 0:
        return [], 0
    num_logical = (seq_len + block_size - 1) // block_size
    compacted: list[int] = []
    retained_tokens = 0
    for lid in range(min(num_logical, len(block_table_row))):
        phys = block_table_row[lid]
        if phys == null_block_id:
            continue
        start = lid * block_size
        end = min(start + block_size, seq_len)
        compacted.append(phys)
        retained_tokens += end - start
    return compacted, retained_tokens

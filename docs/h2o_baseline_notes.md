# H2O-style KV-cache eviction baseline (experimental)

Commit inspected: `9d6500b89` (`v0.20.2rc0-110-g9d6500b89`).

This note describes the **block-granular H2O baseline** added for experiments.
It is **not** mixed-precision compression, RD optimization, or a new research policy.

## Codebase map (V1)

| Area | Location |
| --- | --- |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` (`KVCacheManager`) |
| Per-type managers / allocation | `vllm/v1/core/single_type_kv_cache_manager.py` |
| Block pool / null block | `vllm/v1/core/block_pool.py` (`null_block`, `block_id=0`) |
| Coordinator | `vllm/v1/core/kv_cache_coordinator.py` |
| Scheduler ↔ worker block IDs | `vllm/v1/core/sched/output.py` (`CachedRequestData`) |
| Worker block tables / slot mapping | `vllm/v1/worker/block_table.py` |
| Decode attention (default) | FlashAttention / Triton / FlashInfer under `vllm/v1/attention/backends/` |
| Existing “eviction” analog | Sliding-window `remove_skipped_blocks` (replace with `null_block`) |

**Allocation** happens in `KVCacheManager.allocate_slots` → coordinator →
`SingleTypeKVCacheManager.allocate_new_blocks`.

**Block tables** are updated on the worker from `SchedulerOutput` (`new_block_ids`
append, or full replace for resumed / H2O-refresh requests). Slot mapping uses
**original token positions**: `block_index = position // block_size`.

**Attention score exposure:** production FlashAttention / PagedAttention paths do
**not** return per-token attention probabilities without kernel changes. No prior
H2O / sparse-KV eviction policy existed in this tree (only unrelated “H2OVL”
multimodal models and CPU-offload LRU/ARC policies).

## Implementation approach

1. **Config** (`CacheConfig`): `--kv-eviction-policy none|h2o`,
   `--h2o-max-blocks N`, `--h2o-recent-blocks R`, optional `--h2o-debug`.
   Default remains `none` (stock vLLM). Invalid `R >= N` fails at config validate.
   Prefix caching is forced off when H2O is enabled (eviction vs. hash reuse).

2. **Policy** (`vllm/v1/kv_cache_eviction/h2o.py`): per-request metadata
   (`logical_block_id`, cumulative `h2o_score`, creation order). When retained
   non-null blocks exceed `N`, protect the `R` most recent retained blocks, then
   evict lowest-score candidates. Never evict the block currently being written.

3. **Physical eviction:** reuse sliding-window indirection — replace victims with
   `null_block` and `free_blocks` via the existing allocator. Logical indices stay
   aligned with original token positions so RoPE/slot mapping are unchanged.

4. **Worker sync:** after H2O eviction, mark the request for **full block-table
   refresh** (not append-only), so the worker sees nulls in middle holes.

5. **Attention over holes:** stock FA would read `null_block` pages inside
   `seq_len`. When H2O is enabled, the worker **compacts** non-null pages into the
   attention block table and sets attention `seq_lens` to the retained token count.
   Query **positions** remain the original sequence positions (RoPE identity
   preserved). Documented deviation: attention KV length is compacted; model
   positions are not renumbered.

6. **Score collection (experimental):** FlashAttention is left untouched. An
   optional decode-time reference path
   (`vllm/v1/attention/ops/h2o_score_collector.py`) materializes QKᵀ/softmax for
   the current query against retained pages and accumulates **block attention mass**
   summed over heads (and, when wired, across layers). Scores are returned on
   `ModelRunnerOutput` and applied by the scheduler before the next eviction.
   This path is slower and marked experimental. TODO: block-mass proxy inside FA
   without full matrix materialization.

## Aggregation formula (this baseline)

For retained logical block \(b\) at decode step \(t\):

\[
H_b(t)=H_b(t-1)+\sum_{h}\sum_{j\in b}\alpha_{t,h,j}
\]

Optionally averaged or summed across layers when multiple layer hooks fire.
Prefill may contribute scores when the collector runs; otherwise scores start at
0 and grow during decode (recent window still protects suffix blocks).

## Deviations from canonical token-granular H2O

| Canonical H2O | This baseline | Why |
| --- | --- | --- |
| Token eviction unit | KV **block/page** | Match vLLM paged cache |
| Dense KV tensor gather | Null placeholders + attention compaction | Keep position/`slot_mapping` |
| Attention weights from training-style attn | Experimental score collector | Avoid FA kernel rewrite |
| Fixed HH + recent budgets (often 50/50) | `N` total, `R` recent, rest heavy-hitters | Matches requested knobs |

## Control flow

```text
decode forward
  → (optional) h2o_score_collector → per-req block scores
  → ModelRunnerOutput.h2o_block_scores
scheduler.update
  → H2ORequestState.add_scores
allocate_slots / post-score eviction
  → select_victims(N, R)
  → replace victims with null_block; free physical blocks
  → mark refresh_block_ids
next SchedulerOutput
  → worker replaces full block_ids row
  → slot_mapping still uses original positions
  → attention metadata uses compacted non-null pages
```

## Limitations / blockers

- Faithful token-granular H2O needs per-token scores + gather; block granularity
  is coarser.
- Production FA does not expose \(\alpha\); collector is a slow reference path.
- Prefix caching disabled under H2O.
- Speculative decoding / multi-group hybrid KV not specially validated.
- CUDA-graph + score collector may force eager extras when H2O is on.

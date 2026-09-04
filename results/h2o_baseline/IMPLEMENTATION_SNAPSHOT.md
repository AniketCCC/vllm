# H2O baseline implementation snapshot

**Recorded:** 2026-08-27 (evaluation start)

## Git state at experiment start

| Field | Value |
| --- | --- |
| Commit | `d1d7f48e5b35374b5e0b63b0d3a0e5fc20160c9c` |
| Branch | `exp/h2o-kv-eviction-baseline` |
| Describe | `v0.20.2rc0-111-gd1d7f48e5` |
| Parent upstream | `9d6500b89` |
| Dirty working tree | **No** (clean at evaluation start) |

## vLLM version

V1 KV-cache path on commit above. Scheduler + `GPUModelRunner` + `KVCacheManager`.

## H2O-related files

| Path | Role |
| --- | --- |
| `vllm/config/cache.py` | `kv_eviction_policy`, `h2o_max_blocks`, `h2o_recent_blocks`, `h2o_debug` |
| `vllm/engine/arg_utils.py` | CLI flags |
| `vllm/v1/kv_cache_eviction/h2o.py` | Policy, per-request metadata, victim selection |
| `vllm/v1/core/kv_cache_manager.py` | Eviction on allocate / score update |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `evict_logical_blocks`, `get_retained_logical_block_ids` |
| `vllm/v1/core/sched/scheduler.py` | Score apply, block-table refresh |
| `vllm/v1/core/sched/output.py` | `refresh_block_ids_req_ids` |
| `vllm/v1/worker/gpu_model_runner.py` | Compaction, score accumulator config |
| `vllm/v1/attention/ops/h2o_score_collector.py` | Reference QKᵀ/softmax block mass |
| `vllm/model_executor/layers/attention/attention.py` | Post-attn score hook |
| `vllm/v1/outputs.py` | `h2o_block_scores` on `ModelRunnerOutput` |
| `docs/h2o_baseline_notes.md` | Design notes |
| `tests/v1/kv_cache_eviction/` | Unit / manager tests |

## Configuration flags

```text
--kv-eviction-policy none|h2o
--h2o-max-blocks N          # required when policy=h2o
--h2o-recent-blocks R       # default 1; require 0 <= R < N
--h2o-debug                 # log eviction decisions
```

When H2O is enabled, prefix caching is disabled in `CacheConfig` validation.

Default: `kv_eviction_policy=none` (stock vLLM).

## Score definition

**Block-level** cumulative attention mass (not token-level H2O):

\[
H_b(t) = H_b(t-1) + \sum_{h}\sum_{j \in b} \alpha_{t,h,j}
\]

- Aggregated across layers when the score hook runs on each attention layer.
- Collected via experimental `h2o_score_collector` (materializes softmax over retained KV pages).
- **Not** from production FlashAttention kernels.

## Block size

Default `CacheConfig.block_size = 16` (unless overridden with `--block-size`).

## Recent-window policy

Among **retained non-null** logical blocks:

1. Protect `R = h2o_recent_blocks` most recently **created** blocks (by allocation order).
2. Protect the block currently being written (`retained[-1]` in sync).
3. Among remaining candidates, evict lowest `h2o_score` (tie-break: older order).

## Eviction condition

After allocation or score update, if `num_retained_non_null > h2o_max_blocks`, evict
`excess` lowest-scoring eligible logical blocks.

## How evicted blocks are removed from attention

1. Logical slot replaced with `null_block` (`block_id=0`); physical block `free_blocks`.
2. Worker receives **full** block-table refresh (`refresh_block_ids_req_ids`).
3. **Slot mapping** still uses original token positions (`position // block_size` → logical index).
4. Attention metadata **compacts** non-null physical pages; `seq_lens` = retained token count.

RoPE/query **positions** are **not** renumbered.

## Prefill vs decode

- Eviction runs from `allocate_slots` (prefill growth and decode).
- Score collector armed in `_build_attention_metadata` when H2O enabled (not CUDA graph capture).
- Scores applied in `scheduler.update_from_output` from prior step’s `h2o_block_scores`.

## Attention score source

**Proxy / reference path** in `h2o_score_collector.py` — slow, experimental.
Production FA path unchanged.

## Known slow path

`maybe_accumulate_h2o_scores_from_attention` after each attention layer when accumulator enabled.

## Naive recent-only comparator (evaluation)

No separate policy: **H2O with zero scores** evicts oldest eligible blocks among
non-recent candidates → block-granular “keep N most recent blocks” (with `R` recent
explicitly protected when `R < N`).

## Tests present before evaluation

`tests/v1/kv_cache_eviction/test_h2o_policy.py` (12 cases)
`tests/v1/kv_cache_eviction/test_h2o_kv_manager.py` (5 cases)

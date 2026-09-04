# H2O KV-cache eviction baseline — evaluation report

**Run:** `2026-08-31` (raw data in `results/h2o_baseline/runs/2026-08-31/`)
**Commit:** `d1d7f48e5b35374b5e0b63b0d3a0e5fc20160c9c` (working tree dirty: benchmark harness + `recent` comparator)
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`, dtype `bfloat16`, block size 16
**GPU:** NVIDIA H100 NVL (95.8 GiB) · torch 2.11.0+cu130 · CUDA 13.0
**Decoding:** greedy (`temperature=0`), `seed=42`, prefix caching disabled, `enforce_eager=True`
**Configurations:** 42 runs, 0 failures

> A previous run (`archive_2026-08-27/`) used `facebook/opt-125m`, which scored
> 0% retrieval accuracy even with a full KV cache. That made quality
> measurement impossible, so this run switched to a model that solves the task
> at full KV. No H2O algorithm changes were made.

---

## Executive summary

1. **The implementation is correct.** H2O with a budget large enough to prevent
   eviction reproduces full-KV greedy token IDs **exactly**. Block budgets were
   respected in **every** run (42/42).
2. **H2O genuinely reduces physical KV capacity.** Peak occupied KV blocks scale
   almost exactly with the retention target (671 → 81 blocks from 100% → 12.5%),
   and blocks are returned to the pool after each request.
3. **Quality degrades sharply below 75% retention**, and the damage is
   concentrated in **early-context** facts.
4. **H2O is currently indistinguishable from recent-only eviction on
   long prompts.** This is a real limitation, not a benchmark artifact: 97–99.6%
   of evictions happen at *prefill allocation time*, before any attention scores
   exist, so victim selection falls back to allocation order. See
   [Critical finding](#critical-finding-h2o-is-score-blind-on-long-prompts).
5. **The score collector, not eviction, dominates cost.** H2O at 100% retention
   (zero evictions) is **15× slower** than full KV, while recent-only eviction
   runs at full-KV speed. The overhead scales linearly with context length.

---

## 1. Correctness

| Control | Result |
| --- | --- |
| Unit tests `tests/v1/kv_cache_eviction/` (17 cases) | **17 passed** |
| Inline policy checks (`correctness_checks.py`) | **PASS** |
| H2O-disabled (full KV) execution | **PASS** — 64/64 tokens, no errors |
| H2O with no effective eviction runs | **PASS** — 0 blocks evicted, as intended |
| Long workload triggers real eviction | **PASS** — up to 2172 blocks evicted at 8192 tokens |
| Retained blocks respect the budget | **PASS** — `peak_retained ≤ budget` in 42/42 runs |
| Blocks returned/reusable after request | **PASS** — pool returns to 1 used block of 159 796 |
| NaNs / empty / malformed output | **None observed** |
| Per-request H2O state isolation | **PASS** (unit test + 5-request retrieval batches) |

### Deterministic equivalence control

1024-token prompt, 64 greedy tokens, identical seed:

| Comparison | Token IDs match | Evictions |
| --- | --- | --- |
| Full KV vs H2O with budget ≥ required blocks | **Yes (exact)** | 0 vs 0 |

Greedy token IDs are **bit-identical**, so H2O-100 is a valid control: any quality
difference at lower retention is attributable to eviction, not to backend
numerics or instrumentation. Peak occupancy was also identical (66 blocks each).

---

## 2. Quality — synthetic needle retrieval

2048-token context, 5 facts at fixed positions, 24 generated tokens, identical
prompts and seeds across policies. Quality metric = exact-match retrieval accuracy.

| Policy | Target | Actual retention | Accuracy | Δ vs Full | Blocks evicted | Peak retained | Budget respected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full KV | 100% | 1.000 | **100%** | — | 0 | — | n/a |
| H2O | 100% | 1.000 | **100%** | 0 | 0 | 0 | yes |
| H2O | 75% | 0.748 | **80%** | −20 pts | 164 | 101 | yes |
| H2O | 50% | 0.496 | **40%** | −60 pts | 333 | 67 | yes |
| H2O | 25% | 0.244 | **20%** | −80 pts | 502 | 33 | yes |
| H2O | 12.5% | 0.119 | **20%** | −80 pts | 590 | 16 | yes |
| Recent-only | 75% | 0.748 | **80%** | −20 pts | 164 | 101 | yes |
| Recent-only | 50% | 0.496 | **40%** | −60 pts | 333 | 67 | yes |
| Recent-only | 25% | 0.244 | **20%** | −80 pts | 502 | 33 | yes |
| Recent-only | 12.5% | 0.119 | **20%** | −80 pts | 589 | 16 | yes |

Actual retention tracks the target closely (max deviation 0.006).

### Retrieval accuracy by fact position

`hit` = fact recalled correctly. Position is the fraction of the context before the fact.

| Config | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
| --- | --- | --- | --- | --- | --- |
| Full KV | hit | hit | hit | hit | hit |
| H2O 100% | hit | hit | hit | hit | hit |
| H2O 75% | **miss** | hit | hit | hit | hit |
| H2O 50% | **miss** | **miss** | **miss** | hit | hit |
| H2O 25% | **miss** | **miss** | **miss** | **miss** | hit |
| H2O 12.5% | **miss** | **miss** | **miss** | **miss** | hit |
| Recent-only 75% | **miss** | hit | hit | hit | hit |
| Recent-only 50% | **miss** | **miss** | **miss** | hit | hit |
| Recent-only 25% | **miss** | **miss** | **miss** | **miss** | hit |
| Recent-only 12.5% | **miss** | **miss** | **miss** | **miss** | hit |

**Early-context facts are the first and most severe casualty.** The failure
boundary sweeps strictly left-to-right as retention drops: only facts inside the
surviving recent window are recalled. H2O's attention scores rescue nothing that
recency would not already keep.

---

## 3. Critical finding: H2O is score-blind on long prompts

H2O and recent-only produce **identical accuracy at every retention level and
every fact position**, and near-identical eviction counts. Inspecting the
eviction traces explains why.

Eviction is triggered inside block allocation. For a long prompt, the request is
already far over budget when its prompt blocks are allocated — **before a single
attention score has been computed**. All candidate scores are then `0.0`, so
`select_h2o_victims` falls through to its deterministic tie-break (allocation
order) and evicts the oldest blocks: exactly the recent-only policy.

Fraction of evicted blocks chosen with all-zero scores (2048-token retrieval):

| Config | Score-blind evictions | Total | Share |
| --- | --- | --- | --- |
| H2O 75% | 160 | 164 | **97.6%** |
| H2O 50% | 330 | 333 | **99.1%** |
| H2O 25% | 500 | 502 | **99.6%** |
| H2O 12.5% | 585 | 590 | **99.2%** |

The scoring path itself **does work**: later, decode-time evictions show dense
non-zero scores (max block mass ≈ 300–362) and pick interior blocks
(e.g. victim 113 out of 130) rather than the oldest. But those account for only
2–5 blocks per run.

Consequently the current baseline measures **block-granular recency eviction**
on prompt-dominated workloads, not attention-guided H2O. On a short prompt with
proportionally more decode-time evictions the two policies **do** diverge — the
1024-token control produced different token IDs for H2O-25 vs recent-25.

This is a property of the implementation (evict-at-allocation), not a benchmark
bug, so it was left unchanged and is reported as-is.

---

## 4. Cache usage and memory

2048-token retrieval workload. Page size 8192 B; KV arena 159 796 blocks (1248 MiB).

| Policy | Retention | Peak occupied blocks | Peak occupied KV | Δ vs Full | CUDA reserved | Blocks reused |
| --- | --- | --- | --- | --- | --- | --- |
| Full KV | 100% | 671 | 5.2 MiB | — | 33 294 MiB | yes |
| H2O | 100% | 671 | 5.2 MiB | 0% | 33 296 MiB | yes |
| H2O | 75% | 506 | 4.0 MiB | −24.6% | 33 296 MiB | yes |
| H2O | 50% | 336 | 2.6 MiB | −49.9% | 33 296 MiB | yes |
| H2O | 25% | 166 | 1.3 MiB | −75.3% | 33 294 MiB | yes |
| H2O | 12.5% | 81 | 0.6 MiB | −87.9% | 33 296 MiB | yes |

**Interpretation caveat (important).** CUDA reserved memory is **flat at
~33.3 GiB across every policy**, because vLLM preallocates the KV arena from
`gpu_memory_utilization` at startup regardless of how much is used. Reading
CUDA reserved memory alone would wrongly suggest H2O saves nothing.

The meaningful quantity is **physical KV capacity consumed by active requests**,
which does scale essentially linearly with the retention target (−24.6% / −49.9%
/ −75.3% / −87.9% against targets of −25% / −50% / −75% / −87.5%). After each
request the pool returns to 1 used block out of 159 796, confirming evicted and
released blocks become reusable rather than leaking.

---

## 5. Performance

Prompt 2048 tokens, 128 generated tokens, **5 measured repetitions after 1
discarded warmup**. TTFT is measured as a separate `max_tokens=1` request; decode
latency is the residual. Mean ± standard deviation, n = 5.

| Policy | Retention | TTFT (ms) | Inter-token latency (ms) | Decode tok/s | End-to-end (s) | Slowdown vs Full |
| --- | --- | --- | --- | --- | --- | --- |
| Full KV | 100% | 20.2 ± 4.6 | 13.60 ± 0.11 | **73.5 ± 0.6** | 1.75 ± 0.02 | 1.0× |
| H2O | 100% | 175.5 ± 36.3 | 204.88 ± 1.33 | **4.9 ± 0.0** | 26.20 ± 0.19 | **15.0×** |
| H2O | 75% | 160.3 ± 32.9 | 180.79 ± 0.69 | 5.5 ± 0.0 | 23.12 ± 0.11 | 13.2× |
| H2O | 50% | 131.5 ± 26.5 | 146.27 ± 1.26 | 6.8 ± 0.1 | 18.71 ± 0.18 | 10.7× |
| H2O | 25% | 110.0 ± 0.8 | 112.67 ± 0.37 | 8.9 ± 0.0 | 14.42 ± 0.05 | 8.2× |
| Recent-only | 75% | 18.7 ± 5.0 | 13.78 ± 0.76 | 72.8 ± 4.1 | 1.77 ± 0.10 | 1.01× |
| Recent-only | 50% | 17.2 ± 5.0 | 9.49 ± 2.50 | 111.8 ± 30.5 | 1.22 ± 0.32 | 0.70× |
| Recent-only | 25% | 23.3 ± 0.9 | 14.46 ± 0.22 | 69.2 ± 1.0 | 1.86 ± 0.03 | 1.06× |

**Recent-only eviction runs at full-KV speed** (it shares the same eviction
machinery but sets `h2o_collect_scores=False`). H2O at 100% retention performs
*zero* evictions yet is 15× slower. Therefore **all** of the H2O overhead comes
from the reference attention-score collector, none from the eviction mechanism.

*Anomaly noted:* recent-only at 50% shows high variance (111.8 ± 30.5 tok/s,
faster than full KV). A shorter attention span over fewer retained blocks
plausibly speeds decode, but the wide spread suggests scheduling noise; treat
this single point as low-confidence.

---

## 6. Context-length sweep

Fixed 128 generated tokens, 3 repetitions after 1 warmup. All lengths from 512
to **8192 (maximum tested)** ran successfully.

| Prompt | Policy | E2E (s) | Decode tok/s | Blocks evicted | Peak occupied | Full-cache est. | Budget respected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | Full KV | 1.79 ± 0.02 | 71.7 | 0 | 40 | 40 | n/a |
| 512 | H2O 75% | 9.39 ± 0.05 | 13.6 | 30 | 31 | 40 | yes |
| 512 | H2O 50% | 8.04 ± 0.19 | 15.9 | 90 | 21 | 40 | yes |
| 512 | H2O 25% | 6.73 ± 0.07 | 19.0 | 150 | 11 | 40 | yes |
| 1024 | Full KV | 1.75 ± 0.01 | 73.2 | 0 | 70 | 72 | n/a |
| 1024 | H2O 75% | 14.05 ± 0.06 | 9.1 | 66 | 55 | 72 | yes |
| 1024 | H2O 50% | 11.77 ± 0.06 | 10.9 | 174 | 37 | 72 | yes |
| 1024 | H2O 25% | 9.29 ± 0.03 | 13.8 | 282 | 19 | 72 | yes |
| 2048 | Full KV | 1.69 ± 0.13 | 76.4 | 0 | 131 | 136 | n/a |
| 2048 | H2O 75% | 22.66 ± 0.57 | 5.6 | 144 | 103 | 136 | yes |
| 2048 | H2O 50% | 17.40 ± 2.31 | 7.4 | 348 | 69 | 136 | yes |
| 2048 | H2O 25% | 14.00 ± 0.69 | 9.1 | 552 | 35 | 136 | yes |
| 4096 | Full KV | 1.77 ± 0.03 | 72.8 | 0 | 253 | 264 | n/a |
| 4096 | H2O 75% | 38.83 ± 4.17 | 3.3 | 300 | 199 | 264 | yes |
| 4096 | H2O 50% | 30.72 ± 3.42 | 4.2 | 696 | 133 | 264 | yes |
| 4096 | H2O 25% | 23.79 ± 0.66 | 5.4 | 1092 | 67 | 264 | yes |
| 8192 | Full KV | 1.79 ± 0.01 | 73.3 | 0 | 497 | 520 | n/a |
| 8192 | H2O 75% | 76.15 ± 0.51 | 1.7 | 612 | 391 | 520 | yes |
| 8192 | H2O 50% | 58.55 ± 2.20 | 2.2 | 1392 | 261 | 520 | yes |
| 8192 | H2O 25% | 43.57 ± 0.58 | 2.9 | 2172 | 131 | 520 | yes |

**When eviction begins:** as soon as the prompt's blocks exceed the budget,
i.e. during prefill for every length tested.

**Full-KV latency is flat** (~1.75 s) from 512 to 8192 tokens — decode of 128
tokens dominates and the H100 absorbs the longer attention span. **H2O latency
grows linearly with context** (9.4 s → 76.2 s, a 43× slowdown at 8192), because
the reference collector materializes a softmax over all retained pages for every
layer and every decoded token.

Within a length, **lower retention is consistently faster for H2O** (fewer
retained blocks to score), which is the opposite of the usual quality/speed
trade-off and is another symptom of scoring being the bottleneck.

---

## 7. Long-context benchmark subset

**Not run.** The harness has no LongBench-style task configured, and the task
brief excludes building a new benchmark framework or adding substantial
dependencies. The synthetic needle retrieval above covers the retrieval axis;
long-context QA and summarization were not measured.

---

## 8. Key observations

**At what retention fraction does quality first noticeably decline?**
Between 100% and 75%. H2O-100 is exact; 75% already loses the earliest fact
(100% → 80%).

**How bad is 50% retention?** Severe: accuracy 40% (−60 pts). Only facts in the
newest half of the context survive.

**How bad is 25%?** 20% accuracy (−80 pts) — only the fact at position 0.95 is
recalled. 12.5% is no worse (also 20%), since the last fact still sits inside the
recent window.

**Does H2O preserve long-range retrieval better than recent-only?**
**No — they are identical** at every retention level and every position, because
97–99.6% of evictions are made before any attention scores exist (Section 3).

**Are early-context facts especially vulnerable?** **Yes, decisively.** The
position-0.05 fact is lost at the very first eviction level (75%) and never
recovered; failures propagate strictly from oldest to newest.

**Does H2O really reduce occupied physical KV capacity?** **Yes.** Peak occupied
blocks fall 671 → 506 → 336 → 166 → 81 (−24.6% to −87.9%), tracking targets
closely, and blocks are returned to the pool. Note that CUDA *reserved* memory
does not move, because the KV arena is preallocated.

**What overhead does attention-score collection introduce?** It is the entire
cost: 15× slowdown at 2048 tokens with zero evictions, up to 43× at 8192 tokens.
Recent-only eviction, identical except that scoring is disabled, runs at full-KV
speed.

**Is the implementation compute-bound by a slow/reference attention backend?**
**Yes.** `h2o_score_collector` materializes QKᵀ and softmax in float32 outside
the fused kernel, per layer, per decoded token. Inter-token latency rises from
13.6 ms to 204.9 ms.

**Does H2O-100 match full KV closely enough to validate the evaluation?**
**Yes — exactly.** Identical greedy token IDs, identical retrieval accuracy
(100%), identical peak occupancy (671 blocks). The evaluation is sound.

---

## 9. Suspicious findings and anomalies

| Check | Status |
| --- | --- |
| Zero evictions at low retention | Not observed — evictions scale as expected |
| Actual vs requested retention | Within 0.006 everywhere |
| H2O-100 differing from full KV | Not observed — bit-identical |
| Retained blocks exceeding budget | Never (42/42 respected) |
| Accuracy improving under severe eviction | Not observed (monotone decline) |
| NaNs / empty / malformed output | None |
| GPU OOM for a smaller cache | None |
| Requests sharing H2O state | None (per-request isolation verified) |
| **H2O ≡ recent-only on long prompts** | **Confirmed real** — see Section 3 |
| Timing outlier | recent-only 50%: 111.8 ± 30.5 tok/s, high variance |
| Sequential in-process engines leak GPU memory | Harness bug, fixed by one subprocess per config |

---

## 10. Reproduction

```bash
cd /fast-lab-share/aniketc2/vllm
export PYTHONPATH=. HF_HOME=/fast-lab-share/aniketc2/hf-home
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # instrumentation reads the in-process scheduler

# Unit tests + inline policy checks
.venv/bin/python benchmarks/h2o/correctness_checks.py

# Full suite (42 configurations, ~2 h on one H100)
.venv/bin/python -m benchmarks.h2o.run_benchmarks \
  --run-name 2026-08-31 --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 10240 --stages control retrieval timing length \
  --fractions 1.0 0.75 0.5 0.25 0.125 \
  --retrieval-context 2048 --timing-prompt-tokens 2048 --repetitions 5 \
  --lengths 512 1024 2048 4096 8192 --timeout 5400

# Plots
.venv/bin/python -m benchmarks.h2o.plot_results \
  --run-dir results/h2o_baseline/runs/2026-08-31
```

### Artifacts

| Path | Contents |
| --- | --- |
| `results/h2o_baseline/results.{csv,json}` | 42 aggregate rows |
| `results/h2o_baseline/per_example_results.{csv,json}` | 50 per-example retrieval results with fact positions |
| `results/h2o_baseline/manifest.json` | Full run configuration |
| `results/h2o_baseline/plots/` | 6 plots |
| `results/h2o_baseline/runs/2026-08-31/raw/` | Per-configuration JSON incl. eviction traces |
| `results/h2o_baseline/runs/2026-08-31/logs/` | Per-configuration stdout/stderr |
| `results/h2o_baseline/correctness_controls.json` | Correctness control output |
| `results/h2o_baseline/archive_2026-08-27/` | Previous (invalid, OPT-125m) run, preserved |

---

## 11. Verdict: is this baseline trustworthy?

**Yes for memory and correctness; with one major caveat for policy claims.**

Trustworthy to build on:
- Correctness is established (exact full-KV equivalence, budgets always respected,
  no leaks, clean block reuse).
- Physical KV savings are real, measured, and match retention targets.
- The quality-vs-retention curve and position breakdown are valid and reproducible.

Must be accounted for before drawing conclusions about H2O *as a policy*:
1. **On prompt-dominated workloads this baseline is effectively recency
   eviction**, since ~99% of victims are chosen before any score exists. Any
   claim of the form "H2O beats recency" is not supported by this data — the two
   are identical here. Comparisons should either target decode-dominated
   workloads or the evict-at-allocation behaviour should be revisited.
2. **Absolute timings are not deployment-representative.** The 15–43× slowdown
   is an artifact of the reference score collector, not of KV eviction. Use
   recent-only as the speed reference for the eviction mechanism itself.

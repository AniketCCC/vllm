# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness controls for the H2O baseline.

Checks, in order:
1. H2O-disabled (full KV) execution works.
2. H2O with a budget >= required blocks runs and performs no eviction.
3. Full KV vs H2O-no-eviction produce identical greedy token ids.
4. A long workload under a small budget triggers real eviction, respects the
   budget, and returns non-empty, finite output.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.h2o.common import (
    estimate_full_blocks,
    git_metadata,
    gpu_name,
    make_llm_kwargs,
    results_dir,
    retention_blocks,
    use_inprocess_engine,
    utc_now,
)
from benchmarks.h2o.instrumentation import (
    eviction_stats,
    snapshot_cache,
)


def build_prompt(tokenizer, num_tokens: int) -> tuple[str, list[int]]:
    """Deterministic prompt of approximately ``num_tokens`` tokens."""
    seed_text = (
        "The study of efficient transformer inference considers memory "
        "bandwidth, cache residency, and attention sparsity in detail. "
    )
    ids: list[int] = []
    while len(ids) < num_tokens:
        ids.extend(tokenizer.encode(seed_text, add_special_tokens=False))
    ids = ids[:num_tokens]
    return tokenizer.decode(ids), ids


def generate(llm, prompt: str, max_tokens: int, seed: int):
    from vllm import SamplingParams

    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=seed)
    out = llm.generate([prompt], sampling)[0]
    return list(out.outputs[0].token_ids), out.outputs[0].text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--h2o-recent-blocks", type=int, default=4)
    parser.add_argument("--evict-fraction", type=float, default=0.25)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_inprocess_engine()
    from vllm import LLM

    commit, dirty = git_metadata()
    report: dict = {
        "timestamp_utc": utc_now(),
        "git_commit": commit,
        "git_dirty": dirty,
        "model": args.model,
        "gpu": gpu_name(),
        "dtype": args.dtype,
        "block_size": args.block_size,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "command": sys.argv,
        "checks": {},
    }
    failures: list[str] = []

    def base_kwargs(policy, max_blocks):
        return make_llm_kwargs(
            args.model,
            policy,
            max_blocks,
            args.h2o_recent_blocks,
            args.block_size,
            args.gpu_memory_utilization,
            args.max_model_len,
            args.seed,
            dtype=args.dtype,
        )

    # --- Check 1: full KV runs -------------------------------------------------
    llm_full = LLM(**base_kwargs("full_kv", None))
    tok = llm_full.get_tokenizer()
    prompt, prompt_ids = build_prompt(tok, args.prompt_tokens)
    ids_full, text_full = generate(llm_full, prompt, args.max_tokens, args.seed)
    snap_full = snapshot_cache(llm_full)
    report["checks"]["full_kv_runs"] = {
        "passed": len(ids_full) > 0,
        "generated_tokens": len(ids_full),
        "num_gpu_blocks": snap_full.num_gpu_blocks,
        "page_size_bytes": snap_full.page_size_bytes,
        "sample": text_full[:120],
    }
    if len(ids_full) == 0:
        failures.append("full_kv produced no tokens")
    del llm_full

    total_tokens = len(prompt_ids) + len(ids_full)
    full_blocks = estimate_full_blocks(total_tokens, args.block_size)
    report["full_cache_blocks_est"] = full_blocks

    # --- Check 2 & 3: H2O with no effective eviction ---------------------------
    no_evict_budget = full_blocks + 8
    llm_noev = LLM(**base_kwargs("h2o", no_evict_budget))
    ids_noev, text_noev = generate(llm_noev, prompt, args.max_tokens, args.seed)
    ev_noev = eviction_stats(llm_noev)
    report["checks"]["h2o_no_eviction_runs"] = {
        "passed": len(ids_noev) > 0 and ev_noev.num_blocks_evicted == 0,
        "budget": no_evict_budget,
        "instrumentation_available": ev_noev.available,
        "num_eviction_events": ev_noev.num_eviction_events,
        "num_blocks_evicted": ev_noev.num_blocks_evicted,
        "generated_tokens": len(ids_noev),
    }
    if ev_noev.num_blocks_evicted != 0:
        failures.append(
            f"H2O-no-eviction evicted {ev_noev.num_blocks_evicted} blocks"
        )

    ids_match = ids_full == ids_noev
    report["checks"]["equivalence_full_vs_h2o_no_eviction"] = {
        "passed": ids_match,
        "token_ids_match": ids_match,
        "text_match": text_full == text_noev,
        "num_tokens_full": len(ids_full),
        "num_tokens_h2o": len(ids_noev),
        "first_divergence_index": next(
            (
                i
                for i, (a, b) in enumerate(zip(ids_full, ids_noev))
                if a != b
            ),
            None,
        ),
        "full_sample": text_full[:120],
        "h2o_sample": text_noev[:120],
    }
    if not ids_match:
        failures.append("full KV vs H2O-no-eviction token ids differ")
    del llm_noev

    # --- Check 4: real eviction under a small budget ---------------------------
    budget, actual_frac = retention_blocks(
        full_blocks, args.evict_fraction, args.h2o_recent_blocks
    )
    llm_ev = LLM(**base_kwargs("h2o", budget))
    ids_ev, text_ev = generate(llm_ev, prompt, args.max_tokens, args.seed)
    ev = eviction_stats(llm_ev)
    snap_ev = snapshot_cache(llm_ev)
    budget_ok = ev.budget_respected
    report["checks"]["h2o_real_eviction"] = {
        "passed": bool(
            ev.available
            and ev.num_blocks_evicted > 0
            and budget_ok
            and len(ids_ev) > 0
        ),
        "target_fraction": args.evict_fraction,
        "actual_fraction": actual_frac,
        "budget": budget,
        "recent_window": args.h2o_recent_blocks,
        "num_eviction_events": ev.num_eviction_events,
        "num_blocks_evicted": ev.num_blocks_evicted,
        "peak_retained_blocks": ev.peak_retained_blocks,
        "max_block_count_before_evict": ev.max_block_count_before_evict,
        "budget_respected": budget_ok,
        "generated_tokens": len(ids_ev),
        "output_nonempty": len(text_ev.strip()) > 0,
        "used_blocks_after_run": snap_ev.num_used_blocks,
        "sample": text_ev[:120],
    }
    if not ev.available:
        failures.append("eviction instrumentation unavailable")
    elif ev.num_blocks_evicted == 0:
        failures.append(
            f"no eviction at {args.evict_fraction:.3f} retention (budget={budget})"
        )
    if budget_ok is False:
        failures.append(
            f"peak retained {ev.peak_retained_blocks} exceeds budget {budget}"
        )
    if len(ids_ev) == 0:
        failures.append("H2O under eviction produced no tokens")
    del llm_ev

    report["failures"] = failures
    report["all_passed"] = not failures

    out = results_dir() / "correctness_controls.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["checks"], indent=2))
    print(f"\nWrote {out}")
    if failures:
        print("FAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)
    print("All correctness controls PASSED")


if __name__ == "__main__":
    main()

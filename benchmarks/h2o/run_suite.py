# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main H2O baseline evaluation harness.

Runs retention sweeps, equivalence controls, and length microbenchmarks.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

import torch

from benchmarks.h2o.common import (
    ExperimentRow,
    append_results,
    cuda_version,
    estimate_full_blocks,
    git_metadata,
    gpu_name,
    make_llm_kwargs,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    retention_blocks,
    utc_now,
)


def _generate_once(
    llm,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> tuple[list[int], str, float]:
    from vllm import SamplingParams

    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=seed)
    t0 = time.perf_counter()
    out = llm.generate([prompt], sampling)[0]
    elapsed = time.perf_counter() - t0
    ids = list(out.outputs[0].token_ids)
    text = out.outputs[0].text
    return ids, text, elapsed


def run_equivalence(args: argparse.Namespace) -> ExperimentRow:
    from vllm import LLM

    commit, dirty = git_metadata()
    tok = None
    # Build prompt
    llm_full = LLM(
        **make_llm_kwargs(
            args.model,
            "full_kv",
            None,
            args.h2o_recent_blocks,
            args.block_size,
            args.gpu_memory_utilization,
            args.max_model_len,
            args.seed,
        )
    )
    tok = llm_full.get_tokenizer()
    prompt_ids = tok.encode(args.prompt_text, add_special_tokens=False)
    if len(prompt_ids) > args.prompt_tokens:
        prompt_ids = prompt_ids[:args.prompt_tokens]
    prompt = tok.decode(prompt_ids)

    ids_full, text_full, t_full = _generate_once(
        llm_full, prompt, args.max_tokens, args.seed
    )
    full_blocks = estimate_full_blocks(
        len(prompt_ids) + len(ids_full), args.block_size
    )
    del llm_full

    # H2O 100%: budget equals full blocks (no effective eviction)
    llm_h2o = LLM(
        **make_llm_kwargs(
            args.model,
            "h2o",
            full_blocks,
            args.h2o_recent_blocks,
            args.block_size,
            args.gpu_memory_utilization,
            args.max_model_len,
            args.seed,
        )
    )
    ids_h2o, text_h2o, t_h2o = _generate_once(
        llm_h2o, prompt, args.max_tokens, args.seed
    )
    del llm_h2o

    match = ids_full == ids_h2o
    row = ExperimentRow(
        run_id="equivalence_full_vs_h2o100",
        timestamp_utc=utc_now(),
        command=sys.argv,
        git_commit=commit,
        git_dirty=dirty,
        model=args.model,
        dtype="auto",
        gpu_name=gpu_name(),
        torch_version=torch.__version__,
        cuda_version=cuda_version(),
        vllm_commit=commit,
        policy="equivalence",
        h2o_enabled=True,
        retention_target_fraction=1.0,
        retention_actual_fraction=1.0,
        h2o_max_blocks=full_blocks,
        h2o_recent_blocks=args.h2o_recent_blocks,
        block_size=args.block_size,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(ids_full),
        num_requests=1,
        peak_retained_logical_blocks=full_blocks,
        peak_retained_physical_blocks=full_blocks,
        full_cache_blocks_est=full_blocks,
        num_evictions_est=0,
        estimated_kv_bytes=None,
        gpu_peak_mem_allocated_mb=None,
        gpu_peak_mem_reserved_mb=None,
        wall_clock_s=t_full + t_h2o,
        wall_clock_std=None,
        tokens_per_sec=len(ids_full) / t_full,
        tokens_per_sec_std=None,
        time_to_first_token_s=None,
        repetitions=1,
        seed=args.seed,
        temperature=0.0,
        workload="equivalence",
        output_text_sample=text_full[:120],
        extra={
            "token_ids_match": match,
            "text_match": text_full == text_h2o,
            "full_tokens": ids_full,
            "h2o_tokens": ids_h2o,
        },
    )
    print(f"Equivalence token_ids_match={match}")
    return row


def run_length_sweep(args: argparse.Namespace) -> list[ExperimentRow]:
    from vllm import LLM

    commit, dirty = git_metadata()
    rows: list[ExperimentRow] = []
    policies: list[tuple[str, float | None]] = [("full_kv", 1.0)]
    for frac in args.fractions:
        if frac < 1.0:
            policies.append(("h2o", frac))
            if args.include_recent:
                policies.append(("recent_only", frac))

    tok = None
    for prompt_len in args.lengths:
        for policy_name, frac in policies:
            # Estimate full blocks for this length
            if tok is None:
                llm_tmp = LLM(
                    **make_llm_kwargs(
                        args.model,
                        "full_kv",
                        None,
                        args.h2o_recent_blocks,
                        args.block_size,
                        args.gpu_memory_utilization,
                        args.max_model_len,
                        args.seed,
                    )
                )
                tok = llm_tmp.get_tokenizer()
                del llm_tmp

            prompt_ids = list(range(1000, 1000 + prompt_len))
            prompt = tok.decode(prompt_ids)
            full_blocks = estimate_full_blocks(
                prompt_len + args.max_tokens, args.block_size
            )
            if frac is None or frac >= 1.0:
                max_blocks = full_blocks
                target_frac = 1.0
                actual_frac = 1.0
            else:
                max_blocks, actual_frac = retention_blocks(
                    full_blocks, frac, args.h2o_recent_blocks
                )
                target_frac = frac

            llm = LLM(
                **make_llm_kwargs(
                    args.model,
                    policy_name,
                    max_blocks if policy_name != "full_kv" else None,
                    args.h2o_recent_blocks,
                    args.block_size,
                    args.gpu_memory_utilization,
                    args.max_model_len,
                    args.seed,
                )
            )

            times: list[float] = []
            gen_tokens = 0
            reset_peak_gpu_memory()
            for rep in range(args.repetitions):
                ids, text, elapsed = _generate_once(
                    llm, prompt, args.max_tokens, args.seed + rep
                )
                times.append(elapsed)
                gen_tokens = len(ids)
                if rep == 0:
                    warmup = elapsed  # first rep may include compile

            alloc_mb, res_mb = peak_gpu_memory_mb()
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0.0
            mean_tps = gen_tokens / mean_t if mean_t > 0 else 0.0

            rows.append(
                ExperimentRow(
                    run_id=f"length_{prompt_len}_{policy_name}_{target_frac}",
                    timestamp_utc=utc_now(),
                    command=f"length_sweep len={prompt_len} policy={policy_name}",
                    git_commit=commit,
                    git_dirty=dirty,
                    model=args.model,
                    dtype="auto",
                    gpu_name=gpu_name(),
                    torch_version=torch.__version__,
                    cuda_version=cuda_version(),
                    vllm_commit=commit,
                    policy=policy_name,
                    h2o_enabled=policy_name != "full_kv",
                    retention_target_fraction=target_frac,
                    retention_actual_fraction=actual_frac,
                    h2o_max_blocks=max_blocks if policy_name != "full_kv" else None,
                    h2o_recent_blocks=args.h2o_recent_blocks,
                    block_size=args.block_size,
                    prompt_tokens=prompt_len,
                    generated_tokens=gen_tokens,
                    num_requests=1,
                    peak_retained_logical_blocks=max_blocks,
                    peak_retained_physical_blocks=max_blocks,
                    full_cache_blocks_est=full_blocks,
                    num_evictions_est=max(0, full_blocks - max_blocks),
                    estimated_kv_bytes=None,
                    gpu_peak_mem_allocated_mb=alloc_mb,
                    gpu_peak_mem_reserved_mb=res_mb,
                    wall_clock_s=mean_t,
                    wall_clock_std=std_t,
                    tokens_per_sec=mean_tps,
                    tokens_per_sec_std=None,
                    time_to_first_token_s=None,
                    repetitions=args.repetitions,
                    seed=args.seed,
                    temperature=0.0,
                    workload="length_sweep",
                    output_text_sample=text[:80],
                    extra={"prompt_length": prompt_len},
                )
            )
            del llm
            print(
                f"len={prompt_len} {policy_name} frac={actual_frac:.2f} "
                f"tok/s={mean_tps:.1f} t={mean_t:.3f}s"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--prompt-text", default="The history of machine learning begins with ")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--h2o-recent-blocks", type=int, default=1)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25],
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--include-recent", action="store_true", default=True)
    parser.add_argument("--skip-equivalence", action="store_true")
    parser.add_argument("--skip-length-sweep", action="store_true")
    args = parser.parse_args()

    all_rows: list[ExperimentRow] = []
    if not args.skip_equivalence:
        all_rows.append(run_equivalence(args))
    if not args.skip_length_sweep:
        all_rows.extend(run_length_sweep(args))

    append_results(all_rows)
    print(f"Wrote {len(all_rows)} rows to results/h2o_baseline/")


if __name__ == "__main__":
    main()

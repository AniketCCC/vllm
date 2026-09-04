# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic long-context retrieval benchmark for H2O baseline."""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass

from benchmarks.h2o.common import (
    ExperimentRow,
    append_results,
    estimate_full_blocks,
    git_metadata,
    gpu_name,
    make_llm_kwargs,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    retention_blocks,
    utc_now,
)


@dataclass
class RetrievalCase:
    code: str
    position_fraction: float
    prompt: str
    expected: str


def _distractor_words(n: int, rng: random.Random) -> str:
    words = [
        "algorithm",
        "tensor",
        "memory",
        "context",
        "inference",
        "latency",
        "throughput",
        "parameter",
        "embedding",
        "attention",
    ]
    return " ".join(rng.choice(words) for _ in range(n))


def build_retrieval_cases(
    context_tokens: int,
    tokenizer,
    positions: list[float],
    seed: int,
) -> list[RetrievalCase]:
    rng = random.Random(seed)
    cases: list[RetrievalCase] = []
    for frac in positions:
        code_num = rng.randint(10000, 99999)
        code = f"QX{code_num}"
        secret = str(rng.randint(100000, 999999))
        # Place fact near fraction of context (word-based proxy).
        words_before = max(10, int(context_tokens * frac))
        prefix = _distractor_words(words_before, rng)
        suffix = _distractor_words(context_tokens - words_before, rng)
        text = (
            f"The special code for object {code} is {secret}. "
            f"{prefix} {suffix} "
            f"Question: What is the special code for object {code}? Answer:"
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Trim to target length if needed
        if len(ids) > context_tokens + 50:
            text = tokenizer.decode(ids[:context_tokens + 50])
        cases.append(
            RetrievalCase(
                code=code,
                position_fraction=frac,
                prompt=text,
                expected=secret,
            )
        )
    return cases


def run_retrieval_suite(args: argparse.Namespace) -> list[ExperimentRow]:
    from vllm import LLM, SamplingParams

    commit, dirty = git_metadata()
    rows: list[ExperimentRow] = []

    policies: list[tuple[str, float | None]] = [("full_kv", 1.0)]
    for frac in args.fractions:
        if frac < 1.0:
            policies.append(("h2o", frac))
            if args.include_recent:
                policies.append(("recent_only", frac))

    # Warm run to estimate blocks
    llm0 = LLM(
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
    tok = llm0.get_tokenizer()
    probe = build_retrieval_cases(
        args.context_tokens, tok, [0.5], args.seed
    )[0]
    probe_ids = tok.encode(probe.prompt, add_special_tokens=False)
    full_blocks = estimate_full_blocks(
        len(probe_ids) + args.max_tokens, args.block_size
    )
    del llm0

    positions = args.positions
    for policy_name, frac in policies:
        if frac is None or frac >= 1.0:
            max_blocks = full_blocks
            target_frac = 1.0
            actual_frac = 1.0
        else:
            max_blocks, actual_frac = retention_blocks(
                full_blocks, frac, args.h2o_recent_blocks
            )
            target_frac = frac

        llm_kwargs = make_llm_kwargs(
            args.model,
            policy_name,
            max_blocks if policy_name != "full_kv" else None,
            args.h2o_recent_blocks,
            args.block_size,
            args.gpu_memory_utilization,
            args.max_model_len,
            args.seed,
        )
        if policy_name == "h2o" and args.h2o_debug:
            llm_kwargs["h2o_debug"] = True

        cmd = f"retrieval policy={policy_name} frac={target_frac}"
        llm = LLM(**llm_kwargs)
        tok = llm.get_tokenizer()
        cases = build_retrieval_cases(
            args.context_tokens, tok, positions, args.seed
        )
        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            seed=args.seed,
        )

        correct = 0
        latencies: list[float] = []
        reset_peak_gpu_memory()
        import time

        for case in cases:
            t0 = time.perf_counter()
            out = llm.generate([case.prompt], sampling)[0]
            latencies.append(time.perf_counter() - t0)
            text = out.outputs[0].text
            if case.expected in text or re.search(
                rf"\b{case.expected}\b", text
            ):
                correct += 1

        alloc_mb, res_mb = peak_gpu_memory_mb()
        mean_lat = sum(latencies) / len(latencies)
        row = ExperimentRow(
            run_id=f"retrieval_{policy_name}_{target_frac}",
            timestamp_utc=utc_now(),
            command=cmd,
            git_commit=commit,
            git_dirty=dirty,
            model=args.model,
            dtype=str(args.dtype),
            gpu_name=gpu_name(),
            torch_version="",
            cuda_version=None,
            vllm_commit=commit,
            policy=policy_name,
            h2o_enabled=policy_name != "full_kv",
            retention_target_fraction=target_frac,
            retention_actual_fraction=actual_frac,
            h2o_max_blocks=max_blocks if policy_name != "full_kv" else None,
            h2o_recent_blocks=args.h2o_recent_blocks,
            block_size=args.block_size,
            prompt_tokens=len(probe_ids),
            generated_tokens=args.max_tokens,
            num_requests=len(cases),
            peak_retained_logical_blocks=max_blocks,
            peak_retained_physical_blocks=max_blocks,
            full_cache_blocks_est=full_blocks,
            num_evictions_est=None,
            estimated_kv_bytes=None,
            gpu_peak_mem_allocated_mb=alloc_mb,
            gpu_peak_mem_reserved_mb=res_mb,
            wall_clock_s=sum(latencies),
            wall_clock_std=None,
            tokens_per_sec=(args.max_tokens * len(cases)) / sum(latencies),
            tokens_per_sec_std=None,
            time_to_first_token_s=mean_lat,
            repetitions=1,
            seed=args.seed,
            temperature=0.0,
            workload="synthetic_retrieval",
            output_text_sample=f"acc={correct}/{len(cases)}",
            extra={
                "retrieval_accuracy": correct / len(cases),
                "positions": positions,
            },
        )
        rows.append(row)
        del llm

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--h2o-recent-blocks", type=int, default=1)
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--positions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--include-recent", action="store_true", default=True)
    parser.add_argument("--h2o-debug", action="store_true")
    args = parser.parse_args()

    rows = run_retrieval_suite(args)
    append_results(rows)
    for r in rows:
        print(
            f"{r.policy} target={r.retention_target_fraction} "
            f"acc={r.extra.get('retrieval_accuracy'):.2%} "
            f"tok/s={r.tokens_per_sec:.1f}"
        )


if __name__ == "__main__":
    main()

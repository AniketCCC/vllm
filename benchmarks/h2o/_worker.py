# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run ONE H2O benchmark configuration in a fresh process and dump JSON.

A separate process per configuration is required because vLLM does not release
GPU memory when an in-process ``LLM`` is deleted, and the in-process engine is
itself needed so the eviction/occupancy instrumentation can read scheduler state.

Not intended to be invoked directly; see ``run_benchmarks.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

import torch

from benchmarks.h2o.common import (
    cuda_version,
    estimate_full_blocks,
    git_metadata,
    gpu_name,
    make_llm_kwargs,
    retention_blocks,
    use_inprocess_engine,
    utc_now,
)
from benchmarks.h2o.instrumentation import (
    eviction_stats,
    reset_eviction_trace,
    snapshot_cache,
)

SEED_TEXT = (
    "The study of efficient transformer inference considers memory bandwidth, "
    "cache residency, and attention sparsity in detail. "
)

DISTRACTOR_WORDS = [
    "algorithm", "tensor", "memory", "context", "inference",
    "latency", "throughput", "parameter", "embedding", "attention",
]


def build_filler_ids(tokenizer, num_tokens: int) -> list[int]:
    ids: list[int] = []
    unit = tokenizer.encode(SEED_TEXT, add_special_tokens=False)
    while len(ids) < num_tokens:
        ids.extend(unit)
    return ids[:num_tokens]


class OccupancySampler:
    """Samples used KV blocks from the scheduler while generation runs."""

    def __init__(self, llm, interval_s: float = 0.005):
        self._llm = llm
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_blocks = 0
        self.samples: list[int] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            snap = snapshot_cache(self._llm)
            if snap.num_used_blocks is not None:
                self.samples.append(snap.num_used_blocks)
                self.peak_used_blocks = max(
                    self.peak_used_blocks, snap.num_used_blocks
                )
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def mean_used_blocks(self) -> float | None:
        active = [s for s in self.samples if s > 0]
        return statistics.mean(active) if active else None


def timed_generate(llm, prompts, max_tokens, seed, sample_occupancy=False):
    from vllm import SamplingParams

    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=seed)
    if sample_occupancy:
        with OccupancySampler(llm) as sampler:
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sampling)
            elapsed = time.perf_counter() - t0
        return outs, elapsed, sampler
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0
    return outs, elapsed, None


def make_needle_cases(tokenizer, context_tokens, positions, seed):
    """Needle-in-haystack cases with token-accurate fact placement."""
    import random

    rng = random.Random(seed)
    cases = []
    for frac in positions:
        obj = f"QX{rng.randint(10000, 99999)}"
        secret = str(rng.randint(100000, 999999))
        fact = f" The special code for object {obj} is {secret}. "
        fact_ids = tokenizer.encode(fact, add_special_tokens=False)

        filler = [
            tokenizer.encode(" " + rng.choice(DISTRACTOR_WORDS),
                             add_special_tokens=False)
            for _ in range(64)
        ]
        body: list[int] = []
        while len(body) < context_tokens:
            body.extend(filler[rng.randrange(len(filler))])
        body = body[:context_tokens]

        insert_at = min(len(body), max(0, int(len(body) * frac)))
        ctx_ids = body[:insert_at] + fact_ids + body[insert_at:]
        context = tokenizer.decode(ctx_ids)
        question = (
            f"\n\nQuestion: What is the special code for object {obj}? "
            f"Answer with the number only.\nAnswer:"
        )
        cases.append(
            {
                "object": obj,
                "expected": secret,
                "position_fraction": frac,
                "fact_token_index": insert_at,
                "context_tokens": len(ctx_ids),
                "text": context + question,
            }
        )
    return cases


def apply_chat(tokenizer, text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True,
                   choices=["timing", "retrieval", "gen"])
    p.add_argument("--policy", required=True,
                   choices=["full_kv", "h2o", "recent_only"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-fraction", type=float, default=1.0)
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--h2o-recent-blocks", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--positions", type=float, nargs="+",
                   default=[0.05, 0.25, 0.5, 0.75, 0.95])
    p.add_argument("--run-id", default="run")
    args = p.parse_args()

    use_inprocess_engine()
    from vllm import LLM

    commit, dirty = git_metadata()
    result: dict = {
        "run_id": args.run_id,
        "workload": args.workload,
        "policy": args.policy,
        "timestamp_utc": utc_now(),
        "git_commit": commit,
        "git_dirty": dirty,
        "model": args.model,
        "dtype": args.dtype,
        "gpu_name": gpu_name(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version(),
        "block_size": args.block_size,
        "seed": args.seed,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "h2o_recent_blocks": (
            args.h2o_recent_blocks if args.policy != "full_kv" else None
        ),
        "retention_target_fraction": args.target_fraction,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
    }

    # Tokenize first with a throwaway tokenizer so the budget can be sized
    # from the real prompt length before the engine is built.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.workload in ("timing", "gen"):
        prompt_ids = build_filler_ids(tokenizer, args.prompt_tokens)
        prompts = [tokenizer.decode(prompt_ids)]
        prompt_len = len(prompt_ids)
        total_tokens = prompt_len + args.max_tokens
    else:
        cases = make_needle_cases(
            tokenizer, args.prompt_tokens, args.positions, args.seed
        )
        prompts = [apply_chat(tokenizer, c["text"]) for c in cases]
        prompt_len = max(
            len(tokenizer.encode(t, add_special_tokens=False)) for t in prompts
        )
        total_tokens = prompt_len + args.max_tokens

    full_blocks = estimate_full_blocks(total_tokens, args.block_size)
    result["prompt_tokens"] = prompt_len
    result["full_cache_blocks_est"] = full_blocks

    if args.policy == "full_kv":
        budget = None
        actual_frac = 1.0
    elif args.target_fraction >= 1.0:
        budget = full_blocks
        actual_frac = 1.0
    else:
        budget, actual_frac = retention_blocks(
            full_blocks, args.target_fraction, args.h2o_recent_blocks
        )
    result["h2o_max_blocks"] = budget
    result["retention_actual_fraction"] = actual_frac

    recent = args.h2o_recent_blocks
    if budget is not None and recent >= budget:
        recent = max(0, budget - 1)
        result["h2o_recent_blocks"] = recent
        result["recent_window_clamped"] = True

    llm = LLM(
        **make_llm_kwargs(
            args.model,
            args.policy,
            budget,
            recent,
            args.block_size,
            args.gpu_memory_utilization,
            args.max_model_len,
            args.seed,
            dtype=args.dtype,
        )
    )

    snap0 = snapshot_cache(llm)
    result["kv_arena_blocks"] = snap0.num_gpu_blocks
    result["page_size_bytes"] = snap0.page_size_bytes

    if args.workload == "gen":
        reset_eviction_trace(llm)
        outs, elapsed, sampler = timed_generate(
            llm, prompts, args.max_tokens, args.seed, sample_occupancy=True
        )
        out0 = outs[0].outputs[0]
        result["token_ids"] = list(out0.token_ids)
        result["generated_tokens"] = len(out0.token_ids)
        result["output_text"] = out0.text
        result["e2e_latency_s"] = elapsed
        result["peak_occupied_kv_blocks"] = sampler.peak_used_blocks or None
        result["mean_occupied_kv_blocks"] = sampler.mean_used_blocks
        if sampler.peak_used_blocks and snap0.page_size_bytes:
            result["peak_occupied_kv_bytes"] = (
                sampler.peak_used_blocks * snap0.page_size_bytes
            )

    elif args.workload == "timing":
        # Warmups are discarded entirely, including their eviction traces.
        for w in range(args.warmups):
            timed_generate(llm, prompts, args.max_tokens, args.seed + 1000 + w)

        prefill_times: list[float] = []
        e2e_times: list[float] = []
        gen_counts: list[int] = []
        peak_blocks = 0
        mean_blocks: list[float] = []
        reset_eviction_trace(llm)

        for rep in range(args.repetitions):
            _, t1, _ = timed_generate(llm, prompts, 1, args.seed + rep)
            prefill_times.append(t1)
            outs, tN, sampler = timed_generate(
                llm, prompts, args.max_tokens, args.seed + rep,
                sample_occupancy=True,
            )
            e2e_times.append(tN)
            gen_counts.append(len(outs[0].outputs[0].token_ids))
            if sampler is not None:
                peak_blocks = max(peak_blocks, sampler.peak_used_blocks)
                if sampler.mean_used_blocks is not None:
                    mean_blocks.append(sampler.mean_used_blocks)

        n_gen = statistics.mean(gen_counts)
        decode_times = [
            e - p for e, p in zip(e2e_times, prefill_times)
        ]

        def ms(vals):
            return [v * 1000.0 for v in vals]

        def stat(vals):
            return {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

        n_decode = max(1.0, n_gen - 1)
        result["metrics"] = {
            "ttft_ms": stat(ms(prefill_times)),
            "prefill_latency_ms": stat(ms(prefill_times)),
            "decode_latency_ms": stat(ms(decode_times)),
            "inter_token_latency_ms": stat(
                ms([d / n_decode for d in decode_times])
            ),
            "e2e_latency_s": stat(e2e_times),
            "decode_tokens_per_sec": stat(
                [n_decode / d if d > 0 else 0.0 for d in decode_times]
            ),
            "overall_tokens_per_sec": stat(
                [n_gen / e if e > 0 else 0.0 for e in e2e_times]
            ),
        }
        result["generated_tokens"] = n_gen
        result["peak_occupied_kv_blocks"] = peak_blocks or None
        result["mean_occupied_kv_blocks"] = (
            statistics.mean(mean_blocks) if mean_blocks else None
        )
        if peak_blocks and snap0.page_size_bytes:
            result["peak_occupied_kv_bytes"] = (
                peak_blocks * snap0.page_size_bytes
            )
        result["output_text_sample"] = outs[0].outputs[0].text[:120]

    else:  # retrieval
        from vllm import SamplingParams

        sampling = SamplingParams(
            max_tokens=args.max_tokens, temperature=0.0, seed=args.seed
        )
        reset_eviction_trace(llm)
        per_example = []
        peak_blocks = 0
        t0 = time.perf_counter()
        with OccupancySampler(llm) as sampler:
            outs = llm.generate(prompts, sampling)
        total_time = time.perf_counter() - t0
        peak_blocks = sampler.peak_used_blocks

        correct = 0
        for case, out in zip(cases, outs):
            text = out.outputs[0].text
            hit = case["expected"] in text
            correct += int(hit)
            per_example.append(
                {
                    "object": case["object"],
                    "expected": case["expected"],
                    "position_fraction": case["position_fraction"],
                    "fact_token_index": case["fact_token_index"],
                    "context_tokens": case["context_tokens"],
                    "correct": hit,
                    "output": text[:200],
                    "generated_tokens": len(out.outputs[0].token_ids),
                }
            )
        result["per_example"] = per_example
        result["num_examples"] = len(per_example)
        result["retrieval_accuracy"] = correct / len(per_example)
        result["generated_tokens"] = statistics.mean(
            e["generated_tokens"] for e in per_example
        )
        result["e2e_latency_s"] = total_time
        result["peak_occupied_kv_blocks"] = peak_blocks or None
        result["mean_occupied_kv_blocks"] = sampler.mean_used_blocks
        if peak_blocks and snap0.page_size_bytes:
            result["peak_occupied_kv_bytes"] = (
                peak_blocks * snap0.page_size_bytes
            )

    ev = eviction_stats(llm, keep_events=50)
    result["eviction"] = {
        "instrumentation_available": ev.available,
        "num_eviction_events": ev.num_eviction_events,
        "num_blocks_evicted": ev.num_blocks_evicted,
        "peak_retained_blocks": ev.peak_retained_blocks,
        "max_block_count_before_evict": ev.max_block_count_before_evict,
        "budget": ev.budget,
        "budget_respected": ev.budget_respected,
        "policy": ev.policy,
    }
    result["eviction_trace_tail"] = ev.events

    snap1 = snapshot_cache(llm)
    result["kv_free_blocks_after"] = snap1.num_free_blocks
    result["kv_used_blocks_after"] = snap1.num_used_blocks
    result["blocks_returned_after_run"] = (
        None
        if snap1.num_used_blocks is None
        else (snap0.num_used_blocks or 0) >= snap1.num_used_blocks
    )
    if torch.cuda.is_available():
        result["cuda_mem_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
        result["cuda_mem_reserved_mb"] = (
            torch.cuda.max_memory_reserved() / (1024 ** 2)
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"WORKER_OK {args.run_id}")


if __name__ == "__main__":
    main()

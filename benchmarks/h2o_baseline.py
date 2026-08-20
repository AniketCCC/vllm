# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental H2O KV-cache eviction baseline evaluation.

Compares full KV vs block-granular H2O at several retained-cache fractions.

Example:
  python benchmarks/h2o_baseline.py \\
      --model facebook/opt-125m \\
      --prompt-len 512 --max-tokens 128 \\
      --fractions 1.0 0.75 0.5 0.25
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

from vllm import LLM, SamplingParams
from vllm.utils.math_utils import cdiv


@dataclass
class RunResult:
    mode: str
    model: str
    prompt_length: int
    generated_tokens: int
    retained_kv_blocks: int | None
    peak_kv_blocks_est: int | None
    estimated_kv_bytes: int | None
    wall_clock_s: float
    tokens_per_sec: float
    output_text: str


def _estimate_blocks(num_tokens: int, block_size: int) -> int:
    return cdiv(num_tokens, block_size)


def run_once(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    block_size: int,
    h2o_max_blocks: int | None,
    h2o_recent_blocks: int,
    gpu_memory_utilization: float,
) -> RunResult:
    kwargs: dict = {
        "model": model,
        "block_size": block_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": False,
        "max_model_len": max(2048, len(prompt.split()) * 4 + max_tokens + 64),
        "enforce_eager": True,  # experimental score path safer in eager
    }
    if h2o_max_blocks is None:
        mode = "full_kv"
    else:
        mode = f"h2o_max_blocks={h2o_max_blocks}"
        kwargs.update(
            {
                "kv_eviction_policy": "h2o",
                "h2o_max_blocks": h2o_max_blocks,
                "h2o_recent_blocks": h2o_recent_blocks,
            }
        )

    llm = LLM(**kwargs)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - t0
    text = outs[0].outputs[0].text
    gen_tokens = len(outs[0].outputs[0].token_ids)
    prompt_len = len(outs[0].prompt_token_ids)
    total_tokens = prompt_len + gen_tokens
    full_blocks = _estimate_blocks(total_tokens, block_size)
    retained = h2o_max_blocks if h2o_max_blocks is not None else full_blocks
    # Rough KV byte estimate: blocks * block_size * layers * 2 * kv_heads * head_dim * 2
    # Unknown without loading config deeply; report retained_blocks * block_size tokens.
    estimated_kv_bytes = None
    try:
        cfg = llm.llm_engine.vllm_config.model_config.hf_config
        n_layers = getattr(cfg, "num_hidden_layers", None)
        n_kv = getattr(cfg, "num_key_value_heads", None) or getattr(
            cfg, "num_attention_heads", None
        )
        hidden = getattr(cfg, "hidden_size", None)
        n_heads = getattr(cfg, "num_attention_heads", None)
        if n_layers and n_kv and hidden and n_heads:
            head_dim = hidden // n_heads
            # fp16 bytes
            estimated_kv_bytes = (
                retained * block_size * n_layers * 2 * n_kv * head_dim * 2
            )
    except Exception:
        pass

    del llm
    return RunResult(
        mode=mode,
        model=model,
        prompt_length=prompt_len,
        generated_tokens=gen_tokens,
        retained_kv_blocks=retained,
        peak_kv_blocks_est=full_blocks if h2o_max_blocks is None else retained,
        estimated_kv_bytes=estimated_kv_bytes,
        wall_clock_s=elapsed,
        tokens_per_sec=(gen_tokens / elapsed) if elapsed > 0 else 0.0,
        output_text=text,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--h2o-recent-blocks", type=int, default=1)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25],
        help="Retained KV fractions vs full sequence block count (1.0 = full KV).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    # Deterministic long-ish prompt.
    base = "The history of machine learning begins with "
    prompt = (base * ((args.prompt_len // len(base.split())) + 3)).strip()

    # Warm estimate of full blocks using a cheap tokenizer-less heuristic:
    # run full KV first, then derive N from measured prompt+gen length.
    results: list[RunResult] = []
    full = run_once(
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
        h2o_max_blocks=None,
        h2o_recent_blocks=args.h2o_recent_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    results.append(full)
    full_blocks = max(
        2,
        _estimate_blocks(full.prompt_length + full.generated_tokens, args.block_size),
    )

    for frac in args.fractions:
        if abs(frac - 1.0) < 1e-9:
            continue
        n = max(args.h2o_recent_blocks + 1, int(full_blocks * frac))
        if n >= full_blocks:
            continue
        results.append(
            run_once(
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                block_size=args.block_size,
                h2o_max_blocks=n,
                h2o_recent_blocks=min(args.h2o_recent_blocks, n - 1),
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )

    print("\n=== H2O baseline results ===")
    for r in results:
        print(
            f"{r.mode:24s}  prompt={r.prompt_length} gen={r.generated_tokens}  "
            f"retained_blocks={r.retained_kv_blocks}  "
            f"time={r.wall_clock_s:.3f}s  tok/s={r.tokens_per_sec:.2f}"
        )
        print(f"  output[:120]={r.output_text[:120]!r}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

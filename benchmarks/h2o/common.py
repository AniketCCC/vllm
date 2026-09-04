# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for H2O baseline evaluation."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from vllm.utils.math_utils import cdiv

PolicyMode = Literal["full_kv", "h2o", "recent_only"]


@dataclass
class ExperimentRow:
    """One row in results.csv / results.json."""

    run_id: str
    timestamp_utc: str
    command: str
    git_commit: str
    git_dirty: bool
    model: str
    dtype: str
    gpu_name: str
    torch_version: str
    cuda_version: str | None
    vllm_commit: str
    policy: PolicyMode
    h2o_enabled: bool
    retention_target_fraction: float | None
    retention_actual_fraction: float | None
    h2o_max_blocks: int | None
    h2o_recent_blocks: int | None
    block_size: int
    prompt_tokens: int
    generated_tokens: int
    num_requests: int
    peak_retained_logical_blocks: int | None
    peak_retained_physical_blocks: int | None
    full_cache_blocks_est: int | None
    num_evictions_est: int | None
    estimated_kv_bytes: int | None
    gpu_peak_mem_allocated_mb: float | None
    gpu_peak_mem_reserved_mb: float | None
    wall_clock_s: float
    wall_clock_std: float | None
    tokens_per_sec: float
    tokens_per_sec_std: float | None
    time_to_first_token_s: float | None
    repetitions: int
    seed: int
    temperature: float
    workload: str
    output_text_sample: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {})
        for k, v in extra.items():
            d[f"extra_{k}"] = v
        return d


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    p = repo_root() / "results" / "h2o_baseline"
    p.mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(exist_ok=True)
    (p / "traces").mkdir(exist_ok=True)
    return p


def git_metadata() -> tuple[str, bool]:
    root = repo_root()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return commit, dirty


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def cuda_version() -> str | None:
    if torch.cuda.is_available():
        return torch.version.cuda
    return None


def estimate_full_blocks(num_tokens: int, block_size: int) -> int:
    return max(1, cdiv(num_tokens, block_size))


def retention_blocks(
    full_blocks: int,
    fraction: float,
    recent_blocks: int,
    min_blocks: int = 2,
) -> tuple[int, float]:
    """Return (h2o_max_blocks, actual_fraction) for a target fraction."""
    if fraction >= 1.0:
        n = full_blocks
    else:
        n = max(min_blocks, int(full_blocks * fraction))
        n = min(n, full_blocks)
    if recent_blocks >= n:
        recent_blocks = max(0, n - 1)
    actual = n / full_blocks if full_blocks > 0 else 1.0
    return n, actual


def use_inprocess_engine() -> None:
    """Keep the V1 engine core in this process so instrumentation can read it.

    Must be called before the first ``LLM`` is constructed.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def make_llm_kwargs(
    model: str,
    policy: PolicyMode,
    h2o_max_blocks: int | None,
    h2o_recent_blocks: int,
    block_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
    enforce_eager: bool = True,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "block_size": block_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "enable_prefix_caching": False,
        "seed": seed,
        "enforce_eager": enforce_eager,
        "dtype": dtype,
    }
    if policy == "full_kv":
        return kwargs
    kwargs["kv_eviction_policy"] = "h2o" if policy == "h2o" else "recent"
    kwargs["h2o_max_blocks"] = h2o_max_blocks
    kwargs["h2o_recent_blocks"] = h2o_recent_blocks
    return kwargs


def peak_gpu_memory_mb() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    return (
        torch.cuda.max_memory_allocated() / (1024 ** 2),
        torch.cuda.max_memory_reserved() / (1024 ** 2),
    )


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def append_results(rows: list[ExperimentRow]) -> None:
    out = results_dir()
    json_path = out / "results.json"
    csv_path = out / "results.csv"

    existing: list[dict[str, Any]] = []
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            existing = json.load(f)

    new_dicts = [r.to_flat_dict() for r in rows]
    existing.extend(new_dicts)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    all_rows = existing
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)


def deterministic_prompt(tokenizer, text_seed: str, num_tokens: int) -> list[int]:
    ids = tokenizer.encode(text_seed, add_special_tokens=False)
    if not ids:
        ids = list(range(256))
    out: list[int] = []
    while len(out) < num_tokens:
        out.extend(ids)
    return out[:num_tokens]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def env_summary() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": gpu_name(),
        "cuda": cuda_version() or "",
    }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side instrumentation for the H2O baseline benchmarks.

Reads real KV occupancy and eviction traces from the in-process engine core.
Requires ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the scheduler lives in this
process; otherwise every accessor degrades to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def kv_cache_manager(llm) -> Any | None:
    """Best-effort walk from an ``LLM`` down to the V1 ``KVCacheManager``."""
    try:
        core = llm.llm_engine.engine_core
        core = getattr(core, "engine_core", core)
        return core.scheduler.kv_cache_manager
    except AttributeError:
        return None


def page_size_bytes(kvm) -> int | None:
    try:
        groups = kvm.kv_cache_config.kv_cache_groups
        return sum(g.kv_cache_spec.page_size_bytes for g in groups)
    except (AttributeError, IndexError, TypeError):
        return None


@dataclass
class CacheSnapshot:
    """Physical KV pool state at one instant."""

    num_gpu_blocks: int | None = None
    num_free_blocks: int | None = None
    num_used_blocks: int | None = None
    page_size_bytes: int | None = None

    @property
    def used_bytes(self) -> int | None:
        if self.num_used_blocks is None or self.page_size_bytes is None:
            return None
        return self.num_used_blocks * self.page_size_bytes


def snapshot_cache(llm) -> CacheSnapshot:
    kvm = kv_cache_manager(llm)
    if kvm is None:
        return CacheSnapshot()
    try:
        total = kvm.block_pool.num_gpu_blocks
        free = kvm.block_pool.get_num_free_blocks()
    except AttributeError:
        return CacheSnapshot()
    return CacheSnapshot(
        num_gpu_blocks=total,
        num_free_blocks=free,
        num_used_blocks=total - free,
        page_size_bytes=page_size_bytes(kvm),
    )


@dataclass
class EvictionStats:
    """Aggregated view of ``H2OEvictionManager.trace_events``."""

    available: bool = False
    num_eviction_events: int = 0
    num_blocks_evicted: int = 0
    peak_retained_blocks: int | None = None
    max_block_count_before_evict: int | None = None
    budget: int | None = None
    recent_window: int | None = None
    policy: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def budget_respected(self) -> bool | None:
        if self.peak_retained_blocks is None or self.budget is None:
            return None
        return self.peak_retained_blocks <= self.budget


def eviction_stats(llm, keep_events: int = 0) -> EvictionStats:
    kvm = kv_cache_manager(llm)
    mgr = getattr(kvm, "h2o_manager", None) if kvm is not None else None
    if mgr is None:
        return EvictionStats()

    events = list(mgr.trace_events)
    num_blocks = sum(len(e.get("victims", ())) for e in events)
    retained_peaks = [len(e.get("retained_block_ids", ())) for e in events]
    before_peaks = [e.get("current_block_count", 0) for e in events]
    return EvictionStats(
        available=True,
        num_eviction_events=len(events),
        num_blocks_evicted=num_blocks,
        peak_retained_blocks=max(retained_peaks) if retained_peaks else 0,
        max_block_count_before_evict=max(before_peaks) if before_peaks else 0,
        budget=mgr.max_blocks,
        recent_window=mgr.recent_blocks,
        policy="h2o" if mgr.use_h2o_scores else "recent",
        events=events[-keep_events:] if keep_events else [],
    )


def reset_eviction_trace(llm) -> None:
    """Clear accumulated trace events (call between warmup and measured runs)."""
    kvm = kv_cache_manager(llm)
    mgr = getattr(kvm, "h2o_manager", None) if kvm is not None else None
    if mgr is not None:
        mgr.trace_events.clear()


def peak_cache_tracker(llm):
    """Return a callable that samples used-block count, tracking the max seen."""
    peak = {"blocks": 0}

    def sample() -> int:
        snap = snapshot_cache(llm)
        if snap.num_used_blocks is not None:
            peak["blocks"] = max(peak["blocks"], snap.num_used_blocks)
        return peak["blocks"]

    sample.peak = lambda: peak["blocks"]  # type: ignore[attr-defined]
    return sample

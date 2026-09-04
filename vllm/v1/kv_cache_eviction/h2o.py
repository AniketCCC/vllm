# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental block-granular H2O KV-cache eviction policy.

This is a baseline for research comparisons. It does not implement
mixed-precision compression or novel eviction algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class H2OBlockMeta:
    """Per logical cache block H2O metadata (request-local)."""

    logical_block_id: int
    h2o_score: float = 0.0
    # Monotonic allocation / creation order within the request.
    order: int = 0


@dataclass
class H2ORequestState:
    """H2O state for a single request. Isolated across requests."""

    request_id: str
    # logical_block_id -> metadata for currently retained (non-null) blocks.
    blocks: dict[int, H2OBlockMeta] = field(default_factory=dict)
    _next_order: int = 0

    def note_block_allocated(self, logical_block_id: int) -> None:
        if logical_block_id in self.blocks:
            return
        self.blocks[logical_block_id] = H2OBlockMeta(
            logical_block_id=logical_block_id,
            h2o_score=0.0,
            order=self._next_order,
        )
        self._next_order += 1

    def note_blocks_allocated(self, logical_block_ids: list[int]) -> None:
        for lid in logical_block_ids:
            self.note_block_allocated(lid)

    def add_scores(self, block_scores: dict[int, float]) -> None:
        """Accumulate attention mass into retained blocks."""
        for lid, score in block_scores.items():
            meta = self.blocks.get(lid)
            if meta is None:
                continue
            meta.h2o_score += float(score)

    def discard_blocks(self, logical_block_ids: list[int]) -> None:
        for lid in logical_block_ids:
            self.blocks.pop(lid, None)

    def retained_logical_ids_by_recency(self) -> list[int]:
        """Retained blocks sorted oldest → newest by creation order."""
        return [
            m.logical_block_id
            for m in sorted(self.blocks.values(), key=lambda m: m.order)
        ]

    def num_retained(self) -> int:
        return len(self.blocks)


def select_h2o_victims(
    state: H2ORequestState,
    max_blocks: int,
    recent_blocks: int,
    *,
    protect_logical_ids: set[int] | None = None,
) -> list[int]:
    """Select logical block ids to evict under an H2O budget.

    Rules:
    - Retain at most ``max_blocks`` non-null blocks.
    - The ``recent_blocks`` most recently created retained blocks are protected.
    - Among the remaining candidates, evict lowest cumulative H2O score first.
    - ``protect_logical_ids`` (e.g. the block currently being written) are never
      chosen as victims.

    Returns:
        Victim logical block ids (may be empty if under budget).
    """
    if max_blocks <= 0:
        raise ValueError(f"h2o_max_blocks must be > 0, got {max_blocks}")
    if not (0 <= recent_blocks < max_blocks):
        raise ValueError(
            f"Require 0 <= h2o_recent_blocks < h2o_max_blocks, "
            f"got recent={recent_blocks}, max={max_blocks}"
        )

    protect = set(protect_logical_ids or ())
    retained = state.retained_logical_ids_by_recency()
    excess = len(retained) - max_blocks
    if excess <= 0:
        return []

    # Most recent `recent_blocks` among currently retained.
    protected_recent = set(retained[-recent_blocks:]) if recent_blocks > 0 else set()
    protected = protected_recent | protect

    candidates = [lid for lid in retained if lid not in protected]
    # Lowest score first; tie-break older (smaller order) first for determinism.
    candidates.sort(
        key=lambda lid: (state.blocks[lid].h2o_score, state.blocks[lid].order)
    )

    if len(candidates) < excess:
        # Should not happen if protect set is reasonable; take what we can.
        victims = candidates
    else:
        victims = candidates[:excess]
    return victims


def select_recent_victims(
    state: H2ORequestState,
    max_blocks: int,
    recent_blocks: int,
    *,
    protect_logical_ids: set[int] | None = None,
) -> list[int]:
    """Evict oldest eligible blocks (recent-window naive baseline)."""
    if max_blocks <= 0:
        raise ValueError(f"h2o_max_blocks must be > 0, got {max_blocks}")
    if not (0 <= recent_blocks < max_blocks):
        raise ValueError(
            f"Require 0 <= h2o_recent_blocks < h2o_max_blocks, "
            f"got recent={recent_blocks}, max={max_blocks}"
        )

    protect = set(protect_logical_ids or ())
    retained = state.retained_logical_ids_by_recency()
    excess = len(retained) - max_blocks
    if excess <= 0:
        return []

    protected_recent = set(retained[-recent_blocks:]) if recent_blocks > 0 else set()
    protected = protected_recent | protect
    candidates = [lid for lid in retained if lid not in protected]
    candidates.sort(key=lambda lid: state.blocks[lid].order)
    if len(candidates) < excess:
        victims = candidates
    else:
        victims = candidates[:excess]
    return victims


@dataclass
class H2OEvictionDecision:
    request_id: str
    current_block_count: int
    protected_recent_blocks: list[int]
    candidate_scores: dict[int, float]
    victims: list[int]
    retained_block_ids: list[int]


class H2OEvictionManager:
    """Tracks per-request H2O state and applies victim selection."""

    def __init__(
        self,
        max_blocks: int,
        recent_blocks: int,
        *,
        debug: bool = False,
        use_h2o_scores: bool = True,
    ) -> None:
        if not (0 <= recent_blocks < max_blocks):
            raise ValueError(
                f"Require 0 <= h2o_recent_blocks < h2o_max_blocks, "
                f"got recent={recent_blocks}, max={max_blocks}"
            )
        self.max_blocks = max_blocks
        self.recent_blocks = recent_blocks
        self.debug = debug
        self.use_h2o_scores = use_h2o_scores
        self._states: dict[str, H2ORequestState] = {}
        self.trace_events: list[dict[str, Any]] = []

    def get_or_create(self, request_id: str) -> H2ORequestState:
        state = self._states.get(request_id)
        if state is None:
            state = H2ORequestState(request_id=request_id)
            self._states[request_id] = state
        return state

    def free(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def sync_retained_blocks(
        self, request_id: str, retained_logical_ids: list[int]
    ) -> None:
        """Ensure state tracks exactly the retained logical ids (allocate + prune)."""
        state = self.get_or_create(request_id)
        retained_set = set(retained_logical_ids)
        for lid in retained_logical_ids:
            state.note_block_allocated(lid)
        stale = [lid for lid in list(state.blocks) if lid not in retained_set]
        state.discard_blocks(stale)

    def update_scores(self, request_id: str, block_scores: dict[int, float]) -> None:
        if request_id not in self._states:
            return
        self._states[request_id].add_scores(block_scores)

    def plan_eviction(
        self,
        request_id: str,
        *,
        protect_logical_ids: set[int] | None = None,
    ) -> H2OEvictionDecision | None:
        state = self._states.get(request_id)
        if state is None:
            return None

        retained = state.retained_logical_ids_by_recency()
        protected_recent = (
            retained[-self.recent_blocks :] if self.recent_blocks > 0 else []
        )
        protect = set(protect_logical_ids or ())
        protected = set(protected_recent) | protect
        candidates = [lid for lid in retained if lid not in protected]
        candidate_scores = {lid: state.blocks[lid].h2o_score for lid in candidates}

        victims = (
            select_h2o_victims(
                state,
                self.max_blocks,
                self.recent_blocks,
                protect_logical_ids=protect_logical_ids,
            )
            if self.use_h2o_scores
            else select_recent_victims(
                state,
                self.max_blocks,
                self.recent_blocks,
                protect_logical_ids=protect_logical_ids,
            )
        )
        if not victims:
            return None

        # Apply discard to state now; caller frees physical blocks.
        state.discard_blocks(victims)
        retained_after = state.retained_logical_ids_by_recency()

        decision = H2OEvictionDecision(
            request_id=request_id,
            current_block_count=len(retained),
            protected_recent_blocks=list(protected_recent),
            candidate_scores=candidate_scores,
            victims=victims,
            retained_block_ids=retained_after,
        )
        if self.debug:
            logger.info(
                "H2O eviction: req=%s count=%s protected_recent=%s "
                "candidate_scores=%s victims=%s retained=%s",
                decision.request_id,
                decision.current_block_count,
                decision.protected_recent_blocks,
                decision.candidate_scores,
                decision.victims,
                decision.retained_block_ids,
            )
        if self.debug or self.trace_events is not None:
            self.trace_events.append(
                {
                    "request_id": decision.request_id,
                    "current_block_count": decision.current_block_count,
                    "protected_recent_blocks": decision.protected_recent_blocks,
                    "candidate_scores": decision.candidate_scores,
                    "victims": decision.victims,
                    "retained_block_ids": decision.retained_block_ids,
                    "policy": "h2o" if self.use_h2o_scores else "recent",
                }
            )
        return decision

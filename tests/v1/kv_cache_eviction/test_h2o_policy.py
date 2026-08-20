# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for experimental block-granular H2O eviction policy."""

from __future__ import annotations

import pytest

from vllm.config.cache import CacheConfig
from vllm.v1.attention.ops.h2o_score_collector import compact_block_table_for_attention
from vllm.v1.kv_cache_eviction.h2o import (
    H2OEvictionManager,
    H2ORequestState,
    select_h2o_victims,
)


def test_h2o_disabled_config_defaults():
    cfg = CacheConfig()
    assert cfg.kv_eviction_policy == "none"
    assert not cfg.h2o_enabled


def test_invalid_r_ge_n_rejected():
    with pytest.raises(ValueError, match="h2o_recent_blocks"):
        CacheConfig(
            kv_eviction_policy="h2o",
            h2o_max_blocks=4,
            h2o_recent_blocks=4,
        )


def test_invalid_missing_max_blocks():
    with pytest.raises(ValueError, match="h2o_max_blocks"):
        CacheConfig(kv_eviction_policy="h2o", h2o_recent_blocks=1)


def test_budget_not_exceeded_no_eviction():
    state = H2ORequestState(request_id="r0")
    state.note_blocks_allocated([0, 1, 2])
    state.add_scores({0: 1.0, 1: 2.0, 2: 3.0})
    victims = select_h2o_victims(state, max_blocks=4, recent_blocks=1)
    assert victims == []


def test_recent_blocks_never_victims():
    state = H2ORequestState(request_id="r0")
    # A B C D E newest; N=4 R=1 → protect E; need 1 victim among A-D
    for lid, score in enumerate([10.0, 2.0, 7.0, 1.0, 0.0]):
        state.note_block_allocated(lid)
        state.blocks[lid].h2o_score = score
    victims = select_h2o_victims(
        state, max_blocks=4, recent_blocks=1, protect_logical_ids={4}
    )
    assert 4 not in victims
    # D (score 1) is lowest among unprotected
    assert victims == [3]


def test_deterministic_example_d_is_victim():
    """N=4 R=1; A=10 B=2 C=7 D=1 E=newest → D evicted."""
    mgr = H2OEvictionManager(max_blocks=4, recent_blocks=1)
    state = mgr.get_or_create("req")
    labels = ["A", "B", "C", "D", "E"]
    scores = [10.0, 2.0, 7.0, 1.0, 0.0]
    for i, s in enumerate(scores):
        state.note_block_allocated(i)
        state.blocks[i].h2o_score = s
    decision = mgr.plan_eviction("req", protect_logical_ids={4})
    assert decision is not None
    assert decision.victims == [3]  # D
    assert set(decision.retained_block_ids) == {0, 1, 2, 4}
    assert labels[decision.victims[0]] == "D"


def test_lowest_scoring_eligible_evicted_first():
    state = H2ORequestState(request_id="r0")
    for lid, score in enumerate([5.0, 1.0, 9.0, 3.0]):
        state.note_block_allocated(lid)
        state.blocks[lid].h2o_score = score
    # N=2 R=1 → protect last (3); need 2 victims from {0,1,2} → lowest 1 then 0
    victims = select_h2o_victims(state, max_blocks=2, recent_blocks=1)
    assert victims == [1, 0]


def test_multiple_evictions_retained_set():
    mgr = H2OEvictionManager(max_blocks=3, recent_blocks=1)
    state = mgr.get_or_create("req")
    for i, s in enumerate([4.0, 1.0, 8.0, 2.0, 0.5, 0.0]):
        state.note_block_allocated(i)
        state.blocks[i].h2o_score = s
    decision = mgr.plan_eviction("req", protect_logical_ids={5})
    assert decision is not None
    # excess=3; protect recent{5} and protect{5}; candidates sorted by score:
    # 1(1.0), 4(0.5), 3(2.0), 0(4.0), 2(8.0) → wait sort by (score, order)
    # scores: 1→1.0 order1, 4→0.5 order4, 3→2.0, 0→4.0, 2→8.0
    # lowest three: 4 (0.5), 1 (1.0), 3 (2.0)
    assert set(decision.victims) == {4, 1, 3}
    assert set(decision.retained_block_ids) == {0, 2, 5}


def test_per_request_state_isolation():
    mgr = H2OEvictionManager(max_blocks=2, recent_blocks=1)
    s0 = mgr.get_or_create("a")
    s1 = mgr.get_or_create("b")
    s0.note_blocks_allocated([0, 1, 2])
    s0.add_scores({0: 100.0, 1: 1.0, 2: 0.0})
    s1.note_blocks_allocated([0, 1, 2])
    s1.add_scores({0: 1.0, 1: 100.0, 2: 0.0})
    d0 = mgr.plan_eviction("a", protect_logical_ids={2})
    d1 = mgr.plan_eviction("b", protect_logical_ids={2})
    assert d0 is not None and d1 is not None
    assert d0.victims == [1]
    assert d1.victims == [0]
    assert "a" in mgr._states and "b" in mgr._states


def test_cleanup_after_request_completion():
    mgr = H2OEvictionManager(max_blocks=2, recent_blocks=1)
    mgr.get_or_create("done").note_blocks_allocated([0, 1])
    mgr.free("done")
    assert "done" not in mgr._states


def test_original_positions_via_compaction():
    """Eviction punches nulls; compaction preserves original token accounting."""
    # Logical: blocks 0..4, seq_len=20, block_size=4; evict 1 and 3
    row = [10, 0, 11, 0, 12]  # 0 = null
    compacted, retained = compact_block_table_for_attention(
        row, seq_len=20, block_size=4, null_block_id=0
    )
    assert compacted == [10, 11, 12]
    assert retained == 12  # 3 full blocks
    # Positions for surviving tokens stay tied to logical indices 0,2,4
    # (slot_mapping uses position // block_size into the uncompacted table).


def test_h2o_scores_accumulate():
    state = H2ORequestState(request_id="r")
    state.note_blocks_allocated([0, 1])
    state.add_scores({0: 1.5, 1: 2.0})
    state.add_scores({0: 0.5})
    assert state.blocks[0].h2o_score == pytest.approx(2.0)
    assert state.blocks[1].h2o_score == pytest.approx(2.0)

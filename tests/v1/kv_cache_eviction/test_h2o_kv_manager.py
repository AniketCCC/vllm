# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVCacheManager integration tests for experimental H2O eviction."""

from __future__ import annotations

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request


def _make_manager(enable_h2o: bool, max_blocks: int = 4, recent: int = 1):
    block_size = 16
    num_gpu_blocks = 64
    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=kv_cache_spec)
        ],
    )
    return KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        hash_block_size=block_size,
        enable_caching=False,
        enable_h2o=enable_h2o,
        h2o_max_blocks=max_blocks if enable_h2o else None,
        h2o_recent_blocks=recent,
        h2o_debug=False,
    )


def _make_request(req_id: str, num_tokens: int) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
    )


def test_h2o_disabled_unchanged():
    mgr = _make_manager(enable_h2o=False)
    assert mgr.h2o_manager is None
    req = _make_request("r0", 48)  # 3 blocks
    assert mgr.allocate_slots(req, num_new_tokens=48) is not None
    retained = mgr.coordinator.single_type_managers[0].get_retained_logical_block_ids(
        "r0"
    )
    assert retained == [0, 1, 2]


def test_h2o_evicts_when_over_budget():
    mgr = _make_manager(enable_h2o=True, max_blocks=2, recent=1)
    req = _make_request("r0", 48)  # would need 3 blocks
    assert mgr.allocate_slots(req, num_new_tokens=48) is not None
    type_mgr = mgr.coordinator.single_type_managers[0]
    retained = type_mgr.get_retained_logical_block_ids("r0")
    assert len(retained) == 2
    assert len(type_mgr.req_to_blocks["r0"]) == 3
    assert sum(1 for b in type_mgr.req_to_blocks["r0"] if b.is_null) == 1
    assert "r0" in mgr.take_h2o_refresh_req_ids()


def test_h2o_score_driven_victim():
    mgr = _make_manager(enable_h2o=True, max_blocks=4, recent=1)
    # Allocate exactly 4 blocks first (under/at budget → no eviction)
    req = _make_request("r0", 64)
    assert mgr.allocate_slots(req, num_new_tokens=64) is not None
    assert not mgr.take_h2o_refresh_req_ids()

    type_mgr = mgr.coordinator.single_type_managers[0]
    retained = type_mgr.get_retained_logical_block_ids("r0")
    assert retained == [0, 1, 2, 3]

    h2o = mgr.h2o_manager
    assert h2o is not None
    state = h2o.get_or_create("r0")
    for lid in retained:
        state.blocks[lid].h2o_score = 10.0
    state.blocks[1].h2o_score = 0.1  # lowest among non-recent

    # Grow by one block → force eviction of block 1
    req.num_computed_tokens = 64
    req.append_output_token_ids(list(range(1000, 1016)))
    assert mgr.allocate_slots(req, num_new_tokens=16) is not None
    retained_after = type_mgr.get_retained_logical_block_ids("r0")
    assert 1 not in retained_after
    assert len(retained_after) == 4
    assert type_mgr.req_to_blocks["r0"][1].is_null


def test_h2o_cleanup_on_free():
    mgr = _make_manager(enable_h2o=True, max_blocks=4, recent=1)
    req = _make_request("r0", 32)
    assert mgr.allocate_slots(req, num_new_tokens=32) is not None
    assert "r0" in mgr.h2o_manager._states  # type: ignore[union-attr]
    mgr.free(req)
    assert "r0" not in mgr.h2o_manager._states  # type: ignore[union-attr]


def test_positions_not_renumbered_after_null_replace():
    mgr = _make_manager(enable_h2o=True, max_blocks=2, recent=1)
    req = _make_request("r0", 48)
    assert mgr.allocate_slots(req, num_new_tokens=48) is not None
    blocks = mgr.coordinator.single_type_managers[0].req_to_blocks["r0"]
    assert len(blocks) == 3
    block_size = 16
    pos = 40
    logical = pos // block_size
    assert logical == 2
    assert not blocks[2].is_null

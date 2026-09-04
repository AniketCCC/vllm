# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run correctness checks before H2O baseline benchmarks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.h2o.common import git_metadata, repo_root, utc_now


def run_pytest_unit() -> int:
    root = repo_root()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/v1/kv_cache_eviction/",
        "-v",
        "--tb=short",
        "--noconftest",
    ]
    env = dict(**os.environ)
    # Stub C extensions for policy-only tests when _C mismatches torch
    site = Path("/tmp/vllm_h2o_test_sitecustomize")
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(
        """
import sys
from types import ModuleType
for name in (
    "vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C",
    "vllm.cumem_allocator", "vllm._flashmla_C", "vllm._flashmla_extension_C",
):
    sys.modules.setdefault(name, ModuleType(name))
""",
        encoding="utf-8",
    )
    env["PYTHONPATH"] = f"{site}:{root}:{env.get('PYTHONPATH', '')}"
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=root, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    commit, dirty = git_metadata()
    print(f"git={commit} dirty={dirty} time={utc_now()}")

    rc = 0
    if not args.skip_pytest:
        rc = run_pytest_unit()
        if rc != 0:
            print("Unit tests FAILED")
            sys.exit(rc)
        print("Unit tests PASSED")

    # Import-level policy tests
    from vllm.v1.kv_cache_eviction.h2o import (
        H2OEvictionManager,
        H2ORequestState,
        select_h2o_victims,
        select_recent_victims,
    )

    state = H2ORequestState("r")
    for i, s in enumerate([10, 2, 7, 1, 0]):
        state.note_block_allocated(i)
        state.blocks[i].h2o_score = float(s)
    victims_h2o = select_h2o_victims(state, 4, 1, protect_logical_ids={4})
    assert victims_h2o == [3], f"expected D victim, got {victims_h2o}"

    state2 = H2ORequestState("r2")
    for i in range(6):
        state2.note_block_allocated(i)
    victims_recent = select_recent_victims(state2, 4, 1, protect_logical_ids={5})
    assert victims_recent == [0, 1], f"recent victims {victims_recent}"

    mgr = H2OEvictionManager(2, 1, use_h2o_scores=False)
    mgr.get_or_create("a").note_blocks_allocated([0, 1, 2])
    d = mgr.plan_eviction("a", protect_logical_ids={2})
    assert d and d.victims == [0]

    print("Inline correctness checks PASSED")
    sys.exit(rc)


if __name__ == "__main__":
    main()

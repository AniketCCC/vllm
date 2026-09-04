# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Driver for the H2O baseline benchmark suite.

Each configuration runs in its own subprocess (``_worker.py``) because vLLM does
not release GPU memory when an in-process ``LLM`` is deleted.

Stages: ``control``, ``retrieval``, ``timing``, ``length``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.h2o.common import git_metadata, gpu_name, repo_root, utc_now

WORKER = "benchmarks.h2o._worker"


def run_dir(base: Path, run_name: str) -> Path:
    d = base / "runs" / run_name
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


def launch(cfg: dict[str, Any], out_json: Path, log_path: Path,
           timeout_s: int) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", WORKER, "--out", str(out_json)]
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif isinstance(v, (list, tuple)):
            cmd.append(flag)
            cmd.extend(str(x) for x in v)
        elif v is not None:
            cmd.extend([flag, str(v)])

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root()}:{env.get('PYTHONPATH', '')}"
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd, cwd=repo_root(), env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s}s"
    if proc.returncode != 0:
        return False, f"exit code {proc.returncode} (see {log_path.name})"
    if not out_json.exists():
        return False, "worker produced no result file"
    return True, "ok"


def flatten(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten a worker result into one CSV-friendly row."""
    row: dict[str, Any] = {}
    for k, v in result.items():
        if k in ("per_example", "eviction_trace_tail", "token_ids"):
            continue
        if k == "metrics":
            for mk, mv in v.items():
                row[f"{mk}_mean"] = mv["mean"]
                row[f"{mk}_std"] = mv["std"]
                row[f"{mk}_n"] = mv["n"]
        elif k == "eviction":
            for ek, ev in v.items():
                row[f"eviction_{ek}"] = ev
        elif isinstance(v, (dict, list)):
            row[k] = json.dumps(v)
        else:
            row[k] = v
    return row


def write_outputs(rows: list[dict], results: list[dict], out_dir: Path) -> None:
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if rows:
        fields: list[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(out_dir / "results.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    per_ex = []
    for res in results:
        for ex in res.get("per_example", []):
            per_ex.append(
                {
                    "run_id": res["run_id"],
                    "policy": res["policy"],
                    "retention_target_fraction":
                        res.get("retention_target_fraction"),
                    "retention_actual_fraction":
                        res.get("retention_actual_fraction"),
                    "prompt_tokens": res.get("prompt_tokens"),
                    **ex,
                }
            )
    if per_ex:
        with open(out_dir / "per_example_results.json", "w",
                  encoding="utf-8") as f:
            json.dump(per_ex, f, indent=2)
        with open(out_dir / "per_example_results.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_ex[0].keys()),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(per_ex)


def policy_specs(fractions: list[float], include_recent: bool,
                 recent_fractions: list[float] | None = None):
    """(policy, target_fraction) pairs, full KV first."""
    specs: list[tuple[str, float]] = [("full_kv", 1.0)]
    for frac in fractions:
        specs.append(("h2o", frac))
    if include_recent:
        for frac in (recent_fractions or fractions):
            if frac < 1.0:
                specs.append(("recent_only", frac))
    return specs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stages", nargs="+",
                   default=["control", "retrieval", "timing", "length"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--run-name", default=None)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--h2o-recent-blocks", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[1.0, 0.75, 0.5, 0.25, 0.125])
    p.add_argument("--retrieval-context", type=int, default=2048)
    p.add_argument("--retrieval-max-tokens", type=int, default=24)
    p.add_argument("--positions", type=float, nargs="+",
                   default=[0.05, 0.25, 0.5, 0.75, 0.95])
    p.add_argument("--timing-prompt-tokens", type=int, default=2048)
    p.add_argument("--timing-max-tokens", type=int, default=128)
    p.add_argument("--timing-fractions", type=float, nargs="+",
                   default=[1.0, 0.75, 0.5, 0.25])
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--length-fractions", type=float, nargs="+",
                   default=[0.75, 0.5, 0.25])
    p.add_argument("--length-max-tokens", type=int, default=128)
    p.add_argument("--length-repetitions", type=int, default=3)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base = repo_root() / "results" / "h2o_baseline"
    out_dir = run_dir(base, run_name)
    commit, dirty = git_metadata()

    manifest = {
        "run_name": run_name,
        "timestamp_utc": utc_now(),
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu": gpu_name(),
        "model": args.model,
        "args": vars(args),
        "command": sys.argv,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    common = {
        "model": args.model,
        "block_size": args.block_size,
        "h2o_recent_blocks": args.h2o_recent_blocks,
        "dtype": args.dtype,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }

    jobs: list[tuple[str, dict]] = []

    if "control" in args.stages:
        ctx = min(1024, args.max_model_len - 256)
        for policy, frac, tag in [
            ("full_kv", 1.0, "full"),
            ("h2o", 1.0, "h2o100"),
            ("h2o", 0.25, "h2o25"),
            ("recent_only", 0.25, "recent25"),
        ]:
            jobs.append(
                (
                    f"control_{tag}",
                    {
                        **common,
                        "workload": "gen",
                        "policy": policy,
                        "target_fraction": frac,
                        "prompt_tokens": ctx,
                        "max_tokens": 64,
                    },
                )
            )

    if "retrieval" in args.stages:
        for policy, frac in policy_specs(args.fractions, True):
            tag = f"{policy}_{frac}"
            jobs.append(
                (
                    f"retrieval_{tag}",
                    {
                        **common,
                        "workload": "retrieval",
                        "policy": policy,
                        "target_fraction": frac,
                        "prompt_tokens": args.retrieval_context,
                        "max_tokens": args.retrieval_max_tokens,
                        "positions": args.positions,
                    },
                )
            )

    if "timing" in args.stages:
        for policy, frac in policy_specs(args.timing_fractions, True):
            tag = f"{policy}_{frac}"
            jobs.append(
                (
                    f"timing_{tag}",
                    {
                        **common,
                        "workload": "timing",
                        "policy": policy,
                        "target_fraction": frac,
                        "prompt_tokens": args.timing_prompt_tokens,
                        "max_tokens": args.timing_max_tokens,
                        "repetitions": args.repetitions,
                        "warmups": args.warmups,
                    },
                )
            )

    if "length" in args.stages:
        for length in args.lengths:
            if length + args.length_max_tokens > args.max_model_len:
                print(f"skip length {length}: exceeds max_model_len")
                continue
            for policy, frac in policy_specs(args.length_fractions, False):
                tag = f"{length}_{policy}_{frac}"
                jobs.append(
                    (
                        f"length_{tag}",
                        {
                            **common,
                            "workload": "timing",
                            "policy": policy,
                            "target_fraction": frac,
                            "prompt_tokens": length,
                            "max_tokens": args.length_max_tokens,
                            "repetitions": args.length_repetitions,
                            "warmups": args.warmups,
                        },
                    )
                )

    results: list[dict] = []
    rows: list[dict] = []
    failed: list[dict] = []

    print(f"Running {len(jobs)} configurations -> {out_dir}")
    for i, (run_id, cfg) in enumerate(jobs, 1):
        cfg = {**cfg, "run_id": run_id}
        out_json = out_dir / "raw" / f"{run_id}.json"
        log_path = out_dir / "logs" / f"{run_id}.log"
        t0 = time.perf_counter()
        ok, msg = launch(cfg, out_json, log_path, args.timeout)
        dt = time.perf_counter() - t0

        if ok:
            with open(out_json, encoding="utf-8") as f:
                res = json.load(f)
            results.append(res)
            rows.append(flatten(res))
            summary = ""
            if res["workload"] == "retrieval":
                summary = f"acc={res['retrieval_accuracy']:.0%}"
            elif res["workload"] == "timing":
                m = res["metrics"]["overall_tokens_per_sec"]
                summary = f"tok/s={m['mean']:.1f}±{m['std']:.1f}"
            ev = res.get("eviction", {})
            summary += f" evicted={ev.get('num_blocks_evicted')}"
            print(f"[{i}/{len(jobs)}] {run_id} OK {summary} ({dt:.0f}s)")
        else:
            failed.append({"run_id": run_id, "reason": msg, "config": cfg})
            print(f"[{i}/{len(jobs)}] {run_id} FAILED: {msg} ({dt:.0f}s)")

        # Persist after every job so a later failure cannot lose earlier data.
        write_outputs(rows, results, out_dir)
        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)

    print(f"\nCompleted {len(results)}/{len(jobs)}; failures: {len(failed)}")
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()

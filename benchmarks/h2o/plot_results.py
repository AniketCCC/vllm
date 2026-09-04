# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot H2O baseline results from a run directory produced by run_benchmarks.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.h2o.common import repo_root

POLICY_STYLE = {
    "full_kv": ("Full KV", "black", "o", "--"),
    "h2o": ("H2O", "tab:blue", "o", "-"),
    "recent_only": ("Recent-only", "tab:orange", "s", "-"),
}


def load(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.json", encoding="utf-8") as f:
        return json.load(f)


def by_workload(results, workload):
    return [r for r in results if r.get("workload") == workload]


def full_kv_baseline(rows, key):
    for r in rows:
        if r["policy"] == "full_kv":
            return key(r)
    return None


def plot_retrieval_accuracy(results, plots: Path, plt) -> None:
    rows = by_workload(results, "retrieval")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy in ("h2o", "recent_only"):
        pts = sorted(
            (r["retention_actual_fraction"], r["retrieval_accuracy"])
            for r in rows
            if r["policy"] == policy
        )
        if not pts:
            continue
        label, color, marker, ls = POLICY_STYLE[policy]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=marker, color=color, linestyle=ls, label=label)
    base = full_kv_baseline(rows, lambda r: r["retrieval_accuracy"])
    if base is not None:
        ax.axhline(base, color="black", linestyle="--", label="Full KV")
    ax.set_xlabel("Actual retained-KV fraction")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Needle retrieval accuracy vs retained KV")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "quality_vs_retention.png", dpi=140)
    fig.savefig(plots / "retrieval_accuracy_vs_retention.png", dpi=140)
    plt.close(fig)


def plot_accuracy_by_position(run_dir: Path, plots: Path, plt) -> None:
    path = run_dir / "per_example_results.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    grid = defaultdict(dict)
    for r in rows:
        if r["policy"] == "full_kv":
            key = "Full KV"
        else:
            key = f"{POLICY_STYLE[r['policy']][0]} {r['retention_target_fraction']:.3g}"
        grid[key][r["position_fraction"]] = int(r["correct"])

    positions = sorted({p for v in grid.values() for p in v})
    labels = sorted(grid)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 2))
    data = [[grid[lbl].get(p, float("nan")) for p in positions] for lbl in labels]
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"{p:.2f}" for p in positions])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Fact position in context (fraction)")
    ax.set_title("Retrieval correctness by fact position")
    for i in range(len(labels)):
        for j in range(len(positions)):
            v = data[i][j]
            if v == v:
                ax.text(j, i, "hit" if v else "miss", ha="center",
                        va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="correct")
    fig.tight_layout()
    fig.savefig(plots / "retrieval_accuracy_vs_position.png", dpi=140)
    plt.close(fig)


def plot_occupancy(results, plots: Path, plt) -> None:
    rows = by_workload(results, "retrieval")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax2 = ax.twinx()
    for policy in ("h2o", "recent_only"):
        pts = sorted(
            (r["retention_actual_fraction"], r.get("peak_occupied_kv_blocks"),
             r.get("peak_occupied_kv_bytes"))
            for r in rows
            if r["policy"] == policy and r.get("peak_occupied_kv_blocks")
        )
        if not pts:
            continue
        label, color, marker, ls = POLICY_STYLE[policy]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=marker, color=color, linestyle=ls, label=label)
        ax2.plot([p[0] for p in pts],
                 [(p[2] or 0) / (1024 ** 2) for p in pts], alpha=0)
    base = full_kv_baseline(rows, lambda r: r.get("peak_occupied_kv_blocks"))
    if base:
        ax.axhline(base, color="black", linestyle="--",
                   label=f"Full KV ({base} blocks)")
    ax.set_xlabel("Actual retained-KV fraction")
    ax.set_ylabel("Peak occupied KV blocks")
    ax2.set_ylabel("Peak occupied KV (MiB)")
    page = next((r.get("page_size_bytes") for r in rows
                 if r.get("page_size_bytes")), None)
    if page and base:
        ax2.set_ylim(0, base * page / (1024 ** 2) * 1.1)
    ax.set_title("Physical KV occupancy vs retained KV")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "kv_occupancy_vs_retention.png", dpi=140)
    plt.close(fig)


def plot_throughput(results, plots: Path, plt) -> None:
    rows = [r for r in by_workload(results, "timing")
            if r["run_id"].startswith("timing_")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy in ("h2o", "recent_only"):
        pts = sorted(
            (r["retention_actual_fraction"],
             r["metrics"]["decode_tokens_per_sec"]["mean"],
             r["metrics"]["decode_tokens_per_sec"]["std"])
            for r in rows if r["policy"] == policy
        )
        if not pts:
            continue
        label, color, marker, ls = POLICY_STYLE[policy]
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[p[2] for p in pts], marker=marker, color=color,
                    linestyle=ls, capsize=3, label=label)
    base = full_kv_baseline(
        rows, lambda r: r["metrics"]["decode_tokens_per_sec"]["mean"]
    )
    if base:
        ax.axhline(base, color="black", linestyle="--",
                   label=f"Full KV ({base:.0f} tok/s)")
    ax.set_xlabel("Actual retained-KV fraction")
    ax.set_ylabel("Decode tokens/sec")
    ax.set_title("Decode throughput vs retained KV")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "throughput_vs_retention.png", dpi=140)
    plt.close(fig)


def plot_latency_vs_length(results, plots: Path, plt) -> None:
    rows = [r for r in results if r["run_id"].startswith("length_")]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = defaultdict(list)
    for r in rows:
        key = (
            "Full KV" if r["policy"] == "full_kv"
            else f"H2O {r['retention_target_fraction']:.3g}"
        )
        groups[key].append(r)

    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["prompt_tokens"])
        xs = [r["prompt_tokens"] for r in rs]
        axes[0].errorbar(
            xs, [r["metrics"]["e2e_latency_s"]["mean"] for r in rs],
            yerr=[r["metrics"]["e2e_latency_s"]["std"] for r in rs],
            marker="o", capsize=3, label=key,
            color="black" if key == "Full KV" else None,
            linestyle="--" if key == "Full KV" else "-",
        )
        axes[1].plot(
            xs, [r.get("peak_occupied_kv_blocks") or 0 for r in rs],
            marker="s", label=key,
            color="black" if key == "Full KV" else None,
            linestyle="--" if key == "Full KV" else "-",
        )
    axes[0].set_xlabel("Prompt tokens")
    axes[0].set_ylabel("End-to-end latency (s)")
    axes[0].set_title("Latency vs context length")
    axes[1].set_xlabel("Prompt tokens")
    axes[1].set_ylabel("Peak occupied KV blocks")
    axes[1].set_title("KV occupancy vs context length")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "latency_vs_context_length.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = repo_root() / run_dir
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    results = load(run_dir)
    plot_retrieval_accuracy(results, plots, plt)
    plot_accuracy_by_position(run_dir, plots, plt)
    plot_occupancy(results, plots, plt)
    plot_throughput(results, plots, plt)
    plot_latency_vs_length(results, plots, plt)

    made = sorted(f.name for f in plots.glob("*.png"))
    print(f"Wrote {len(made)} plots to {plots}:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()

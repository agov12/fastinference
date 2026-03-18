#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "plots"

INPUTS = [
    (
        "HF baseline",
        ROOT / "results" / "speculative_qwen3" / "gcp_a100_g128" / "hf_baseline_qwen3-8b_g128.json",
        "#6c757d",
    ),
    (
        "HF spec\n1.7B k=4",
        ROOT / "results" / "speculative_qwen3" / "gcp_a100_g128" / "spec_Qwen3-1_7B_k4_g128.json",
        "#c44e52",
    ),
    (
        "TRT-LLM\nbaseline",
        ROOT / "results" / "profiling_speculative_qwen3_1p7b_k4" / "trtllm_baseline_g128.json",
        "#4c72b0",
    ),
    (
        "TRT-LLM spec\n1.7B k=4",
        ROOT / "results" / "profiling_speculative_qwen3_1p7b_k4" / "trtllm_native_spec_g128.json",
        "#55a868",
    ),
]


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["summary"]


def annotate_bars(ax, bars, suffix: str, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{fmt.format(height)}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    colors: list[str] = []
    summaries: list[dict] = []
    for label, path, color in INPUTS:
        labels.append(label)
        colors.append(color)
        summaries.append(load_summary(path))

    total_latency_ms = [s["total_generate_ms"] for s in summaries]
    tok_per_sec = [s["end_to_end_tokens_per_sec"] for s in summaries]
    prefill_ms = [s.get("prefill_ms") or 0.0 for s in summaries]
    decode_ms = [s.get("decode_ms") or 0.0 for s in summaries]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = list(range(len(labels)))

    bars_latency = axes[0].bar(x, total_latency_ms, color=colors, width=0.65)
    axes[0].set_title("Qwen3-8B Speculative vs Baseline\nPrompt 4096, Generate 128, A100")
    axes[0].set_ylabel("End-to-End Latency (ms)")
    axes[0].set_xticks(x, labels)
    annotate_bars(axes[0], bars_latency, " ms", "{:.0f}")

    bars_tps = axes[1].bar(x, tok_per_sec, color=colors, width=0.65)
    axes[1].set_title("Throughput")
    axes[1].set_ylabel("End-to-End Tokens / s")
    axes[1].set_xticks(x, labels)
    annotate_bars(axes[1], bars_tps, " tok/s")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "qwen3_speculative_vs_baseline_g128.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars_prefill = ax.bar(x, prefill_ms, color="#8da0cb", width=0.65, label="Prefill")
    bars_decode = ax.bar(
        x,
        decode_ms,
        bottom=prefill_ms,
        color="#fc8d62",
        width=0.65,
        label="Decode",
    )
    ax.set_title("Prefill vs Decode Split\nQwen3-8B Prompt 4096, Generate 128, A100")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x, labels)
    ax.legend()

    for idx, (prefill, decode) in enumerate(zip(prefill_ms, decode_ms)):
        total = prefill + decode
        if total <= 0:
            ax.text(idx, 50, "n/a", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(idx, total, f"{total:.0f} ms", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "qwen3_speculative_prefill_decode_split_g128.png", dpi=180)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

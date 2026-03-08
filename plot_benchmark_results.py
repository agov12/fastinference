#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
HF_RESULTS_DIR = ROOT / "results" / "hf_sweep_results"
VLLM_RESULTS_DIR = ROOT / "results" / "sweep_results"
PLOTS_DIR = ROOT / "results" / "plots"


@dataclass
class BenchmarkSummary:
    backend: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float | None
    decode_ms: float | None
    total_generate_ms: float
    decode_tokens_per_sec: float | None
    end_to_end_tokens_per_sec: float
    max_gpu_mem_gb: float
    source_json: Path

    @property
    def generated_tok_s_per_gpu(self) -> float:
        # All current runs are single-GPU, so per-GPU rate equals the run rate.
        return self.end_to_end_tokens_per_sec

    @property
    def total_processed_tok_s_per_gpu(self) -> float:
        total_tokens = self.batch_size * (self.prompt_tokens + self.generated_tokens)
        return total_tokens / (self.total_generate_ms / 1000.0)


def load_summaries(results_dir: Path) -> list[BenchmarkSummary]:
    summaries: list[BenchmarkSummary] = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        summary = payload["summary"]
        summaries.append(
            BenchmarkSummary(
                backend=summary["backend"],
                batch_size=summary["batch_size"],
                prompt_tokens=summary["prompt_tokens"],
                generated_tokens=summary["generated_tokens"],
                prefill_ms=summary["prefill_ms"],
                decode_ms=summary["decode_ms"],
                total_generate_ms=summary["total_generate_ms"],
                decode_tokens_per_sec=summary["decode_tokens_per_sec"],
                end_to_end_tokens_per_sec=summary["end_to_end_tokens_per_sec"],
                max_gpu_mem_gb=summary["max_gpu_mem_gb"],
                source_json=path,
            )
        )
    return summaries


def filter_summaries(
    summaries: list[BenchmarkSummary],
    *,
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
) -> list[BenchmarkSummary]:
    return [
        summary
        for summary in summaries
        if (batch_size is None or summary.batch_size == batch_size)
        and (prompt_tokens is None or summary.prompt_tokens == prompt_tokens)
    ]


def by_batch_size(summaries: list[BenchmarkSummary]) -> list[BenchmarkSummary]:
    return sorted(summaries, key=lambda summary: summary.batch_size)


def by_prompt_tokens(summaries: list[BenchmarkSummary]) -> list[BenchmarkSummary]:
    return sorted(summaries, key=lambda summary: summary.prompt_tokens)


def line_plot(
    ax,
    x_values,
    y_values,
    *,
    label: str,
    color: str,
    marker: str = "o",
):
    ax.plot(x_values, y_values, marker=marker, linewidth=2, markersize=7, label=label, color=color)


def save_figure(fig, filename: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_hf_phase_plots(hf_summaries: list[BenchmarkSummary]):
    prompt_fixed = by_batch_size(filter_summaries(hf_summaries, prompt_tokens=1024))
    x_batch = [summary.batch_size for summary in prompt_fixed]

    fig, ax = plt.subplots(figsize=(8, 5))
    line_plot(ax, x_batch, [summary.prefill_ms for summary in prompt_fixed], label="Prefill", color="#1f77b4")
    line_plot(ax, x_batch, [summary.decode_ms for summary in prompt_fixed], label="Decode", color="#d62728")
    line_plot(
        ax,
        x_batch,
        [summary.total_generate_ms for summary in prompt_fixed],
        label="Total Generate",
        color="#2ca02c",
    )
    ax.set_title("HF Latency vs Batch Size (Prompt = 1024)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "hf_batch_size_phase_times.png")

    batch_fixed = by_prompt_tokens(filter_summaries(hf_summaries, batch_size=1))
    x_prompt = [summary.prompt_tokens for summary in batch_fixed]

    fig, ax = plt.subplots(figsize=(8, 5))
    line_plot(ax, x_prompt, [summary.prefill_ms for summary in batch_fixed], label="Prefill", color="#1f77b4")
    line_plot(ax, x_prompt, [summary.decode_ms for summary in batch_fixed], label="Decode", color="#d62728")
    line_plot(
        ax,
        x_prompt,
        [summary.total_generate_ms for summary in batch_fixed],
        label="Total Generate",
        color="#2ca02c",
    )
    ax.set_title("HF Latency vs Prompt Tokens (Batch Size = 1)")
    ax.set_xlabel("Prompt Tokens")
    ax.set_ylabel("Latency (ms)")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "hf_prompt_tokens_phase_times.png")


def make_backend_comparison_plots(
    hf_summaries: list[BenchmarkSummary],
    vllm_summaries: list[BenchmarkSummary],
):
    hf_batch = by_batch_size(filter_summaries(hf_summaries, prompt_tokens=1024))
    vllm_batch = by_batch_size(filter_summaries(vllm_summaries, prompt_tokens=1024))
    batch_sizes = [summary.batch_size for summary in hf_batch]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    line_plot(
        axes[0, 0],
        batch_sizes,
        [summary.total_generate_ms for summary in hf_batch],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[0, 0],
        batch_sizes,
        [summary.total_generate_ms for summary in vllm_batch],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[0, 0].set_title("Total Latency vs Batch Size")
    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 0].set_ylabel("Total Generate (ms)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    line_plot(
        axes[0, 1],
        batch_sizes,
        [summary.generated_tok_s_per_gpu for summary in hf_batch],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[0, 1],
        batch_sizes,
        [summary.generated_tok_s_per_gpu for summary in vllm_batch],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[0, 1].set_title("Generated Tok/s/GPU vs Batch Size")
    axes[0, 1].set_xlabel("Batch Size")
    axes[0, 1].set_ylabel("Generated Tok/s/GPU")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    line_plot(
        axes[1, 0],
        batch_sizes,
        [summary.total_processed_tok_s_per_gpu for summary in hf_batch],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[1, 0],
        batch_sizes,
        [summary.total_processed_tok_s_per_gpu for summary in vllm_batch],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[1, 0].set_title("Total Processed Tok/s/GPU vs Batch Size")
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Total Processed Tok/s/GPU")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    line_plot(
        axes[1, 1],
        batch_sizes,
        [summary.max_gpu_mem_gb for summary in hf_batch],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[1, 1],
        batch_sizes,
        [summary.max_gpu_mem_gb for summary in vllm_batch],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[1, 1].set_title("GPU Memory vs Batch Size")
    axes[1, 1].set_xlabel("Batch Size")
    axes[1, 1].set_ylabel("Max GPU Memory (GB)")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    save_figure(fig, "hf_vs_vllm_batch_size_comparison.png")

    hf_prompt = by_prompt_tokens(filter_summaries(hf_summaries, batch_size=1))
    vllm_prompt = by_prompt_tokens(filter_summaries(vllm_summaries, batch_size=1))
    prompt_tokens = [summary.prompt_tokens for summary in hf_prompt]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    line_plot(
        axes[0, 0],
        prompt_tokens,
        [summary.total_generate_ms for summary in hf_prompt],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[0, 0],
        prompt_tokens,
        [summary.total_generate_ms for summary in vllm_prompt],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[0, 0].set_title("Total Latency vs Prompt Tokens")
    axes[0, 0].set_xlabel("Prompt Tokens")
    axes[0, 0].set_ylabel("Total Generate (ms)")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    line_plot(
        axes[0, 1],
        prompt_tokens,
        [summary.generated_tok_s_per_gpu for summary in hf_prompt],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[0, 1],
        prompt_tokens,
        [summary.generated_tok_s_per_gpu for summary in vllm_prompt],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[0, 1].set_title("Generated Tok/s/GPU vs Prompt Tokens")
    axes[0, 1].set_xlabel("Prompt Tokens")
    axes[0, 1].set_ylabel("Generated Tok/s/GPU")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    line_plot(
        axes[1, 0],
        prompt_tokens,
        [summary.total_processed_tok_s_per_gpu for summary in hf_prompt],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[1, 0],
        prompt_tokens,
        [summary.total_processed_tok_s_per_gpu for summary in vllm_prompt],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[1, 0].set_title("Total Processed Tok/s/GPU vs Prompt Tokens")
    axes[1, 0].set_xlabel("Prompt Tokens")
    axes[1, 0].set_ylabel("Total Processed Tok/s/GPU")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    line_plot(
        axes[1, 1],
        prompt_tokens,
        [summary.max_gpu_mem_gb for summary in hf_prompt],
        label="HF",
        color="#1f77b4",
    )
    line_plot(
        axes[1, 1],
        prompt_tokens,
        [summary.max_gpu_mem_gb for summary in vllm_prompt],
        label="vLLM",
        color="#ff7f0e",
    )
    axes[1, 1].set_title("GPU Memory vs Prompt Tokens")
    axes[1, 1].set_xlabel("Prompt Tokens")
    axes[1, 1].set_ylabel("Max GPU Memory (GB)")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    save_figure(fig, "hf_vs_vllm_prompt_tokens_comparison.png")


def make_backend_phase_availability_plot(
    hf_summaries: list[BenchmarkSummary],
    vllm_summaries: list[BenchmarkSummary],
):
    hf_batch = by_batch_size(filter_summaries(hf_summaries, prompt_tokens=1024))
    batch_sizes = [summary.batch_size for summary in hf_batch]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    line_plot(
        axes[0],
        batch_sizes,
        [summary.prefill_ms for summary in hf_batch],
        label="HF Prefill",
        color="#1f77b4",
    )
    line_plot(
        axes[0],
        batch_sizes,
        [summary.decode_ms for summary in hf_batch],
        label="HF Decode",
        color="#d62728",
    )
    axes[0].set_title("HF Phase Times vs Batch Size")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.85,
        "Phase-Level Comparison Note",
        fontsize=14,
        fontweight="bold",
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.02,
        0.45,
        (
            "The current vLLM benchmark path does not expose\n"
            "separate prefill/decode timings through its offline API.\n\n"
            "This is why the comparison plots focus on:\n"
            "- total generate latency\n"
            "- generated tok/s per GPU\n"
            "- total processed tok/s per GPU\n"
            "- GPU memory\n\n"
            f"Loaded {len(vllm_summaries)} vLLM summaries and {len(hf_summaries)} HF summaries."
        ),
        fontsize=11,
        transform=axes[1].transAxes,
        va="top",
    )

    save_figure(fig, "phase_timing_availability_note.png")


def main():
    hf_summaries = load_summaries(HF_RESULTS_DIR)
    vllm_summaries = load_summaries(VLLM_RESULTS_DIR)

    if not hf_summaries:
        raise SystemExit(f"No HF JSON summaries found in {HF_RESULTS_DIR}")
    if not vllm_summaries:
        raise SystemExit(f"No vLLM JSON summaries found in {VLLM_RESULTS_DIR}")

    plt.style.use("seaborn-v0_8-whitegrid")

    make_hf_phase_plots(hf_summaries)
    make_backend_comparison_plots(hf_summaries, vllm_summaries)
    make_backend_phase_availability_plot(hf_summaries, vllm_summaries)

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

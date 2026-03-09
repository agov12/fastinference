#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIRS = {
    "hf": ROOT / "results" / "hf_sweep_results",
    "vllm": ROOT / "results" / "sweep_results",
    "trtllm": ROOT / "results" / "trtllm_sweep_results",
}
PLOTS_DIR = ROOT / "results" / "plots"
BACKEND_ORDER = ["hf", "vllm", "trtllm"]
BACKEND_LABELS = {
    "hf": "HF",
    "vllm": "vLLM",
    "trtllm": "TensorRT-LLM",
}
BACKEND_COLORS = {
    "hf": "#1f77b4",
    "vllm": "#ff7f0e",
    "trtllm": "#2ca02c",
}


@dataclass
class BenchmarkSummary:
    backend: str
    model_name: str
    dtype: str
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
        return self.end_to_end_tokens_per_sec

    @property
    def total_processed_tok_s_per_gpu(self) -> float:
        total_tokens = self.batch_size * (self.prompt_tokens + self.generated_tokens)
        return total_tokens / (self.total_generate_ms / 1000.0)


def load_summaries(results_dir: Path) -> list[BenchmarkSummary]:
    if not results_dir.exists():
        return []

    summaries: list[BenchmarkSummary] = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        summary = payload["summary"]
        summaries.append(
            BenchmarkSummary(
                backend=summary["backend"],
                model_name=summary["model_name"],
                dtype=summary["dtype"],
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


def write_summary_csv(results_dir: Path, summaries: list[BenchmarkSummary]) -> None:
    if not summaries:
        return

    out_path = results_dir / "summary.csv"
    fieldnames = [
        "backend",
        "model_name",
        "dtype",
        "batch_size",
        "prompt_tokens",
        "generated_tokens",
        "prefill_ms",
        "decode_ms",
        "total_generate_ms",
        "decode_tokens_per_sec",
        "end_to_end_tokens_per_sec",
        "generated_tok_s_per_gpu",
        "total_processed_tok_s_per_gpu",
        "max_gpu_mem_gb",
        "source_json",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "backend": summary.backend,
                    "model_name": summary.model_name,
                    "dtype": summary.dtype,
                    "batch_size": summary.batch_size,
                    "prompt_tokens": summary.prompt_tokens,
                    "generated_tokens": summary.generated_tokens,
                    "prefill_ms": summary.prefill_ms,
                    "decode_ms": summary.decode_ms,
                    "total_generate_ms": summary.total_generate_ms,
                    "decode_tokens_per_sec": summary.decode_tokens_per_sec,
                    "end_to_end_tokens_per_sec": summary.end_to_end_tokens_per_sec,
                    "generated_tok_s_per_gpu": summary.generated_tok_s_per_gpu,
                    "total_processed_tok_s_per_gpu": summary.total_processed_tok_s_per_gpu,
                    "max_gpu_mem_gb": summary.max_gpu_mem_gb,
                    "source_json": summary.source_json.name,
                }
            )


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


def line_plot(ax, x_values, y_values, *, label: str, color: str, marker: str = "o") -> None:
    ax.plot(
        x_values,
        y_values,
        marker=marker,
        linewidth=2,
        markersize=7,
        label=label,
        color=color,
    )


def save_figure(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_hf_phase_plots(hf_summaries: list[BenchmarkSummary]) -> None:
    prompt_fixed = by_batch_size(filter_summaries(hf_summaries, prompt_tokens=1024))
    x_batch = [summary.batch_size for summary in prompt_fixed]

    fig, ax = plt.subplots(figsize=(8, 5))
    line_plot(ax, x_batch, [summary.prefill_ms for summary in prompt_fixed], label="Prefill", color="#1f77b4")
    line_plot(ax, x_batch, [summary.decode_ms for summary in prompt_fixed], label="Decode", color="#d62728")
    line_plot(ax, x_batch, [summary.total_generate_ms for summary in prompt_fixed], label="Total Generate", color="#2ca02c")
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
    line_plot(ax, x_prompt, [summary.total_generate_ms for summary in batch_fixed], label="Total Generate", color="#2ca02c")
    ax.set_title("HF Latency vs Prompt Tokens (Batch Size = 1)")
    ax.set_xlabel("Prompt Tokens")
    ax.set_ylabel("Latency (ms)")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "hf_prompt_tokens_phase_times.png")


def make_hf_vs_trtllm_phase_plots(
    hf_summaries: list[BenchmarkSummary],
    trtllm_summaries: list[BenchmarkSummary],
) -> None:
    hf_batch = by_batch_size(filter_summaries(hf_summaries, prompt_tokens=1024))
    trt_batch = by_batch_size(filter_summaries(trtllm_summaries, prompt_tokens=1024))
    hf_prompt = by_prompt_tokens(filter_summaries(hf_summaries, batch_size=1))
    trt_prompt = by_prompt_tokens(filter_summaries(trtllm_summaries, batch_size=1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for backend, summaries in [("hf", hf_batch), ("trtllm", trt_batch)]:
        x_values = [summary.batch_size for summary in summaries]
        line_plot(
            axes[0],
            x_values,
            [summary.prefill_ms for summary in summaries],
            label=f"{BACKEND_LABELS[backend]} Prefill",
            color=BACKEND_COLORS[backend],
            marker="o",
        )
        line_plot(
            axes[0],
            x_values,
            [summary.decode_ms for summary in summaries],
            label=f"{BACKEND_LABELS[backend]} Decode",
            color=BACKEND_COLORS[backend],
            marker="s",
        )
    axes[0].set_title("HF vs TensorRT-LLM Phase Times (Prompt = 1024)")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for backend, summaries in [("hf", hf_prompt), ("trtllm", trt_prompt)]:
        x_values = [summary.prompt_tokens for summary in summaries]
        line_plot(
            axes[1],
            x_values,
            [summary.prefill_ms for summary in summaries],
            label=f"{BACKEND_LABELS[backend]} Prefill",
            color=BACKEND_COLORS[backend],
            marker="o",
        )
        line_plot(
            axes[1],
            x_values,
            [summary.decode_ms for summary in summaries],
            label=f"{BACKEND_LABELS[backend]} Decode",
            color=BACKEND_COLORS[backend],
            marker="s",
        )
    axes[1].set_title("HF vs TensorRT-LLM Phase Times (Batch Size = 1)")
    axes[1].set_xlabel("Prompt Tokens")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    save_figure(fig, "hf_vs_trtllm_phase_times.png")


def make_backend_comparison_plots(summaries_by_backend: dict[str, list[BenchmarkSummary]]) -> None:
    active_backends = [
        backend for backend in BACKEND_ORDER if summaries_by_backend.get(backend)
    ]
    if len(active_backends) < 2:
        return

    metric_specs = [
        ("total_generate_ms", "Total Latency", "Total Generate (ms)"),
        ("generated_tok_s_per_gpu", "Generated Tok/s/GPU", "Generated Tok/s/GPU"),
        ("total_processed_tok_s_per_gpu", "Total Processed Tok/s/GPU", "Total Processed Tok/s/GPU"),
        ("max_gpu_mem_gb", "Peak GPU Memory Used", "Peak GPU Memory Used (GB)"),
    ]

    def plot_grid(selector_name: str, filename: str, *, x_is_prompt: bool) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes_flat = axes.ravel()

        per_backend = {}
        for backend in active_backends:
            summaries = summaries_by_backend[backend]
            filtered = (
                by_prompt_tokens(filter_summaries(summaries, batch_size=1))
                if x_is_prompt
                else by_batch_size(filter_summaries(summaries, prompt_tokens=1024))
            )
            if filtered:
                per_backend[backend] = filtered

        for axis, (metric_name, title, ylabel) in zip(axes_flat, metric_specs):
            for backend in active_backends:
                summaries = per_backend.get(backend, [])
                if not summaries:
                    continue
                x_values = [
                    summary.prompt_tokens if x_is_prompt else summary.batch_size
                    for summary in summaries
                ]
                y_values = [getattr(summary, metric_name) for summary in summaries]
                line_plot(
                    axis,
                    x_values,
                    y_values,
                    label=BACKEND_LABELS[backend],
                    color=BACKEND_COLORS[backend],
                )
            axis.set_title(f"{title} vs {selector_name}")
            axis.set_xlabel(selector_name)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
            if x_is_prompt:
                axis.set_xscale("log", base=2)
            axis.legend()

        save_figure(fig, filename)

    plot_grid("Batch Size", "backend_batch_size_comparison.png", x_is_prompt=False)
    plot_grid("Prompt Tokens", "backend_prompt_tokens_comparison.png", x_is_prompt=True)


def make_phase_timing_note_plot(
    hf_summaries: list[BenchmarkSummary],
    vllm_summaries: list[BenchmarkSummary],
    trtllm_summaries: list[BenchmarkSummary],
) -> None:
    hf_batch = by_batch_size(filter_summaries(hf_summaries, prompt_tokens=1024))
    trt_batch = by_batch_size(filter_summaries(trtllm_summaries, prompt_tokens=1024))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    line_plot(
        axes[0],
        [summary.batch_size for summary in hf_batch],
        [summary.prefill_ms for summary in hf_batch],
        label="HF Prefill",
        color=BACKEND_COLORS["hf"],
    )
    line_plot(
        axes[0],
        [summary.batch_size for summary in hf_batch],
        [summary.decode_ms for summary in hf_batch],
        label="HF Decode",
        color="#d62728",
        marker="s",
    )
    if trt_batch:
        line_plot(
            axes[0],
            [summary.batch_size for summary in trt_batch],
            [summary.prefill_ms for summary in trt_batch],
            label="TensorRT-LLM Prefill",
            color=BACKEND_COLORS["trtllm"],
        )
        line_plot(
            axes[0],
            [summary.batch_size for summary in trt_batch],
            [summary.decode_ms for summary in trt_batch],
            label="TensorRT-LLM Decode",
            color="#9467bd",
            marker="s",
        )
    axes[0].set_title("Phase Timing Coverage")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axis("off")
    axes[1].text(0.02, 0.86, "Phase-Level Comparison Note", fontsize=14, fontweight="bold", transform=axes[1].transAxes)
    axes[1].text(
        0.02,
        0.10,
        (
            "HF reports explicit prefill and decode timings.\n\n"
            "TensorRT-LLM phase timing is derived from request perf metrics:\n"
            "arrival -> first token, then first token -> last token.\n\n"
            "The current vLLM offline API still does not expose a comparable\n"
            "prefill/decode split, so backend-wide plots compare total latency,\n"
            "throughput, and peak GPU memory instead.\n\n"
            f"Loaded summaries: HF={len(hf_summaries)}, vLLM={len(vllm_summaries)}, "
            f"TensorRT-LLM={len(trtllm_summaries)}."
        ),
        fontsize=11,
        transform=axes[1].transAxes,
        va="bottom",
    )

    save_figure(fig, "phase_timing_availability_note.png")


def main() -> None:
    summaries_by_backend = {
        backend: load_summaries(results_dir)
        for backend, results_dir in RESULTS_DIRS.items()
    }

    hf_summaries = summaries_by_backend["hf"]
    vllm_summaries = summaries_by_backend["vllm"]
    trtllm_summaries = summaries_by_backend["trtllm"]

    if not hf_summaries:
        raise SystemExit(f"No HF JSON summaries found in {RESULTS_DIRS['hf']}")
    if not vllm_summaries:
        raise SystemExit(f"No vLLM JSON summaries found in {RESULTS_DIRS['vllm']}")

    for backend, summaries in summaries_by_backend.items():
        write_summary_csv(RESULTS_DIRS[backend], summaries)

    plt.style.use("seaborn-v0_8-whitegrid")

    make_hf_phase_plots(hf_summaries)
    if trtllm_summaries:
        make_hf_vs_trtllm_phase_plots(hf_summaries, trtllm_summaries)
    make_backend_comparison_plots(summaries_by_backend)
    make_phase_timing_note_plot(hf_summaries, vllm_summaries, trtllm_summaries)

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

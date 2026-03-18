#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_row(path: Path) -> dict:
    payload = load_json(path)
    config = payload.get("config", {})
    summary = payload.get("summary", {})
    draft_model = config.get("hf_speculative_draft_model")
    total_generate_ms = summary.get("total_generate_ms")
    end_to_end_tok_s = summary.get("end_to_end_tokens_per_sec")
    return {
        "variant": "baseline" if draft_model is None else "speculative",
        "draft_model": draft_model or "-",
        "k": config.get("hf_speculative_k"),
        "prompt_len": config.get("prompt_len"),
        "max_new_tokens": config.get("max_new_tokens"),
        "total_generate_ms": total_generate_ms,
        "end_to_end_tokens_per_sec": end_to_end_tok_s,
        "ttft_ms": summary.get("ttft_ms"),
        "inter_token_latency_ms": summary.get("inter_token_latency_ms"),
        "spec_acceptance_rate": summary.get("spec_acceptance_rate"),
        "spec_drafted_per_target_forward": summary.get("spec_drafted_per_target_forward"),
        "spec_avg_accepted_tokens_per_verification": summary.get(
            "spec_avg_accepted_tokens_per_verification"
        ),
        "max_gpu_mem_gb": summary.get("max_gpu_mem_gb"),
        "prefill_ms": summary.get("prefill_ms"),
        "decode_ms": summary.get("decode_ms"),
        "path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize speculative Qwen3 sweep JSON outputs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/speculative_qwen3/sweep"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/speculative_qwen3/sweep_summary.csv"),
    )
    args = parser.parse_args()

    json_paths = sorted(args.input_dir.glob("*.json"))
    if not json_paths:
        print(f"No JSON files found in {args.input_dir}")
        return 1

    rows = [build_row(path) for path in json_paths]

    baseline = next((row for row in rows if row["variant"] == "baseline"), None)
    baseline_latency = baseline["total_generate_ms"] if baseline is not None else None
    baseline_tok_s = baseline["end_to_end_tokens_per_sec"] if baseline is not None else None

    for row in rows:
        latency = row["total_generate_ms"]
        tok_s = row["end_to_end_tokens_per_sec"]
        row["latency_speedup_vs_baseline"] = (
            (baseline_latency / latency)
            if baseline_latency not in (None, 0) and latency not in (None, 0)
            else None
        )
        row["tok_s_speedup_vs_baseline"] = (
            (tok_s / baseline_tok_s)
            if baseline_tok_s not in (None, 0) and tok_s not in (None, 0)
            else None
        )

    rows.sort(
        key=lambda row: (
            row["variant"] != "baseline",
            float("inf") if row["total_generate_ms"] is None else row["total_generate_ms"],
        )
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "draft_model",
        "k",
        "prompt_len",
        "max_new_tokens",
        "total_generate_ms",
        "latency_speedup_vs_baseline",
        "end_to_end_tokens_per_sec",
        "tok_s_speedup_vs_baseline",
        "ttft_ms",
        "inter_token_latency_ms",
        "spec_acceptance_rate",
        "spec_drafted_per_target_forward",
        "spec_avg_accepted_tokens_per_verification",
        "max_gpu_mem_gb",
        "prefill_ms",
        "decode_ms",
        "path",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'variant':<12} {'draft_model':<18} {'k':>3} {'lat_ms':>10} {'lat_x':>8} "
        f"{'tok/s':>10} {'tok_x':>8} {'ttft_ms':>10} {'itl_ms':>10} {'accept':>8} {'mem_gb':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['variant']:<12} "
            f"{row['draft_model'].split('/')[-1]:<18} "
            f"{fmt(row['k'], 0):>3} "
            f"{fmt(row['total_generate_ms']):>10} "
            f"{fmt(row['latency_speedup_vs_baseline']):>8} "
            f"{fmt(row['end_to_end_tokens_per_sec']):>10} "
            f"{fmt(row['tok_s_speedup_vs_baseline']):>8} "
            f"{fmt(row['ttft_ms']):>10} "
            f"{fmt(row['inter_token_latency_ms']):>10} "
            f"{fmt(row['spec_acceptance_rate'], 3):>8} "
            f"{fmt(row['max_gpu_mem_gb']):>8}"
        )

    print(f"\nWrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

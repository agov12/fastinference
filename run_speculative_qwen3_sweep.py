#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_DRAFT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]
DEFAULT_K_VALUES = [1, 2, 4, 8]


def sanitize_model_name(model_name: str) -> str:
    tail = model_name.split("/")[-1]
    return tail.replace(".", "_").replace("-", "-")


def default_output_stem(target_model: str, max_new_tokens: int) -> str:
    return f"hf_baseline_{sanitize_model_name(target_model).lower()}_g{max_new_tokens}"


def speculative_output_stem(draft_model: str, k_value: int, max_new_tokens: int) -> str:
    return f"spec_{sanitize_model_name(draft_model)}_k{k_value}_g{max_new_tokens}"


def run_case(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n==> Running: {shlex.join(cmd)}")
    print(f"    Log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def build_base_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(args.benchmark_script),
        "--backend",
        "hf",
        "--model",
        args.target_model,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--prompt-len",
        str(args.prompt_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-iters",
        str(args.warmup_iters),
        "--benchmark-iters",
        str(args.benchmark_iters),
        "--seed",
        str(args.seed),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable speculative decoding sweep for Qwen3 on a single GPU."
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path("hf_a100_benchmark.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/speculative_qwen3/sweep"),
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--draft-model",
        dest="draft_models",
        action="append",
        default=None,
        help="Repeat to add draft models. Defaults to the Qwen3 0.6B/1.7B/4B set.",
    )
    parser.add_argument(
        "--k",
        dest="k_values",
        type=int,
        action="append",
        default=None,
        help="Repeat to add draft lengths. Defaults to 1,2,4,8.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the plain HF baseline run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON/log files instead of skipping completed cases.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.batch_size != 1:
        parser.error("This sweep runner expects --batch-size 1 because the speculative path is bs=1 only.")

    if not args.benchmark_script.exists():
        parser.error(f"Benchmark script not found: {args.benchmark_script}")

    draft_models = args.draft_models or list(DEFAULT_DRAFT_MODELS)
    k_values = args.k_values or list(DEFAULT_K_VALUES)
    if any(k < 1 for k in k_values):
        parser.error("All --k values must be >= 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    base_cmd = build_base_command(args)

    if not args.skip_baseline:
        stem = default_output_stem(args.target_model, args.max_new_tokens)
        json_path = args.output_dir / f"{stem}.json"
        log_path = args.output_dir / f"{stem}.log"
        if args.overwrite or not json_path.exists():
            cmd = [*base_cmd, "--output-json", str(json_path)]
            rc = run_case(cmd, log_path)
            if rc != 0:
                failures.append(stem)
        else:
            print(f"Skipping existing baseline: {json_path}")

    for draft_model in draft_models:
        for k_value in k_values:
            stem = speculative_output_stem(draft_model, k_value, args.max_new_tokens)
            json_path = args.output_dir / f"{stem}.json"
            log_path = args.output_dir / f"{stem}.log"
            if not args.overwrite and json_path.exists():
                print(f"Skipping existing case: {json_path}")
                continue

            cmd = [
                *base_cmd,
                "--hf-speculative-draft-model",
                draft_model,
                "--hf-speculative-k",
                str(k_value),
                "--output-json",
                str(json_path),
            ]
            rc = run_case(cmd, log_path)
            if rc != 0:
                failures.append(stem)

    if failures:
        print("\nSweep completed with failures:")
        for stem in failures:
            print(f"  - {stem}")
        return 1

    print("\nSweep completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

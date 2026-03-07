#!/usr/bin/env python3
"""
LLM inference benchmark for a single NVIDIA A100 with Hugging Face Transformers + PyTorch.

Example baseline runs:
  python hf_a100_benchmark.py --dtype bf16 --batch-size 1
  python hf_a100_benchmark.py --dtype bf16 --batch-size 16
  python hf_a100_benchmark.py --dtype fp16 --batch-size 1

Example Nsight Systems profile runs (decode-focused):
  nsys profile -o nsys_bs1_decode --trace=cuda,nvtx \
    python hf_a100_benchmark.py --batch-size 1 --dtype bf16 --prompt-len 1024 \
      --max-new-tokens 64 --profile-decode

  nsys profile -o nsys_bs16_decode --trace=cuda,nvtx \
    python hf_a100_benchmark.py --batch-size 16 --dtype bf16 --prompt-len 1024 \
      --max-new-tokens 64 --profile-decode
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class IterationMetrics:
    prefill_ms: float
    decode_ms: float
    total_generate_ms: float
    decode_tokens_per_sec: float
    end_to_end_tokens_per_sec: float
    prompt_tokens: int
    generated_tokens: int
    batch_size: int
    dtype: str
    model_name: str
    max_gpu_mem_gb: float


def load_model_and_tokenizer(model_name: str, dtype: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark, but no GPU was detected.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: {list(dtype_map)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # Decoder-only models often do not define PAD; EOS is standard fallback.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map[dtype],
    )
    model.to("cuda")
    model.eval()
    return model, tokenizer


def make_synthetic_batch(
    tokenizer,
    batch_size: int,
    prompt_len: int,
    device: str,
):
    # Create token IDs from a repeated pattern so prompt length is controlled by tokens,
    # not characters/words. This makes runs reproducible across models/tokenizers.
    pattern_text = "benchmark token stream for deterministic inference throughput. "
    pattern_ids = tokenizer.encode(pattern_text, add_special_tokens=False)
    if not pattern_ids:
        raise RuntimeError("Tokenizer returned empty pattern tokenization.")

    repeats = (prompt_len + len(pattern_ids) - 1) // len(pattern_ids)
    ids = (pattern_ids * repeats)[:prompt_len]

    encoded_samples = [{"input_ids": ids} for _ in range(batch_size)]
    batch = tokenizer.pad(encoded_samples, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_tokens = int(attention_mask[0].sum().item())
    return input_ids, attention_mask, prompt_tokens


def _time_cuda_callable(fn):
    # Use CUDA events for device-side timing accuracy.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), result


def time_prefill(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    with torch.no_grad():
        ms, outputs = _time_cuda_callable(
            lambda: model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        )
    return ms, outputs


def time_manual_decode(
    model,
    prefill_outputs,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
):
    if max_new_tokens < 1:
        return 0.0, 0

    past_key_values = prefill_outputs.past_key_values
    # First token from prompt logits (already computed during prefill).
    next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)

    current_attention_mask = attention_mask
    generated = 0

    def _decode_loop():
        nonlocal past_key_values, next_token, current_attention_mask, generated
        for _ in range(max_new_tokens):
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.size(0), 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )

            outputs = model(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated += 1

    with torch.no_grad():
        ms, _ = _time_cuda_callable(_decode_loop)

    return ms, generated


def time_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    pad_token_id: int,
):
    with torch.no_grad():
        ms, out = _time_cuda_callable(
            lambda: model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        )
    return ms, out


def summarize_results(all_iters: list[IterationMetrics]):
    def avg(field: str) -> float:
        values = [getattr(m, field) for m in all_iters]
        return float(statistics.mean(values))

    sample = all_iters[0]
    summary = IterationMetrics(
        prefill_ms=avg("prefill_ms"),
        decode_ms=avg("decode_ms"),
        total_generate_ms=avg("total_generate_ms"),
        decode_tokens_per_sec=avg("decode_tokens_per_sec"),
        end_to_end_tokens_per_sec=avg("end_to_end_tokens_per_sec"),
        prompt_tokens=sample.prompt_tokens,
        generated_tokens=sample.generated_tokens,
        batch_size=sample.batch_size,
        dtype=sample.dtype,
        model_name=sample.model_name,
        max_gpu_mem_gb=avg("max_gpu_mem_gb"),
    )
    return summary


def _fmt_row(name: str, value: Any, unit: str = "") -> str:
    return f"{name:<30} {value:>14}{(' ' + unit) if unit else ''}"


def _print_iteration(idx: int, metrics: IterationMetrics):
    print(
        f"iter={idx:02d} "
        f"prefill_ms={metrics.prefill_ms:.2f} "
        f"decode_ms={metrics.decode_ms:.2f} "
        f"generate_ms={metrics.total_generate_ms:.2f} "
        f"decode_tok_s={metrics.decode_tokens_per_sec:.2f} "
        f"e2e_tok_s={metrics.end_to_end_tokens_per_sec:.2f} "
        f"max_mem_gb={metrics.max_gpu_mem_gb:.2f}"
    )


def _print_summary(summary: IterationMetrics):
    print("\n=== Benchmark Summary (Averages) ===")
    print(_fmt_row("model_name", summary.model_name))
    print(_fmt_row("dtype", summary.dtype))
    print(_fmt_row("batch_size", summary.batch_size))
    print(_fmt_row("prompt_tokens", summary.prompt_tokens))
    print(_fmt_row("generated_tokens", summary.generated_tokens))
    print(_fmt_row("prefill_ms", f"{summary.prefill_ms:.2f}", "ms"))
    print(_fmt_row("decode_ms", f"{summary.decode_ms:.2f}", "ms"))
    print(_fmt_row("total_generate_ms", f"{summary.total_generate_ms:.2f}", "ms"))
    print(_fmt_row("decode_tokens_per_sec", f"{summary.decode_tokens_per_sec:.2f}", "tok/s"))
    print(
        _fmt_row(
            "end_to_end_tokens_per_sec",
            f"{summary.end_to_end_tokens_per_sec:.2f}",
            "tok/s",
        )
    )
    print(_fmt_row("max_gpu_mem_allocated", f"{summary.max_gpu_mem_gb:.2f}", "GB"))


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="HF + PyTorch LLM inference benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile-decode", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main():
    args = _build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
        input_ids, attention_mask, prompt_tokens = make_synthetic_batch(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            device="cuda",
        )
    except RuntimeError as e:
        print(f"Initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.profile_decode:
        print("Running profile-decode mode (single short measured pass).")
        try:
            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            print("START_PREFILL", flush=True)
            prefill_ms, prefill_outputs = time_prefill(model, input_ids, attention_mask)
            print("END_PREFILL", flush=True)

            print("START_DECODE", flush=True)
            decode_ms, generated_tokens = time_manual_decode(
                model=model,
                prefill_outputs=prefill_outputs,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
            )
            print("END_DECODE", flush=True)

            max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            decode_tok_s = (
                (generated_tokens * args.batch_size) / (decode_ms / 1000.0)
                if decode_ms > 0
                else 0.0
            )

            print(
                f"profile_result prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f} "
                f"decode_tokens_per_sec={decode_tok_s:.2f} max_mem_gb={max_mem_gb:.2f}"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    "CUDA OOM during profile pass. Try lowering --batch-size, --prompt-len, or --max-new-tokens.",
                    file=sys.stderr,
                )
            else:
                print(f"Runtime error during profile pass: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Warmup passes
    print(
        f"Warmup: {args.warmup_iters} iters | Benchmark: {args.benchmark_iters} iters | "
        f"model={args.model} dtype={args.dtype} batch_size={args.batch_size}"
    )
    for i in range(args.warmup_iters):
        try:
            prefill_ms, prefill_outputs = time_prefill(model, input_ids, attention_mask)
            decode_ms, _ = time_manual_decode(
                model=model,
                prefill_outputs=prefill_outputs,
                attention_mask=attention_mask,
                max_new_tokens=min(16, args.max_new_tokens),
            )
            _ = time_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(16, args.max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
            )
            print(f"warmup={i+1}/{args.warmup_iters} prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    "CUDA OOM during warmup. Reduce --batch-size, --prompt-len, or --max-new-tokens.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    metrics_list: list[IterationMetrics] = []

    for i in range(args.benchmark_iters):
        try:
            torch.cuda.reset_peak_memory_stats()

            prefill_ms, prefill_outputs = time_prefill(model, input_ids, attention_mask)
            decode_ms, generated_tokens = time_manual_decode(
                model=model,
                prefill_outputs=prefill_outputs,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
            )
            gen_ms, _ = time_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

            decode_toks_total = generated_tokens * args.batch_size
            e2e_toks_total = generated_tokens * args.batch_size

            decode_tok_s = decode_toks_total / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
            e2e_tok_s = e2e_toks_total / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

            iter_metrics = IterationMetrics(
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                total_generate_ms=gen_ms,
                decode_tokens_per_sec=decode_tok_s,
                end_to_end_tokens_per_sec=e2e_tok_s,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                batch_size=args.batch_size,
                dtype=args.dtype,
                model_name=args.model,
                max_gpu_mem_gb=max_mem_gb,
            )
            metrics_list.append(iter_metrics)
            _print_iteration(i + 1, iter_metrics)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"CUDA OOM on benchmark iteration {i+1}. "
                    "Try lowering --batch-size, --prompt-len, or --max-new-tokens.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    if not metrics_list:
        print("No benchmark iterations completed.", file=sys.stderr)
        sys.exit(1)

    summary = summarize_results(metrics_list)
    _print_summary(summary)

    if args.output_json:
        out_path = Path(args.output_json)
        payload = {
            "config": vars(args),
            "iterations": [asdict(m) for m in metrics_list],
            "summary": asdict(summary),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to: {out_path}")


if __name__ == "__main__":
    main()

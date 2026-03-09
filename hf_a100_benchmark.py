#!/usr/bin/env python3
"""
LLM inference benchmark for a single NVIDIA A100.

Supported backends:
  - Hugging Face Transformers + PyTorch
  - vLLM offline engine
  - TensorRT-LLM Python LLM API

Example baseline runs:
  python hf_a100_benchmark.py --backend hf --dtype bf16 --batch-size 1
  python hf_a100_benchmark.py --backend hf --dtype bf16 --batch-size 16
  python hf_a100_benchmark.py --backend vllm --dtype bf16 --batch-size 1
  python hf_a100_benchmark.py --backend vllm --dtype bf16 --batch-size 16
  python hf_a100_benchmark.py --backend trtllm --dtype bf16 --batch-size 1
  python hf_a100_benchmark.py --backend trtllm --dtype bf16 --batch-size 16

Example Nsight Systems profile runs (HF decode-focused only):
  nsys profile -o nsys_bs1_decode --trace=cuda,nvtx \
    python hf_a100_benchmark.py --backend hf --batch-size 1 --dtype bf16 \
      --prompt-len 1024 --max-new-tokens 64 --profile-decode

  nsys profile -o nsys_bs16_decode --trace=cuda,nvtx \
    python hf_a100_benchmark.py --backend hf --batch-size 16 --dtype bf16 \
      --prompt-len 1024 --max-new-tokens 64 --profile-decode
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import statistics
import sys
import time
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class IterationMetrics:
    backend: str
    prefill_ms: float | None
    decode_ms: float | None
    total_generate_ms: float
    decode_tokens_per_sec: float | None
    end_to_end_tokens_per_sec: float
    prompt_tokens: int
    generated_tokens: int
    batch_size: int
    dtype: str
    model_name: str
    max_gpu_mem_gb: float


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # Decoder-only models often do not define PAD; EOS is standard fallback.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_hf_model_and_tokenizer(model_name: str, dtype: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark, but no GPU was detected.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: {list(dtype_map)}")

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map[dtype],
    )
    model.to("cuda")
    model.eval()
    return model, tokenizer


def load_vllm_engine_and_tokenizer(args):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested, but the 'vllm' package is not installed."
        ) from exc

    dtype_map = {"bf16": "bfloat16", "fp16": "float16"}
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{args.dtype}'. Use one of: {list(dtype_map)}")

    llm_kwargs = {
        "model": args.model,
        "dtype": dtype_map[args.dtype],
        "tensor_parallel_size": 1,
        "seed": args.seed,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        # Disable prefix caching by default so repeated iterations do not
        # skip prefill work and distort apples-to-apples comparisons.
        "enable_prefix_caching": args.vllm_enable_prefix_caching,
        "enforce_eager": args.vllm_enforce_eager,
    }
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.vllm_max_model_len

    llm = LLM(**llm_kwargs)
    tokenizer = load_tokenizer(args.model)
    return llm, tokenizer


def _site_packages_dir() -> Path:
    return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _prepend_env_path(var_name: str, paths: list[Path]) -> None:
    existing = os.environ.get(var_name, "")
    ordered = [str(path) for path in paths if path.exists()]
    if existing:
        ordered.append(existing)
    if ordered:
        os.environ[var_name] = ":".join(ordered)


def _ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        link_path.symlink_to(target)
    except OSError:
        # Best-effort only: if the environment is read-only, rely on the
        # caller to provide a system CUDA install instead.
        pass


def configure_trtllm_runtime_environment() -> Path | None:
    site_packages = _site_packages_dir()
    detected_cuda_home = site_packages / "nvidia" / "cu13"
    cuda_home = Path(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or detected_cuda_home)
    if cuda_home.exists():
        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ["CUDA_PATH"] = str(cuda_home)

        cuda_lib_dir = cuda_home / "lib"
        cuda_lib64_dir = cuda_home / "lib64"
        if cuda_lib_dir.exists():
            _ensure_symlink(cuda_lib64_dir, cuda_lib_dir)
            libcudart_major = next(cuda_lib_dir.glob("libcudart.so.*"), None)
            if libcudart_major is not None:
                _ensure_symlink(cuda_lib_dir / "libcudart.so", Path(libcudart_major.name))

    ld_library_paths: list[Path] = []
    nvidia_root = site_packages / "nvidia"
    if nvidia_root.exists():
        ld_library_paths.extend(sorted(nvidia_root.glob("*/lib")))

    tensorrt_libs_dir = site_packages / "tensorrt_libs"
    if tensorrt_libs_dir.exists():
        ld_library_paths.insert(0, tensorrt_libs_dir)

    _prepend_env_path("LD_LIBRARY_PATH", ld_library_paths)
    return cuda_home if cuda_home.exists() else None


def ensure_trtllm_runtime_ready_for_current_process() -> None:
    configure_trtllm_runtime_environment()
    if os.environ.get("_TRTLLM_ENV_READY") == "1":
        return

    os.environ["_TRTLLM_ENV_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def load_trtllm_engine_and_tokenizer(args):
    cuda_home = configure_trtllm_runtime_environment()
    try:
        from tensorrt_llm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT-LLM backend requested, but the 'tensorrt_llm' package is not installed."
        ) from exc

    dtype_map = {"bf16": "bfloat16", "fp16": "float16"}
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{args.dtype}'. Use one of: {list(dtype_map)}")
    if cuda_home is None:
        raise RuntimeError(
            "TensorRT-LLM requires CUDA_HOME/CUDA_PATH or a pip-installed CUDA 13 toolkit."
        )

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=dtype_map[args.dtype],
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_batch_size=args.batch_size,
        max_num_tokens=args.prompt_len + args.max_new_tokens,
        max_seq_len=args.prompt_len + args.max_new_tokens,
    )
    tokenizer = load_tokenizer(args.model)
    return llm, tokenizer


def make_synthetic_prompt_ids(tokenizer, prompt_len: int) -> list[int]:
    # Create token IDs from a repeated pattern so prompt length is controlled by
    # tokens, not characters/words. This makes runs reproducible across backends.
    pattern_text = "benchmark token stream for deterministic inference throughput. "
    pattern_ids = tokenizer.encode(pattern_text, add_special_tokens=False)
    if not pattern_ids:
        raise RuntimeError("Tokenizer returned empty pattern tokenization.")

    repeats = (prompt_len + len(pattern_ids) - 1) // len(pattern_ids)
    return (pattern_ids * repeats)[:prompt_len]


def make_hf_batch(
    tokenizer,
    prompt_token_ids: list[int],
    batch_size: int,
    device: str,
):
    encoded_samples = [{"input_ids": prompt_token_ids} for _ in range(batch_size)]
    batch = tokenizer.pad(encoded_samples, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_tokens = int(attention_mask[0].sum().item())
    return input_ids, attention_mask, prompt_tokens


def make_vllm_inputs(prompt_token_ids: list[int], batch_size: int):
    return [{"prompt_token_ids": list(prompt_token_ids)} for _ in range(batch_size)]


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


def _time_host_callable(fn):
    # vLLM's offline API does not expose separate prefill/decode timings here,
    # so use synchronized wall clock timing for total request latency.
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, result


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


def build_vllm_sampling_params(max_new_tokens: int, seed: int):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested, but the 'vllm' package is not installed."
        ) from exc

    # Match the HF benchmark as closely as possible: greedy generation and no
    # output text detokenization overhead in the timed path.
    return SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
        detokenize=False,
    )


def build_trtllm_sampling_params(max_new_tokens: int, seed: int):
    try:
        from tensorrt_llm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT-LLM backend requested, but the 'tensorrt_llm' package is not installed."
        ) from exc

    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=seed,
        detokenize=False,
        return_perf_metrics=True,
    )


def time_vllm_generate(llm, prompts, sampling_params):
    return _time_host_callable(
        lambda: llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    )


def time_trtllm_generate(llm, prompts, sampling_params):
    return _time_host_callable(lambda: llm.generate(prompts, sampling_params=sampling_params))


def extract_vllm_generated_tokens(outputs) -> tuple[int, int]:
    per_request_counts = []
    for output in outputs:
        if not output.outputs:
            per_request_counts.append(0)
            continue
        per_request_counts.append(len(output.outputs[0].token_ids))

    if not per_request_counts:
        return 0, 0

    if all(count == per_request_counts[0] for count in per_request_counts):
        generated_tokens = per_request_counts[0]
    else:
        generated_tokens = int(round(statistics.mean(per_request_counts)))
    return generated_tokens, sum(per_request_counts)


def extract_trtllm_generated_tokens(outputs) -> tuple[int, int]:
    per_request_counts = []
    for output in outputs:
        completion_outputs = getattr(output, "outputs", [])
        if not completion_outputs:
            per_request_counts.append(0)
            continue
        per_request_counts.append(len(getattr(completion_outputs[0], "token_ids", [])))

    if not per_request_counts:
        return 0, 0

    if all(count == per_request_counts[0] for count in per_request_counts):
        generated_tokens = per_request_counts[0]
    else:
        generated_tokens = int(round(statistics.mean(per_request_counts)))
    return generated_tokens, sum(per_request_counts)


def _timedelta_ms(value) -> float | None:
    if value is None:
        return None
    total_seconds = getattr(value, "total_seconds", None)
    if total_seconds is None:
        return None
    return total_seconds() * 1000.0


def extract_trtllm_phase_timings(outputs) -> tuple[float | None, float | None]:
    arrivals = []
    first_tokens = []
    last_tokens = []

    for output in outputs:
        completion_outputs = getattr(output, "outputs", [])
        if not completion_outputs:
            continue
        perf_metrics = getattr(completion_outputs[0], "request_perf_metrics", None)
        timing_metrics = getattr(perf_metrics, "timing_metrics", None)
        if timing_metrics is None:
            continue

        arrival_ms = _timedelta_ms(getattr(timing_metrics, "arrival_time", None))
        first_token_ms = _timedelta_ms(getattr(timing_metrics, "first_token_time", None))
        last_token_ms = _timedelta_ms(getattr(timing_metrics, "last_token_time", None))
        if arrival_ms is None or first_token_ms is None or last_token_ms is None:
            continue

        arrivals.append(arrival_ms)
        first_tokens.append(first_token_ms)
        last_tokens.append(last_token_ms)

    if not arrivals or not first_tokens or not last_tokens:
        return None, None

    prefill_ms = max(first_tokens) - min(arrivals)
    decode_ms = max(last_tokens) - min(first_tokens)
    return prefill_ms, decode_ms


def _find_child_process_ids() -> set[int]:
    try:
        import psutil
    except ImportError:
        return set()

    try:
        return {child.pid for child in psutil.Process().children(recursive=True)}
    except psutil.Error:
        return set()


def get_child_worker_memory_gb() -> float | None:
    worker_pids = _find_child_process_ids()
    if not worker_pids:
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    total_mib = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if pid in worker_pids:
            total_mib += used_mib

    if total_mib <= 0:
        return None
    return total_mib / 1024.0


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(statistics.mean(present))


def summarize_results(all_iters: list[IterationMetrics]):
    def avg(field: str) -> float | None:
        return _mean_or_none([getattr(m, field) for m in all_iters])

    sample = all_iters[0]
    summary = IterationMetrics(
        backend=sample.backend,
        prefill_ms=avg("prefill_ms"),
        decode_ms=avg("decode_ms"),
        total_generate_ms=float(statistics.mean([m.total_generate_ms for m in all_iters])),
        decode_tokens_per_sec=avg("decode_tokens_per_sec"),
        end_to_end_tokens_per_sec=float(
            statistics.mean([m.end_to_end_tokens_per_sec for m in all_iters])
        ),
        prompt_tokens=sample.prompt_tokens,
        generated_tokens=sample.generated_tokens,
        batch_size=sample.batch_size,
        dtype=sample.dtype,
        model_name=sample.model_name,
        max_gpu_mem_gb=float(statistics.mean([m.max_gpu_mem_gb for m in all_iters])),
    )
    return summary


def _fmt_row(name: str, value: Any, unit: str = "") -> str:
    return f"{name:<30} {value:>14}{(' ' + unit) if unit else ''}"


def _fmt_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _print_iteration(idx: int, metrics: IterationMetrics):
    print(
        f"iter={idx:02d} "
        f"prefill_ms={_fmt_optional_float(metrics.prefill_ms)} "
        f"decode_ms={_fmt_optional_float(metrics.decode_ms)} "
        f"generate_ms={metrics.total_generate_ms:.2f} "
        f"decode_tok_s={_fmt_optional_float(metrics.decode_tokens_per_sec)} "
        f"e2e_tok_s={metrics.end_to_end_tokens_per_sec:.2f} "
        f"max_mem_gb={metrics.max_gpu_mem_gb:.2f}"
    )


def _print_summary(summary: IterationMetrics):
    print("\n=== Benchmark Summary (Averages) ===")
    print(_fmt_row("backend", summary.backend))
    print(_fmt_row("model_name", summary.model_name))
    print(_fmt_row("dtype", summary.dtype))
    print(_fmt_row("batch_size", summary.batch_size))
    print(_fmt_row("prompt_tokens", summary.prompt_tokens))
    print(_fmt_row("generated_tokens", summary.generated_tokens))
    print(_fmt_row("prefill_ms", _fmt_optional_float(summary.prefill_ms), "ms"))
    print(_fmt_row("decode_ms", _fmt_optional_float(summary.decode_ms), "ms"))
    print(_fmt_row("total_generate_ms", f"{summary.total_generate_ms:.2f}", "ms"))
    print(
        _fmt_row(
            "decode_tokens_per_sec",
            _fmt_optional_float(summary.decode_tokens_per_sec),
            "tok/s",
        )
    )
    print(
        _fmt_row(
            "end_to_end_tokens_per_sec",
            f"{summary.end_to_end_tokens_per_sec:.2f}",
            "tok/s",
        )
    )
    print(_fmt_row("max_gpu_mem_allocated", f"{summary.max_gpu_mem_gb:.2f}", "GB"))


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="LLM inference benchmark on a single GPU")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm", "trtllm"])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--profile-decode",
        action="store_true",
        help="HF only: emit explicit prefill/decode markers for external profilers.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM only: fraction of GPU memory reserved by the engine.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="vLLM only: optional max model length override.",
    )
    parser.add_argument(
        "--vllm-enable-prefix-caching",
        action="store_true",
        help="vLLM only: enable prefix caching across requests/iterations.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="vLLM only: disable CUDA graph capture and use eager execution only.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def run_hf_benchmark(args) -> list[IterationMetrics]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        model, tokenizer = load_hf_model_and_tokenizer(args.model, args.dtype)
        prompt_token_ids = make_synthetic_prompt_ids(tokenizer, args.prompt_len)
        input_ids, attention_mask, prompt_tokens = make_hf_batch(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            batch_size=args.batch_size,
            device="cuda",
        )
    except RuntimeError as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
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
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    "CUDA OOM during profile pass. Try lowering --batch-size, --prompt-len, or --max-new-tokens.",
                    file=sys.stderr,
                )
            else:
                print(f"Runtime error during profile pass: {exc}", file=sys.stderr)
            sys.exit(1)
        return []

    print(
        f"Warmup: {args.warmup_iters} iters | Benchmark: {args.benchmark_iters} iters | "
        f"backend=hf model={args.model} dtype={args.dtype} batch_size={args.batch_size}"
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
            print(
                f"warmup={i+1}/{args.warmup_iters} "
                f"prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f}"
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
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
                backend="hf",
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

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"CUDA OOM on benchmark iteration {i+1}. "
                    "Try lowering --batch-size, --prompt-len, or --max-new-tokens.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    return metrics_list


def run_vllm_benchmark(args) -> list[IterationMetrics]:
    if args.profile_decode:
        print("--profile-decode is only supported with --backend hf.", file=sys.stderr)
        sys.exit(2)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Another library may have already fixed the start method in this process.
        pass

    try:
        llm, tokenizer = load_vllm_engine_and_tokenizer(args)
        prompt_token_ids = make_synthetic_prompt_ids(tokenizer, args.prompt_len)
        prompt_tokens = len(prompt_token_ids)
        prompts = make_vllm_inputs(prompt_token_ids, args.batch_size)
    except RuntimeError as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    warmup_params = build_vllm_sampling_params(
        max_new_tokens=min(16, args.max_new_tokens),
        seed=args.seed,
    )
    benchmark_params = build_vllm_sampling_params(
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    print(
        f"Warmup: {args.warmup_iters} iters | Benchmark: {args.benchmark_iters} iters | "
        f"backend=vllm model={args.model} dtype={args.dtype} batch_size={args.batch_size}"
    )
    for i in range(args.warmup_iters):
        try:
            torch.cuda.reset_peak_memory_stats()
            gen_ms, outputs = time_vllm_generate(llm, prompts, warmup_params)
            generated_tokens, total_generated_tokens = extract_vllm_generated_tokens(outputs)
            e2e_tok_s = total_generated_tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
            print(
                f"warmup={i+1}/{args.warmup_iters} "
                f"generate_ms={gen_ms:.2f} e2e_tok_s={e2e_tok_s:.2f} "
                f"generated_tokens={generated_tokens}"
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    "CUDA OOM during warmup. Reduce --batch-size, --prompt-len, "
                    "--max-new-tokens, or --vllm-gpu-memory-utilization.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    metrics_list: list[IterationMetrics] = []
    max_observed_mem_gb: float | None = get_child_worker_memory_gb()

    for i in range(args.benchmark_iters):
        try:
            torch.cuda.reset_peak_memory_stats()
            gen_ms, outputs = time_vllm_generate(llm, prompts, benchmark_params)
            generated_tokens, total_generated_tokens = extract_vllm_generated_tokens(outputs)
            e2e_tok_s = total_generated_tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
            current_mem_gb = get_child_worker_memory_gb()
            if current_mem_gb is not None:
                max_observed_mem_gb = (
                    current_mem_gb
                    if max_observed_mem_gb is None
                    else max(max_observed_mem_gb, current_mem_gb)
                )

            iter_metrics = IterationMetrics(
                backend="vllm",
                prefill_ms=None,
                decode_ms=None,
                total_generate_ms=gen_ms,
                decode_tokens_per_sec=None,
                end_to_end_tokens_per_sec=e2e_tok_s,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                batch_size=args.batch_size,
                dtype=args.dtype,
                model_name=args.model,
                max_gpu_mem_gb=max_observed_mem_gb,
            )
            metrics_list.append(iter_metrics)
            _print_iteration(i + 1, iter_metrics)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"CUDA OOM on benchmark iteration {i+1}. "
                    "Try lowering --batch-size, --prompt-len, --max-new-tokens, "
                    "or --vllm-gpu-memory-utilization.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    return metrics_list


def run_trtllm_benchmark(args) -> list[IterationMetrics]:
    if args.profile_decode:
        print("--profile-decode is only supported with --backend hf.", file=sys.stderr)
        sys.exit(2)

    ensure_trtllm_runtime_ready_for_current_process()

    try:
        llm, tokenizer = load_trtllm_engine_and_tokenizer(args)
        prompt_token_ids = make_synthetic_prompt_ids(tokenizer, args.prompt_len)
        prompt_tokens = len(prompt_token_ids)
        prompts = [{"prompt_token_ids": list(prompt_token_ids)} for _ in range(args.batch_size)]
    except RuntimeError as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    warmup_params = build_trtllm_sampling_params(
        max_new_tokens=min(16, args.max_new_tokens),
        seed=args.seed,
    )
    benchmark_params = build_trtllm_sampling_params(
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    print(
        f"Warmup: {args.warmup_iters} iters | Benchmark: {args.benchmark_iters} iters | "
        f"backend=trtllm model={args.model} dtype={args.dtype} batch_size={args.batch_size}"
    )
    for i in range(args.warmup_iters):
        try:
            gen_ms, outputs = time_trtllm_generate(llm, prompts, warmup_params)
            generated_tokens, total_generated_tokens = extract_trtllm_generated_tokens(outputs)
            e2e_tok_s = total_generated_tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
            print(
                f"warmup={i+1}/{args.warmup_iters} "
                f"generate_ms={gen_ms:.2f} e2e_tok_s={e2e_tok_s:.2f} "
                f"generated_tokens={generated_tokens}"
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    "CUDA OOM during TensorRT-LLM warmup. Reduce --batch-size, "
                    "--prompt-len, or --max-new-tokens.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    metrics_list: list[IterationMetrics] = []
    max_observed_mem_gb: float | None = get_child_worker_memory_gb()

    for i in range(args.benchmark_iters):
        try:
            gen_ms, outputs = time_trtllm_generate(llm, prompts, benchmark_params)
            generated_tokens, total_generated_tokens = extract_trtllm_generated_tokens(outputs)
            prefill_ms, decode_ms = extract_trtllm_phase_timings(outputs)
            current_mem_gb = get_child_worker_memory_gb()
            if current_mem_gb is not None:
                max_observed_mem_gb = (
                    current_mem_gb
                    if max_observed_mem_gb is None
                    else max(max_observed_mem_gb, current_mem_gb)
                )

            decode_tok_s = (
                total_generated_tokens / (decode_ms / 1000.0)
                if decode_ms is not None and decode_ms > 0
                else None
            )
            e2e_tok_s = total_generated_tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0

            iter_metrics = IterationMetrics(
                backend="trtllm",
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
                max_gpu_mem_gb=max_observed_mem_gb or 0.0,
            )
            metrics_list.append(iter_metrics)
            _print_iteration(i + 1, iter_metrics)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"CUDA OOM on TensorRT-LLM benchmark iteration {i+1}. "
                    "Try lowering --batch-size, --prompt-len, or --max-new-tokens.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    return metrics_list


def main():
    args = _build_arg_parser().parse_args()

    if args.backend == "hf":
        metrics_list = run_hf_benchmark(args)
        if args.profile_decode:
            return
    elif args.backend == "vllm":
        metrics_list = run_vllm_benchmark(args)
    else:
        metrics_list = run_trtllm_benchmark(args)

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

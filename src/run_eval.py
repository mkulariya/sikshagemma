"""
lm-evaluation-harness wrapper for Gemma-4-E4B w/ 4-bit bnb quantization.

Avoids CLI escape hell — configure here, run `python run_eval.py`.

Usage:
    python run_eval.py
    python run_eval.py --tasks gpqa_diamond_zeroshot jeebench
    python run_eval.py --batch-size 4 --no-quant
    python run_eval.py --limit 20    # quick sanity run
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoTokenizer, BitsAndBytesConfig


DEFAULT_TASKS = [
    "gpqa_diamond_zeroshot",
    "milu_hindi_stem",
    "indicmmlu_pro_hindi_stem",
]

# Only these tasks benefit from Gemma 4 reasoning. Loglikelihood scoring
# (multiple_choice) breaks if thinking is forced — model thinks first then
# choice-logprobs are computed after a <think> block. Keep thinking off there.
DEFAULT_THINKING_TASKS = set()  # all off by default; opt-in via --thinking-tasks

DEFAULT_MODEL = "unsloth/gemma-4-E4B-it"
DEFAULT_MODEL = "merged_stage2"  # local path to merged Stage 1 adapter (for Stage 2 training)
DEFAULT_INCLUDE_PATH = "custom_tasks"
DEFAULT_OUTPUT_DIR = "final_results"


def _enable_thinking_on_tokenizer(tokenizer):
    """
    Monkey-patch tokenizer.apply_chat_template so every call defaults to
    enable_thinking=True. Returns the original function so caller can restore.
    """
    orig = tokenizer.apply_chat_template
    # Store the unwrapped original so toggling never stacks patches.
    base = getattr(tokenizer, "_orig_apply_chat_template", orig)
    tokenizer._orig_apply_chat_template = base

    def patched(*args, **kwargs):
        kwargs.setdefault("enable_thinking", True)
        try:
            return base(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return base(*args, **kwargs)

    tokenizer.apply_chat_template = patched
    return tokenizer


def _disable_thinking_on_tokenizer(tokenizer):
    """Restore tokenizer to non-thinking baseline (or leave alone if never patched)."""
    base = getattr(tokenizer, "_orig_apply_chat_template", None)
    if base is not None:
        tokenizer.apply_chat_template = base
    return tokenizer


def build_model(model_name: str, use_quant: bool, batch_size, trust_remote_code: bool,
                enable_thinking: bool = False):
    """
    Build HFLM with chat-template + thinking enabled (Gemma 4 reasoning mode).
    For quant + multimodal model classes (Gemma4ForConditionalGeneration),
    construct model externally to avoid HFLM's internal `quantization_config` collision
    and `load_in_4bit` kwarg leak into model __init__.
    """
    if not use_quant:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        if enable_thinking:
            _enable_thinking_on_tokenizer(tokenizer)

        return HFLM(
            pretrained=model_name,
            tokenizer=tokenizer,
            dtype="bfloat16",
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
            apply_chat_template=True,
        )

    # Pre-load model with bnb 4-bit. Try CausalLM first, then ImageTextToText.
    import torch
    from transformers import AutoModelForCausalLM

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    common_kwargs = dict(
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    except (ValueError, KeyError, TypeError):
        # Multimodal config (e.g. Gemma4ForConditionalGeneration). Fall back.
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_name, **common_kwargs)
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(model_name, **common_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if enable_thinking:
        _enable_thinking_on_tokenizer(tokenizer)

    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        apply_chat_template=True,
    )


def main():
    parser = argparse.ArgumentParser(description="lm-eval-harness runner for Gemma-4-E4B")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HF model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help="Tasks to run")
    parser.add_argument("--include-path", default=DEFAULT_INCLUDE_PATH,
                        help="Path to custom task YAML dir")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory base (timestamped subdir created inside)")
    parser.add_argument("--batch-size", default="auto",
                        help="Batch size (int or 'auto')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (debug)")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable 4-bit quantization (use plain bfloat16)")
    parser.add_argument("--no-trust-remote-code", action="store_true",
                        help="Disable trust_remote_code")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Override num_fewshot for tasks that support it")
    parser.add_argument("--log-samples", action="store_true", default=True,
                        help="Log per-sample predictions (default: on)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Force thinking OFF for ALL tasks (override per-task default)")
    parser.add_argument("--all-thinking", action="store_true",
                        help="Force thinking ON for ALL tasks (override per-task default)")
    parser.add_argument("--thinking-tasks", nargs="+", default=None,
                        help=f"Which tasks need thinking. Default: {sorted(DEFAULT_THINKING_TASKS)}")
    args = parser.parse_args()

    # Resolve batch size
    try:
        batch_size = int(args.batch_size)
    except (ValueError, TypeError):
        batch_size = args.batch_size  # 'auto'

    # Output dir w/ timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"run_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve include_path (must exist if user passed a custom one)
    include_path = None
    if args.include_path:
        p = Path(args.include_path).resolve()
        if p.exists():
            include_path = str(p)
        else:
            print(f"  WARN: include_path '{p}' not found — running stock tasks only.")

    # Decide which tasks get thinking
    if args.no_thinking:
        thinking_set = set()
    elif args.all_thinking:
        thinking_set = set(args.tasks)
    elif args.thinking_tasks is not None:
        thinking_set = set(args.thinking_tasks)
    else:
        thinking_set = set(DEFAULT_THINKING_TASKS)

    thinking_tasks = [t for t in args.tasks if t in thinking_set]
    no_thinking_tasks = [t for t in args.tasks if t not in thinking_set]

    print("=" * 60)
    print("lm-eval-harness runner")
    print("=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Quantization: {'BF16 (no quant)' if args.no_quant else '4-bit NF4 bnb'}")
    print(f"  No-think    : {no_thinking_tasks or '(none)'}")
    print(f"  Thinking    : {thinking_tasks or '(none)'}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Limit       : {args.limit or 'ALL'}")
    print(f"  Include     : {include_path or '(none)'}")
    print(f"  Output      : {out_root}")
    print("=" * 60)

    # Build model ONCE. Thinking is toggled on the tokenizer per phase.
    print("\nLoading model...")
    lm = build_model(
        model_name=args.model,
        use_quant=not args.no_quant,
        batch_size=batch_size,
        trust_remote_code=not args.no_trust_remote_code,
        enable_thinking=False,  # start clean; toggle per phase
    )
    print("Model loaded.\n")

    tokenizer = lm.tokenizer

    if include_path:
        task_manager = TaskManager(include_path=include_path)
    else:
        task_manager = TaskManager()

    def run_phase(phase_tasks, phase_thinking, phase_label):
        if not phase_tasks:
            return {}
        if phase_thinking:
            _enable_thinking_on_tokenizer(tokenizer)
        else:
            _disable_thinking_on_tokenizer(tokenizer)
        print("\n" + "─" * 60)
        print(f"PHASE: {phase_label}  thinking={'ON' if phase_thinking else 'OFF'}")
        print(f"  tasks: {phase_tasks}")
        print("─" * 60)
        kwargs = dict(
            model=lm,
            tasks=phase_tasks,
            task_manager=task_manager,
            log_samples=args.log_samples,
        )
        if args.limit is not None:
            kwargs["limit"] = args.limit
        if args.num_fewshot is not None:
            kwargs["num_fewshot"] = args.num_fewshot
        return lm_eval.simple_evaluate(**kwargs)

    # Phase 1: no-thinking tasks first (loglikelihood, fast)
    res_no_think = run_phase(no_thinking_tasks, False, "NO-THINKING")
    # Phase 2: thinking tasks (slow generation)
    res_think = run_phase(thinking_tasks, True, "THINKING")

    # Merge results across phases
    merged_results = {}
    merged_samples = {}
    merged_configs = {}
    merged_versions = {}
    merged_nsamples = {}
    for r in (res_no_think, res_think):
        if not r:
            continue
        merged_results.update(r.get("results", {}) or {})
        merged_samples.update(r.get("samples", {}) or {})
        merged_configs.update(r.get("configs", {}) or {})
        merged_versions.update(r.get("versions", {}) or {})
        merged_nsamples.update(r.get("n-samples", {}) or {})

    # Save outputs
    results_path = out_root / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    config_path = out_root / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "tasks": args.tasks,
            "thinking_tasks": sorted(thinking_set & set(args.tasks)),
            "no_thinking_tasks": no_thinking_tasks,
            "batch_size": str(batch_size),
            "limit": args.limit,
            "quantization": "none" if args.no_quant else "nf4-4bit",
            "include_path": include_path,
            "timestamp": timestamp,
            "num_fewshot": args.num_fewshot,
            "configs": merged_configs,
            "versions": merged_versions,
            "n-samples": merged_nsamples,
        }, f, indent=2, ensure_ascii=False, default=str)

    for task_name, task_samples in merged_samples.items():
        sample_path = out_root / f"samples_{task_name}.jsonl"
        with open(sample_path, "w", encoding="utf-8") as f:
            for s in task_samples:
                f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(merged_results, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()

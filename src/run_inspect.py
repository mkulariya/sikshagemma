"""
Wrapper to run inspect-ai JEE benchmarks against Gemma-4-E4B (or any HF model).

Defaults:
  - HF backend (transformers + bnb 4-bit NF4).
  - Thinking mode OFF (monkey-patches tokenizer.apply_chat_template to default
    enable_thinking=False, so inspect-ai's HF provider never enables it).
  - Runs both jee_main_2026 and jee_advanced_2025.

Usage:
    python3 src/run_inspect.py
    python3 src/run_inspect.py --tasks jee_main_2026
    python3 src/run_inspect.py --limit 5
    python3 src/run_inspect.py --no-quant
    python3 src/run_inspect.py --thinking          # opt-in
    python3 src/run_inspect.py --backend vllm \
            --vllm-base-url http://localhost:8000/v1 \
            --vllm-model unsloth/gemma-4-E4B-it
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


# ── Pre-import: tokenizer monkey-patch (thinking OFF by default) ──────────

_ENABLE_THINKING_DEFAULT = False


def _msg_to_dict(m):
    """Convert pydantic ChatMessage (inspect-ai) to plain dict for jinja templates."""
    if isinstance(m, dict):
        return m
    # Pydantic v2
    if hasattr(m, "model_dump"):
        d = m.model_dump()
    elif hasattr(m, "dict"):
        d = m.dict()
    else:
        return m
    # Gemma 4 template expects 'content' to be a string. Inspect-ai sometimes
    # represents content as a list of content parts (text/image/etc.). Flatten.
    content = d.get("content")
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif "text" in p:
                    parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append(str(p))
        d["content"] = "".join(parts)
    return d


def _normalize_messages(conv):
    """Recursively turn ChatMessage objects into dicts inside conv list/batch."""
    if isinstance(conv, list) and conv:
        first = conv[0]
        if isinstance(first, list):  # batch of conversations
            return [[_msg_to_dict(m) for m in inner] for inner in conv]
        return [_msg_to_dict(m) for m in conv]
    return conv


def _install_tokenizer_patch(enable_thinking: bool):
    """
    Intercept AutoTokenizer.from_pretrained so every loaded tokenizer's
    apply_chat_template:
      (a) defaults to enable_thinking=<flag>,
      (b) converts pydantic ChatMessage objects (used by inspect-ai) into
          plain dicts before forwarding, so Gemma 4 jinja template (which
          calls message.get()) works.
    """
    from transformers import AutoTokenizer

    if getattr(AutoTokenizer, "_jee_patched", False):
        AutoTokenizer._jee_enable_thinking = enable_thinking
        return

    AutoTokenizer._jee_enable_thinking = enable_thinking
    orig_from_pretrained = AutoTokenizer.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, *args, **kwargs):
        tok = orig_from_pretrained.__func__(cls, *args, **kwargs) \
            if hasattr(orig_from_pretrained, "__func__") \
            else orig_from_pretrained(*args, **kwargs)
        orig_apply = tok.apply_chat_template

        def patched_apply(*a, **kw):
            # Normalize messages (first positional arg) to plain dicts.
            if a:
                a = (_normalize_messages(a[0]),) + a[1:]
            elif "conversation" in kw:
                kw["conversation"] = _normalize_messages(kw["conversation"])
            elif "messages" in kw:
                kw["messages"] = _normalize_messages(kw["messages"])

            kw.setdefault("enable_thinking", AutoTokenizer._jee_enable_thinking)
            try:
                return orig_apply(*a, **kw)
            except TypeError:
                kw.pop("enable_thinking", None)
                return orig_apply(*a, **kw)

        tok.apply_chat_template = patched_apply
        return tok

    AutoTokenizer.from_pretrained = patched_from_pretrained
    AutoTokenizer._jee_patched = True


# ── Main ──────────────────────────────────────────────────────────────────

DEFAULT_TASKS = ["jee_main_2026", "jee_advanced_2025"]
DEFAULT_MODEL = "unsloth/gemma-4-E4B-it"
DEFAULT_OUTPUT_DIR = "final_results"


def build_model_args(args) -> dict:
    """
    Build HF backend model args. bnb 4-bit on by default.

    Inspect-ai's hf provider passes these to AutoModel.from_pretrained.
    """
    if args.backend != "hf":
        return {}

    # inspect-ai HF provider owns device_map, dtype, trust_remote_code internally.
    # Only pass quantization_config here.
    model_args: dict = {}

    if not args.no_quant:
        import torch
        from transformers import BitsAndBytesConfig
        model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return model_args


def main():
    ap = argparse.ArgumentParser(description="inspect-ai runner for JEE Hindi benchmarks")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                    choices=["jee_main_2026", "jee_advanced_2025"],
                    help="Which task(s) to run")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model id for hf backend (default: {DEFAULT_MODEL})")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="Model backend (default: hf)")
    ap.add_argument("--vllm-base-url", default=None,
                    help="Base URL for vllm OpenAI-compatible endpoint (backend=vllm)")
    ap.add_argument("--vllm-model", default=None,
                    help="Model name as registered in vllm server (defaults to --model)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-task sample limit (debug)")
    ap.add_argument("--no-quant", action="store_true",
                    help="Disable bnb 4-bit; load BF16 (hf backend only)")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable Gemma 4 thinking mode (default: OFF)")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Max generation tokens per sample")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Output base dir (timestamped subdir created inside)")
    ap.add_argument("--log-level", default="info",
                    help="inspect-ai log level")
    args = ap.parse_args()

    # 1. Install tokenizer patch BEFORE importing inspect-ai.
    _install_tokenizer_patch(enable_thinking=args.thinking)

    # 2. Now safe to import inspect-ai.
    from inspect_ai import eval as inspect_eval

    # 3. Import our task definitions.
    sys.path.insert(0, str(Path(__file__).parent))
    from inspect_tasks.jee_main_2026 import jee_main_2026
    from inspect_tasks.jee_advanced_2025 import jee_advanced_2025

    task_registry = {
        "jee_main_2026": jee_main_2026,
        "jee_advanced_2025": jee_advanced_2025,
    }
    tasks = [task_registry[name]() for name in args.tasks]

    # 4. Build model spec.
    if args.backend == "hf":
        model_spec = f"hf/{args.model}"
        model_args = build_model_args(args)
    else:  # vllm
        if not args.vllm_base_url:
            ap.error("--vllm-base-url required when --backend vllm")
        # inspect-ai openai-compatible provider requires 'openai-api/<service>/<model>'.
        # The <service> token drives env var lookup: <SERVICE>_BASE_URL + <SERVICE>_API_KEY.
        vllm_model = args.vllm_model or args.model
        service = "vllm"
        model_spec = f"openai-api/{service}/{vllm_model}"
        os.environ[f"{service.upper()}_BASE_URL"] = args.vllm_base_url
        os.environ.setdefault(f"{service.upper()}_API_KEY", "EMPTY")
        model_args = {}

    # 5. Output dir.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"inspect_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("inspect-ai JEE benchmark runner")
    print("=" * 60)
    print(f"  Backend       : {args.backend}")
    print(f"  Model         : {model_spec}")
    print(f"  Quantization  : {'BF16 (no quant)' if args.no_quant else '4-bit NF4 bnb'}")
    print(f"  Thinking      : {'ON' if args.thinking else 'OFF'} "
          f"({'via vllm chat_template_kwargs' if args.backend == 'vllm' else 'via HF tokenizer patch'})")
    print(f"  Tasks         : {args.tasks}")
    print(f"  Max tokens    : {args.max_tokens}")
    print(f"  Limit         : {args.limit or 'ALL'}")
    print(f"  Output dir    : {out_dir}")
    print("=" * 60)

    # 6. Run.
    eval_kwargs = dict(
        tasks=tasks,
        model=model_spec,
        model_args=model_args,
        log_dir=str(out_dir),
        log_level=args.log_level,
        max_tokens=args.max_tokens,
    )
    # vllm: thinking toggle lives on the server-side chat template; forward via
    # OpenAI extra_body -> chat_template_kwargs (vllm 0.6+).
    if args.backend == "vllm":
        eval_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": bool(args.thinking)}
        }
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    logs = inspect_eval(**eval_kwargs)

    # 7. Persist a compact summary alongside the inspect log files.
    def _extract_metrics(m_field):
        out = {}
        if m_field is None:
            return out
        if isinstance(m_field, dict):
            for name, mv in m_field.items():
                out[name] = getattr(mv, "value", mv)
        elif isinstance(m_field, list):
            for mv in m_field:
                name = getattr(mv, "name", None) or (mv if isinstance(mv, str) else "?")
                out[str(name)] = getattr(mv, "value", mv)
        return out

    summary = []
    for log in logs:
        task_name = getattr(log.eval, "task", "unknown")
        results = getattr(log, "results", None)
        scores_block = []
        if results and getattr(results, "scores", None):
            for s in results.scores:
                metrics = _extract_metrics(getattr(s, "metrics", None))
                scores_block.append({"name": getattr(s, "name", "?"), "metrics": metrics})
        dataset_info = getattr(log.eval, "dataset", None)
        samples_count = None
        if isinstance(dataset_info, dict):
            samples_count = dataset_info.get("samples")
        elif dataset_info is not None:
            samples_count = getattr(dataset_info, "samples", None)
        summary.append({
            "task": task_name,
            "samples": samples_count,
            "scores": scores_block,
        })

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nLog dir: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

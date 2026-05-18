"""
Stage 3: GRPO + RLVR training on JEE Advanced / BSc-MSc hard questions.
Uses FastVisionModel with fast_inference=False (required for Gemma-4 GRPO).
Based on official Unsloth Gemma-4 E2B GRPO notebook.

Reward functions (all self-contained):
  correctness       2.0 max  — PRIMARY: MCQ exact, numerical ±1%
  format            0.35 max — step structure + math working + <answer> tag
  cot_quality       0.5 max  — reasoning depth, formulas, logical connectors
  solution_length   0.3 max  — penalise too short / too long
  anti_repetition   0.2 max  — penalise degenerate / looping outputs
  subject_accuracy  0.2 max  — subject-specific terminology check
  hindi_fluency     0.3 max  — Hindi response quality (Hindi questions only)

Total max: 3.75
"""

import os
import transformers.utils.hub as _hub

# Provide the missing TRANSFORMERS_CACHE attribute for llm_blender
def _transformers_cache():
    return os.path.join(os.path.expanduser("~"), ".cache/huggingface/transformers")
_hub.TRANSFORMERS_CACHE = _transformers_cache

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import random
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template

DEFAULT_MODEL = "merged_stage2"
DEFAULT_DATASET_PATH = "sikshagemma_stage3_training.jsonl"
DEFAULT_OUTPUT_DIR = "stage3_gemma4_e4b"
RANDOM_SEED = 3407


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> str:
    """Priority: <answer> tag → \boxed → 'Final answer:' marker → last bare line."""
    # 1. <answer> tag
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. \boxed{X}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        val = m.group(1).strip()
        if re.match(r'^[A-D]$', val, re.IGNORECASE):
            return val.upper()
        return val

    # 3. "Final answer: X" / "अंतिम उत्तर: X"
    for pat in [
        r'(?:Final\s+answer|अंतिम\s+उत्तर|उत्तर)\s*[:ः\-]\s*(.+?)(?:\n|$)',
        r'(?:answer)\s*[:\=]\s*\(?([A-D])\)?(?:\s|$)',
        r'(?:answer)\s*[:\=]\s*(-?\d+\.?\d*)(?:\s|$)',
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip().rstrip('.')

    # 4. Last non-empty line that looks like a bare answer
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        if re.match(r'^[A-D]$', last, re.IGNORECASE):
            return last.upper()
        if re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', last):
            return last

    return ""


def normalize_answer(ans: str) -> str:
    a = ans.strip().lower().rstrip('.').rstrip(')').strip()
    m = re.match(r'^\(?([a-d])\)?$', a)
    if m:
        return m.group(1)
    return a


def try_eval_numerical(val: str) -> float | None:
    """
    Try to convert a model-generated answer string to float.
    Handles: decimals, fractions, LaTeX fractions, sqrt, pi, powers.
    Returns None if not parseable as a number.
    """
    val = val.strip()
    if not val:
        return None

    # Plain float / scientific notation (e.g. 4.25, 2e-6)
    try:
        return float(val)
    except ValueError:
        pass

    # Simple fraction: 1/2, -3/4, 6/11
    m = re.match(r'^(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)$', val)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return num / den if den != 0 else None

    # LaTeX fraction: \frac{a}{b} or \frac ab
    m = re.match(r'^\\frac\{?\s*(-?[\d.]+)\s*\}?\{?\s*(-?[\d.]+)\s*\}?$', val)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return num / den if den != 0 else None

    # LaTeX fraction with pi numerator: \frac{3\pi}{4}
    m = re.match(r'^\\frac\{\s*(-?\d*\.?\d*)\s*\\pi\s*\}\{\s*(-?\d+\.?\d*)\s*\}$', val)
    if m:
        coeff = float(m.group(1)) if m.group(1) not in ('', '-') else (1.0 if m.group(1) == '' else -1.0)
        den = float(m.group(2))
        return coeff * math.pi / den if den != 0 else None

    # Bare pi expression: \pi, 3\pi, \pi/4, 3\pi/4
    m = re.match(r'^(-?\d*\.?\d*)\s*\\?pi\s*(?:/\s*(-?\d+\.?\d*))?$', val, re.IGNORECASE)
    if m:
        coeff_str = m.group(1)
        coeff = float(coeff_str) if coeff_str not in ('', '-') else (1.0 if coeff_str == '' else -1.0)
        den = float(m.group(2)) if m.group(2) else 1.0
        return coeff * math.pi / den

    # Power notation: 2^{-1}, 10^3, 8500^{-1}
    m = re.match(r'^(-?[\d.]+)\^\{?(-?[\d.]+)\}?$', val)
    if m:
        base, exp = float(m.group(1)), float(m.group(2))
        return base ** exp

    # Sqrt: \sqrt{10}, sqrt(10)
    m = re.match(r'^\\?sqrt\{?(\d+\.?\d*)\}?$', val, re.IGNORECASE)
    if m:
        return math.sqrt(float(m.group(1)))

    # Expression: A - \sqrt{B}, e.g. 4-\sqrt{10}
    m = re.match(r'^(-?[\d.]+)\s*([+\-])\s*\\?sqrt\{?(\d+\.?\d*)\}?$', val)
    if m:
        a = float(m.group(1))
        op = m.group(2)
        b = math.sqrt(float(m.group(3)))
        return a + b if op == '+' else a - b

    # Scientific notation with \times: 2.0 \times 10^{3}
    m = re.match(r'^(-?[\d.]+)\s*\\times\s*10\^\{?(-?\d+)\}?$', val)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))

    # Percentage: 1.5%
    m = re.match(r'^(-?[\d.]+)\s*%$', val)
    if m:
        return float(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Reward 1: Correctness (2.0 max)
# ---------------------------------------------------------------------------

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    2.0 if correct, 0.5 if close (numerical ±5%), 0.0 otherwise.
    Model answer is evaluated via try_eval_numerical to handle fractions,
    LaTeX expressions, sqrt, pi etc. before numeric comparison.
    """
    rewards = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        extracted = extract_final_answer(response)
        ground_truth = str(answer[i] if isinstance(answer, list) else answer).strip()

        norm_ext = normalize_answer(extracted)
        norm_gt = normalize_answer(ground_truth)

        # MCQ: exact letter match
        if norm_gt in {'a', 'b', 'c', 'd'}:
            rewards.append(2.0 if norm_ext == norm_gt else 0.0)
            continue

        # Numerical: try converting both to float (handles fractions, LaTeX, etc.)
        true_num = try_eval_numerical(norm_gt)
        ext_num = try_eval_numerical(norm_ext)

        if true_num is not None and ext_num is not None:
            if true_num == 0:
                rewards.append(2.0 if abs(ext_num) < 1e-6 else 0.0)
            else:
                rel_err = abs(ext_num - true_num) / abs(true_num)
                rewards.append(2.0 if rel_err < 0.01 else 0.5 if rel_err < 0.05 else 0.0)
        else:
            # Fallback: string match
            rewards.append(2.0 if norm_ext == norm_gt else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward 2: Format (0.35 max)
# ---------------------------------------------------------------------------

def format_reward(completions, **kwargs):
    """
    Evaluate reasoning structure, not the answer tag content.
    - Numbered / step-by-step reasoning in body  → 0.15
    - Mathematical working (=, formulas, ops)     → 0.10
    - <answer> tag present with non-empty content → 0.10
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0

        # Extract reasoning body (everything before <answer>)
        ans_pos = text.lower().find('<answer>')
        body = text[:ans_pos] if ans_pos > -1 else text

        # Step structure in body
        if re.search(r'(Step\s*\d|चरण\s*\d|पद\s*\d|\d+[\.\)]\s)', body):
            score += 0.15

        # Mathematical working in body
        if re.search(r'[=+\-×÷]|\\frac|\\int|\^|_\{|\bmod\b', body):
            score += 0.10

        # <answer> tag with non-empty content
        m = re.search(r'<answer>\s*(.+?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            score += 0.10

        rewards.append(min(score, 0.35))
    return rewards


# ---------------------------------------------------------------------------
# Reward 3: CoT Quality (0.5 max)
# ---------------------------------------------------------------------------

def cot_quality_reward_func(completions, **kwargs):
    """Reward depth of reasoning: step count, formulas, logical connectors."""
    rewards = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0

        # Step count
        step_count = len(re.findall(r'(?:Step\s*\d|चरण\s*\d|पद\s*\d)', response))
        numbered = len(re.findall(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        total_steps = max(step_count, numbered)

        if total_steps >= 5:
            score += 0.20
        elif total_steps >= 3:
            score += 0.15
        elif total_steps >= 1:
            score += 0.05

        # Mathematical expressions
        if re.search(r'[=×÷∫∑√∞∂±]|\\frac|\\int|\\sum|[a-zA-Z]\^|[a-zA-Z]_\d', response):
            score += 0.10

        # Logical connectors (English + Hindi)
        connectors = len(re.findall(
            r'(because|therefore|hence|thus|since|अतः|इसलिए|क्योंकि|चूँकि|इस\s*प्रकार)',
            response, re.IGNORECASE
        ))
        if connectors >= 2:
            score += 0.10
        elif connectors >= 1:
            score += 0.05

        # Penalise: answer before reasoning
        ans_pos = response.lower().find('final answer')
        step_pos = response.lower().find('step')
        if step_pos > -1 and ans_pos > -1 and ans_pos < step_pos:
            score -= 0.10

        rewards.append(max(0.0, min(score, 0.5)))
    return rewards


# ---------------------------------------------------------------------------
# Reward 4: Anti-Repetition (0.2 max)
# ---------------------------------------------------------------------------

def repetition_penalty_reward_func(completions, **kwargs):
    """Penalise duplicate lines and repeated trigrams."""
    rewards = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        if len(response.strip()) < 10:
            rewards.append(0.0)
            continue
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if len(lines) < 2:
            rewards.append(0.0)
            continue
        line_dup_ratio = len(set(lines)) / len(lines)
        words = response.split()
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
        ngram_dup_ratio = len(set(trigrams)) / max(len(trigrams), 1)
        score = 0.2
        if line_dup_ratio < 0.6:
            score -= 0.10
        if ngram_dup_ratio < 0.5:
            score -= 0.10
        rewards.append(max(0.0, score))
    return rewards


# ---------------------------------------------------------------------------
# Reward 6: Subject Accuracy (0.2 max)
# ---------------------------------------------------------------------------

def subject_accuracy_reward_func(prompts, completions, answer, subject, **kwargs):
    """Subject-specific terminology bonus. Requires 'subject' column in dataset."""
    rewards = []
    subject_list = [subject] if isinstance(subject, str) else subject
    for i, completion in enumerate(completions):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        subj = (subject_list[i] if isinstance(subject_list, list) and i < len(subject_list)
                else subject_list[0] if subject_list else "").lower()
        score = 0.0
        if 'physics' in subj:
            if re.search(r'(?:m/s|kg|N\b|J\b|W\b|V\b|Ω|T\b|Hz|Pa|eV)', response):
                score += 0.10
            if re.search(r'(?:F\s*=|E\s*=|V\s*=|P\s*=|λ\s*=|ω\s*=)', response):
                score += 0.10
        elif 'chemistry' in subj:
            if re.search(r'[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+|→|⇌|mol\b|pH\b', response):
                score += 0.10
            if re.search(r'(?:oxidation|reduction|equilibrium|enthalpy|entropy)', response, re.IGNORECASE):
                score += 0.10
        elif 'math' in subj:
            if re.search(r'(?:∫|∑|lim|∂|∇|∈|∀|∃|⇒|⇔|∞)', response):
                score += 0.10
            if re.search(r'(?:derivative|integral|limit|theorem|matrix|eigenvalue)', response, re.IGNORECASE):
                score += 0.10
        elif 'biology' in subj:
            if re.search(r'(?:cell|DNA|RNA|protein|enzyme|mitosis|meiosis|gene|allele)', response, re.IGNORECASE):
                score += 0.10
            if re.search(r'(?:membrane|nucleus|mitochondria|ribosome|ATP|NADH)', response, re.IGNORECASE):
                score += 0.10
        rewards.append(min(score, 0.2))
    return rewards


# ---------------------------------------------------------------------------
# Reward 7: Hindi Fluency — Hindi questions only (0.3 max)
# ---------------------------------------------------------------------------

def hindi_fluency_reward(prompts, completions, **kwargs):
    """
    Reward Hindi response quality, but ONLY when the question is in Hindi.
    Hindi detection: ≥20 Devanagari characters in the prompt.
    Returns 0.0 silently for English questions (no penalty).
    """
    rewards = []
    prompt_list = prompts if isinstance(prompts, list) else [prompts] * len(completions)
    for i, completion in enumerate(completions):
        prompt_str = prompt_list[i] if i < len(prompt_list) else ""
        if isinstance(prompt_str, list):
            # list of message dicts — flatten to string
            prompt_str = " ".join(m.get("content", "") for m in prompt_str)

        # Skip entirely for English questions
        devanagari_in_prompt = len(re.findall(r'[ऀ-ॿ]', prompt_str))
        if devanagari_in_prompt < 20:
            rewards.append(0.0)
            continue

        response = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        devanagari_in_response = len(re.findall(r'[ऀ-ॿ]', response))
        total_chars = len(response.strip())

        if total_chars > 0:
            ratio = devanagari_in_response / total_chars
            if ratio >= 0.5:
                score += 0.15
            elif ratio >= 0.2:
                score += 0.10
            elif ratio >= 0.05:
                score += 0.03

        hindi_markers = len(re.findall(
            r'(का|की|के|है|हैं|था|थी|थे|होता|होती|होते|करता|करती|करते|'
            r'जाता|जाती|जाते|रहा|रही|रहे|गया|गई|गए)',
            response
        ))
        if hindi_markers >= 5:
            score += 0.10
        elif hindi_markers >= 2:
            score += 0.05

        rewards.append(min(score, 0.3))
    return rewards


# ---------------------------------------------------------------------------
# All reward functions
# ---------------------------------------------------------------------------

REWARD_FUNCS = [
    correctness_reward_func,        # 2.0 max
    format_reward,                  # 0.35 max
    cot_quality_reward_func,        # 0.5 max
    repetition_penalty_reward_func, # 0.2 max
    subject_accuracy_reward_func,   # 0.2 max
    hindi_fluency_reward,           # 0.3 max (Hindi questions only)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_stats():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for GRPO training.")
    gpu = torch.cuda.get_device_properties(0)
    reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    total = round(gpu.total_memory / 1024**3, 3)
    print(f"GPU: {gpu.name}  |  Total: {total} GB  |  Reserved: {reserved} GB")
    return reserved, total


def save_run_metadata(path: str, payload: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 GRPO training – Gemma-4")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-adapter-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", default=False,
                        help="Load in 4-bit (saves VRAM but may reduce RL quality)")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="GRPO rollouts per prompt. Reduce to 4 if OOM.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--sample-size", type=int, default=0, help="0 = use full dataset")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    if args.sample_size and args.sample_size > 0:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    print(f"Dataset: {len(dataset)} GRPO records")

    # Model — FastVisionModel required for Gemma-4
    # fast_inference=False: vLLM not supported for Gemma-4 GRPO
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Gemma-4 thinking chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")

    gc.collect()
    torch.cuda.empty_cache()

    # Pre-process prompts with enable_thinking=True
    # GRPOTrainer accepts pre-formatted string prompts
    def apply_template(example):
        return {
            "prompt": tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                #enable_thinking=True,
            )
        }

    print("Applying chat template (enable_thinking=True)...")
    dataset = dataset.map(apply_template, desc="Formatting prompts")

    _, max_mem = print_gpu_stats()

    save_adapter_dir = args.save_adapter_dir or os.path.join(args.output_dir, "final_adapter")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        # GRPO
        num_generations=args.num_generations,
        max_prompt_length=args.max_seq_length - args.max_completion_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        # Optimizer
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        # Batch
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Training length
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        # Gemma-4 specific (from official Unsloth Gemma-4 GRPO notebook)
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="bnpo",
        mask_truncated_completions=True,
        # Logging + saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    model.warnings_issued = {}
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Starting GRPO training ({len(REWARD_FUNCS)} reward functions):")
    for rf in REWARD_FUNCS:
        print(f"  • {rf.__name__}")

    trainer_stats = trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    runtime = trainer_stats.metrics.get("train_runtime", 0)
    print(f"Training complete: {round(runtime/60, 2)} min")
    print(f"Peak VRAM: {used_mem} GB / {max_mem} GB ({round(used_mem/max_mem*100, 1)}%)")
    print(f"Train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")

    os.makedirs(save_adapter_dir, exist_ok=True)
    model.save_lora(save_adapter_dir)
    tokenizer.save_pretrained(save_adapter_dir)
    print(f"Saved adapter: {save_adapter_dir}")

    save_run_metadata(
        os.path.join(args.output_dir, "stage3_run_summary.json"),
        {
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "dataset_size": len(dataset),
            "output_dir": args.output_dir,
            "save_adapter_dir": save_adapter_dir,
            "max_seq_length": args.max_seq_length,
            "num_generations": args.num_generations,
            "reward_functions": [rf.__name__ for rf in REWARD_FUNCS],
            "train_runtime_seconds": runtime,
            "train_loss": trainer_stats.metrics.get("train_loss"),
        },
    )


if __name__ == "__main__":
    main()

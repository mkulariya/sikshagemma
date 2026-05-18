"""
GYANDEEP Solution Generator (Language-Matched) — JEE/NEET Step-by-Step Solutions
==================================================================================
Exact copy of solution_generator.py with modified prompts:
  - Language-matched output: English question → English solution, Hindi question → Hindi solution
  - Deep step-by-step reasoning with full working
  - No translation task

Input:  Train JSONL (questions with option-only answers — stage3_jee_advanced.jsonl or similar)
Output: JSONL records matching generate_stage3_questions.py format:
        question, options, answer, answer_type, solution, subject, level, group,
        chunk_id, source_chunks_file

Uses: Azure OpenAI (gpt-5.4-mini) or DeepSeek (via openai package)
Parallel: 5 workers, thread-safe rate limiter, resume support

Usage:
    python src/solution_generator_lang.py stage3_jee_advanced.jsonl \
        --output data/jee_solutions_lang.jsonl \
        --provider azure \
        --workers 5

    # With DeepSeek:
    python src/solution_generator_lang.py stage3_jee_advanced.jsonl \
        --output data/jee_solutions_lang.jsonl \
        --provider deepseek \
        --workers 5
"""

import json
import sys
import time
import os
import re
import threading
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, AzureOpenAI


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Azure defaults
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_MODEL = "gpt-5.4-mini"

# DeepSeek defaults
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-v4-pro"

# Generation knobs
TEMPERATURE = 0.3          # Low for solution accuracy
MAX_OUTPUT_TOKENS = 65536
DEFAULT_WORKERS = 5
DEFAULT_RPM = 10

# Resume — per-question progress files go here
CHECKPOINT_DIRNAME = "solution_lang_checkpoints"


# ═══════════════════════════════════════════════════════════════════════════
# THE PROMPT — Language-Matched Step-by-Step Solution Generation
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert JEE Advanced / NEET problem solver with deep mastery in Physics, Chemistry, Mathematics, and Biology.

VERY IMPORTANT: STRICTLY MATCH THE LANGUAGE OF THE QUESTION IN YOUR SOLUTION.
- If the question is written in English → your ENTIRE solution must be in English.
- If the question is written in Hindi → your ENTIRE solution must be in Hindi.
- You may use English technical/scientific terms even in a Hindi solution if the standard Hindi equivalent is not known to you.

VERY CRITICAL: THINK HARD AND REASON DEEPLY. Show complete step-by-step working. No shortcuts, no skipped steps. Every step must be independently followable by a student.

SOLVING RULES:
1. Identify the core concept/law/formula required first.
2. For numerical problems: state the formula → substitute all values with units → compute → state final answer with units.
3. For MCQ: after solving, explicitly state WHY the correct option is right AND briefly explain why each other option is wrong.
4. For conceptual/theoretical problems: derive from first principles. Do not just recall — reason through it.
5. Show ALL intermediate calculations. Never combine multiple steps silently.
6. LaTeX for math: $...$ inline, $$...$$ display. Backslash on every command: \\frac, \\sqrt, \\int, \\sum. NEVER write bare LaTeX commands (\\times, \\approx, \\frac, \\sigma, etc.) outside math delimiters — every mathematical symbol, expression, and equation MUST be wrapped in $...$ or $$...$$.
7. If a diagram is referenced but not provided, work from the text description alone.
8. NEVER just state the answer. The path to the answer is the value.

VERY IMPORTANT: Be thorough and rigorous. JEE Advanced questions require multi-step reasoning — do not cut corners.
VERY IMPORTANT: Strictly match the language of the question to solution.

VERY CRITICAL: OUTPUT FORMAT — Valid JSON only. No markdown fences, no preamble, no explanation outside JSON.

{
  "solution": "Full step-by-step solution in the SAME language as the question...",
  "answer": "<final answer in the exact format described below>",
  "answer_type": "mcq_single | mcq_multi | numerical",
  "subject": "physics|chemistry|mathematics|biology",
  "topic_tags": ["tag1", "tag2"],
  "difficulty": "easy|medium|hard|jee_advanced",
  "requires_diagram": true/false,
  "solution_language": "english|hindi"
}

ANSWER FIELD RULES — STRICT:
- For single-correct MCQ:        answer = "A" | "B" | "C" | "D"   (one capital letter, no punctuation, no parentheses)
- For multi-correct MCQ:         answer = sorted concatenated letters, e.g. "AC", "BD", "ABD"   (uppercase, no separators)
- For numerical / integer:       answer = the numeric value as a string, e.g. "9.5", "-12", "3.14"   (no units, no scientific notation prefix, just the number)
- NEVER include extra words. NEVER include explanations in the answer field. Solution goes in "solution", answer goes in "answer".
- If the question is purely conceptual without a determinable single value/letter, still put your best determination.

NEVER REFUSE. If ambiguous, pick the most reasonable interpretation and solve completely."""


def build_solution_prompt(question_json: dict) -> str:
    """Build the user prompt for a single JEE/NEET question."""

    q_text = question_json.get("question", "")
    q_no = question_json.get("question_no", "?")
    subject = question_json.get("subject", "unknown")
    options = question_json.get("options", [])
    has_diagrams = bool(question_json.get("diagram_paths", []))
    section = question_json.get("section", "")
    language = question_json.get("language", "unknown")
    # NOTE: input `answer` is intentionally NOT shown to the LLM — model must solve
    # from scratch and populate `answer` itself. Any answer carried in input JSONL
    # is discarded downstream.

    # Format options nicely
    if options:
        option_labels = []
        for i, opt in enumerate(options):
            label = f"({i+1})"
            option_labels.append(f"  {label}  {opt}")
        options_block = "\n".join(option_labels)
    else:
        options_block = "  (no options provided)"

    diagram_note = ""
    if has_diagrams:
        diagram_note = (
            "\n⚠️  This question has an attached diagram not available in text. "
            "Solve using the text description alone."
        )

    return f"""Solve this JEE/NEET question with complete step-by-step reasoning.

QUESTION #{q_no} | SUBJECT: {subject} | SECTION: {section} | LANGUAGE: {language}
ANSWER: (NOT provided — you must determine it yourself from the question and options)
DIAGRAM: {"YES (not provided — use text only)" if has_diagrams else "NO"}

📝 QUESTION:
{q_text}

📋 OPTIONS:
{options_block}
{diagram_note}

TASK: Produce a complete, rigorous, step-by-step solution in the SAME LANGUAGE as the question above.
Determine the correct answer entirely from your own reasoning. Explain why your chosen answer is correct, and (for MCQ) why each other option is wrong.
Think deeply. Show ALL working. No shortcuts.

ALSO: Populate the "answer" field with the final answer using the STRICT format rules:
  • single-correct MCQ → one capital letter (e.g. "B")
  • multi-correct MCQ → sorted concatenated letters (e.g. "AC", "ABD")
  • numerical / integer → the numeric value as a plain string (e.g. "9.5", "-12")
Set "answer_type" accordingly: "mcq_single", "mcq_multi", or "numerical"."""


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITER — Thread-safe token bucket (shared across workers)
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, max_rpm: int = 10):
        self.min_interval = 60.0 / max_rpm
        self.last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            wait = self.last_call + self.min_interval - now
            self.last_call = max(self.last_call, now) + self.min_interval
        if wait > 0:
            time.sleep(wait)


# ═══════════════════════════════════════════════════════════════════════════
# LLM CLIENT — Azure OpenAI or DeepSeek
# ═══════════════════════════════════════════════════════════════════════════

_client: Optional[OpenAI] = None
_client_type: str = ""  # "azure" or "deepseek"
_client_model: str = ""


def init_client(
    provider: str = "azure",
    azure_api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
):
    """Initialize the LLM client — Azure OpenAI or DeepSeek."""
    global _client, _client_type, _client_model

    provider = provider.lower().strip()

    if provider == "azure":
        key = azure_api_key or AZURE_API_KEY
        ep = azure_endpoint or AZURE_ENDPOINT
        if not key:
            raise ValueError("No Azure API key. Set AZURE_OPENAI_API_KEY or pass --azure-api-key.")
        if not ep:
            raise ValueError("No Azure endpoint. Set AZURE_OPENAI_ENDPOINT or pass --azure-endpoint.")
        _client = AzureOpenAI(
            api_key=key,
            azure_endpoint=ep,
            api_version=AZURE_API_VERSION,
        )
        _client_type = "azure"
        _client_model = AZURE_MODEL
        print(f"  ✅ Azure OpenAI client ready — model: {_client_model} | endpoint: {ep}")

    elif provider == "deepseek":
        key = deepseek_api_key or DEEPSEEK_API_KEY
        if not key:
            raise ValueError("No DeepSeek API key. Set DEEPSEEK_API_KEY or pass --deepseek-api-key.")
        _client = OpenAI(
            api_key=key,
            base_url=DEEPSEEK_BASE_URL,
        )
        _client_type = "deepseek"
        _client_model = DEEPSEEK_MODEL
        print(f"  ✅ DeepSeek client ready — model: {_client_model} | base_url: {DEEPSEEK_BASE_URL}")

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'azure' or 'deepseek'.")


def call_llm(
    messages: list[dict],
    rate_limiter: Optional[RateLimiter] = None,
) -> str:
    """Call the LLM (Azure or DeepSeek) with rate limiting. Returns raw response text."""
    global _client, _client_type, _client_model

    if rate_limiter:
        rate_limiter.acquire()

    if _client_type == "azure":
        response = _client.chat.completions.create(
            model=_client_model,
            messages=messages,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            reasoning_effort="medium",
            response_format={"type": "json_object"},
        )
    else:
        # DeepSeek
        response = _client.chat.completions.create(
            model=_client_model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )

    return response.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════
# SOLUTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_solution(
    question: dict,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Generate a step-by-step solution for one JEE/NEET question.

    Returns: Dict with keys: solution, subject, topic_tags, difficulty, requires_diagram, solution_language
             Returns None on failure after all retries.
    """
    q_no = question.get("question_no", "?")

    user_prompt = build_solution_prompt(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            raw = call_llm(messages, rate_limiter=rate_limiter)
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            parsed = json.loads(raw)

            missing = []
            if "solution" not in parsed or not parsed["solution"]:
                missing.append("solution")
            if "answer" not in parsed or (isinstance(parsed.get("answer"), str) and not parsed["answer"].strip()):
                missing.append("answer")
            if missing:
                print(f"  ⚠️  Q{q_no}: missing fields {missing}, retry {attempt+1}/{max_retries}")
                time.sleep(3)
                continue

            # Check solution isn't too short
            sol = parsed["solution"]
            if len(sol.split()) < 15:
                print(f"  ⚠️  Q{q_no}: solution too short ({len(sol.split())} words), retry {attempt+1}/{max_retries}")
                time.sleep(3)
                continue

            # Normalize answer field — strip whitespace, uppercase MCQ letters
            ans = str(parsed["answer"]).strip()
            atype = (parsed.get("answer_type") or "").strip().lower()
            if atype in ("mcq_single", "mcq_multi"):
                ans = "".join(sorted(set(c for c in ans.upper() if c in "ABCD")))
            parsed["answer"] = ans

            return parsed

        except json.JSONDecodeError as e:
            print(f"  🔄 Q{q_no}: JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(3)
        except Exception as e:
            print(f"  🔄 Q{q_no}: API error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(5 * (attempt + 1))

    print(f"  ❌ Q{q_no}: FAILED after {max_retries} attempts")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# WORKER — Processes one question (for parallel execution)
# ═══════════════════════════════════════════════════════════════════════════

def _process_one_question(
    question: dict,
    checkpoint_dir: Path,
    rate_limiter: RateLimiter,
    all_records: list,
    lock: threading.Lock,
    progress: list,
    total: int,
) -> None:
    """
    Worker: process one question → save checkpoint → append to shared list.
    Thread-safe for parallel execution.
    """
    q_id = _question_id(question)
    checkpoint_file = checkpoint_dir / f"{q_id}.json"

    # ── Resume: skip if already processed ──
    if checkpoint_file.exists():
        try:
            existing = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if existing.get("status") == "ok" and existing.get("record"):
                with lock:
                    all_records.append(existing["record"])
                    progress[0] += 1
                    done = progress[0]
                    print(f"  [{done}/{total}] Q{q_id} ↪ resumed from checkpoint")
                return
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupt checkpoint → regenerate

    # ── Generate solution ──
    solution_data = generate_solution(question, rate_limiter=rate_limiter)

    if solution_data:
        subject = solution_data.get("subject") or question.get("subject", "unknown")
        level = question.get("level", "JEE_Advanced")
        # IMPORTANT: discard any input `answer` / `answer_type` — use only LLM output.
        record = {
            "question":           question.get("question", ""),
            "options":            question.get("options", []),
            "answer":             solution_data.get("answer", ""),
            "answer_type":        solution_data.get("answer_type", "mcq_single"),
            "solution":           solution_data["solution"],
            "subject":            subject,
            "level":              level,
            "group":              f"{subject}_{level}",
            "chunk_id":           question.get("question_no", "?"),
            "source_chunks_file": question.get("source_file", ""),
        }

        # Save checkpoint
        try:
            checkpoint_file.write_text(
                json.dumps({"status": "ok", "record": record}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except IOError:
            pass

        with lock:
            all_records.append(record)
            progress[0] += 1
            done = progress[0]
            diff = solution_data.get("difficulty", "?")
            words = len(solution_data.get("solution", "").split())
            sol_lang = solution_data.get("solution_language", "?")
            print(f"  [{done}/{total}] {q_id} ✅ {words}w | diff={diff} | lang={sol_lang}")

    else:
        # Failed — save failure checkpoint to avoid retrying forever
        try:
            checkpoint_file.write_text(
                json.dumps({"status": "failed", "question_no": q_id}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except IOError:
            pass

        with lock:
            progress[0] += 1
            done = progress[0]
            print(f"  [{done}/{total}] {q_id} ❌ FAILED (checkpoint saved)")


def _question_id(question: dict) -> str:
    """Generate a unique, filesystem-safe ID for a question."""
    pdf = question.get("pdf_name", "unknown")
    qno = question.get("question_no", "?")
    safe_pdf = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(pdf).stem)[:60]
    return f"{safe_pdf}_Q{qno}"


def infer_subject(pdf_name: str) -> str:
    """Infer subject from PDF filename. Returns 'physics', 'chemistry', 'mathematics', 'biology', or 'unknown'."""
    name = pdf_name.lower()
    if "biology" in name:
        return "biology"
    if "chemistry" in name:
        return "chemistry"
    if "physics" in name:
        return "physics"
    if "math" in name:
        return "mathematics"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_solution_pipeline(
    input_jsonl: str,
    output_path: str,
    provider: str = "azure",
    workers: int = DEFAULT_WORKERS,
    max_rpm: int = DEFAULT_RPM,
    azure_api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Full pipeline: read train JSONL → generate solutions → output JSONL.

    Args:
        input_jsonl: Path to the training JSONL (questions with option-only answers)
        output_path: Where to save the output JSONL
        provider: "azure" or "deepseek"
        workers: Number of parallel workers
        max_rpm: Maximum API calls per minute
        azure_api_key: Azure OpenAI API key (env fallback)
        azure_endpoint: Azure OpenAI endpoint (env fallback)
        deepseek_api_key: DeepSeek API key (env fallback)
        limit: Only process first N questions (for testing)

    Returns:
        Path to the output JSONL
    """
    # ── Load questions ──
    input_path = Path(input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")

    questions = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                # Infer subject if missing
                if "subject" not in q or not q["subject"]:
                    q["subject"] = infer_subject(q.get("pdf_name", ""))
                questions.append(q)

    total_qs = len(questions)
    print(f"\n{'='*70}")
    print(f"SOLUTION GENERATION PIPELINE (LANGUAGE-MATCHED)")
    print(f"{'='*70}")
    print(f"  Input:       {input_path} ({total_qs} questions)")
    print(f"  Provider:    {provider}")
    print(f"  Workers:     {workers}")
    print(f"  Max RPM:     {max_rpm}")
    print(f"  Output:      {output_path}")
    print(f"{'='*70}\n")

    if limit and limit < total_qs:
        questions = questions[:limit]
        total_qs = len(questions)
        print(f"  ⚠️  LIMIT mode: only processing first {total_qs} questions\n")

    # ── Initialize client ──
    init_client(
        provider=provider,
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        deepseek_api_key=deepseek_api_key,
    )

    # ── Setup checkpoint directory ──
    output_path_obj = Path(output_path)
    checkpoint_dir = output_path_obj.parent / CHECKPOINT_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Count already-processed questions from checkpoints
    already_done = 0
    for q in questions:
        q_id = _question_id(q)
        ckpt = checkpoint_dir / f"{q_id}.json"
        if ckpt.exists():
            try:
                ckpt_data = json.loads(ckpt.read_text(encoding="utf-8"))
                if ckpt_data.get("status") == "ok":
                    already_done += 1
            except (json.JSONDecodeError, KeyError):
                pass

    if already_done:
        print(f"  📋 Resume: {already_done}/{total_qs} already solved (checkpoints found)\n")

    # ── Auto-cap workers ──
    workers = min(workers, total_qs)
    if workers < 1:
        workers = 1

    # ── Parallel processing ──
    all_records = []
    lock = threading.Lock()
    progress = [0]
    rate_limiter = RateLimiter(max_rpm=max_rpm)
    start_clock = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_one_question,
                q, checkpoint_dir, rate_limiter, all_records, lock, progress, total_qs,
            ): q
            for q in questions
        }

        for future in as_completed(futures):
            q = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  💥 FATAL worker error on Q{_question_id(q)}: {e}")

    # ── Sort by group then chunk_id for deterministic output ──
    all_records.sort(key=lambda r: (r["group"], r["chunk_id"]))

    # ── Write output ──
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Summary ──
    elapsed = time.time() - start_clock

    # Stats
    by_group = {}
    by_type = {}
    for r in all_records:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
        by_type[r["answer_type"]] = by_type.get(r["answer_type"], 0) + 1

    failed = total_qs - len(all_records)

    print(f"\n{'─'*50}")
    print(f"DONE ({elapsed:.0f}s)")
    print(f"  Generated:   {len(all_records)} / {total_qs}")
    print(f"  Failed:      {failed}")
    print(f"  By type:     {by_type}")
    print(f"  By group:")
    for g, n in sorted(by_group.items()):
        print(f"    {g:<30} {n}")
    print(f"  Output:      {output_path_obj}")
    print(f"{'─'*50}\n")

    # ── Save run summary ──
    summary_path = output_path_obj.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({
            "input": str(input_path),
            "output": str(output_path_obj),
            "provider": provider,
            "model": _client_model,
            "total_questions": total_qs,
            "solutions_generated": len(all_records),
            "failed": failed,
            "success_rate": round(len(all_records) / max(total_qs, 1) * 100, 1),
            "by_group": dict(sorted(by_group.items())),
            "by_type": dict(sorted(by_type.items())),
            "elapsed_seconds": round(elapsed, 0),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(output_path_obj)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GYANDEEP Solution Generator (Language-Matched) — JEE/NEET step-by-step solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Azure OpenAI (default)
  python src/solution_generator_lang.py stage3_jee_advanced.jsonl -o solutions_lang.jsonl

  # DeepSeek
  python src/solution_generator_lang.py stage3_jee_advanced.jsonl -o solutions_lang.jsonl --provider deepseek

  # Test with 10 questions
  python src/solution_generator_lang.py stage3_jee_advanced.jsonl -o test.jsonl --limit 10

  # Custom RPM for rate limiting
  python src/solution_generator_lang.py stage3_jee_advanced.jsonl -o solutions_lang.jsonl --rpm 5 --workers 3
        """,
    )

    parser.add_argument("input", help="Path to train JSONL (questions with option-only answers)")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument(
        "--provider", "-p",
        choices=["azure", "deepseek"],
        default="azure",
        help="LLM provider: azure (gpt-5.4-mini) or deepseek (deepseek-chat)",
    )
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM,
                        help=f"Max API calls per minute (default: {DEFAULT_RPM})")

    # Azure-specific
    group_azure = parser.add_argument_group("Azure OpenAI")
    group_azure.add_argument("--azure-api-key", help="Azure OpenAI API key (env: AZURE_OPENAI_API_KEY)")
    group_azure.add_argument("--azure-endpoint", help="Azure OpenAI endpoint (env: AZURE_OPENAI_ENDPOINT)")

    # DeepSeek-specific
    group_ds = parser.add_argument_group("DeepSeek")
    group_ds.add_argument("--deepseek-api-key", help="DeepSeek API key (env: DEEPSEEK_API_KEY)")

    # Misc
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N questions (for testing)")

    args = parser.parse_args()

    run_solution_pipeline(
        input_jsonl=args.input,
        output_path=args.output,
        provider=args.provider,
        workers=args.workers,
        max_rpm=args.rpm,
        azure_api_key=args.azure_api_key,
        azure_endpoint=args.azure_endpoint,
        deepseek_api_key=args.deepseek_api_key,
        limit=args.limit,
    )

"""
GYANDEEP Formatter — V2 (Simplified)
======================================
Input:  Raw QA pairs from qa_generator.py (merged JSON)
Output: ShareGPT-format JSONL (universal, works with Unsloth/Axolotl/TRL)

Does ONLY:
  1. Format validation — required fields, valid types
  2. Length check — answer 75-1500 words, question ≥ 8 words
  3. Convert to ShareGPT JSONL format

Dedup, benchmark hash, and distribution balancing happen in quality_gate.py
(Step 5, after all books are merged).

ShareGPT format:
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Optional


# ── Configuration ──────────────────────────────────────────────────────────

MIN_ANSWER_WORDS = 20
MAX_ANSWER_WORDS = 1500
MIN_QUESTION_WORDS = 8

GYANDEEP_SYSTEM_PROMPT = (
    "You are Gyandeep (ग्यानदीप), India's most expert Hindi-medium STEM teacher. "
    "You are like a patient, experienced professor at a Rajasthan government college "
    "who has been teaching for 25 years. Explain concepts in natural Hindi/Hinglish. "
    "Always give step-by-step reasoning with formulas, diagrams, and examples. "
    "Use English for technical terms with Hindi transliteration in parentheses on first use. "
    "Keep math notation in LaTeX. Structure your answers as: "
    "1) 'चलिए पहले समझते हैं कि प्रश्न क्या पूछ रहा है' "
    "2) 'हमें पता है कि...' "
    "3) 'अब step-by-step करते हैं:' "
    "4) 'तो हमारा final answer है...' "
    "5) 'संक्षेप में'"
)

REQUIRED_FIELDS = ["question", "answer", "type"]
VALID_TYPES = {
    "conceptual", "numerical", "derivation", "comparative",
    "application", "misconception", "multi_step", "true_false",
}
VALID_DIFFICULTIES = {"school", "bsc_early", "bsc_advanced", "msc"}


# ── Gate 1: Format Validation ─────────────────────────────────────────────

def gate_format(qa: dict) -> tuple[bool, str]:
    """Check if QA has all required fields with valid values."""
    for field in REQUIRED_FIELDS:
        if field not in qa or not qa[field]:
            return False, f"missing field: {field}"

    if not isinstance(qa["question"], str) or not isinstance(qa["answer"], str):
        return False, "question/answer must be strings"

    # Normalize type
    qa_type = qa.get("type", "").lower().strip()
    if qa_type not in VALID_TYPES:
        for vt in VALID_TYPES:
            if vt in qa_type or qa_type in vt:
                qa["type"] = vt
                break
        else:
            return False, f"invalid type: {qa_type}"

    # Normalize difficulty (don't reject, just fix)
    difficulty = qa.get("difficulty", "").lower().strip()
    if difficulty not in VALID_DIFFICULTIES:
        qa["difficulty"] = "bsc_early"  # Safe default

    return True, "ok"


# ── Gate 2: Length Check ──────────────────────────────────────────────────

def gate_length(qa: dict) -> tuple[bool, str]:
    """Check if answer and question meet length requirements."""
    q_words = len(qa["question"].split())
    a_words = len(qa["answer"].split())

    if q_words < MIN_QUESTION_WORDS:
        return False, f"question too short: {q_words} words (min {MIN_QUESTION_WORDS})"

    if a_words < MIN_ANSWER_WORDS:
        return False, f"answer too short: {a_words} words (min {MIN_ANSWER_WORDS})"

    if a_words > MAX_ANSWER_WORDS:
        return False, f"answer too long: {a_words} words (max {MAX_ANSWER_WORDS})"

    return True, "ok"


# ── ShareGPT Format Conversion ────────────────────────────────────────────

def to_sharegpt(qa: dict, include_system: bool = False) -> dict:
    """
    Convert a QA pair to ShareGPT format with metadata.

    ShareGPT format is the universal standard for LLM fine-tuning datasets.
    Unsloth, Axolotl, TRL, LLaMA-Factory all support it natively.
    Unsloth's get_chat_template() converts it to Gemma 4's chat template at training time.
    """
    conversations = []

    if include_system:
        conversations.append({"from": "system", "value": GYANDEEP_SYSTEM_PROMPT})

    conversations.append({"from": "human", "value": qa["question"]})
    conversations.append({"from": "gpt", "value": qa["answer"]})

    record = {
        "conversations": conversations,
        "metadata": {
            "source": qa.get("book_title", "unknown"),
            "source_chunk_id": qa.get("source_chunk_id", qa.get("chunk_id", "unknown")),
            "subject": qa.get("subject", "unknown"),
            "topic": qa.get("topic", "unknown"),
            "type": qa.get("type", "unknown"),
            "difficulty": qa.get("difficulty", "unknown"),
            "level": qa.get("level", "unknown"),
            "key_concepts": qa.get("key_concepts", []),
        },
    }

    return record


# ── Main Formatter ────────────────────────────────────────────────────────

def format_and_validate(
    qa_json_path: str,
    output_path: Optional[str] = None,
    include_system_prompt: bool = False,
) -> dict:
    """
    Validate QA pairs and convert to ShareGPT JSONL.

    Args:
        qa_json_path: Path to merged QA JSON from qa_generator.py
        output_path: Where to save the JSONL (default: same dir)
        include_system_prompt: Whether to include system prompt (Stage 2)

    Returns:
        Stats dict with counts
    """
    qa_path = Path(qa_json_path)
    with open(qa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", data if isinstance(data, list) else [])
    book_title = data.get("book_title", "unknown")

    if output_path is None:
        output_path = qa_path.with_suffix('.sharegpt.jsonl')
    else:
        output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"FORMATTING: {book_title}")
    print(f"Input QA pairs: {len(qa_pairs)}")
    print(f"{'='*60}\n")

    stats = {
        "total_input": len(qa_pairs),
        "passed": 0,
        "failed_format": 0,
        "failed_length": 0,
    }

    passed_qa = []
    rejected_log = []

    for i, qa in enumerate(qa_pairs):
        # Gate 1: Format validation
        ok, reason = gate_format(qa)
        if not ok:
            stats["failed_format"] += 1
            rejected_log.append({"index": i, "gate": "format", "reason": reason})
            continue

        # Gate 2: Length check
        ok, reason = gate_length(qa)
        if not ok:
            stats["failed_length"] += 1
            rejected_log.append({"index": i, "gate": "length", "reason": reason})
            continue

        passed_qa.append(qa)
        stats["passed"] += 1

    # Convert to ShareGPT and write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in passed_qa:
            record = to_sharegpt(qa, include_system=include_system_prompt)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save rejection log
    reject_log_path = output_path.with_suffix('.rejected.json')
    with open(reject_log_path, 'w', encoding='utf-8') as f:
        json.dump(rejected_log, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"{'─'*40}")
    print(f"FORMATTING COMPLETE")
    print(f"  Input:          {stats['total_input']}")
    print(f"  Passed:         {stats['passed']} ({stats['passed']/max(stats['total_input'],1)*100:.1f}%)")
    print(f"  Failed format:  {stats['failed_format']}")
    print(f"  Failed length:  {stats['failed_length']}")
    print(f"  Output: {output_path}")

    if passed_qa:
        type_dist = Counter(qa.get("type") for qa in passed_qa)
        diff_dist = Counter(qa.get("difficulty") for qa in passed_qa)
        subj_dist = Counter(qa.get("subject") for qa in passed_qa)
        print(f"  Types: {dict(type_dist)}")
        print(f"  Difficulty: {dict(diff_dist)}")
        print(f"  Subjects: {dict(subj_dist)}")

    print(f"{'─'*40}\n")

    stats["output_path"] = str(output_path)
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GYANDEEP Formatter — QA JSON → ShareGPT JSONL")
    parser.add_argument("qa_json", help="Path to merged QA JSON from qa_generator.py")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument("--include-system-prompt", action="store_true",
                        help="Include Gyandeep system prompt (for Stage 2 Hindi)")

    args = parser.parse_args()

    format_and_validate(args.qa_json, args.output, args.include_system_prompt)

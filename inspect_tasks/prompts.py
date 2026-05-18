"""
Prompt builders for JEE Hindi benchmarks.

System prompt is English (clearer instructions to model).
Model is instructed to solve in the SAME language as the question (Hindi).
A strict 'Final answer:' anchor format is required for reliable extraction.
"""

from typing import List, Dict


SYSTEM_PROMPT = (
    "You are an expert tutor for Indian engineering entrance exams (JEE Main and JEE Advanced).\n"
    "You are given a STEM problem in Hindi.\n"
    "\n"
    "Instructions:\n"
    "1. Solve the problem step-by-step. Write your reasoning in the SAME language as the question (Hindi).\n"
    "2. Keep the reasoning concise but complete — show key formulas, substitutions, and the final calculation.\n"
    "3. After the reasoning, output the final answer on its OWN LINE, in the exact format:\n"
    "       Final answer: <answer>\n"
    "\n"
    "Answer format rules (very important):\n"
    "  - single_correct  : exactly one uppercase letter from {A,B,C,D}.\n"
    "                      Example -> Final answer: B\n"
    "  - multi_correct   : the correct letters concatenated, sorted A->D, no spaces or punctuation.\n"
    "                      Example -> Final answer: AC\n"
    "  - numerical       : a plain number (integer or decimal). No units, no commas.\n"
    "                      Fractions are allowed only when the exact value cannot be a clean decimal,\n"
    "                      then write as 'a/b' with no spaces.\n"
    "                      Example -> Final answer: 9.5\n"
    "                      Example -> Final answer: 3/4\n"
    "\n"
    "Do not write anything after the 'Final answer:' line.\n"
)


def _format_options(options: List[Dict[str, str]]) -> str:
    if not options:
        return ""
    lines = []
    for o in options:
        label = (o.get("label") or "").strip()
        text = (o.get("text") or "").strip()
        lines.append(f"({label}) {text}")
    return "\n".join(lines)


def build_user_prompt(record: dict) -> str:
    """
    Assemble a user-turn prompt from a benchmark record.

    The record schema is the unified one produced by build_benchmark.py:
      { id, exam, year, paper, subject, question_type, language,
        question, options:[{label,text}], answer, has_diagram, ... }
    """
    qtype = record.get("question_type", "single_correct")
    subject = record.get("subject", "")
    question = (record.get("question") or "").strip()
    options = record.get("options") or []

    header = f"Subject: {subject}\nQuestion type: {qtype}\n"

    body = f"Question (Hindi):\n{question}\n"

    if qtype != "numerical" and options:
        body += "\nOptions:\n" + _format_options(options) + "\n"

    if qtype == "single_correct":
        body += "\nPick exactly ONE option. Respond with one capital letter (A/B/C/D)."
    elif qtype == "multi_correct":
        body += "\nOne or more options are correct. Respond with sorted concatenated letters (e.g. 'AC')."
    elif qtype == "numerical":
        body += "\nThe answer is a number. Respond with a plain number (or 'a/b' fraction only if needed)."

    body += "\n\nRemember: end your response with a line of the exact form 'Final answer: <answer>'.\n"

    return header + "\n" + body

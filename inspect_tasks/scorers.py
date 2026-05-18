"""
Scorer for JEE Hindi benchmarks. Dispatches on metadata.question_type.

Three modes:
  - single_correct : exact single-letter match, after 'Final answer:' anchor.
  - multi_correct  : sorted-letter-set equality.
  - numerical      : 1% relative tolerance; supports 'a/b' fraction strings.
"""

import re
from typing import Tuple

from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    grouped,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState


# ── Extraction ────────────────────────────────────────────────────────────

_FINAL_RE = re.compile(r"Final\s+answer\s*[:\-=]\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FALLBACK_RE = re.compile(r"(?:answer\s+is|\\boxed\{)\s*([^\n}]+)", re.IGNORECASE)

_LETTER_RE = re.compile(r"[A-D]")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")
_FRAC_RE = re.compile(r"(-?\d+)\s*/\s*(\d+)")

# LaTeX preprocessing — convert common math wrappers into plain forms the
# downstream extractors already understand.
_LATEX_FRAC_RE = re.compile(r"\\d?frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}")
_LATEX_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_LATEX_TEXT_RE = re.compile(r"\\text(?:bf|it|rm|sf|tt)?\s*\{([^{}]*)\}")
_LATEX_CMDS_RE = re.compile(r"\\(?:left|right|displaystyle|mathrm|operatorname|cdot|times|approx|pm|mp|frac|boxed|sqrt|tag|quad|qquad|;|:|,|!|\.)\b")


def _latex_normalize(span: str) -> str:
    """Flatten common LaTeX so plain number/fraction regex can score it."""
    if not span:
        return ""
    s = span
    # Unwrap \boxed{X}, \text{X}, repeatedly (nested-light).
    for _ in range(3):
        new = _LATEX_BOXED_RE.sub(r"\1", s)
        new = _LATEX_TEXT_RE.sub(r"\1", new)
        if new == s:
            break
        s = new
    # \frac{a}{b} -> a/b   (also \dfrac, \tfrac)
    s = _LATEX_FRAC_RE.sub(r"\1/\2", s)
    # Strip leftover bare LaTeX commands that confuse number-grabbing.
    s = _LATEX_CMDS_RE.sub(" ", s)
    # Strip $, $$, \(, \), \[, \], stray braces.
    s = re.sub(r"\${1,2}", " ", s)
    s = re.sub(r"\\[\(\)\[\]]", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    return s


def extract_final_span(text: str) -> str:
    """Return text after the LAST 'Final answer:' anchor; fallback to tail."""
    if not text:
        return ""
    matches = list(_FINAL_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    m = _FALLBACK_RE.search(text)
    if m:
        return m.group(1).strip()
    # last-resort: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _to_float(s: str):
    """Parse number, allowing 'a/b' fractions. Returns float or None."""
    if not s:
        return None
    s = s.strip().rstrip(".")
    fm = _FRAC_RE.fullmatch(s)
    if fm:
        num, den = float(fm.group(1)), float(fm.group(2))
        return num / den if den != 0 else None
    try:
        return float(s)
    except ValueError:
        return None


def extract_letters(span: str) -> str:
    """Return sorted unique uppercase letters in [A-D] found in span."""
    letters = set(_LETTER_RE.findall((span or "").upper()))
    return "".join(sorted(letters))


def extract_number(span: str):
    """
    Try fraction first (LaTeX or plain), then last number in normalized span.
    Returns (parsed_float, raw_str) or (None, "").
    """
    if not span:
        return None, ""
    norm = _latex_normalize(span)
    # Prefer the LAST fraction in the normalized span (matches "final answer"
    # written at the tail when reasoning earlier mentioned other fractions).
    fracs = list(_FRAC_RE.finditer(norm))
    if fracs:
        fm = fracs[-1]
        raw = f"{fm.group(1)}/{fm.group(2)}"
        return _to_float(raw), raw
    nums = _NUM_RE.findall(norm)
    if not nums:
        return None, ""
    raw = nums[-1]
    return _to_float(raw), raw


# ── Match logic per type ──────────────────────────────────────────────────

def _match_single(pred_text: str, gold: str) -> Tuple[bool, str, str]:
    span = extract_final_span(pred_text)
    letters = extract_letters(span)
    pick = letters[0] if len(letters) == 1 else letters
    ok = len(letters) == 1 and letters == gold.strip().upper()
    return ok, pick, f"pred_letters={pick!r} gold={gold!r}"


def _match_multi(pred_text: str, gold: str) -> Tuple[bool, str, str]:
    span = extract_final_span(pred_text)
    pred = extract_letters(span)
    target = "".join(sorted(set(gold.strip().upper())))
    ok = pred == target and bool(pred)
    return ok, pred, f"pred_set={pred!r} gold_set={target!r}"


def _match_numeric(pred_text: str, gold: str, rel_tol: float = 0.01) -> Tuple[bool, str, str]:
    span = extract_final_span(pred_text)
    pred_val, pred_raw = extract_number(span)
    gold_val = _to_float(gold.strip())
    if pred_val is None or gold_val is None:
        return False, pred_raw, f"parse_fail pred={pred_raw!r} gold={gold!r}"
    ok = abs(pred_val - gold_val) <= rel_tol * max(abs(gold_val), 1.0)
    return ok, pred_raw, f"pred={pred_val} gold={gold_val} tol={rel_tol}"


# ── Inspect-ai scorer ─────────────────────────────────────────────────────

@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(accuracy(), "subject"),
        grouped(accuracy(), "question_type"),
    ]
)
def jee_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        qtype = (state.metadata or {}).get("question_type", "single_correct")
        pred_text = state.output.completion if state.output else ""
        gold = target.text if target else ""

        if qtype == "single_correct":
            ok, answer, expl = _match_single(pred_text, gold)
        elif qtype == "multi_correct":
            ok, answer, expl = _match_multi(pred_text, gold)
        elif qtype == "numerical":
            ok, answer, expl = _match_numeric(pred_text, gold)
        else:
            ok, answer, expl = False, "", f"unknown qtype={qtype!r}"

        return Score(
            value=1.0 if ok else 0.0,
            answer=answer,
            explanation=expl,
        )

    return score

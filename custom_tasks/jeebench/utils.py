"""
JEEBench utilities for lm-evaluation-harness.

Place this file in the same directory as jeebench.yaml (the `--include-path`
dir you pass to lm-eval / run_eval.py). YAML references `jeebench_utils.X`.
"""

import re


# ── Prompt construction ───────────────────────────────────────────────────

def doc_to_text(doc):
    """JEEBench prompt with explicit `Final answer:` anchor for parsing."""
    q = doc["question"]
    t = doc.get("type", "")
    if t == "MCQ":
        suffix = (
            "Answer with a single letter (A, B, C, or D). "
            "End your response with: Final answer: <letter>"
        )
    elif t == "MCQ(multiple)":
        suffix = (
            "Multiple letters from A, B, C, D may be correct. "
            "End your response with: Final answer: <letters concatenated, e.g. AC>"
        )
    elif t in ("Integer", "Numeric"):
        suffix = "End your response with: Final answer: <number>"
    else:
        suffix = "End your response with: Final answer: <answer>"
    return f"{q}\n\n{suffix}\n"


def doc_to_target(doc):
    """Pass gold through as-is. Scoring fn does per-type normalization."""
    return str(doc.get("gold", "")).strip()


# ── Answer extraction helpers ─────────────────────────────────────────────

_LETTERS_RE = re.compile(r"[A-Da-d]")
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+\-]?\d+)?")


def _normalize_letter_set(s):
    """Normalize a string of MCQ letters → sorted unique uppercase ('AB', 'C', ...)."""
    if not s:
        return ""
    return "".join(sorted(set(c.upper() for c in _LETTERS_RE.findall(s))))


def _extract_number(s):
    """Pull a number from response. Prefer `Final answer:` anchor, else last number."""
    if not s:
        return None
    m = re.search(r"final\s+answer\s*[:=\s]\s*(-?\d+\.?\d*)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = _NUM_RE.findall(s)
    return nums[-1] if nums else None


# ── Scoring ──────────────────────────────────────────────────────────────

def _score_one(gold, pred):
    """Score a single (gold, pred) pair. Returns 1.0 or 0.0."""
    gold = str(gold).strip() if gold is not None else ""
    pred = str(pred).strip() if pred is not None else ""

    if not gold:
        return 0.0

    # Numeric gold?
    try:
        g_val = float(gold)
        p_str = _extract_number(pred)
        if p_str is None:
            return 0.0
        try:
            p_val = float(p_str)
        except ValueError:
            return 0.0
        tol = 0.01 * max(abs(g_val), 1.0)
        return 1.0 if abs(p_val - g_val) < tol else 0.0
    except ValueError:
        pass

    # Letter-set gold (MCQ or MCQ-multi)
    gold_set = _normalize_letter_set(gold)
    pred_set = _normalize_letter_set(pred)
    if not gold_set:
        return 0.0
    return 1.0 if pred_set == gold_set else 0.0


def jee_score(items):
    """
    lm-eval custom metric. lm-eval passes either a list of (gold, pred) tuples
    OR a single (gold, pred) per call (per-sample aggregation=mean).
    Handle both signatures defensively.
    """
    # Per-sample call: items is a (gold, pred) tuple/list
    if isinstance(items, (tuple, list)) and len(items) == 2 and not isinstance(items[0], (list, tuple)):
        return _score_one(items[0], items[1])

    # Batch call: list of (gold, pred) pairs
    if not items:
        return 0.0
    scores = [_score_one(g, p) for g, p in items]
    return sum(scores) / len(scores)

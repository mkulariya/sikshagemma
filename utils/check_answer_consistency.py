"""
Sanity-check: does record.answer match the final answer mentioned in record.solution?

For each row in solutions JSONL:
  - extract candidate final answer from solution text (heuristics: tail letters/numbers,
    "अंतिम उत्तर" / "final answer" / "उत्तर है" / "answer is" / \\boxed{} anchors)
  - compare against record.answer using type-aware match (mcq letters set / numeric tolerance)
  - report mismatches

Usage:
  python3 src/check_answer_consistency.py
  python3 src/check_answer_consistency.py --input <path>
"""

import argparse
import json
import re
from pathlib import Path


# ── Anchors that often precede the final answer in solutions ──────────────
ANCHOR_PATTERNS = [
    # English
    r"final\s+answer\s*[:\-=]\s*([^\n]+)",
    r"the\s+answer\s+is\s*[:\-=]?\s*([^\n]+)",
    r"answer\s*[:\-=]\s*([^\n]+)",
    # Hindi
    r"अंतिम\s+उत्तर\s*[:\-=]\s*([^\n]+)",
    r"उत्तर\s+है\s*[:\-=]?\s*([^\n]+)",
    r"सही\s+विकल्प\s*[:\-=]?\s*([^\n]+)",
    r"उत्तर\s*[:\-=]\s*([^\n]+)",
    # LaTeX boxed
    r"\\boxed\s*\{([^}]+)\}",
]


def extract_from_solution(sol: str) -> str:
    """Pull a candidate final-answer fragment from solution text."""
    if not sol:
        return ""
    for pat in ANCHOR_PATTERNS:
        m = re.search(pat, sol, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: last non-empty line, stripped
    lines = [ln.strip() for ln in sol.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ── Normalization helpers ────────────────────────────────────────────────

_LETTER_RE = re.compile(r"[A-Da-d]")
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+\-]?\d+)?")


def normalize_letter_set(s: str) -> str:
    if not s:
        return ""
    return "".join(sorted(set(c.upper() for c in _LETTER_RE.findall(s))))


def extract_number(s: str):
    if not s:
        return None
    nums = _NUM_RE.findall(s)
    return nums[-1] if nums else None


NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}
_OPT_PAT = re.compile(r"(?:option|विकल्प)\s*[\(\[\{]?\s*([1-4])|[\(\[\{]\s*([1-4])\s*[\)\]\}]")


_FINAL_LINE_ANCHORS = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer\s+is|"
    r"अंतिम\s+उत्तर|उत्तर\s+है|सही\s+उत्तर|सही\s+विकल्प|"
    r"correct\s+(?:options?|answers?))\s*[:\-=]?\s*(.{0,150})",
    re.IGNORECASE,
)


def extract_option_letters_from_text(text: str) -> str:
    """
    Pick the FINAL conclusion in the solution and extract option letters from it.
    Priority:
      1. Numbered option refs like '(1)', 'विकल्प (2)' in last conclusion line.
      2. Bare A/B/C/D letters, only inside an anchor span (after 'answer is' / 'उत्तर है').
    Avoids picking up stray chemistry letters like 'B' in $B_2$.
    """
    if not text:
        return ""
    tail = text[-600:] if len(text) > 600 else text

    # Try anchor-bounded extraction first (most reliable)
    anchor_spans = [m.group(1) for m in _FINAL_LINE_ANCHORS.finditer(tail)]
    search_space = " ".join(anchor_spans) if anchor_spans else tail

    letters = set()

    # Option-number refs → letter
    for m in _OPT_PAT.finditer(search_space):
        n = m.group(1) or m.group(2)
        if n in NUM_TO_LETTER:
            letters.add(NUM_TO_LETTER[n])

    # Bare letter refs — only if anchor span produced no number refs,
    # to avoid raw chemistry letters contaminating multi.
    if not letters and anchor_spans:
        for span in anchor_spans:
            for ch in re.findall(r"\b([A-D])\b", span):
                letters.add(ch.upper())

    return "".join(sorted(letters))


_DOLLAR_NUM = re.compile(r"\$[^$]*?(-?\d+\.?\d*)[^$]*?\$")
_AFTER_EQ = re.compile(r"[=≈]\s*(-?\d+\.?\d*)")


def extract_numeric_candidates(text: str):
    """
    Return list of candidate numeric strings from solution tail.
    Priority: inside $...$ math, after = or ≈, then any in the tail.
    """
    if not text:
        return []
    tail = text[-600:] if len(text) > 600 else text
    cands = []
    cands += [m.group(1) for m in _DOLLAR_NUM.finditer(tail)]
    cands += [m.group(1) for m in _AFTER_EQ.finditer(tail)]
    cands += _NUM_RE.findall(tail)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def matches(answer: str, solution: str, answer_type: str) -> tuple:
    """
    Compare gold record.answer vs whatever the solution concludes.
    Returns (is_match, debug_str).
    Scans entire solution (not just one extracted candidate) for the conclusion.
    """
    if not answer or not solution:
        return False, "empty"
    answer = str(answer).strip()

    if answer_type in ("mcq_single", "mcq_multi"):
        a_set = normalize_letter_set(answer)
        c_set = extract_option_letters_from_text(solution)
        if answer_type == "mcq_single":
            ok = bool(a_set) and a_set in c_set and len(a_set) == 1
        else:
            ok = a_set == c_set
        return ok, f"answer={a_set!r} extracted_letters={c_set!r}"

    if answer_type == "numerical":
        cands = extract_numeric_candidates(solution)
        try:
            a = float(answer)
        except ValueError:
            return answer in cands, f"answer={answer!r} cands(top)={cands[:5]}"
        for c in cands:
            try:
                cf = float(c)
            except ValueError:
                continue
            if abs(cf - a) < 0.01 * max(abs(a), 1.0):
                return True, f"answer={a} matched candidate={cf}"
        return False, f"answer={a} no candidate within tol; cands(top)={cands[:5]}"

    return answer.lower() in solution.lower(), "fallback substring"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/jee_advanced-2025/jee_advanced_2025_solutions.jsonl")
    ap.add_argument("--show-mismatches", type=int, default=20,
                    help="How many mismatches to print in detail")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(rows)} rows from {args.input}\n")

    stats = {"match": 0, "mismatch": 0, "missing_answer": 0, "missing_solution": 0}
    by_type_match = {}
    by_type_total = {}
    mismatches = []

    for r in rows:
        ans = (r.get("answer") or "").strip()
        sol = r.get("solution") or ""
        atype = (r.get("answer_type") or "").strip().lower()
        chunk_id = r.get("chunk_id", "?")
        subj = r.get("subject", "?")

        by_type_total[atype] = by_type_total.get(atype, 0) + 1

        if not ans:
            stats["missing_answer"] += 1
            mismatches.append((chunk_id, subj, atype, ans, "<no answer>", ""))
            continue
        if not sol:
            stats["missing_solution"] += 1
            continue

        ok, debug = matches(ans, sol, atype)
        if ok:
            stats["match"] += 1
            by_type_match[atype] = by_type_match.get(atype, 0) + 1
        else:
            stats["mismatch"] += 1
            mismatches.append((chunk_id, subj, atype, ans, debug, sol[-300:]))

    print("=" * 70)
    print("CONSISTENCY CHECK")
    print("=" * 70)
    print(f"  total       : {len(rows)}")
    print(f"  match       : {stats['match']}")
    print(f"  mismatch    : {stats['mismatch']}")
    print(f"  no answer   : {stats['missing_answer']}")
    print(f"  no solution : {stats['missing_solution']}")
    print()
    print("Per-type match rate:")
    for t in sorted(by_type_total):
        n = by_type_total[t]
        m = by_type_match.get(t, 0)
        pct = 100.0 * m / max(n, 1)
        print(f"  {t:14s}  {m:3d}/{n:3d}  ({pct:5.1f}%)")

    if mismatches:
        print(f"\n=== Showing first {min(args.show_mismatches, len(mismatches))} mismatches ===\n")
        for chunk_id, subj, atype, ans, candidate, sol_tail in mismatches[: args.show_mismatches]:
            print(f"[{chunk_id}] subj={subj} type={atype}")
            print(f"   record.answer : {ans!r}")
            print(f"   extracted     : {candidate!r}")
            if sol_tail:
                print(f"   solution tail : ...{sol_tail.strip()[-280:]}")
            print()


if __name__ == "__main__":
    main()

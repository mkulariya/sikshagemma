"""
Convert JEE Advanced parsed JSON (per-paper dict) → Mains-compatible JSONL
for use with solution_generator_lang.py.

Inputs (defaults):
  jee_papers/jee_advanced_parsed/jee_advanced_2025_paper_1_clean.json
  jee_papers/jee_advanced_parsed/jee_advanced_2025_paper_2_clean.json

Output (default):
  data/jee_advanced/jee_advanced_2025_clean.jsonl  (combined, both papers)

Per-line schema (flat, Mains-compatible):
{
  "question":      "<question_text>",
  "question_no":   "<question_number prefixed with paper-id for uniqueness>",
  "options":       ["text1","text2",...],        # flattened from [{label,text}]
  "subject":       "physics|chemistry|mathematics",   # lowercased
  "question_type": "single_correct|multi_correct|numerical",
  "section":       "A" | "",                     # mapped from question_type
  "language":      "Hindi",
  "diagram_paths": [],
  "source_file":   "jee_advanced_2025_paper_<n>.json",
  "paper":         1 | 2,
  "year":          2025,
  "exam":          "jee_advanced",
  "answer":        ""                            # intentionally empty — LLM fills it
}
"""

import argparse
import json
from pathlib import Path


# JEE Advanced question_type → section letter (cosmetic, mirrors Mains convention)
TYPE_TO_SECTION = {
    "single_correct": "A",
    "multi_correct": "A",
    "numerical": "B",
}


def convert_paper(input_path: Path, paper_num: int, year: int) -> list:
    """Convert one paper JSON → list of flat per-question dicts."""
    d = json.loads(input_path.read_text(encoding="utf-8"))
    paper_language = d.get("language", "Hindi")
    out = []

    for q in d.get("questions", []):
        # Flatten options: [{label,text}] → [text]. Preserve original order.
        opts_in = q.get("options", []) or []
        opts_flat = [(o.get("text") or "").strip() for o in opts_in]

        q_num = q.get("question_number", "?")
        qtype = q.get("question_type", "single_correct")

        # Subject abbreviation for unique ID. JEE Advanced numbers questions
        # 1..16 per subject per paper, so question_number alone is NOT unique.
        subj_raw = (q.get("subject") or "unknown").lower()
        subj_abbr = {
            "mathematics": "math",
            "physics":     "phys",
            "chemistry":   "chem",
            "biology":     "bio",
        }.get(subj_raw, subj_raw[:4])

        flat = {
            "question":      q.get("question_text", ""),
            "question_no":   f"p{paper_num}_{subj_abbr}_q{q_num}",
            "options":       opts_flat,
            "subject":       (q.get("subject") or "unknown").lower(),
            "question_type": qtype,
            "section":       TYPE_TO_SECTION.get(qtype, "A"),
            "language":      paper_language,
            "diagram_paths": [],
            "source_file":   input_path.name,
            "paper":         paper_num,
            "year":          year,
            "exam":          "jee_advanced",
            "answer":        "",
        }
        out.append(flat)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper1", default="jee_papers/jee_advanced_parsed/jee_advanced_2025_paper_1_clean.json")
    ap.add_argument("--paper2", default="jee_papers/jee_advanced_parsed/jee_advanced_2025_paper_2_clean.json")
    ap.add_argument("--output", default="data/jee_advanced/jee_advanced_2025_clean.jsonl")
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for paper_num, in_p in [(1, args.paper1), (2, args.paper2)]:
        p = Path(in_p)
        if not p.exists():
            print(f"  ⚠️  missing: {p}")
            continue
        rows = convert_paper(p, paper_num=paper_num, year=args.year)
        all_rows.extend(rows)
        print(f"  paper {paper_num}: {len(rows):3d} questions  ← {p.name}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Sanity stats
    from collections import Counter
    by_subj = Counter(r["subject"] for r in all_rows)
    by_type = Counter(r["question_type"] for r in all_rows)
    by_paper = Counter(r["paper"] for r in all_rows)

    print(f"\n✅ Wrote {len(all_rows)} rows → {out_path}")
    print(f"   subjects : {dict(by_subj)}")
    print(f"   types    : {dict(by_type)}")
    print(f"   papers   : {dict(by_paper)}")


if __name__ == "__main__":
    main()

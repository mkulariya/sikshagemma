#!/usr/bin/env python3
"""Extract answers from solutions for empty-answer records."""
import json
import re
import sys

NUM_TO_LET = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

def convert_option(x):
    """Convert 1-4 or A-D to A-D."""
    x = x.strip().upper()
    if x in NUM_TO_LET:
        return NUM_TO_LET[x]
    if x in ('A', 'B', 'C', 'D'):
        return x
    return None


def frac_to_slash(val):
    """Convert \\frac{a}{b} → a/b, including negatives."""
    m = re.match(r'^(-?)\\frac\{([^}]+)\}\{([^}]+)\}$', val.strip())
    if m:
        sign = m.group(1)
        num = m.group(2).strip()
        den = m.group(3).strip()
        if re.match(r'^[\d.]+$', num) and re.match(r'^[\d.]+$', den):
            return f"{sign}{num}/{den}"
    return val


def clean_numerical_boxed(val):
    """Strip units and variable prefixes from a \\boxed{} content."""
    val = val.strip()
    # Strip LaTeX thin/thick spaces
    val = re.sub(r'\\[,;:! ]', '', val)
    val = re.sub(r'\\ +', ' ', val)
    # Strip trailing units: \text{...}, \mathrm{...}, \rm{...}
    val = re.sub(r'[\s,\\]*\\(?:text|mathrm|rm|mbox|hbox)\{[^}]*\}', '', val)
    # Strip \approx N (approximation after main value, e.g. \approx 6.47)
    val = re.sub(r'\s*\\approx\s*[\d.]+', '', val)
    # Strip trailing plain LaTeX unit commands: \Omega, \text{Hz}, \text{m}, etc.
    val = re.sub(r'[\s\\,]*\\[A-Za-z]+\s*$', '', val)
    val = val.strip().rstrip(',').strip()
    # Try frac conversion
    val = frac_to_slash(val)
    # Strip variable= prefix: single letter/word followed by =
    m = re.match(r'^\\?[A-Za-z_][A-Za-z_0-9]*\s*=\s*(.+)$', val)
    if m:
        candidate = m.group(1).strip()
        candidate = frac_to_slash(candidate)
        # Accept only if no more = signs (not another equation)
        if '=' not in candidate:
            val = candidate
        else:
            return None  # equation, skip
    val = val.strip()
    return val


def is_valid_numerical(val):
    """Check if cleaned value is suitable to store as numerical answer."""
    if not val:
        return False
    # Must contain at least one digit
    if not re.search(r'\d', val):
        return False
    # Skip: has remaining = (equation)
    if '=' in val:
        return False
    # Skip: matching patterns
    if '\\to' in val or '\\rightarrow' in val:
        return False
    # Skip: matrix/environment
    if '\\begin' in val:
        return False
    # Skip: multi-item
    if '\\quad' in val or '\\ldots' in val or '\\cdots' in val:
        return False
    # Skip: mappings like I→(ii),Q
    if re.search(r'[IVX]+\s*\\to', val):
        return False
    # Skip: too long
    if len(val) > 60:
        return False
    # Skip: "cannot", "error", "no solution"
    low = val.lower()
    if any(w in low for w in ('cannot', 'error', 'no solution', 'undefined', 'does not')):
        return False
    return True


def extract_numerical(sol):
    """Extract numerical answer from solution text."""
    if not sol:
        return None

    # Check "cannot be determined" / incomplete question
    low = sol.lower()
    cannot_phrases = [
        'cannot be determined', 'cannot be solved', 'question has error',
        'full problem statement', 'share the full', 'missing reaction',
        'question text is required', 'please provide', 'if you share',
        'complete question', 'not enough information',
        'question is incomplete',
    ]
    if any(p in low for p in cannot_phrases):
        return None
    # "no solution" only when it's a standalone phrase near end, not "number of solutions is N"
    if 'no solution' in low and not re.search(r'(?:number|count)\s+of\s+(?:real\s+)?solutions', low):
        return None

    # Find all \boxed{...} in solution — use last one
    # Must handle nested braces carefully (simple greedy won't work for nested)
    # Use a stack-based approach
    boxed_vals = []
    for m in re.finditer(r'\\boxed\{', sol):
        start = m.end()
        depth = 1
        i = start
        while i < len(sol) and depth > 0:
            if sol[i] == '{':
                depth += 1
            elif sol[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            boxed_vals.append(sol[start:i-1])

    if boxed_vals:
        # Try from last to first
        for raw in reversed(boxed_vals):
            cleaned = clean_numerical_boxed(raw)
            if cleaned and is_valid_numerical(cleaned):
                return cleaned

    # Fallback: look in tail for text patterns
    tail = sol[-600:]

    # "Final answer: N" or "Final Answer:\n$$N$$" — allow $N$ wrapping
    for pat in [
        r'[Ff]inal\s+[Aa]nswer\s*[:\-]\s*\$\$?\s*(-?[\d\.]+)\s*\$?\$?',
        r'[Ff]inal\s+[Aa]nswer\s*[:\-]\s*(-?[\d]+\.?[\d]*)',
        r'\*\*[Ff]inal\s+[Aa]nswer\**\s*[:\-]\s*\$?\s*(-?[\d]+\.?[\d]*)\s*\$?',
        r'[Ff]inal\s+[Aa]nswer\s*:\s*\$(-?[\d]+\.?[\d]*)\$',
    ]:
        m = re.search(pat, tail)
        if m:
            return m.group(1)

    # Hindi: अंतिम उत्तर: N  (with optional $N$ wrapping)
    for pat in [
        r'अंतिम\s+उत्तर\s*[:\-]?\s*\$?\s*(-?[\d]+\.?[\d]*)\s*\$?',
        r'अंतिम\s+उत्तर\s*[:\-]?\s*(-?[\d]+\.?[\d]*)',
        r'उत्तर\s*[:=]\s*\$\s*(-?[\d]+\.?[\d]*)\s*\$',
    ]:
        m = re.search(pat, tail)
        if m:
            return m.group(1)

    # "comes out to be $N$" / "the value is $N$" / "solutions is $2$" — with optional $ wrapping
    m = re.search(
        r'(?:comes?\s+out\s+to\s+be|the\s+(?:required\s+)?value\s+is|the\s+answer\s+is|संख्या\s*=|solutions?\s+is|solutions?\s+are)\s*\$?\s*(-?\d+\.?\d*)\s*\$?',
        tail, re.IGNORECASE)
    if m:
        return m.group(1)

    # Last resort: near "answer" keyword, allow $N$
    end = sol[-300:]
    m = re.search(r'(?:answer|उत्तर|result)\D{0,30}?\$?\s*(-?\d+\.?\d*)\s*\$?', end, re.IGNORECASE)
    if m:
        val = m.group(1)
        if re.match(r'^-?\d+\.?\d*$', val):
            return val

    return None


def extract_mcq(sol):
    """Extract MCQ letter answer (A/B/C/D) from solution text."""
    if not sol:
        return None

    tail = sol[-800:]

    # 1. \boxed{(N)} or \boxed{N} alone where N is 1-4 or A-D
    # Also: \boxed{(N) something} — extract N from start of content
    boxed_vals = []
    for m in re.finditer(r'\\boxed\{', sol):
        start = m.end()
        depth = 1
        i = start
        while i < len(sol) and depth > 0:
            if sol[i] == '{':
                depth += 1
            elif sol[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            boxed_vals.append(sol[start:i-1])

    if boxed_vals:
        for raw in reversed(boxed_vals):
            stripped = raw.strip()
            # Remove full \text{...} wrapper: \text{(2)}
            stripped = re.sub(r'^\\text\{(.+)\}$', r'\1', stripped).strip()
            # Also handle \text{(N) ...} prefix — extract N from start of \text block
            m2 = re.match(r'^\\text\{\s*\(?([1-4A-Da-d])\)?', stripped)
            if m2:
                letter = convert_option(m2.group(1))
                if letter:
                    return letter
            # Match (N) alone or (N) followed by non-digit content
            m = re.match(r'^\(?([1-4A-Da-d])\)?\s*(?:[^0-9]|$)', stripped)
            if m:
                letter = convert_option(m.group(1))
                if letter:
                    return letter
            # Match single letter/number alone
            m = re.match(r'^([1-4A-Da-d])$', stripped)
            if m:
                letter = convert_option(m.group(1))
                if letter:
                    return letter

    # helper: N can be (N), N, $(N)$, $N$ — for option extraction
    _EN = r'(?:\$\s*)?\(?([1-4A-Da-d])\)?(?:\s*\$)?'

    # 2. Text patterns in tail (English)
    english_patterns = [
        # "the correct answer is (N)" / "correct answer is N" / "correct answer is option (N)"
        r'(?:correct|right)\s+(?:answer|option)\s+(?:is\s+)?(?:option\s+)?' + _EN,
        # "option (N) is correct" / "option N is correct"
        r'option\s+\(?([1-4A-Da-d])\)?\s+(?:is\s+)?(?:correct|right)',
        # "Hence, the answer is option (N)"
        r'(?:[Hh]ence|[Tt]herefore|[Tt]hus)[,.]?\s+(?:the\s+)?(?:correct\s+)?(?:answer|option)\s+(?:is\s+)?(?:option\s+)?' + _EN,
        # "So the correct answer is option (N)"
        r'[Ss]o\s+(?:the\s+)?correct\s+(?:answer|option)\s+(?:is\s+)?(?:option\s+)?' + _EN,
        # "Only option (N) matches" / "Only option (N) is correct"
        r'[Oo]nly\s+option\s+\(?([1-4A-Da-d])\)?\s+(?:matches|is)',
        # "corresponds to option (N)"
        r'corresponds\s+to\s+option\s+\(?([1-4A-Da-d])\)?',
        # "answer is **option (D)**"
        r'(?:answer|option)\s+(?:is\s+)?\*{1,2}\(?([1-4A-Da-d])\)?\*{1,2}',
        # "**option (N)**" near end
        r'\*{1,2}option\s+\(?([1-4A-Da-d])\)?\*{1,2}',
        # "the correct option is $(N)$"
        r'(?:correct|right)\s+(?:answer|option)\s+is\s+\$\(?([1-4A-Da-d])\)?\$',
    ]
    for pat in english_patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            letter = convert_option(m.group(1))
            if letter:
                return letter

    # 3. Hindi patterns  — \$? handles $(N)$ wrapping (LaTeX inline math in Hindi text)
    _N = r'(?:\$\s*)?\(?([1-4A-Da-d])\)?(?:\s*\$)?'  # matches (N), N, $(N)$, $N$
    hindi_patterns = [
        # "सही विकल्प है: विकल्प (C)"  or "सही उत्तर है: विकल्प (C)"
        r'(?:सही|correct)\s+(?:विकल्प|उत्तर)\s+(?:है)?(?:[:\s]+)?(?:विकल्प\s+)?' + _N,
        # "विकल्प (N)" near end with "सही" nearby
        r'(?:सही|correct).{0,50}विकल्प\s+' + _N,
        # "अतः सही उत्तर है: विकल्प (C)।"
        r'अतः\s+(?:सही|correct)\s+(?:विकल्प|उत्तर)\s+(?:है)?[:\s]*(?:विकल्प\s+)?' + _N,
        # "इसलिए सही विकल्प है **(N)**" or "$(N)$"
        r'इसलिए\s+सही\s+(?:विकल्प|उत्तर)\s+(?:है)?\s*\*{0,2}' + _N + r'\*{0,2}',
        # "विकल्प **(N)**" near "सही"
        r'विकल्प\s+\*{1,2}\(?\s*([1-4A-Da-d])\s*\)?\*{1,2}',
        # Matching: "यह **विकल्प (N)** है"
        r'यह\s+\*{1,2}विकल्प\s+\(?([1-4A-Da-d])\)?\*{1,2}',
        # "सही विकल्प है $(N)$" — explicit dollar-wrapped
        r'(?:सही|correct)\s+(?:विकल्प|उत्तर)\s+है\s+\$\(?([1-4A-Da-d])\)?\$',
        # "इसलिए सही विकल्प $(N)$ है"
        r'(?:इसलिए|अतः)\s+सही\s+विकल्प\s+\$\(?([1-4A-Da-d])\)?\$',
        # "सही विकल्प है **(N)**" or "सही विकल्प है **(N) ...**" — bold option
        r'(?:सही|correct)\s+(?:विकल्प|उत्तर)\s+(?:है)?\s*\*{1,2}\(?([1-4A-Da-d])\)?',
        # "यह विकल्प (N) है" / "विकल्पों में यह विकल्प (N) है"
        r'यह\s+विकल्प\s+\(?([1-4A-Da-d])\)?\s+है',
        # "**(N)** ... सही है" — bold option number at start of item
        r'\*{1,2}\(?([1-4A-Da-d])\)?\*{1,2}[^।.]{0,80}सही\s+है',
        # "N. ... — यही प्राप्त हुआ, इसलिए सही" — numbered list item marked correct
        r'([1-4])\.\s+\$[^।.]{0,120}?(?:यही\s+प्राप्त|यही\s+सही|सही\s+है|correct)',
    ]
    for pat in hindi_patterns:
        m = re.search(pat, tail)
        if m:
            letter = convert_option(m.group(1))
            if letter:
                return letter

    # 4. Broader fallback: "answer is **(N)**" or "answer: (N)" in last 500 chars
    m = re.search(r'(?:answer|option)[^.?!]{0,40}\(?([1-4A-Da-d])\)?\s*[\.\s]', tail[-500:], re.IGNORECASE)
    if m:
        letter = convert_option(m.group(1))
        if letter:
            return letter

    return None


def run_dry(path):
    """Dry run: show what would be extracted."""
    with open(path) as f:
        records = [json.loads(l) for l in f]

    empty = [r for r in records if not r.get('answer', '').strip()]
    print(f"Empty answer records: {len(empty)}")

    extracted = 0
    failed = []
    for r in empty:
        sol = r.get('solution', '')
        cid = r['chunk_id']
        atype = r['answer_type']
        if atype == 'numerical':
            ans = extract_numerical(sol)
        else:
            ans = extract_mcq(sol)

        if ans:
            extracted += 1
            if '--verbose' in sys.argv:
                print(f"OK  [{atype}] {cid}: {repr(ans)}")
        else:
            failed.append((cid, atype, sol[-200:]))

    print(f"\nExtracted: {extracted}/{len(empty)}")
    print(f"Failed: {len(failed)}/{len(empty)}")
    print("\n=== FAILED ===")
    for cid, atype, tail in failed:
        print(f"\n[{atype}] {cid}:")
        print(repr(tail[-150:]))


def apply(path, out_path):
    """Apply extraction and write updated file."""
    with open(path) as f:
        records = [json.loads(l) for l in f]

    updated = 0
    for r in records:
        if r.get('answer', '').strip():
            continue
        sol = r.get('solution', '')
        atype = r['answer_type']
        if atype == 'numerical':
            ans = extract_numerical(sol)
        else:
            ans = extract_mcq(sol)
        if ans:
            r['answer'] = ans
            updated += 1

    with open(out_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"Updated {updated} records. Written to {out_path}")


if __name__ == '__main__':
    path = '/home/m.kulariya/phi_agents/kaggle-hackthon-zapclaw/zapclaw_workspace/jee-advance-sol/azure/stage3_jee_solutions.jsonl'
    if '--apply' in sys.argv:
        apply(path, path)
    else:
        run_dry(path)

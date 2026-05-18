"""
GYANDEEP Stage 1 Training Dataset Builder
==========================================
Downloads, filters, converts, and merges all Stage 1 training datasets
into a single ShareGPT-format JSONL file ready for fine-tuning.

Datasets (with verified schemas):
  1. nvidia/OpenMathReasoning      (split=cot)     → 30K  math CoT
  2. open-r1/OpenR1-Math-220k      (default) → 18K  math reasoning
  3. nvidia/Nemotron-SFT-Math-v3             → 12K  math instruction
  4. open-thoughts/OpenThoughts3-1.2M        → 15K  science+math CoT
  5. nvidia/Nemotron-Science-v1    (MCQ/RQA) → 12K  graduate science
  6. derek-thomas/ScienceQA        (train)   → 10K  school science
  7. allenai/sciq                  (train)   → 12K  science MCQ
  8. KadamParth/Ncert_dataset      (train)   → 30K  NCERT foundation
  + synthetic_dataset_stage_1.jsonl           → all   synthetic textbook QA

Output: ShareGPT format JSONL
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

Usage:
    python build_stage1_dataset.py
    python build_stage1_dataset.py --output my_dataset.jsonl
    python build_stage1_dataset.py --skip D5 D8     # Skip specific datasets
    python build_stage1_dataset.py --synthetic-path /path/to/synthetic.jsonl
"""

import json
import os
import sys
import time
import random
import hashlib
import re
import argparse
import itertools
from pathlib import Path
from collections import Counter
from datetime import datetime

random.seed(42)


# ── Configuration ─────────────────────────────────────────────────────────

TARGETS = {
    "D1": 30000,   # OpenMathReasoning
    "D2": 18000,   # OpenR1-Math-220k
    "D3": 12000,   # Nemotron-SFT-Math-v3
    "D4": 15000,   # OpenThoughts3-1.2M
    "D5": 12000,   # Nemotron-Science-v1
    "D6": 10000,   # ScienceQA
    "D7": 12000,   # SciQ
    "D8": 30000,   # NCERT
}

D1_PROGRESS_EVERY = 10000
D3_PROGRESS_EVERY = 25000
D4_PROGRESS_EVERY = 5000
D5_PROGRESS_EVERY = 25000
MINHASH_THRESHOLD = 0.80
MINHASH_NUM_PERM = 128
MINHASH_NGRAM_SIZES = (2, 3)
MINHASH_CONTAINMENT_THRESHOLD = 0.90


# ── Helpers ───────────────────────────────────────────────────────────────

def word_count(text):
    return len(text.split()) if text else 0


def normalize_for_hash(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def normalize_for_semantic_dedup(text):
    """Normalize question text for near-duplicate detection."""
    text = str(text or "").lower().strip()
    text = re.sub(r'\s+', ' ', text)

    # Drop common instruction wrappers that vary across datasets.
    text = re.sub(r'^(please\s+)?(help me|solve|answer|analyze|read carefully|select the best answer).*?:\s*', '', text)
    text = re.sub(r'(the\s+)?final answer\s+is.*$', '', text)

    # Normalize MCQ option prefixes so reordered punctuation does not matter as much.
    text = re.sub(r'[\(\[]?\b[a-j]\b[\)\].:]\s*', ' ', text)
    text = re.sub(r'\boption\s+[a-j]\b', ' ', text)

    # Keep alphanumerics and basic separators only.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_semantic_shingles(text, n_sizes=MINHASH_NGRAM_SIZES):
    """Build token n-grams for MinHash/LSH dedup."""
    normalized = normalize_for_semantic_dedup(text)
    tokens = normalized.split()
    if not tokens:
        return set()

    shingles = set(tokens)
    for n in n_sizes:
        if len(tokens) < n:
            continue
        shingles.update(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return shingles


def jaccard_similarity(a, b):
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def containment_similarity(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def hash_question(text):
    return hashlib.sha256(normalize_for_hash(text).encode()).hexdigest()


def normalize_subject(subject):
    subject = str(subject or "").strip().lower()
    aliases = {
        "math": "mathematics",
        "maths": "mathematics",
        "mathematics": "mathematics",
        "bio": "biology",
        "chem": "chemistry",
        "phy": "physics",
    }
    return aliases.get(subject, subject)


def keep_ncert_subject(subject, grade):
    subject = normalize_subject(subject)
    try:
        grade = int(grade)
    except (TypeError, ValueError):
        return False

    if subject == "science" and grade in {6, 7, 8, 9, 10}:
        return True

    if subject == "mathematics" and grade in {6, 7, 8, 9, 10, 11, 12}:
        return True

    if subject in {"physics", "chemistry", "biology"} and grade in {11, 12}:
        return True

    return False


def to_sharegpt(question, answer, metadata=None):
    """Convert a Q/A pair to ShareGPT format."""
    record = {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
    }
    if metadata:
        record["metadata"] = metadata
    return record


def explore_schema(ds_or_items, name, n=2):
    """Print actual schema of a dataset for verification."""
    print(f"\n  Schema for {name}:")
    
    if hasattr(ds_or_items, 'column_names'):
        print(f"    Columns: {ds_or_items.column_names}")
        items = [ds_or_items[i] for i in range(min(n, len(ds_or_items)))]
    elif hasattr(ds_or_items, '__iter__'):
        items = list(itertools.islice(ds_or_items, n))
    else:
        print(f"    Cannot inspect schema")
        return
    
    for i, item in enumerate(items):
        print(f"    --- Sample {i+1} ---")
        for k, v in item.items():
            val_str = str(v)[:150]
            print(f"      {k}: {val_str}")


def save_filtered(data, path):
    """Save filtered data as JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} items → {path}")


def filtered_output_path(output_dir, ds_id, ds_name):
    filename = f"{ds_id}_{ds_name.lower().replace(' ', '_').replace('-', '_')}.jsonl"
    return output_dir / filename


def load_hf_dataset(path, *, split=None, name=None, streaming=False):
    """
    Thin wrapper around datasets.load_dataset without trust_remote_code.
    These builders are Hub-hosted tabular datasets, so remote code execution
    should not be required.
    """
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    if name is not None:
        kwargs["name"] = name
    return load_dataset(path, **kwargs)


# ── Global dedup tracker ──────────────────────────────────────────────────

class DedupTracker:
    """Track question hashes across all datasets to prevent duplicates."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.seen = set()
        self.semantic_available = False
        self.semantic_index = None
        self.semantic_signatures = {}
        self.semantic_shingles = {}

        if not self.enabled:
            print("  Dedup: disabled (--skip-dedup)")
            self._MinHash = None
            return

        try:
            from datasketch import MinHash, MinHashLSH
            self._MinHash = MinHash
            self.semantic_index = MinHashLSH(
                threshold=MINHASH_THRESHOLD,
                num_perm=MINHASH_NUM_PERM,
            )
            self.semantic_available = True
            print(
                f"  Semantic dedup: enabled (MinHashLSH, threshold={MINHASH_THRESHOLD}, "
                f"num_perm={MINHASH_NUM_PERM}, ngrams={MINHASH_NGRAM_SIZES})"
            )
        except Exception:
            self._MinHash = None
            print("  Semantic dedup: disabled (datasketch not installed, exact dedup only)")
    
    def is_duplicate(self, question_text):
        if not self.enabled:
            return False

        h = hash_question(question_text)
        if h in self.seen:
            return True

        if self.semantic_available:
            shingles = get_semantic_shingles(question_text)
            if shingles:
                minhash = self._MinHash(num_perm=MINHASH_NUM_PERM)
                for shingle in sorted(shingles):
                    minhash.update(shingle.encode("utf-8"))

                for candidate_hash in self.semantic_index.query(minhash):
                    candidate_shingles = self.semantic_shingles.get(candidate_hash)
                    if candidate_shingles:
                        jaccard = jaccard_similarity(shingles, candidate_shingles)
                        containment = containment_similarity(shingles, candidate_shingles)
                        if jaccard >= MINHASH_THRESHOLD or containment >= MINHASH_CONTAINMENT_THRESHOLD:
                            return True

                self.semantic_index.insert(h, minhash)
                self.semantic_signatures[h] = minhash
                self.semantic_shingles[h] = shingles

        self.seen.add(h)
        return False

    def add_existing(self, question_text):
        """Seed the dedup index with an already accepted question."""
        if not self.enabled or not question_text:
            return

        h = hash_question(question_text)
        if h in self.seen:
            return

        if self.semantic_available:
            shingles = get_semantic_shingles(question_text)
            if shingles:
                minhash = self._MinHash(num_perm=MINHASH_NUM_PERM)
                for shingle in sorted(shingles):
                    minhash.update(shingle.encode("utf-8"))
                self.semantic_index.insert(h, minhash)
                self.semantic_signatures[h] = minhash
                self.semantic_shingles[h] = shingles

        self.seen.add(h)
    
    @property
    def count(self):
        return len(self.seen)


# ── D1: nvidia/OpenMathReasoning ──────────────────────────────────────────

def load_d1_openmath(target, dedup, source_cap=None):
    """
    nvidia/OpenMathReasoning, split="cot", streaming
    Verified schema:
      problem, generated_solution, problem_type, expected_answer, problem_source
    Filter: problem_type == "has_answer_extracted", length gate, optional
    source cap, dedup
    """
    print(f"\n{'='*60}")
    print(f"[D1] nvidia/OpenMathReasoning (target: {target:,})")
    print(f"{'='*60}")

    # Current HF layout exposes cot/tir/genselect as splits under the default
    # builder. Older examples may refer to "cot" as a config, so keep a local
    # fallback for compatibility without requiring remote code.
    try:
        ds = load_hf_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
        reload_kwargs = {"split": "cot", "streaming": True}
    except Exception as e:
        print(f"  Split-based load failed ({e}); retrying legacy config layout")
        ds = load_hf_dataset("nvidia/OpenMathReasoning", name="cot", split="train", streaming=True)
        reload_kwargs = {"name": "cot", "split": "train", "streaming": True}
    
    # Verify schema
    sample = list(itertools.islice(ds, 2))
    print(f"  Columns: {list(sample[0].keys())}")
    
    # Re-create stream (consumed by sample)
    ds = load_hf_dataset("nvidia/OpenMathReasoning", **reload_kwargs)
    
    filtered = []
    scanned = 0
    source_counts = Counter()

    if source_cap is None:
        print("  Source diversity cap: disabled")
    else:
        print(f"  Source diversity cap: {source_cap} per problem_source")
    
    for item in ds:
        scanned += 1
        
        problem = item.get("problem", "")
        solution = item.get("generated_solution", "")
        problem_type = item.get("problem_type", "")
        source = item.get("problem_source", "unknown")
        
        if not problem or not solution:
            continue
        
        # Keep only verified-correct items
        if problem_type != "has_answer_extracted":
            continue
        
        # Length gate
        sol_words = word_count(solution)
        if sol_words < 50 or sol_words > 3000:
            continue
        
        # Source diversity cap
        if source_cap is not None and source_counts[source] >= source_cap:
            continue
        
        # Cross-dataset dedup
        if dedup.is_duplicate(problem):
            continue
        
        source_counts[source] += 1
        filtered.append(to_sharegpt(problem, solution, {"source": "openmath_reasoning", "problem_source": source}))
        
        if len(filtered) >= target:
            break
        
        if scanned % D1_PROGRESS_EVERY == 0:
            print(f"  Scanned {scanned:,}, kept {len(filtered):,}")
    
    print(f"  DONE: scanned {scanned:,}, kept {len(filtered):,}")
    print(f"  Top sources: {source_counts.most_common(5)}")
    return filtered


# ── D2: open-r1/OpenR1-Math-220k ─────────────────────────────────────────

def load_d2_openr1(target, dedup):
    """
    open-r1/OpenR1-Math-220k, config="default"
    Verified schema:
      problem, solution, answer, problem_type, question_type, source,
      correctness_count, correctness_math_verify, messages
    Filter: correctness_count >= 1, length gate, dedup against D1
    """
    print(f"\n{'='*60}")
    print(f"[D2] open-r1/OpenR1-Math-220k (target: {target:,})")
    print(f"{'='*60}")

    ds = load_hf_dataset("open-r1/OpenR1-Math-220k", name="default", split="train")
    
    print(f"  Total items: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    
    filtered = []
    
    for item in ds:
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        correctness_count = item.get("correctness_count", 0)
        
        if not problem or not solution:
            continue
        
        # Only keep verified-correct items
        if correctness_count < 1:
            continue
        
        # Length gate
        sol_words = word_count(solution)
        if sol_words < 50 or sol_words > 3000:
            continue
        
        # Cross-dataset dedup
        if dedup.is_duplicate(problem):
            continue
        
        filtered.append(to_sharegpt(problem, solution, {"source": "openr1_math"}))
    
    # If we have more than target, sample (bias toward harder = longer solutions)
    if len(filtered) > target:
        # Sort by solution length (longer = harder), take top 70% + random 30%
        filtered.sort(key=lambda x: word_count(x["conversations"][1]["value"]), reverse=True)
        hard_take = int(target * 0.7)
        easy_pool = filtered[hard_take:]
        easy_take = target - hard_take
        filtered = filtered[:hard_take] + random.sample(easy_pool, min(easy_take, len(easy_pool)))
        random.shuffle(filtered)
    
    print(f"  DONE: kept {len(filtered):,}")
    return filtered


# ── D3: nvidia/Nemotron-SFT-Math-v3 ──────────────────────────────────────

def load_d3_nemotron_math(target, dedup):
    """
    nvidia/Nemotron-SFT-Math-v3, streaming
    Verified schema:
      problem, messages, expected_answer, data_source, tool_usage, ...
    Filter: length gate, dedup against D1+D2
    """
    print(f"\n{'='*60}")
    print(f"[D3] nvidia/Nemotron-SFT-Math-v3 (target: {target:,})")
    print(f"{'='*60}")

    ds = load_hf_dataset("nvidia/Nemotron-SFT-Math-v3", split="train", streaming=True)
    
    # Verify schema
    sample = list(itertools.islice(ds, 2))
    print(f"  Columns: {list(sample[0].keys())}")
    
    ds = load_hf_dataset("nvidia/Nemotron-SFT-Math-v3", split="train", streaming=True)
    
    filtered = []
    scanned = 0
    empty_problem = 0
    missing_answer = 0
    short_answer = 0
    deduped = 0
    
    for item in ds:
        scanned += 1
        
        problem = item.get("problem", "")
        messages = item.get("messages", [])

        solution = ""
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).lower()
                content = msg.get("content", "")
                if role == "assistant" and content:
                    solution = content
        
        if not problem or not solution:
            if not problem:
                empty_problem += 1
            else:
                missing_answer += 1
            continue
        
        sol_words = word_count(solution)
        if sol_words < 50 or sol_words > 2500:
            short_answer += 1
            continue
        
        if dedup.is_duplicate(problem):
            deduped += 1
            continue
        
        metadata = {
            "source": "nemotron_sft_math_v3",
            "data_source": item.get("data_source"),
            "tool_usage": item.get("tool_usage"),
        }
        filtered.append(to_sharegpt(problem, solution, metadata))
        
        if len(filtered) >= target:
            break
        
        if scanned % D3_PROGRESS_EVERY == 0:
            print(f"  Scanned {scanned:,}, kept {len(filtered):,}")
    
    print(f"  DONE: scanned {scanned:,}, kept {len(filtered):,}")
    print(
        "  Rejections: "
        f"empty_problem={empty_problem:,}, "
        f"missing_answer={missing_answer:,}, "
        f"length_gate={short_answer:,}, "
        f"dedup={deduped:,}"
    )
    return filtered


# ── D4: open-thoughts/OpenThoughts3-1.2M ──────────────────────────────────

def load_d4_openthoughts(target, dedup):
    """
    open-thoughts/OpenThoughts3-1.2M, streaming
    Verified schema:
      difficulty (int), source (str), domain (str),
      conversations (list of {from, value})  ← ALREADY ShareGPT format
    Filter: skip code domain, keep science+math, length gate, dedup
    """
    print(f"\n{'='*60}")
    print(f"[D4] open-thoughts/OpenThoughts3-1.2M (target: {target:,})")
    print(f"{'='*60}")

    ds = load_hf_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)
    
    # Verify schema
    sample = list(itertools.islice(ds, 2))
    print(f"  Columns: {list(sample[0].keys())}")
    
    ds = load_hf_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)
    
    filtered = []
    scanned = 0
    domain_counts = Counter()
    skipped_code = 0
    missing_turns = 0
    length_gate = 0
    deduped = 0
    
    for item in ds:
        scanned += 1
        
        domain = item.get("domain", "").lower()
        conversations = item.get("conversations", [])
        
        # Skip code — we only want science and math
        if "code" in domain:
            skipped_code += 1
            continue
        
        # Must have human + gpt turn
        if len(conversations) < 2:
            missing_turns += 1
            continue
        
        # Extract question for dedup
        human_text = ""
        gpt_text = ""
        for turn in conversations:
            role = str(turn.get("from", "")).lower()
            if role in {"human", "user"} and not human_text:
                human_text = turn.get("value", "")
            if role in {"gpt", "assistant", "model"} and not gpt_text:
                gpt_text = turn.get("value", "")
        
        if not human_text or not gpt_text:
            missing_turns += 1
            continue
        
        # Length check on gpt response
        gpt_words = word_count(gpt_text)
        if gpt_words < 50 or gpt_words > 5000:
            length_gate += 1
            continue
        
        if dedup.is_duplicate(human_text):
            deduped += 1
            continue
        
        domain_counts[domain] += 1
        
        # Already ShareGPT format — keep as-is
        record = {"conversations": conversations, "metadata": {"source": "openthoughts3", "domain": domain}}
        filtered.append(record)
        
        if len(filtered) >= target:
            break
        
        if scanned % D4_PROGRESS_EVERY == 0:
            print(
                f"  Scanned {scanned:,}, kept {len(filtered):,}, "
                f"skipped_code={skipped_code:,}, "
                f"missing_turns={missing_turns:,}, "
                f"length_gate={length_gate:,}"
            )
    
    print(f"  DONE: scanned {scanned:,}, kept {len(filtered):,}")
    print(f"  Domains: {domain_counts.most_common(10)}")
    print(
        "  Rejections: "
        f"code={skipped_code:,}, "
        f"missing_turns={missing_turns:,}, "
        f"length_gate={length_gate:,}, "
        f"dedup={deduped:,}"
    )
    return filtered


# ── D5: nvidia/Nemotron-Science-v1 ────────────────────────────────────────

def load_d5_nemotron_science(target, dedup):
    """
    nvidia/Nemotron-Science-v1, split="MCQ" (174K GPQA-style) + split="RQA" (52K chemistry)
    Schema: messages-based JSONL
    Filter: length gate, dedup
    """
    print(f"\n{'='*60}")
    print(f"[D5] nvidia/Nemotron-Science-v1 (target: {target:,})")
    print(f"{'='*60}")
    
    filtered = []
    
    split_aliases = {
        "MCQ": ["MCQ"],
        "RQA": ["RQA"],
    }

    for split_name in ["MCQ", "RQA"]:
        per_split_target = target // 2 if split_name == "MCQ" else target - len(filtered)
        ds = None
        loaded_split = None

        # Current HF layout exposes MCQ/RQA as splits under the default builder.
        for alias in split_aliases[split_name]:
            try:
                ds = load_hf_dataset("nvidia/Nemotron-Science-v1", split=alias, streaming=True)
                loaded_split = alias
                break
            except Exception as e:
                print(f"  ERROR loading split '{alias}': {e}")

        if ds is None:
            # Legacy fallback in case a cached older layout uses builder configs.
            if split_name == "MCQ":
                try:
                    ds = load_hf_dataset("nvidia/Nemotron-Science-v1", name="MCQ", split="train", streaming=True)
                    loaded_split = "MCQ(train)"
                except Exception as e2:
                    print(f"  ERROR: {e2}")
                    continue
            else:
                try:
                    ds = load_hf_dataset("nvidia/Nemotron-Science-v1", name="RQA", split="train", streaming=True)
                    loaded_split = "RQA(train)"
                except Exception as e2:
                    print(f"  ERROR: {e2}")
                    continue
        
        # Explore schema
        sample = list(itertools.islice(ds, 3))
        if sample:
            print(f"\n  Split '{loaded_split}' columns: {list(sample[0].keys())}")
            print(f"  Sample keys detail:")
            for k, v in sample[0].items():
                print(f"    {k}: {str(v)[:100]}")
        
        # Re-create stream
        if loaded_split in {"MCQ", "RQA"}:
            ds = load_hf_dataset("nvidia/Nemotron-Science-v1", split=loaded_split, streaming=True)
        elif loaded_split == "MCQ(train)":
            ds = load_hf_dataset("nvidia/Nemotron-Science-v1", name="MCQ", split="train", streaming=True)
        else:
            ds = load_hf_dataset("nvidia/Nemotron-Science-v1", name="RQA", split="train", streaming=True)
        
        scanned = 0
        split_kept = 0
        missing_messages = 0
        missing_answer = 0
        short_answer = 0
        deduped = 0
        
        for item in ds:
            scanned += 1

            msgs = item.get("messages", [])
            if not isinstance(msgs, list) or len(msgs) < 2:
                missing_messages += 1
                continue

            convs = []
            first_user = ""
            first_assistant = ""
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).lower()
                content = msg.get("content", "")
                if role in ("user", "human") and content:
                    convs.append({"from": "human", "value": content})
                    if not first_user:
                        first_user = content
                elif role in ("assistant", "model", "gpt") and content:
                    convs.append({"from": "gpt", "value": content})
                    if not first_assistant:
                        first_assistant = content

            if not first_user or not first_assistant:
                missing_answer += 1
                continue

            if word_count(first_assistant) < 20:
                short_answer += 1
                continue

            if dedup.is_duplicate(first_user):
                deduped += 1
                continue

            filtered.append({
                "conversations": convs,
                "metadata": {
                    "source": f"nemotron_science_{split_name.lower()}",
                    "used_in": item.get("used_in"),
                },
            })
            split_kept += 1

            if split_kept >= per_split_target:
                break
            
            if scanned % D5_PROGRESS_EVERY == 0:
                print(f"  [{split_name}] Scanned {scanned:,}, kept {split_kept:,}")
        
        print(f"  [{split_name}] scanned {scanned:,}, kept {split_kept:,}")
        print(
            f"  [{split_name}] Rejections: "
            f"missing_messages={missing_messages:,}, "
            f"missing_answer={missing_answer:,}, "
            f"length_gate={short_answer:,}, "
            f"dedup={deduped:,}"
        )
    
    print(f"  DONE: total {len(filtered):,}")
    return filtered


# ── D6: derek-thomas/ScienceQA ────────────────────────────────────────────

def load_d6_scienceqa(target, dedup):
    """
    derek-thomas/ScienceQA, split="train"
    Verified schema:
      image, question, choices (list), answer (int index),
      hint, task, grade, subject, topic, category, skill,
      lecture, solution
    Filter: text-only (no image), has lecture, science subjects
    """
    print(f"\n{'='*60}")
    print(f"[D6] derek-thomas/ScienceQA (target: {target:,})")
    print(f"{'='*60}")

    ds = load_hf_dataset("derek-thomas/ScienceQA", split="train")
    
    print(f"  Total items: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    
    filtered = []
    
    for item in ds:
        # Text-only: skip items with images
        image = item.get("image", None)
        if image is not None:
            continue
        
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_idx = item.get("answer", -1)
        lecture = item.get("lecture", "")
        solution = item.get("solution", "")
        
        if not question or not lecture:
            continue
        
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(choices):
            continue
        
        correct_answer = choices[answer_idx]
        
        # Build answer with explanation
        answer_text = f"The correct answer is: {correct_answer}\n\n{lecture}"
        if solution:
            answer_text += f"\n\nExplanation: {solution}"
        
        if word_count(answer_text) < 30:
            continue
        
        # Build question with choices
        choice_lines = []
        for idx, c in enumerate(choices):
            letter = chr(65 + idx)
            choice_lines.append(f"{letter}) {c}")
        
        full_question = f"{question}\n\n" + "\n".join(choice_lines)
        
        if dedup.is_duplicate(full_question):
            continue
        
        filtered.append(to_sharegpt(full_question, answer_text, {"source": "scienceqa"}))
    
    if len(filtered) > target:
        filtered = random.sample(filtered, target)
    
    print(f"  DONE: kept {len(filtered):,}")
    return filtered


# ── D7: allenai/sciq ─────────────────────────────────────────────────────

def load_d7_sciq(target, dedup):
    """
    allenai/sciq, split="train"
    Verified schema:
      question, correct_answer, support, distractor1, distractor2, distractor3
    Filter: has support text (explanation), length gate
    """
    print(f"\n{'='*60}")
    print(f"[D7] allenai/sciq (target: {target:,})")
    print(f"{'='*60}")

    ds = load_hf_dataset("allenai/sciq", split="train")
    
    print(f"  Total items: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    
    filtered = []
    
    for item in ds:
        question = item.get("question", "")
        correct = item.get("correct_answer", "")
        support = item.get("support", "")
        
        if not question or not correct:
            continue
        
        # Only keep items with explanation
        if not support or word_count(support) < 10:
            continue
        
        answer_text = f"{correct}\n\nExplanation: {support}"
        
        if dedup.is_duplicate(question):
            continue
        
        filtered.append(to_sharegpt(question, answer_text, {"source": "sciq"}))
    
    if len(filtered) > target:
        filtered = random.sample(filtered, target)
    
    print(f"  DONE: kept {len(filtered):,}")
    return filtered


# ── D8: KadamParth/Ncert_dataset ─────────────────────────────────────────

def load_d8_ncert(target, dedup):
    """
    KadamParth/Ncert_dataset
    Schema: auto-detected — print columns first
    Expected: question, answer, possibly subject/class fields
    Filter: length gate, dedup, subject balance
    """
    print(f"\n{'='*60}")
    print(f"[D8] KadamParth/Ncert_dataset (target: {target:,})")
    print(f"{'='*60}")
    
    try:
        ds = load_hf_dataset("KadamParth/Ncert_dataset", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return []
    
    print(f"  Total items: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    
    # Print 2 samples to see actual schema
    explore_schema(ds, "NCERT", n=2)
    
    # Auto-detect question and answer column names
    cols = set(ds.column_names)
    q_col = None
    a_col = None
    
    for candidate in ["question", "Question", "input", "instruction", "prompt", "text"]:
        if candidate in cols:
            q_col = candidate
            break
    
    for candidate in ["answer", "Answer", "output", "response", "target", "completion"]:
        if candidate in cols:
            a_col = candidate
            break
    
    if not q_col or not a_col:
        print(f"  ERROR: Could not detect question/answer columns from: {cols}")
        print(f"  Detected q_col={q_col}, a_col={a_col}")
        return []
    
    print(f"  Using: question_col='{q_col}', answer_col='{a_col}'")
    
    # Detect subject column
    subj_col = None
    for candidate in ["subject", "Subject", "category", "topic"]:
        if candidate in cols:
            subj_col = candidate
            break

    # Detect grade/class column separately
    grade_col = None
    for candidate in ["grade", "Grade", "class", "Class"]:
        if candidate in cols:
            grade_col = candidate
            break
    
    # Group by subject for balanced sampling
    by_subject = {}
    seen_exact = set()
    
    for item in ds:
        q = item.get(q_col, "") or ""
        a = item.get(a_col, "") or ""
        
        if not q or not a:
            continue
        
        # Exact dedup within NCERT
        exact_key = normalize_for_hash(q)
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        
        if word_count(a) < 10:
            continue
        
        # Cross-dataset dedup
        if dedup.is_duplicate(q):
            continue
        
        subject = normalize_subject(item.get(subj_col, "general") if subj_col else "general")
        grade = item.get(grade_col) if grade_col else None

        if not keep_ncert_subject(subject, grade):
            continue

        if subject not in by_subject:
            by_subject[subject] = []
        
        by_subject[subject].append(
            to_sharegpt(q, a, {"source": "ncert", "subject": subject, "grade": grade})
        )
    
    # Balance across subjects
    filtered = []
    if by_subject:
        per_subject = max(1, target // len(by_subject))
        for subj, items in by_subject.items():
            take = min(per_subject, len(items))
            filtered.extend(random.sample(items, take) if len(items) > take else items)
        
        # Fill remaining from largest subjects
        remaining = target - len(filtered)
        if remaining > 0:
            all_extra = []
            for subj, items in by_subject.items():
                already = min(per_subject, len(items))
                all_extra.extend(items[already:])
            if all_extra:
                filtered.extend(random.sample(all_extra, min(remaining, len(all_extra))))
    
    if len(filtered) > target:
        filtered = random.sample(filtered, target)
    
    print(f"  Exact dedup: {len(seen_exact):,} unique questions")
    print(f"  Subjects: {list(by_subject.keys())[:10]}")
    print(f"  DONE: kept {len(filtered):,}")
    return filtered


# ── Synthetic Dataset Loader ──────────────────────────────────────────────

def load_synthetic(path, dedup):
    """
    Load synthetic dataset from our pipeline.
    Handles both ShareGPT and Alpaca formats.
    """
    print(f"\n{'='*60}")
    print(f"[SYN] Synthetic dataset: {path}")
    print(f"{'='*60}")
    
    if not os.path.exists(path):
        print(f"  WARNING: File not found: {path}")
        print(f"  Skipping synthetic data. Run the pipeline first.")
        return []
    
    data = []
    alpaca_converted = 0
    sharegpt_loaded = 0
    skipped = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            
            if "conversations" in item:
                # ShareGPT format
                convs = item["conversations"]
                if isinstance(convs, list) and len(convs) >= 2:
                    human_text = ""
                    for turn in convs:
                        if turn.get("from") == "human":
                            human_text = turn.get("value", "")
                            break
                    
                    if human_text and not dedup.is_duplicate(human_text):
                        data.append(item)
                        sharegpt_loaded += 1
                    else:
                        skipped += 1
                        
            elif "instruction" in item and "output" in item:
                # Alpaca format → convert
                q = item["instruction"]
                a = item["output"]
                
                if q and a and not dedup.is_duplicate(q):
                    data.append(to_sharegpt(q, a, {"source": "synthetic_textbook"}))
                    alpaca_converted += 1
                else:
                    skipped += 1
            else:
                skipped += 1
    
    print(f"  Loaded: {len(data):,} items")
    print(f"    ShareGPT: {sharegpt_loaded:,}, Alpaca→ShareGPT: {alpaca_converted:,}, Skipped: {skipped:,}")
    return data


def load_reused_filtered(path, dedup, ds_id, ds_name):
    """Reuse a previously saved filtered JSONL and seed dedup from it."""
    print(f"\n{'='*60}")
    print(f"[{ds_id}] Reusing filtered dataset: {path}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"  ERROR: Reuse file not found: {path}")
        return []

    data = []
    skipped = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            convs = item.get("conversations", [])
            if not isinstance(convs, list) or len(convs) < 2:
                skipped += 1
                continue

            human_text = ""
            for turn in convs:
                if isinstance(turn, dict) and turn.get("from") == "human":
                    human_text = turn.get("value", "")
                    if human_text:
                        break

            if not human_text:
                skipped += 1
                continue

            dedup.add_existing(human_text)
            data.append(item)

    print(f"  Loaded: {len(data):,} items from cached filtered file")
    if skipped:
        print(f"  Skipped malformed rows: {skipped:,}")
    return data


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GYANDEEP Stage 1 Dataset Builder")
    parser.add_argument("--output", default="gyandeep_stage1_training.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip specific datasets: D1 D2 D3 D4 D5 D6 D7 D8 SYN")
    parser.add_argument("--synthetic-path", default="synthetic_dataset_stage_1.jsonl",
                        help="Path to synthetic dataset JSONL")
    parser.add_argument("--output-dir", default="./stage1_filtered",
                        help="Directory to save individual filtered datasets")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Disable both exact and semantic dedup across all datasets")
    parser.add_argument("--d1-source-cap", type=int, default=None,
                        help="Optional per-problem_source cap for D1/OpenMathReasoning. Disabled by default.")
    parser.add_argument("--reuse-filtered", nargs="*", default=[],
                        help="Reuse previously saved filtered JSONL files from --output-dir for specific datasets")
    
    args = parser.parse_args()
    
    output_path = args.output
    skip_list = set(s.upper() for s in args.skip)
    reuse_list = set(s.upper() for s in args.reuse_filtered)
    synthetic_path = args.synthetic_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GYANDEEP STAGE 1 TRAINING DATASET BUILDER")
    print(f"{'='*70}")
    print(f"  Output: {output_path}")
    print(f"  Filtered dir: {output_dir}")
    print(f"  Synthetic: {synthetic_path}")
    print(f"  Skip: {skip_list or 'none'}")
    print(f"  Reuse filtered: {reuse_list or 'none'}")
    print(f"  Dedup: {'disabled' if args.skip_dedup else 'enabled'}")
    print(f"{'='*70}\n")
    
    dedup = DedupTracker(enabled=not args.skip_dedup)
    all_data = {}
    start_time = time.time()
    
    # Load each dataset in order (order matters for dedup priority)
    loaders = [
        ("D1", "OpenMathReasoning",     lambda: load_d1_openmath(TARGETS["D1"], dedup, source_cap=args.d1_source_cap)),
        ("D2", "OpenR1-Math-220k",      lambda: load_d2_openr1(TARGETS["D2"], dedup)),
        ("D3", "Nemotron-SFT-Math-v3",  lambda: load_d3_nemotron_math(TARGETS["D3"], dedup)),
        ("D4", "OpenThoughts3-1.2M",    lambda: load_d4_openthoughts(TARGETS["D4"], dedup)),
        ("D5", "Nemotron-Science-v1",   lambda: load_d5_nemotron_science(TARGETS["D5"], dedup)),
        ("D6", "ScienceQA",            lambda: load_d6_scienceqa(TARGETS["D6"], dedup)),
        ("D7", "SciQ",                 lambda: load_d7_sciq(TARGETS["D7"], dedup)),
        ("D8", "NCERT",                lambda: load_d8_ncert(TARGETS["D8"], dedup)),
        ("SYN", "Synthetic",           lambda: load_synthetic(synthetic_path, dedup)),
    ]
    
    for ds_id, ds_name, loader_fn in loaders:
        if ds_id in reuse_list:
            reuse_path = filtered_output_path(output_dir, ds_id, ds_name)
            data = load_reused_filtered(reuse_path, dedup, ds_id, ds_name)
            all_data[ds_id] = data
            continue

        if ds_id in skip_list:
            print(f"\n  SKIPPED: {ds_id} ({ds_name})")
            continue
        
        try:
            data = loader_fn()
            all_data[ds_id] = data
            
            # Save individual filtered file
            if data:
                save_filtered(data, filtered_output_path(output_dir, ds_id, ds_name))
        
        except Exception as e:
            print(f"\n  ERROR in {ds_id} ({ds_name}): {e}")
            import traceback
            traceback.print_exc()
            all_data[ds_id] = []
    
    # Merge all
    print(f"\n{'='*70}")
    print(f"MERGING ALL DATASETS")
    print(f"{'='*70}")
    
    merged = []
    for ds_id, ds_name, _ in loaders:
        data = all_data.get(ds_id, [])
        count = len(data)
        merged.extend(data)
        pct = count / max(sum(len(d) for d in all_data.values()), 1) * 100
        print(f"  {ds_id} {ds_name:30s} → {count:>6,} items ({pct:4.1f}%)")
    
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':33s} → {len(merged):>6,} items")
    print(f"  Unique questions tracked: {dedup.count:,}")
    
    # Shuffle
    random.shuffle(merged)
    
    # Save merged
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print(f"STAGE 1 DATASET BUILD COMPLETE")
    print(f"{'='*70}")
    print(f"  Total items:  {len(merged):,}")
    print(f"  File size:    {file_size:.1f} MB")
    print(f"  Output:       {output_path}")
    print(f"  Time:         {elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    # Save build log
    log = {
        "timestamp": datetime.now().isoformat(),
        "output": output_path,
        "total_items": len(merged),
        "file_size_mb": round(file_size, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "unique_questions": dedup.count,
        "per_dataset": {
            ds_id: len(all_data.get(ds_id, []))
            for ds_id, _, _ in loaders
        },
        "targets": TARGETS,
    }
    log_path = output_dir / "build_log.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Build log: {log_path}")


if __name__ == "__main__":
    main()

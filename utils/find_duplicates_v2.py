"""
Multi-gate near-duplicate detection pipeline with checkpointing.

Gate 1 — Exact hash     : MD5 of normalized full question
Gate 2 — WordLlama l3   : 1024-dim cosine, threshold 0.85
Gate 3 — Jaccard filter : word-level Jaccard on full question
  >= 0.60  → confirmed dup
  0.50-0.60 → borderline → escalate to Gate 4
  < 0.50   → rejected
Gate 4 — Nomic (Ollama) : 768-dim on borderline pairs only
  >= 0.88  → confirmed dup
  < 0.88   → rejected

Checkpoints saved after each gate — safe to kill and resume.
Output: duplicate_mapping_v2.json
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

DATASET = Path("/home/m.kulariya/phi_agents/kaggle-hackthon-zapclaw/zapclaw_workspace/gyandeep_stage1_training.jsonl")
OUTPUT  = Path("/home/m.kulariya/phi_agents/kaggle-hackthon-zapclaw/zapclaw_workspace/duplicate_mapping_v2.json")
CKPT_DIR = Path("/home/m.kulariya/phi_agents/kaggle-hackthon-zapclaw/zapclaw_workspace/dedup_checkpoints")

WORDLLAMA_THRESH      = 0.85
JACCARD_CONFIRMED     = 0.60
JACCARD_BORDERLINE_LO = 0.50   # raised from 0.30 → fewer nomic calls
NOMIC_THRESH          = 0.88
BATCH_SIZE            = 2000
NOMIC_BATCH           = 16
N_WORKERS             = 4
CHUNK                 = 256

_args = None  # set in main


# ── helpers ──────────────────────────────────────────────────────────────────

def get_human(obj):
    return next((c["value"] for c in obj["conversations"] if c.get("from") == "human"), "")

def get_gpt(obj):
    return next((c["value"] for c in obj["conversations"] if c.get("from") == "gpt"), "")

def normalize(text):
    return " ".join(text.lower().split())

def md5(text):
    return hashlib.md5(text.encode()).hexdigest()

def nomic_embed(texts, retries=3):
    payload = json.dumps({"model": "nomic-embed-text:v1.5", "input": texts}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["embeddings"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Ollama retry {attempt+1}: {e}", flush=True)
            time.sleep(2)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def save_ckpt(name, data):
    CKPT_DIR.mkdir(exist_ok=True)
    p = CKPT_DIR / name
    with open(p, "w") as f:
        json.dump(data, f)
    print(f"  checkpoint saved: {p}", flush=True)

def load_ckpt(name):
    p = CKPT_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def save_embs(embs_norm):
    CKPT_DIR.mkdir(exist_ok=True)
    p = CKPT_DIR / "gate2_embs.npy"
    np.save(str(p), embs_norm)
    print(f"  embeddings saved: {p} ({embs_norm.shape})", flush=True)

def load_embs():
    p = CKPT_DIR / "gate2_embs.npy"
    if p.exists():
        return np.load(str(p))
    return None


# ── multiprocessing workers (must be module-level) ───────────────────────────

_worker_wl = None

def _init_worker(config, dim):
    global _worker_wl
    from wordllama import WordLlama
    _worker_wl = WordLlama.load(config=config, dim=dim)

def _embed_chunk(texts):
    return _worker_wl.embed(texts)


# ── union-find ────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a

    def groups(self):
        from collections import defaultdict
        g = defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return {k: v for k, v in g.items() if len(v) > 1}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global _args
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-nomic", action="store_true", help="Skip Gate 4")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints, start fresh")
    _args = parser.parse_args()

    if _args.fresh:
        for f in CKPT_DIR.glob("*.json"):
            f.unlink()
        p = CKPT_DIR / "gate2_embs.npy"
        if p.exists():
            p.unlink()
        print("Checkpoints cleared.", flush=True)

    t0 = time.time()

    # ── load dataset ─────────────────────────────────────────────────────────
    print("Loading dataset...", flush=True)
    samples, questions, answers = [], [], []
    with open(DATASET) as f:
        for i, line in enumerate(f):
            if _args.limit and i >= _args.limit:
                break
            obj = json.loads(line.strip())
            samples.append(obj)
            questions.append(get_human(obj))
            answers.append(get_gpt(obj))
    n = len(samples)
    print(f"  {n} samples in {time.time()-t0:.1f}s", flush=True)

    # ── gate 1: exact hash ────────────────────────────────────────────────────
    ckpt1 = load_ckpt("gate1.json") if not _args.fresh else None
    if ckpt1:
        exact_pairs = [tuple(p) for p in ckpt1["exact_pairs"]]
        print(f"\nGate 1: loaded from checkpoint ({len(exact_pairs)} exact pairs)", flush=True)
    else:
        print("\nGate 1: exact hash...", flush=True)
        t1 = time.time()
        hash_map = {}
        exact_pairs = []
        for i, q in enumerate(questions):
            h = md5(normalize(q))
            if h in hash_map:
                exact_pairs.append((hash_map[h], i))
            else:
                hash_map[h] = i
        print(f"  exact pairs: {len(exact_pairs)} in {time.time()-t1:.1f}s", flush=True)
        save_ckpt("gate1.json", {"exact_pairs": exact_pairs})

    # ── gate 2: wordllama embed + cosine search ───────────────────────────────
    ckpt2 = load_ckpt("gate2_candidates.json") if not _args.fresh else None
    if ckpt2:
        wl_candidates = [tuple(p) for p in ckpt2["wl_candidates"]]
        print(f"\nGate 2: loaded from checkpoint ({len(wl_candidates)} candidates)", flush=True)
    else:
        print(f"\nGate 2: WordLlama l3 1024-dim (thresh={WORDLLAMA_THRESH})...", flush=True)
        from multiprocessing import Pool

        # embedding — check if embs already saved
        embs_norm = load_embs()
        if embs_norm is not None:
            print(f"  embeddings loaded from checkpoint {embs_norm.shape}", flush=True)
        else:
            chunks = [questions[s:s+CHUNK] for s in range(0, n, CHUNK)]
            print(f"  {N_WORKERS} workers, {len(chunks)} chunks of {CHUNK}", flush=True)
            print(f"  initializing workers (model load ~15s)...", flush=True)
            t2 = time.time()
            embs_list = []
            with Pool(N_WORKERS, initializer=_init_worker, initargs=("l3_supercat", 1024)) as pool:
                for done, result in enumerate(pool.imap(_embed_chunk, chunks, chunksize=1), 1):
                    embs_list.append(result)
                    elapsed = time.time() - t2
                    pct = done / len(chunks) * 100
                    eta = elapsed / max(pct, 0.01) * (100 - pct)
                    print(f"  embed {min(done*CHUNK,n)}/{n} ({pct:.1f}%) | ETA: {eta:.0f}s", flush=True)
            embs = np.vstack(embs_list)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            embs_norm = (embs / norms).astype(np.float32)
            print(f"  embedding done in {time.time()-t2:.1f}s", flush=True)
            save_embs(embs_norm)

        # cosine search
        print(f"  cosine search...", flush=True)
        t2b = time.time()
        all_i, all_j, all_s = [], [], []
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            if end >= n:
                break
            sim = embs_norm[start:end] @ embs_norm[end:].T
            rows, cols = np.where(sim >= WORDLLAMA_THRESH)
            all_i.append(rows + start)
            all_j.append(cols + end)
            all_s.append(sim[rows, cols])
            elapsed = time.time() - t2b
            pct = end / n * 100
            eta = elapsed / max(pct, 0.1) * (100 - pct)
            ncands = sum(len(x) for x in all_i)
            print(f"  search {end}/{n} ({pct:.1f}%) | candidates: {ncands} | ETA: {eta:.0f}s", flush=True)

        if all_i:
            gi = np.concatenate(all_i)
            gj = np.concatenate(all_j)
            gs = np.concatenate(all_s)
            wl_candidates = list(zip(gi.tolist(), gj.tolist(), gs.tolist()))
        else:
            wl_candidates = []
        print(f"  WordLlama candidates: {len(wl_candidates)} in {time.time()-t2b:.1f}s", flush=True)
        save_ckpt("gate2_candidates.json", {"wl_candidates": wl_candidates})

    # ── gate 3: jaccard filter ────────────────────────────────────────────────
    ckpt3 = load_ckpt("gate3.json") if not _args.fresh else None
    if ckpt3:
        confirmed_pairs  = [tuple(p) for p in ckpt3["confirmed_pairs"]]
        borderline_pairs = [tuple(p) for p in ckpt3["borderline_pairs"]]
        rejected_count   = ckpt3["rejected_count"]
        print(f"\nGate 3: loaded from checkpoint (confirmed={len(confirmed_pairs)}, borderline={len(borderline_pairs)})", flush=True)
    else:
        print("\nGate 3: Jaccard filter...", flush=True)
        t3 = time.time()
        involved = set()
        for i, j, _ in wl_candidates:
            involved.add(i); involved.add(j)
        token_cache = {idx: set(questions[idx].lower().split()) for idx in involved}

        confirmed_pairs, borderline_pairs, rejected_count = [], [], 0
        for i, j, wl_sim in wl_candidates:
            sa, sb = token_cache[i], token_cache[j]
            union = len(sa | sb)
            jac = len(sa & sb) / union if union else 1.0
            if jac >= JACCARD_CONFIRMED:
                confirmed_pairs.append((i, j, "wordllama_jaccard"))
            elif jac >= JACCARD_BORDERLINE_LO:
                borderline_pairs.append((i, j, wl_sim))
            else:
                rejected_count += 1

        print(f"  confirmed: {len(confirmed_pairs)}", flush=True)
        print(f"  borderline→nomic: {len(borderline_pairs)}", flush=True)
        print(f"  rejected: {rejected_count}", flush=True)
        print(f"  done in {time.time()-t3:.1f}s", flush=True)
        save_ckpt("gate3.json", {
            "confirmed_pairs":  confirmed_pairs,
            "borderline_pairs": borderline_pairs,
            "rejected_count":   rejected_count,
        })

    # ── gate 4: nomic on borderline ───────────────────────────────────────────
    ckpt4 = load_ckpt("gate4.json") if not _args.fresh else None
    if ckpt4:
        nomic_confirmed = [tuple(p) for p in ckpt4["nomic_confirmed"]]
        print(f"\nGate 4: loaded from checkpoint ({len(nomic_confirmed)} nomic confirmed)", flush=True)
    elif _args.skip_nomic:
        nomic_confirmed = []
        print(f"\nGate 4: skipped (--skip-nomic)", flush=True)
    else:
        print(f"\nGate 4: Nomic on {len(borderline_pairs)} borderline pairs...", flush=True)
        t4 = time.time()
        nomic_confirmed = []
        nomic_rejected  = 0

        if borderline_pairs:
            border_indices = sorted(set(idx for pair in borderline_pairs for idx in pair[:2]))
            print(f"  unique samples to embed: {len(border_indices)}", flush=True)

            nomic_embs = {}
            for batch_start in range(0, len(border_indices), NOMIC_BATCH):
                batch_idxs = border_indices[batch_start:batch_start + NOMIC_BATCH]
                texts = [questions[idx] for idx in batch_idxs]
                vecs = nomic_embed(texts)
                for idx, vec in zip(batch_idxs, vecs):
                    nomic_embs[idx] = vec
                done = min(batch_start + NOMIC_BATCH, len(border_indices))
                elapsed = time.time() - t4
                eta = elapsed / max(done, 1) * (len(border_indices) - done)
                print(f"  embedded {done}/{len(border_indices)} | ETA: {eta:.0f}s", flush=True)

            for i, j, _ in borderline_pairs:
                sim = cosine_sim(nomic_embs[i], nomic_embs[j])
                if sim >= NOMIC_THRESH:
                    nomic_confirmed.append((i, j, "nomic_confirmed"))
                else:
                    nomic_rejected += 1

            print(f"  nomic confirmed: {len(nomic_confirmed)}", flush=True)
            print(f"  nomic rejected:  {nomic_rejected}", flush=True)
        print(f"  Gate 4 done in {time.time()-t4:.1f}s", flush=True)
        save_ckpt("gate4.json", {"nomic_confirmed": nomic_confirmed})

    # ── merge + output ────────────────────────────────────────────────────────
    print("\nMerging groups...", flush=True)
    uf = UnionFind(n)
    pair_confidence = {}

    for i, j in exact_pairs:
        uf.union(i, j)
        pair_confidence[(min(i,j), max(i,j))] = "exact_hash"
    for i, j, label in confirmed_pairs:
        uf.union(i, j)
        pair_confidence[(min(i,j), max(i,j))] = label
    for i, j, label in nomic_confirmed:
        uf.union(i, j)
        pair_confidence[(min(i,j), max(i,j))] = label

    raw_groups = uf.groups()
    groups_out = {}
    total_remove = 0

    for gid, members in raw_groups.items():
        best = max(members, key=lambda idx: len(answers[idx]))
        to_remove = [idx for idx in members if idx != best]
        confs = set()
        for a in members:
            for b in members:
                if a >= b: continue
                c = pair_confidence.get((min(a,b), max(a,b)))
                if c:
                    confs.add(c)
        conf = ("exact_hash" if "exact_hash" in confs
                else "nomic_confirmed" if "nomic_confirmed" in confs
                else "wordllama_jaccard")
        groups_out[str(gid)] = {
            "keep": best, "remove": to_remove,
            "size": len(members), "confidence": conf,
        }
        total_remove += len(to_remove)

    by_conf = {}
    for g in groups_out.values():
        c = g["confidence"]
        by_conf[c] = by_conf.get(c, 0) + len(g["remove"])

    output = {
        "dataset":         str(DATASET),
        "total_samples":   n,
        "total_remove":    total_remove,
        "total_keep":      n - total_remove,
        "reduction_pct":   round(total_remove / n * 100, 2),
        "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "removed_by_gate": by_conf,
        "groups":          groups_out,
    }
    with open(OUTPUT, "w") as f:
        json.dump(output, f)

    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"  Groups:          {len(groups_out)}", flush=True)
    print(f"  Remove:          {total_remove} ({total_remove/n*100:.1f}%)", flush=True)
    print(f"  Keep:            {n - total_remove}", flush=True)
    print(f"  Removed by gate: {by_conf}", flush=True)
    print(f"  Mapping saved:   {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

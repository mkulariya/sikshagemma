"""
Clean dataset using gate1 (exact hash) + gate2 (WordLlama >= WL_THRESH) checkpoints.
Keeps the sample with the longest GPT answer in each duplicate group.
Never modifies the original dataset.
Output: <dataset>_deduped.jsonl
"""

import json
from pathlib import Path

DATASET   = Path("/home/m.kulariya/phi_agents/kaggle-hackthon-zapclaw/zapclaw_workspace/gyandeep_stage1_training.jsonl")
GATE1     = Path("/home/m.kulariya/phi_agents/kaggle-hackthon-zapclaw/zapclaw_workspace/dedup_checkpoints/gate1.json")
GATE2     = Path("/home/m.kulariya/phi_agents/kaggle-hackthon-zapclaw/zapclaw_workspace/dedup_checkpoints/gate2_candidates.json")
WL_THRESH = 0.95


def get_gpt_length(obj):
    for c in obj.get("conversations", []):
        if c.get("from") == "gpt":
            return len(c.get("value", ""))
    return 0


def union_find_groups(pairs, n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i, j in pairs:
        union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return {k: v for k, v in groups.items() if len(v) > 1}


def main():
    # 1. Load dataset
    print("Loading dataset...", flush=True)
    samples = []
    with open(DATASET) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    n = len(samples)
    print(f"  {n} samples", flush=True)

    # 2. Load gate1 pairs (exact hash)
    print("Loading gate1 (exact hash) pairs...", flush=True)
    with open(GATE1) as f:
        g1_pairs = json.load(f)["exact_pairs"]   # [[i, j], ...]
    print(f"  {len(g1_pairs)} exact pairs", flush=True)

    # 3. Load gate2 pairs, filter by threshold
    print(f"Loading gate2 (WordLlama >= {WL_THRESH}) pairs...", flush=True)
    with open(GATE2) as f:
        g2_all = json.load(f)["wl_candidates"]   # [[i, j, sim], ...]
    g2_pairs = [(i, j) for i, j, s in g2_all if s >= WL_THRESH]
    print(f"  {len(g2_pairs)} pairs >= {WL_THRESH} (from {len(g2_all)} total)", flush=True)

    # 4. Combine and deduplicate pair set
    all_pairs = {(min(i, j), max(i, j)) for i, j in g1_pairs}
    all_pairs.update((min(i, j), max(i, j)) for i, j in g2_pairs)
    print(f"  {len(all_pairs)} unique pairs combined", flush=True)

    # 5. Union-Find → duplicate groups
    print("Running Union-Find...", flush=True)
    dup_groups = union_find_groups(all_pairs, n)
    print(f"  {len(dup_groups)} duplicate groups found", flush=True)

    # 6. Within each group keep sample with longest GPT answer, mark rest for removal
    remove = set()
    for group_idxs in dup_groups.values():
        best = max(group_idxs, key=lambda i: get_gpt_length(samples[i]))
        for idx in group_idxs:
            if idx != best:
                remove.add(idx)

    kept    = [s for i, s in enumerate(samples) if i not in remove]
    removed = n - len(kept)

    # 7. Write output — original file never touched
    output = str(DATASET).replace(".jsonl", "_deduped.jsonl")
    assert output != str(DATASET), "BUG: output path equals input path"

    print(f"\nOriginal:  {n}", flush=True)
    print(f"Removed:   {removed} ({removed / n * 100:.1f}%)", flush=True)
    print(f"Kept:      {len(kept)} ({len(kept) / n * 100:.1f}%)", flush=True)
    print(f"Writing → {output}", flush=True)

    with open(output, "w") as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

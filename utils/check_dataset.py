import json
import sys
from collections import Counter

dataset_path = sys.argv[1] if len(sys.argv) > 1 else "gyandeep_stage1_training.jsonl"

issues = Counter()
bad_examples = []
total = 0

with open(dataset_path) as f:
    for lineno, line in enumerate(f, 1):
        total += 1
        ex = json.loads(line)
        convo = ex.get("conversations", [])
        role_key = "role" if convo and "role" in convo[0] else "from"
        human_alias = {"user", "human"}
        assistant_alias = {"assistant", "gpt", "model"}

        raw_roles = [m.get(role_key, "") for m in convo]
        roles = []
        for r in raw_roles:
            if r in human_alias:
                roles.append("user")
            elif r in assistant_alias:
                roles.append("assistant")
            elif r == "system":
                pass  # skip system
            else:
                roles.append(r)

        if not roles:
            issues["empty_after_system"] += 1
            bad_examples.append((lineno, "empty_after_system", raw_roles))
        elif roles[0] != "user":
            key = f"starts_with_{roles[0]}"
            issues[key] += 1
            bad_examples.append((lineno, key, raw_roles[:4]))
        else:
            bad = False
            for i in range(len(roles) - 1):
                if roles[i] == roles[i + 1]:
                    key = f"consecutive_{roles[i]}"
                    issues[key] += 1
                    bad_examples.append((lineno, key, raw_roles[:6]))
                    bad = True
                    break
            if not bad:
                issues["valid"] += 1

print(f"\nDataset: {dataset_path}")
print(f"Total conversations: {total}")
print(f"\n--- Issue breakdown ---")
for k, v in sorted(issues.items(), key=lambda x: -x[1]):
    pct = v / total * 100
    tag = "OK" if k == "valid" else "BAD"
    print(f"  [{tag}] {k}: {v} ({pct:.2f}%)")

bad_total = total - issues["valid"]
print(f"\nTotal invalid: {bad_total} ({bad_total/total*100:.2f}%)")
print(f"Total valid:   {issues['valid']} ({issues['valid']/total*100:.2f}%)")

if bad_examples:
    print(f"\n--- First 10 bad examples (line, issue, roles) ---")
    for lineno, issue, roles in bad_examples[:10]:
        print(f"  line {lineno:6d} | {issue:30s} | {roles}")

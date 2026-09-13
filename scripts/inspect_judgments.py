import json
from pathlib import Path

path = Path("data/processed/semantic_llm_judgments.jsonl")

rows = [
    json.loads(line)
    for line in path.open("r", encoding="utf-8")
    if line.strip()
]

rows.sort(
    key=lambda r: r.get("retrieval_similarity", 0),
    reverse=True,
)

print(f"Total judgments: {len(rows)}")
print()

for r in rows:
    judge = r["judge"]

    print(
        f"{r['case_id']} | "
        f"sim={r.get('retrieval_similarity', 0):.4f} | "
        f"score={judge['overall_score']} | "
        f"{judge['decision']} | "
        f"{judge['reason']}"
    )

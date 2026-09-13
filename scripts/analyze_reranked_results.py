import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "semantic_reranked_results.jsonl"
)


def main():
    total = 0
    top1_quality = Counter()
    top5_quality = Counter()
    top1_resolution = Counter()
    top5_resolution = Counter()
    rank_changes = 0

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            retrieved = record.get("retrieved", [])

            if not retrieved:
                continue

            total += 1

            first = retrieved[0]

            top1_quality[first.get("inferred_evidence_quality", "unknown")] += 1
            top1_resolution[first.get("resolution_signal", "unknown")] += 1

            for item in retrieved[:5]:
                top5_quality[
                    item.get("inferred_evidence_quality", "unknown")
                ] += 1

                top5_resolution[
                    item.get("resolution_signal", "unknown")
                ] += 1

            if first.get("original_rank") != first.get("rank"):
                rank_changes += 1

    print(f"Total records: {total}")
    print(f"Rank changed in top result: {rank_changes}")

    print("\nTop-1 inferred quality:")
    for key, value in top1_quality.items():
        print(f"  {key}: {value}")

    print("\nTop-5 inferred quality:")
    for key, value in top5_quality.items():
        print(f"  {key}: {value}")

    print("\nTop-1 resolution signal:")
    for key, value in top1_resolution.items():
        print(f"  {key}: {value}")

    print("\nTop-5 resolution signal:")
    for key, value in top5_resolution.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
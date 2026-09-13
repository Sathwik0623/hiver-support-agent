import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "semantic_retrieval_results.jsonl"
)

RERANKED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "semantic_reranked_results.jsonl"
)


def load_records(path: Path):
    records = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            records[record["case_id"]] = record

    return records


def conversation_id(case_id):
    """
    Extract the support conversation ID from an Apple case ID.

    Example:
    APPLE_CASE_2503735_711875 -> 711875
    """

    if not case_id:
        return None

    parts = str(case_id).split("_")
    return parts[-1] if parts else None


def get_gold_conversation_ids(record):
    gold_ids = record.get("mapped_historical_case_ids", [])

    return {
        conversation_id(case_id)
        for case_id in gold_ids
        if conversation_id(case_id)
    }


def is_same_conversation(item, gold_conversation_ids):
    retrieved_case_id = item.get("case_id")

    return (
        conversation_id(retrieved_case_id)
        in gold_conversation_ids
    )


def evaluate(records):
    top1_conversation_match = 0
    top5_conversation_match = 0
    top1_resolved = 0
    top5_resolved = 0

    total = len(records)

    for record in records.values():
        retrieved = record.get("retrieved", [])

        if not retrieved:
            continue

        gold_conversation_ids = get_gold_conversation_ids(record)

        top1 = retrieved[0]

        if is_same_conversation(top1, gold_conversation_ids):
            top1_conversation_match += 1

        if any(
            is_same_conversation(item, gold_conversation_ids)
            for item in retrieved[:5]
        ):
            top5_conversation_match += 1

        if top1.get("resolution_signal") == "RESOLVED":
            top1_resolved += 1

        if any(
            item.get("resolution_signal") == "RESOLVED"
            for item in retrieved[:5]
        ):
            top5_resolved += 1

    return {
        "total": total,
        "top1_conversation_match": top1_conversation_match,
        "top5_conversation_match": top5_conversation_match,
        "top1_resolved": top1_resolved,
        "top5_resolved": top5_resolved,
    }


def print_metrics(name, metrics):
    total = metrics["total"]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Total records: {total}")

    for key, value in metrics.items():
        if key == "total":
            continue

        percentage = value / total * 100 if total else 0

        print(
            f"{key}: {value} "
            f"({percentage:.2f}%)"
        )


def main():
    original = load_records(ORIGINAL_FILE)
    reranked = load_records(RERANKED_FILE)

    print_metrics(
        "Original semantic retrieval",
        evaluate(original),
    )

    print_metrics(
        "Reranked semantic retrieval",
        evaluate(reranked),
    )


if __name__ == "__main__":
    main()
import json
from pathlib import Path

from src.retrieval.reranker import rerank_evidence


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "semantic_retrieval_results.jsonl"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "semantic_reranked_results.jsonl"
)


def main():
    processed = 0

    with INPUT_FILE.open("r", encoding="utf-8") as source, \
            OUTPUT_FILE.open("w", encoding="utf-8") as target:

        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)

            reranked = rerank_evidence(
                query=record.get("query", ""),
                retrieved=record.get("retrieved", []),
                top_k=5,
            )

            record["retrieved"] = reranked
            record["reranking_applied"] = True

            target.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            processed += 1

    print(f"Processed records: {processed}")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
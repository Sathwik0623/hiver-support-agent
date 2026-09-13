from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_retrieval_results.jsonl"
)

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "apple_support_golden_set.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_error_analysis.json"
)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_golden(path: Path) -> dict[str, dict]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return {
            str(row["case_id"]): row
            for row in csv.DictReader(f)
        }


def main() -> None:

    results = load_jsonl(RESULTS_PATH)
    golden = load_golden(GOLDEN_PATH)

    analyses = []

    for result in results:

        case_id = str(result["case_id"])

        gold = golden.get(case_id)

        if not gold:
            continue

        retrieved = result.get("retrieved", [])

        if not retrieved:
            continue

        top1 = retrieved[0]

        top5 = retrieved[:5]

        resolved_rank = None

        for item in top5:

            if (
                str(
                    item.get(
                        "resolution_signal",
                        "UNKNOWN"
                    )
                ).upper()
                == "RESOLVED"
            ):
                resolved_rank = item.get("rank")
                break

        analyses.append(
            {
                "case_id": case_id,
                "source_tweet_id": gold[
                    "source_tweet_id"
                ],
                "gold_intent": gold[
                    "annotator_intent"
                ],
                "gold_action": gold[
                    "expected_action"
                ],
                "query": result.get(
                    "query",
                    gold.get(
                        "customer_message",
                        ""
                    )
                ),
                "top1_similarity": top1.get(
                    "similarity",
                    0.0
                ),
                "top1_resolution_signal": top1.get(
                    "resolution_signal",
                    "UNKNOWN"
                ),
                "top1_resolution_confidence": top1.get(
                    "resolution_confidence",
                    0.0
                ),
                "top1_customer_problem": top1.get(
                    "customer_problem",
                    ""
                ),
                "top1_apple_response": top1.get(
                    "apple_response",
                    ""
                ),
                "top1_case_id": top1.get(
                    "case_id",
                    ""
                ),
                "resolved_rank_top5": resolved_rank,
                "top5": top5,
            }
        )

    # -------------------------------------------------------------
    # Interesting slices
    # -------------------------------------------------------------

    high_similarity_unknown = sorted(
        [
            x for x in analyses
            if str(
                x["top1_resolution_signal"]
            ).upper() == "UNKNOWN"
        ],
        key=lambda x: x["top1_similarity"],
        reverse=True
    )[:10]

    high_similarity_resolved = sorted(
        [
            x for x in analyses
            if str(
                x["top1_resolution_signal"]
            ).upper() == "RESOLVED"
        ],
        key=lambda x: x["top1_similarity"],
        reverse=True
    )[:10]

    low_similarity_resolved = sorted(
        [
            x for x in analyses
            if any(
                str(
                    item.get(
                        "resolution_signal",
                        "UNKNOWN"
                    )
                ).upper()
                == "RESOLVED"
                for item in x["top5"]
            )
        ],
        key=lambda x: x["top1_similarity"]
    )[:10]

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    output = {
        "high_similarity_unknown": (
            high_similarity_unknown
        ),
        "high_similarity_resolved": (
            high_similarity_resolved
        ),
        "low_similarity_with_resolved_in_top5": (
            low_similarity_resolved
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 70)
    print("TF-IDF RETRIEVAL ERROR ANALYSIS")
    print("=" * 70)

    print()
    print(
        "High-similarity UNKNOWN cases :",
        len(high_similarity_unknown)
    )

    print(
        "High-similarity RESOLVED cases:",
        len(high_similarity_resolved)
    )

    print(
        "Low-similarity cases with "
        "RESOLVED evidence in Top-5 :",
        len(low_similarity_resolved)
    )

    print()
    print(
        "Analysis written to:"
    )

    print(OUTPUT_PATH)

    print()
    print("=" * 70)
    print("TOP 5 HIGH-SIMILARITY / UNKNOWN CASES")
    print("=" * 70)

    for i, item in enumerate(
        high_similarity_unknown[:5],
        start=1
    ):

        print()
        print(f"[{i}] {item['case_id']}")
        print(
            f"Similarity: "
            f"{item['top1_similarity']:.4f}"
        )

        print(
            f"Intent: "
            f"{item['gold_intent']}"
        )

        print(
            "Customer:"
        )

        print(
            item["query"][:500]
        )

        print(
            "\nRetrieved:"
        )

        print(
            item[
                "top1_customer_problem"
            ][:500]
        )

        print(
            "\nApple response:"
        )

        print(
            item[
                "top1_apple_response"
            ][:500]
        )

    print()
    print(
        "Full analysis is available in:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()

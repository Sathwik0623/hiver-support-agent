"""
Leakage-safe semantic retrieval baseline.

Uses a local SentenceTransformer model to retrieve historically
similar AppleSupport cases.

Important experimental controls:
- Same 200 golden cases as TF-IDF baseline
- Same historical evidence corpus
- Entire historical case excluded for every golden case
- Only customer problem + conversation context are embedded
- Historical Apple responses are NOT embedded
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_evidence.jsonl"
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
    / "semantic_retrieval_results.jsonl"
)

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

BATCH_SIZE = 128


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def load_golden(path: Path) -> list[dict]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ---------------------------------------------------------------------
# Embedding text
# ---------------------------------------------------------------------

def build_embedding_text(record: dict) -> str:
    """
    Build the representation used for semantic retrieval.

    We deliberately exclude apple_response.

    Retrieval should be driven by the customer's problem and
    conversation context, not by similarity between support-agent
    wording.
    """

    problem = (
        record.get(
            "customer_problem",
            ""
        )
        or ""
    ).strip()

    context = (
        record.get(
            "conversation_context",
            ""
        )
        or ""
    ).strip()

    return (
        "Customer problem:\n"
        f"{problem}\n\n"
        "Conversation context:\n"
        f"{context}"
    )


def build_golden_query(row: dict) -> str:

    message = (
        row.get(
            "customer_message",
            ""
        )
        or ""
    ).strip()

    followup = (
        row.get(
            "customer_followup",
            ""
        )
        or ""
    ).strip()

    if followup:

        return (
            "Customer problem:\n"
            f"{message}\n\n"
            "Customer follow-up:\n"
            f"{followup}"
        )

    return (
        "Customer problem:\n"
        f"{message}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("LEAKAGE-SAFE SEMANTIC RETRIEVAL BASELINE")
    print("=" * 70)

    print()
    print(f"Evidence : {EVIDENCE_PATH}")
    print(f"Golden   : {GOLDEN_PATH}")
    print(f"Model    : {MODEL_NAME}")
    print(f"Top-K    : {TOP_K}")

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    evidence = load_jsonl(
        EVIDENCE_PATH
    )

    golden = load_golden(
        GOLDEN_PATH
    )

    print()
    print(
        f"Historical evidence loaded : "
        f"{len(evidence):,}"
    )

    print(
        f"Golden records loaded      : "
        f"{len(golden):,}"
    )

    # -------------------------------------------------------------
    # Map source tweet -> historical case
    # -------------------------------------------------------------

    source_to_case = {}

    for record in evidence:

        source_tweet_id = str(
            record.get(
                "source_tweet_id",
                ""
            )
        )

        case_id = str(
            record.get(
                "case_id",
                ""
            )
        )

        if (
            source_tweet_id
            and case_id
        ):
            source_to_case[
                source_tweet_id
            ] = case_id

    # -------------------------------------------------------------
    # Map golden cases to historical cases that must be excluded
    # -------------------------------------------------------------

    golden_excluded_cases = {}

    mapped = 0
    unmapped = 0

    for row in golden:

        golden_case_id = str(
            row["case_id"]
        )

        source_tweet_id = str(
            row["source_tweet_id"]
        )

        historical_case_id = (
            source_to_case.get(
                source_tweet_id
            )
        )

        if historical_case_id:

            golden_excluded_cases[
                golden_case_id
            ] = {
                historical_case_id
            }

            mapped += 1

        else:

            golden_excluded_cases[
                golden_case_id
            ] = set()

            unmapped += 1

    print()
    print("LEAKAGE MAPPING")
    print("-" * 70)

    print(
        f"Golden cases mapped       : "
        f"{mapped}"
    )

    print(
        f"Golden cases unmapped     : "
        f"{unmapped}"
    )

    all_excluded_cases = set()

    for case_ids in (
        golden_excluded_cases.values()
    ):
        all_excluded_cases.update(
            case_ids
        )

    # -------------------------------------------------------------
    # Build retrieval corpus
    # -------------------------------------------------------------

    retrieval_records = [
        record
        for record in evidence
        if str(
            record.get(
                "case_id",
                ""
            )
        ) not in all_excluded_cases
    ]

    print(
        f"Historical cases excluded: "
        f"{len(all_excluded_cases):,}"
    )

    print()
    print("RETRIEVAL CORPUS")
    print("-" * 70)

    print(
        f"Original evidence records : "
        f"{len(evidence):,}"
    )

    print(
        f"Evidence records removed  : "
        f"{len(evidence) - len(retrieval_records):,}"
    )

    print(
        f"Retrieval corpus           : "
        f"{len(retrieval_records):,}"
    )

    # -------------------------------------------------------------
    # Build embedding text
    # -------------------------------------------------------------

    corpus_texts = [
        build_embedding_text(
            record
        )
        for record in retrieval_records
    ]

    query_texts = [
        build_golden_query(
            row
        )
        for row in golden
    ]

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print()
    print(
        "Loading embedding model..."
    )

    start = time.time()

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Model loaded in "
        f"{time.time() - start:.1f}s"
    )

    # -------------------------------------------------------------
    # Encode historical evidence
    # -------------------------------------------------------------

    print()
    print(
        "Encoding historical evidence..."
    )

    start = time.time()

    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(
        f"Corpus encoding completed in "
        f"{time.time() - start:.1f}s"
    )

    # -------------------------------------------------------------
    # Encode golden queries
    # -------------------------------------------------------------

    print()
    print(
        "Encoding golden queries..."
    )

    start = time.time()

    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(
        f"Query encoding completed in "
        f"{time.time() - start:.1f}s"
    )

    # -------------------------------------------------------------
    # Retrieval
    #
    # Since embeddings are normalized, dot product == cosine
    # similarity.
    # -------------------------------------------------------------

    print()
    print(
        "Running semantic retrieval..."
    )

    start = time.time()

    output_records = []

    leakage_violations = 0

    for index, row in enumerate(golden):

        golden_case_id = str(
            row["case_id"]
        )

        excluded_cases = (
            golden_excluded_cases.get(
                golden_case_id,
                set()
            )
        )

        query_vector = query_embeddings[
            index
        ]

        scores = (
            corpus_embeddings
            @ query_vector
        )

        # Get enough candidates, then apply an explicit case-level
        # leakage check.
        candidate_count = min(
            len(scores),
            TOP_K + 50
        )

        candidate_indices = np.argpartition(
            -scores,
            candidate_count - 1
        )[
            :candidate_count
        ]

        candidate_indices = sorted(
            candidate_indices,
            key=lambda i: scores[i],
            reverse=True
        )

        retrieved = []

        for candidate_index in candidate_indices:

            record = retrieval_records[
                candidate_index
            ]

            historical_case_id = str(
                record.get(
                    "case_id",
                    ""
                )
            )

            if (
                historical_case_id
                in excluded_cases
            ):

                leakage_violations += 1

                continue

            retrieved.append(
                {
                    "rank": len(retrieved) + 1,
                    "similarity": round(
                        float(
                            scores[
                                candidate_index
                            ]
                        ),
                        6
                    ),
                    "evidence_id": record.get(
                        "evidence_id"
                    ),
                    "case_id": historical_case_id,
                    "source_tweet_id": record.get(
                        "source_tweet_id"
                    ),
                    "customer_problem": record.get(
                        "customer_problem",
                        ""
                    ),
                    "conversation_context": record.get(
                        "conversation_context",
                        ""
                    ),
                    "apple_response": record.get(
                        "apple_response",
                        ""
                    ),
                    "follow_up": record.get(
                        "follow_up",
                        ""
                    ),
                    "resolution_signal": record.get(
                        "resolution_signal",
                        "UNKNOWN"
                    ),
                    "resolution_confidence": record.get(
                        "resolution_confidence",
                        0.0
                    ),
                    "resolution_reason": record.get(
                        "resolution_reason",
                        ""
                    ),
                }
            )

            if len(retrieved) >= TOP_K:
                break

        output_records.append(
            {
                "case_id": golden_case_id,
                "source_tweet_id": str(
                    row["source_tweet_id"]
                ),
                "query": row.get(
                    "customer_message",
                    ""
                ),
                "gold_intent": row.get(
                    "annotator_intent",
                    ""
                ),
                "gold_action": row.get(
                    "expected_action",
                    ""
                ),
                "mapped_historical_case_ids": sorted(
                    excluded_cases
                ),
                "retrieved": retrieved,
                "leakage_check": {
                    "passed": (
                        len(retrieved) == 0
                        or all(
                            item["case_id"]
                            not in excluded_cases
                            for item in retrieved
                        )
                    ),
                },
            }
        )

    runtime = time.time() - start

    # -------------------------------------------------------------
    # Final leakage audit
    # -------------------------------------------------------------

    exact_excluded_hits = 0

    for result in output_records:

        excluded = set(
            result[
                "mapped_historical_case_ids"
            ]
        )

        for item in result["retrieved"]:

            if (
                item["case_id"]
                in excluded
            ):
                exact_excluded_hits += 1

    # -------------------------------------------------------------
    # Write results
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in output_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    successful = sum(
        bool(
            record["retrieved"]
        )
        for record in output_records
    )

    top1_scores = [
        record["retrieved"][0]["similarity"]
        for record in output_records
        if record["retrieved"]
    ]

    print()
    print("=" * 70)
    print("SEMANTIC BASELINE COMPLETE")
    print("=" * 70)

    print(
        f"Successful queries         : "
        f"{successful}"
    )

    print(
        f"Empty queries              : "
        f"{len(output_records) - successful}"
    )

    print(
        f"Runtime                    : "
        f"{runtime:.1f}s"
    )

    print(
        f"Leakage violations         : "
        f"{leakage_violations}"
    )

    print(
        f"Exact excluded-case hits   : "
        f"{exact_excluded_hits}"
    )

    print()
    print("SIMILARITY DIAGNOSTICS")
    print("-" * 70)

    if top1_scores:

        print(
            f"Top-1 similarity mean      : "
            f"{np.mean(top1_scores):.4f}"
        )

        print(
            f"Top-1 similarity median    : "
            f"{np.median(top1_scores):.4f}"
        )

        print(
            f"Top-1 similarity P90       : "
            f"{np.percentile(top1_scores, 90):.4f}"
        )

        print(
            f"Top-1 similarity P10       : "
            f"{np.percentile(top1_scores, 10):.4f}"
        )

    print()
    print("LEAKAGE SANITY CHECK")
    print("-" * 70)

    if exact_excluded_hits == 0:

        print(
            "PASS: no excluded historical case "
            "was retrieved."
        )

    else:

        print(
            "FAIL: excluded historical cases "
            "were retrieved."
        )

    print()
    print(
        "Results written to:"
    )

    print(OUTPUT_PATH)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Cosine similarity is a retrieval diagnostic, "
        "not retrieval accuracy."
    )


if __name__ == "__main__":
    main()

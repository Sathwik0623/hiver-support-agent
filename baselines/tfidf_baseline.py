"""
TF-IDF retrieval baseline for the Hiver AI Support Agent.

Purpose:
    Retrieve historically similar AppleSupport cases using TF-IDF cosine similarity.

Evaluation design:
    - Golden-set cases are excluded at the HISTORICAL CASE level.
    - Each golden source_tweet_id is mapped to its historical evidence case_id.
    - The entire historical case is removed from the retrieval corpus.
    - Retrieved results are checked again for leakage.
    - Similarity is reported as a diagnostic, not as retrieval accuracy.

This is a retrieval baseline only.
It does NOT:
    - classify intent
    - generate an LLM response
    - decide escalation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EVIDENCE = (
    PROJECT_ROOT / "data" / "processed" / "historical_evidence.jsonl"
)

DEFAULT_GOLDEN = (
    PROJECT_ROOT / "data" / "golden" / "apple_support_golden_set.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "tfidf_retrieval_results.jsonl"
)


def normalize_text(text: str) -> str:
    """
    Normalize text for retrieval comparison.

    URLs and @mentions are removed because they are generally not useful
    for matching the underlying support problem.
    """
    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_evidence(path: Path) -> list[dict]:
    """Load historical evidence records from JSONL."""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            records.append(record)

    return records


def load_golden_set(path: Path) -> list[dict]:
    """Load golden evaluation records from CSV."""
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_query_text(record: dict) -> str:
    """
    Extract the customer problem used as the retrieval query.

    Prefer customer_message from the golden set.
    """
    text = (
        record.get("customer_message")
        or record.get("customer_problem")
        or record.get("text")
        or ""
    )

    return normalize_text(text)


def get_evidence_text(record: dict) -> str:
    """
    Build the searchable representation of a historical case.

    The customer problem is intentionally repeated once so that it has
    more influence than surrounding conversation context.

    The Apple response is excluded from the retrieval representation.
    """
    problem = record.get("customer_problem", "")
    context = record.get("conversation_context", "")

    text = f"{problem} {problem} {context}"

    return normalize_text(text)


def build_vectorizer() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=100_000,
        sublinear_tf=True,
    )


def build_source_to_case_map(
    evidence: list[dict],
) -> dict[str, set[str]]:
    """
    Map historical source_tweet_id -> case_id.

    This is the key bridge between the golden set and the historical
    evidence corpus.
    """
    mapping: dict[str, set[str]] = {}

    for record in evidence:
        source_tweet_id = str(record.get("source_tweet_id", "")).strip()
        case_id = str(record.get("case_id", "")).strip()

        if not source_tweet_id or not case_id:
            continue

        mapping.setdefault(source_tweet_id, set()).add(case_id)

    return mapping


def map_golden_cases_to_historical_cases(
    golden: list[dict],
    source_to_case: dict[str, set[str]],
) -> tuple[dict[str, set[str]], list[str]]:
    """
    Map each golden case to the historical case(s) containing its
    source_tweet_id.

    Returns:
        golden_to_historical:
            golden case_id -> historical case_ids

        unmapped_golden:
            golden case_ids for which no historical case was found
    """
    golden_to_historical: dict[str, set[str]] = {}
    unmapped_golden: list[str] = []

    for record in golden:
        golden_case_id = str(record.get("case_id", "")).strip()
        source_tweet_id = str(
            record.get("source_tweet_id", "")
        ).strip()

        if not golden_case_id:
            continue

        historical_cases = source_to_case.get(
            source_tweet_id,
            set(),
        )

        if historical_cases:
            golden_to_historical[golden_case_id] = historical_cases
        else:
            unmapped_golden.append(golden_case_id)

    return golden_to_historical, unmapped_golden


def retrieve(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    evidence: list[dict],
    top_k: int,
) -> list[dict]:
    """Retrieve the top-k most similar historical cases."""

    if not query:
        return []

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix,
    ).ravel()

    if len(scores) == 0:
        return []

    top_k = min(top_k, len(scores))

    candidate_indices = np.argpartition(
        -scores,
        top_k - 1,
    )[:top_k]

    candidate_indices = candidate_indices[
        np.argsort(-scores[candidate_indices])
    ]

    results = []

    for rank, index in enumerate(
        candidate_indices,
        start=1,
    ):
        record = evidence[index]

        results.append(
            {
                "rank": rank,
                "similarity": round(
                    float(scores[index]),
                    6,
                ),
                "evidence_id": record.get("evidence_id"),
                "case_id": record.get("case_id"),
                "source_tweet_id": record.get(
                    "source_tweet_id"
                ),
                "customer_problem": record.get(
                    "customer_problem"
                ),
                "conversation_context": record.get(
                    "conversation_context"
                ),
                "apple_response": record.get(
                    "apple_response"
                ),
                "follow_up": record.get(
                    "follow_up"
                ),
                "resolution_signal": record.get(
                    "resolution_signal"
                ),
                "resolution_confidence": record.get(
                    "resolution_confidence"
                ),
                "resolution_reason": record.get(
                    "resolution_reason"
                ),
            }
        )

    return results


def compute_retrieval_metrics(
    results: list[dict],
    golden_to_historical: set[str],
) -> dict:
    """
    Compute simple case-level retrieval metrics.

    A retrieved case is considered a direct historical-case match only
    if its case_id belongs to the golden example's mapped historical
    case IDs.

    Note:
        This is NOT a semantic relevance judgment. It measures whether
        the retriever recovered the exact historical case associated
        with the golden example. Because that case is deliberately
        excluded from the corpus, these values are expected to be zero
        in a leakage-safe evaluation.

    The metrics are therefore included primarily as a leakage sanity
    check, not as the primary quality metric.
    """
    ranks = []

    for result in results:
        case_id = str(result.get("case_id", ""))

        if case_id in golden_to_historical:
            ranks.append(result["rank"])

    if ranks:
        reciprocal_ranks = [
            1.0 / rank
            for rank in ranks
        ]

        return {
            "exact_case_recall_at_1": 1.0
            if any(rank <= 1 for rank in ranks)
            else 0.0,
            "exact_case_recall_at_3": 1.0
            if any(rank <= 3 for rank in ranks)
            else 0.0,
            "exact_case_recall_at_5": 1.0
            if any(rank <= 5 for rank in ranks)
            else 0.0,
            "exact_case_mrr": max(
                reciprocal_ranks
            ),
        }

    return {
        "exact_case_recall_at_1": 0.0,
        "exact_case_recall_at_3": 0.0,
        "exact_case_recall_at_5": 0.0,
        "exact_case_mrr": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe TF-IDF retrieval baseline."
    )

    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )

    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    print("=" * 70)
    print("LEAKAGE-SAFE TF-IDF RETRIEVAL BASELINE")
    print("=" * 70)

    print(f"Evidence : {args.evidence}")
    print(f"Golden   : {args.golden}")
    print(f"Top-K    : {args.top_k}")
    print()

    # ------------------------------------------------------------------
    # Load evidence
    # ------------------------------------------------------------------

    evidence = load_evidence(args.evidence)

    print(
        f"Historical evidence loaded : "
        f"{len(evidence):,}"
    )

    # ------------------------------------------------------------------
    # Load golden set
    # ------------------------------------------------------------------

    golden = load_golden_set(args.golden)

    print(
        f"Golden records loaded      : "
        f"{len(golden):,}"
    )

    if not golden:
        raise ValueError(
            "Golden set is empty."
        )

    # ------------------------------------------------------------------
    # Map golden examples to historical cases
    # ------------------------------------------------------------------

    source_to_case = build_source_to_case_map(
        evidence
    )

    (
        golden_to_historical,
        unmapped_golden,
    ) = map_golden_cases_to_historical_cases(
        golden,
        source_to_case,
    )

    excluded_case_ids = set()

    for case_ids in golden_to_historical.values():
        excluded_case_ids.update(case_ids)

    print()
    print(
        "LEAKAGE MAPPING"
    )
    print("-" * 70)

    print(
        f"Golden cases mapped       : "
        f"{len(golden_to_historical):,}"
    )

    print(
        f"Golden cases unmapped     : "
        f"{len(unmapped_golden):,}"
    )

    print(
        f"Historical cases excluded: "
        f"{len(excluded_case_ids):,}"
    )

    if unmapped_golden:
        print()
        print(
            "WARNING: Some golden cases could not be mapped "
            "to historical evidence."
        )

        print(
            "Unmapped examples:"
        )

        for case_id in unmapped_golden[:20]:
            print(
                f"  - {case_id}"
            )

        if len(unmapped_golden) > 20:
            print(
                f"  ... and "
                f"{len(unmapped_golden) - 20:,} more"
            )

    # ------------------------------------------------------------------
    # Remove entire historical cases
    # ------------------------------------------------------------------

    retrieval_evidence = [
        record
        for record in evidence
        if record.get("case_id")
        not in excluded_case_ids
    ]

    removed_records = (
        len(evidence)
        - len(retrieval_evidence)
    )

    print()
    print(
        "RETRIEVAL CORPUS"
    )
    print("-" * 70)

    print(
        f"Original evidence records : "
        f"{len(evidence):,}"
    )

    print(
        f"Evidence records removed  : "
        f"{removed_records:,}"
    )

    print(
        f"Retrieval corpus           : "
        f"{len(retrieval_evidence):,}"
    )

    if not retrieval_evidence:
        raise ValueError(
            "Retrieval corpus is empty after leakage exclusion."
        )

    # ------------------------------------------------------------------
    # Build TF-IDF corpus
    # ------------------------------------------------------------------

    print()
    print(
        "Building TF-IDF matrix..."
    )

    corpus = [
        get_evidence_text(record)
        for record in retrieval_evidence
    ]

    vectorizer = build_vectorizer()

    matrix = vectorizer.fit_transform(
        corpus
    )

    print(
        f"Matrix shape               : "
        f"{matrix.shape}"
    )

    print(
        f"Vocabulary size            : "
        f"{len(vectorizer.vocabulary_):,}"
    )

    # ------------------------------------------------------------------
    # Retrieve for every golden query
    # ------------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful_queries = 0
    empty_queries = 0

    similarity_values = []

    leakage_violations = []

    exact_case_hits = []

    top1_case_ids = Counter()

    print()
    print(
        "Running retrieval..."
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for golden_record in golden:

            golden_case_id = str(
                golden_record.get(
                    "case_id",
                    "",
                )
            ).strip()

            query = get_query_text(
                golden_record
            )

            if not query:

                empty_queries += 1

                result = {
                    "case_id": golden_case_id,
                    "source_tweet_id": golden_record.get(
                        "source_tweet_id"
                    ),
                    "query": "",
                    "gold_intent": golden_record.get(
                        "annotator_intent"
                    ),
                    "gold_action": golden_record.get(
                        "expected_action"
                    ),
                    "retrieved": [],
                    "leakage_check": {
                        "passed": True,
                        "violations": [],
                    },
                }

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                continue

            retrieved = retrieve(
                query=query,
                vectorizer=vectorizer,
                matrix=matrix,
                evidence=retrieval_evidence,
                top_k=args.top_k,
            )

            if retrieved:
                successful_queries += 1

                similarity_values.append(
                    retrieved[0]["similarity"]
                )

                top1_case_ids[
                    str(
                        retrieved[0].get(
                            "case_id",
                            "",
                        )
                    )
                ] += 1

            # ----------------------------------------------------------
            # Leakage audit
            # ----------------------------------------------------------

            forbidden_case_ids = (
                golden_to_historical.get(
                    golden_case_id,
                    set(),
                )
            )

            violations = []

            for item in retrieved:
                if item.get("case_id") in forbidden_case_ids:
                    violations.append(
                        {
                            "retrieved_case_id": item.get(
                                "case_id"
                            ),
                            "rank": item.get(
                                "rank"
                            ),
                        }
                    )

            if violations:
                leakage_violations.append(
                    golden_case_id
                )

            # ----------------------------------------------------------
            # Exact historical-case hit
            # ----------------------------------------------------------

            exact_case_rank = None

            for item in retrieved:

                if item.get("case_id") in forbidden_case_ids:
                    exact_case_rank = item.get(
                        "rank"
                    )
                    exact_case_hits.append(
                        golden_case_id
                    )
                    break

            result = {
                "case_id": golden_case_id,
                "source_tweet_id": golden_record.get(
                    "source_tweet_id"
                ),
                "query": golden_record.get(
                    "customer_message",
                    golden_record.get(
                        "customer_problem",
                        "",
                    ),
                ),
                "gold_intent": golden_record.get(
                    "annotator_intent"
                ),
                "gold_action": golden_record.get(
                    "expected_action"
                ),
                "mapped_historical_case_ids": sorted(
                    forbidden_case_ids
                ),
                "retrieved": retrieved,
                "exact_historical_case_rank": exact_case_rank,
                "leakage_check": {
                    "passed": not bool(
                        violations
                    ),
                    "violations": violations,
                },
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)

    print(
        f"Successful queries         : "
        f"{successful_queries:,}"
    )

    print(
        f"Empty queries              : "
        f"{empty_queries:,}"
    )

    print(
        f"Leakage violations         : "
        f"{len(leakage_violations):,}"
    )

    print(
        f"Exact excluded-case hits  : "
        f"{len(exact_case_hits):,}"
    )

    if similarity_values:

        similarities = np.array(
            similarity_values
        )

        print()
        print(
            "SIMILARITY DIAGNOSTICS"
        )
        print("-" * 70)

        print(
            f"Top-1 similarity mean      : "
            f"{similarities.mean():.4f}"
        )

        print(
            f"Top-1 similarity median    : "
            f"{np.median(similarities):.4f}"
        )

        print(
            f"Top-1 similarity P90       : "
            f"{np.percentile(similarities, 90):.4f}"
        )

        print(
            f"Top-1 similarity P10       : "
            f"{np.percentile(similarities, 10):.4f}"
        )

    print()
    print(
        "LEAKAGE SANITY CHECK"
    )
    print("-" * 70)

    if leakage_violations:
        print(
            "FAIL: leakage detected."
        )

        print(
            "Violating golden cases:"
        )

        for case_id in leakage_violations[:20]:
            print(
                f"  - {case_id}"
            )

        raise RuntimeError(
            "Leakage detected in retrieval results."
        )

    print(
        "PASS: no excluded historical case "
        "was retrieved."
    )

    print()
    print(
        "Results written to:"
    )

    print(args.output)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Similarity is a retrieval diagnostic, "
        "not retrieval accuracy."
    )

    print(
        "The previous contaminated 0.9180 result "
        "must not be used."
    )


if __name__ == "__main__":
    main()

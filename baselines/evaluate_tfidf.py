"""
Evaluate the leakage-safe TF-IDF retrieval baseline.

Metrics:
1. Top-K similarity diagnostics
2. Historical analogue Hit@K using normalized customer problems
3. Resolution-evidence coverage in retrieved results
4. Leakage sanity check

Important:
Historical analogue matching is a proxy metric, NOT ground-truth retrieval accuracy.
The TWCS dataset does not provide manually labeled intent/relevance labels.
"""

from __future__ import annotations
import sys
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_retrieval_results.jsonl"
)

RESULTS_PATH = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else DEFAULT_RESULTS_PATH
)

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


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Conservative normalization used only for approximate
    historical-analogue matching.
    """
    text = text or ""
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter handles
    text = re.sub(r"@\w+", " ", text)

    # Normalize punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def load_golden(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return values[lower] + (
        values[upper] - values[lower]
    ) * fraction


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("TF-IDF RETRIEVAL EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    results = load_jsonl(RESULTS_PATH)
    evidence = load_jsonl(EVIDENCE_PATH)
    golden = load_golden(GOLDEN_PATH)

    print()
    print(f"Retrieval results loaded : {len(results):,}")
    print(f"Historical evidence     : {len(evidence):,}")
    print(f"Golden records          : {len(golden):,}")

    # -------------------------------------------------------------
    # Index historical evidence
    # -------------------------------------------------------------

    evidence_by_id = {
        str(record["evidence_id"]): record
        for record in evidence
    }

    evidence_by_case = defaultdict(list)

    for record in evidence:
        evidence_by_case[
            str(record["case_id"])
        ].append(record)

    # -------------------------------------------------------------
    # Build normalized historical-problem index
    # -------------------------------------------------------------

    problem_to_cases = defaultdict(set)

    for record in evidence:

        problem = normalize_text(
            record.get("customer_problem", "")
        )

        if not problem:
            continue

        problem_to_cases[problem].add(
            str(record["case_id"])
        )

    print(
        f"Historical problem groups : "
        f"{len(problem_to_cases):,}"
    )

    # -------------------------------------------------------------
    # Golden case -> historical case mapping
    #
    # The retrieval baseline already removed these cases.
    # This mapping is used only to ensure we don't count the
    # source case as a historical analogue.
    # -------------------------------------------------------------

    golden_by_case = {
        str(row["case_id"]): row
        for row in golden
    }

    # -------------------------------------------------------------
    # Retrieve top-K results
    # -------------------------------------------------------------

    top1_similarities = []
    top5_similarities = []

    analogue_hits = {
        1: 0,
        3: 0,
        5: 0,
    }

    eligible_analogue_queries = {
        1: 0,
        3: 0,
        5: 0,
    }

    resolution_evidence_hits = {
        1: 0,
        3: 0,
        5: 0,
    }

    resolution_confident_hits = {
        1: 0,
        3: 0,
        5: 0,
    }

    outcome_counts = Counter()

    leakage_violations = 0

    # -----------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------

    for result in results:

        golden_case_id = str(
            result.get("case_id", "")
        )

        golden_record = golden_by_case.get(
            golden_case_id
        )

        if golden_record is None:
            continue

        retrieved = result.get("retrieved", [])

        if not retrieved:
            continue

        # -------------------------------------------------------------
        # This is a valid evaluated query
        # -------------------------------------------------------------

        top1_similarities.append(
            float(
                retrieved[0].get(
                    "similarity",
                    0.0
                )
            )
        )

        top5_similarities.append(
            mean(
                [
                    float(
                        item.get(
                            "similarity",
                            0.0
                        )
                    )
                    for item in retrieved[:5]
                ]
            )
        )

        # -------------------------------------------------------------
        # Historical cases intentionally excluded by the baseline
        # -------------------------------------------------------------

        excluded_case_ids = {
            str(case_id)
            for case_id in result.get(
                "mapped_historical_case_ids",
                []
            )
        }

        # -------------------------------------------------------------
        # Leakage check
        # -------------------------------------------------------------

        for item in retrieved:

            retrieved_case_id = str(
                item.get("case_id", "")
            )

            if retrieved_case_id in excluded_case_ids:
                leakage_violations += 1

        # -------------------------------------------------------------
        # Golden problem
        # -------------------------------------------------------------

        golden_problem = normalize_text(
            golden_record.get(
                "customer_message",
                ""
            )
        )

        # -------------------------------------------------------------
        # Find historical analogues.
        #
        # IMPORTANT:
        # We build the analogue set independently from the excluded
        # source case. If another historical case has the exact same
        # normalized customer problem, it is considered a proxy
        # analogue.
        # -------------------------------------------------------------

        analogue_cases = set()

        for historical_case_id, records in evidence_by_case.items():

            if historical_case_id in excluded_case_ids:
                continue

            for record in records:

                historical_problem = normalize_text(
                    record.get(
                        "customer_problem",
                        ""
                    )
                )

                if (
                    historical_problem
                    and historical_problem == golden_problem
                ):
                    analogue_cases.add(
                        historical_case_id
                    )
                    break

        # -------------------------------------------------------------
        # Historical analogue eligibility
        # -------------------------------------------------------------

        if analogue_cases:

            for k in (1, 3, 5):

                eligible_analogue_queries[k] += 1

                retrieved_case_ids = {
                    str(
                        item.get(
                            "case_id",
                            ""
                        )
                    )
                    for item in retrieved[:k]
                }

                if (
                    retrieved_case_ids
                    & analogue_cases
                ):
                    analogue_hits[k] += 1

        # -------------------------------------------------------------
        # Evaluate retrieved evidence
        # -------------------------------------------------------------

        for k in (1, 3, 5):

            top_k = retrieved[:k]

            has_resolution_evidence = False
            has_confident_resolution = False

            for item in top_k:

                signal = str(
                    item.get(
                        "resolution_signal",
                        "UNKNOWN"
                    )
                ).upper()

                confidence = float(
                    item.get(
                        "resolution_confidence",
                        0.0
                    ) or 0.0
                )

                outcome_counts[signal] += 1

                if signal == "RESOLVED":

                    has_resolution_evidence = True

                    if confidence >= 0.70:
                        has_confident_resolution = True

            if has_resolution_evidence:
                resolution_evidence_hits[k] += 1

            if has_confident_resolution:
                resolution_confident_hits[k] += 1

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------

    total = len(top1_similarities)

    print()
    print("=" * 70)
    print("SIMILARITY DIAGNOSTICS")
    print("=" * 70)

    print(
        f"Queries evaluated       : {total:,}"
    )

    print(
        f"Top-1 similarity mean   : "
        f"{mean(top1_similarities):.4f}"
    )

    print(
        f"Top-1 similarity median : "
        f"{statistics.median(top1_similarities):.4f}"
    )

    print(
        f"Top-1 similarity P90    : "
        f"{percentile(top1_similarities, 0.90):.4f}"
    )

    print(
        f"Top-5 similarity mean   : "
        f"{mean(top5_similarities):.4f}"
    )

    # -----------------------------------------------------------------
    # Historical analogue metrics
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("HISTORICAL ANALOGUE RETRIEVAL")
    print("=" * 70)

    print(
        "Metric definition:"
    )

    print(
        "A hit means the retrieved case contains the same "
        "normalized customer problem as the golden case."
    )

    print(
        "This is a proxy metric, NOT ground-truth relevance."
    )

    for k in (1, 3, 5):

        eligible = eligible_analogue_queries[k]

        if eligible:
            score = (
                analogue_hits[k] / eligible
            )
        else:
            score = 0.0

        print(
            f"Analogue Hit@{k:<2}      : "
            f"{analogue_hits[k]:>3}/{eligible:<3} "
            f"({score:.2%})"
        )

    # -----------------------------------------------------------------
    # Resolution evidence
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESOLUTION EVIDENCE COVERAGE")
    print("=" * 70)

    for k in (1, 3, 5):

        if total:
            score = (
                resolution_evidence_hits[k]
                / total
            )
        else:
            score = 0.0

        confident_score = (
            resolution_confident_hits[k]
            / total
            if total
            else 0.0
        )

        print(
            f"Top-{k} contains RESOLVED case : "
            f"{resolution_evidence_hits[k]:>3}/{total:<3} "
            f"({score:.2%})"
        )

        print(
            f"Top-{k} contains confident "
            f"resolution                  : "
            f"{resolution_confident_hits[k]:>3}/{total:<3} "
            f"({confident_score:.2%})"
        )

    # -----------------------------------------------------------------
    # Outcome distribution
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVED EVIDENCE OUTCOMES")
    print("=" * 70)

    for signal, count in outcome_counts.most_common():

        print(
            f"{signal:<15} : {count:,}"
        )

    # -----------------------------------------------------------------
    # Leakage
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("LEAKAGE SANITY CHECK")
    print("=" * 70)

    print(
        f"Leakage violations : {leakage_violations}"
    )

    if leakage_violations == 0:
        print(
            "PASS: no golden case was retrieved."
        )
    else:
        print(
            "FAIL: retrieval leakage detected."
        )

    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        "Do NOT report cosine similarity as retrieval accuracy."
    )

    print(
        "Analogue Hit@K is only a proxy because normalized "
        "text equality does not guarantee semantic relevance."
    )

    print(
        "Resolution evidence coverage measures whether retrieval "
        "surfaces historically resolved examples; it does not "
        "prove that the retrieved response is correct for the "
        "current customer."
    )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()

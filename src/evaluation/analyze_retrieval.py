"""
Unified retrieval analysis for the Hiver Support Agent take-home.

Inputs:
    data/processed/tfidf_retrieval_results.jsonl
    data/processed/semantic_retrieval_results.jsonl
    data/processed/semantic_llm_judgments.jsonl
    data/golden/apple_support_golden_set.csv

Outputs:
    data/processed/retrieval_comparison.json
    data/processed/retrieval_comparison.csv

Purpose:
    Compare lexical TF-IDF retrieval against semantic retrieval and
    incorporate the available LLM-as-judge sample.

Important:
    - Retrieval similarity is NOT treated as accuracy.
    - "Resolved evidence coverage" is a proxy metric, not response correctness.
    - LLM judgments are qualitative evidence only.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TFIDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_retrieval_results.jsonl"
)

SEMANTIC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_retrieval_results.jsonl"
)

LLM_JUDGMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_llm_judgments.jsonl"
)

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "apple_support_golden_set.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_comparison.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_comparison.csv"
)


# ---------------------------------------------------------------------
# GENERIC LOADERS
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[WARNING] Could not parse {path.name} "
                    f"line {line_number}: {exc}"
                )

    return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# NUMERIC HELPERS
# ---------------------------------------------------------------------

def percentile(values: List[float], p: float) -> Optional[float]:
    """
    Calculate a percentile using linear interpolation.

    p should be between 0 and 100.
    """
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    rank = (p / 100.0) * (len(values) - 1)

    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)

    weight = rank - lower

    return values[lower] + weight * (values[upper] - values[lower])


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float safely."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """Convert a value to int safely."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# SIMILARITY STATISTICS
# ---------------------------------------------------------------------

def similarity_stats(
    results: List[Dict[str, Any]]
) -> Dict[str, Optional[float]]:
    """Calculate retrieval similarity statistics."""
    similarities: List[float] = []

    for row in results:
        similarity = safe_float(row.get("top1_similarity"))

        # Some retrieval outputs may not have a top1_similarity field.
        # Fall back to the first retrieved item's similarity.
        if similarity is None:
            retrieved = row.get("retrieved", [])

            if retrieved:
                similarity = safe_float(
                    retrieved[0].get("similarity")
                )

        if similarity is not None:
            similarities.append(similarity)

    if not similarities:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": statistics.mean(similarities),
        "median": statistics.median(similarities),
        "p10": percentile(similarities, 10),
        "p90": percentile(similarities, 90),
        "min": min(similarities),
        "max": max(similarities),
    }


# ---------------------------------------------------------------------
# RETRIEVED EVIDENCE HELPERS
# ---------------------------------------------------------------------

def get_retrieved(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return retrieved evidence.

    Current expected schema:
        {
            "retrieved": [...]
        }

    Older/alternate outputs may use "results", so support that too.
    """
    retrieved = row.get("retrieved")

    if isinstance(retrieved, list):
        return retrieved

    results = row.get("results")

    if isinstance(results, list):
        return results

    return []


def has_resolved_evidence(
    row: Dict[str, Any],
    k: int,
) -> bool:
    """
    Check whether at least one of the top-k retrieved cases has
    resolution_signal == RESOLVED.
    """
    retrieved = get_retrieved(row)[:k]

    for item in retrieved:
        if item.get("resolution_signal") == "RESOLVED":
            return True

    return False


def resolved_evidence_coverage(
    results: List[Dict[str, Any]],
    k: int,
) -> float:
    """Return percentage of queries with resolved evidence in top-k."""
    if not results:
        return 0.0

    hits = sum(
        1
        for row in results
        if has_resolved_evidence(row, k)
    )

    return (hits / len(results)) * 100.0


def outcome_distribution(
    results: List[Dict[str, Any]],
    k: int = 5,
) -> Counter:
    """
    Count resolution outcomes across retrieved evidence.

    This counts retrieved evidence rows, not unique cases.
    """
    counter: Counter = Counter()

    for row in results:
        for item in get_retrieved(row)[:k]:
            outcome = item.get(
                "resolution_signal",
                "UNKNOWN",
            )

            counter[outcome] += 1

    return counter


# ---------------------------------------------------------------------
# LEAKAGE / CASE MAPPING
# ---------------------------------------------------------------------

def get_mapped_historical_cases(
    row: Dict[str, Any]
) -> set[str]:
    """
    Return historical case IDs excluded/mapped for the golden query.

    Expected field:
        mapped_historical_case_ids

    Used to identify whether a retrieved item belongs to the
    source historical case and therefore must not be considered
    a valid retrieval hit.
    """
    mapped = row.get("mapped_historical_case_ids", [])

    if not isinstance(mapped, list):
        return set()

    return {
        str(case_id)
        for case_id in mapped
        if case_id is not None
    }


def count_leakage_violations(
    results: List[Dict[str, Any]]
) -> int:
    """Count retrieval results containing an excluded historical case."""
    violations = 0

    for row in results:
        excluded_cases = get_mapped_historical_cases(row)

        if not excluded_cases:
            continue

        for item in get_retrieved(row):
            retrieved_case_id = item.get("case_id")

            if (
                retrieved_case_id is not None
                and str(retrieved_case_id) in excluded_cases
            ):
                violations += 1
                break

    return violations


# ---------------------------------------------------------------------
# ANALOGUE HIT
# ---------------------------------------------------------------------

def analogue_hit(
    row: Dict[str, Any],
    k: int,
) -> bool:
    """
    Determine whether the expected historical analogue appears
    in top-k.

    This metric is only meaningful where the golden set explicitly
    maps to a historical analogue.
    """
    mapped_cases = get_mapped_historical_cases(row)

    if not mapped_cases:
        return False

    for item in get_retrieved(row)[:k]:
        retrieved_case_id = item.get("case_id")

        if (
            retrieved_case_id is not None
            and str(retrieved_case_id) in mapped_cases
        ):
            return True

    return False


def analogue_metrics(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate analogue Hit@K only for cases with a known mapped
    historical case.

    Important:
        This is intentionally reported as a secondary diagnostic
        because only a small subset of the golden set may have a
        clearly defined historical analogue.
    """
    eligible = [
        row
        for row in results
        if get_mapped_historical_cases(row)
    ]

    output: Dict[str, Any] = {
        "eligible_cases": len(eligible),
        "total_cases": len(results),
    }

    for k in (1, 3, 5):
        hits = sum(
            1
            for row in eligible
            if analogue_hit(row, k)
        )

        output[f"hit_at_{k}"] = hits
        output[f"hit_at_{k}_pct"] = (
            (hits / len(eligible)) * 100.0
            if eligible
            else 0.0
        )

    return output


# ---------------------------------------------------------------------
# LLM JUDGE HANDLING
# ---------------------------------------------------------------------

def build_llm_judgment_map(
    llm_rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Build:

        case_id -> flattened judge information

    Current LLM schema:

        {
            "case_id": "...",
            "retrieval_rank": 1,
            "historical_case_id": "...",
            "historical_evidence_id": "...",
            "retrieval_similarity": 0.907284,
            "judge": {
                "problem_relevance": 4,
                "resolution_usefulness": 2,
                "context_compatibility": 4,
                "grounding_value": 3,
                "overall_score": 3,
                "decision": "USE_WITH_CAUTION",
                "reason": "..."
            },
            "judge_model": "gemini-3.5-flash"
        }
    """
    judgment_map: Dict[str, Dict[str, Any]] = {}

    for row in llm_rows:
        case_id = row.get("case_id")

        if not case_id:
            continue

        judge = row.get("judge", {})

        if not isinstance(judge, dict):
            continue

        judgment_map[str(case_id)] = {
            "overall_score": safe_int(
                judge.get("overall_score")
            ),
            "decision": judge.get("decision"),
            "problem_relevance": safe_int(
                judge.get("problem_relevance")
            ),
            "resolution_usefulness": safe_int(
                judge.get("resolution_usefulness")
            ),
            "context_compatibility": safe_int(
                judge.get("context_compatibility")
            ),
            "grounding_value": safe_int(
                judge.get("grounding_value")
            ),
            "reason": judge.get("reason", ""),
            "retrieval_similarity": safe_float(
                row.get("retrieval_similarity")
            ),
            "historical_case_id": row.get(
                "historical_case_id"
            ),
            "historical_evidence_id": row.get(
                "historical_evidence_id"
            ),
            "judge_model": row.get(
                "judge_model"
            ),
        }

    return judgment_map


def llm_judge_summary(
    judgment_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarize available LLM judge results."""
    scores = [
        value["overall_score"]
        for value in judgment_map.values()
        if value.get("overall_score") is not None
    ]

    decisions = Counter(
        value["decision"]
        for value in judgment_map.values()
        if value.get("decision")
    )

    dimensions = [
        "problem_relevance",
        "resolution_usefulness",
        "context_compatibility",
        "grounding_value",
        "overall_score",
    ]

    dimension_means: Dict[str, Optional[float]] = {}

    for dimension in dimensions:
        values = [
            value[dimension]
            for value in judgment_map.values()
            if value.get(dimension) is not None
        ]

        dimension_means[dimension] = (
            statistics.mean(values)
            if values
            else None
        )

    return {
        "judged_cases": len(judgment_map),
        "scores": dict(Counter(scores)),
        "decisions": dict(decisions),
        "dimension_means": dimension_means,
    }


# ---------------------------------------------------------------------
# QUALITATIVE SLICES
# ---------------------------------------------------------------------

def get_top1_similarity(row: Dict[str, Any]) -> Optional[float]:
    """Get top-1 similarity from a retrieval row."""
    similarity = safe_float(row.get("top1_similarity"))

    if similarity is not None:
        return similarity

    retrieved = get_retrieved(row)

    if retrieved:
        return safe_float(
            retrieved[0].get("similarity")
        )

    return None


def get_top1_evidence(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return top-1 retrieved evidence."""
    retrieved = get_retrieved(row)

    if not retrieved:
        return None

    return retrieved[0]


def high_similarity_poor_evidence(
    semantic_results: List[Dict[str, Any]],
    llm_map: Dict[str, Dict[str, Any]],
    similarity_threshold: float = 0.80,
    max_score: int = 1,
) -> List[Dict[str, Any]]:
    """
    Find high-similarity cases where the LLM judge considered the
    evidence poor.

    Criteria:
        similarity >= threshold
        overall_score <= max_score
    """
    output: List[Dict[str, Any]] = []

    for row in semantic_results:
        case_id = str(row.get("case_id", ""))

        judgment = llm_map.get(case_id)

        if not judgment:
            continue

        similarity = get_top1_similarity(row)
        score = judgment.get("overall_score")

        if similarity is None or score is None:
            continue

        if (
            similarity >= similarity_threshold
            and score <= max_score
        ):
            top1 = get_top1_evidence(row) or {}

            output.append({
                "case_id": case_id,
                "similarity": similarity,
                "overall_score": score,
                "decision": judgment.get("decision"),
                "resolution_usefulness": judgment.get(
                    "resolution_usefulness"
                ),
                "grounding_value": judgment.get(
                    "grounding_value"
                ),
                "reason": judgment.get("reason", ""),
                "historical_case_id": top1.get(
                    "case_id"
                ),
                "resolution_signal": top1.get(
                    "resolution_signal"
                ),
                "historical_response": top1.get(
                    "apple_response",
                    ""
                ),
            })

    output.sort(
        key=lambda x: x["similarity"],
        reverse=True,
    )

    return output


def lower_similarity_strong_evidence(
    semantic_results: List[Dict[str, Any]],
    llm_map: Dict[str, Dict[str, Any]],
    similarity_threshold: float = 0.75,
    min_score: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find lower-similarity cases where the LLM judge found useful
    evidence.

    Criteria:
        similarity < threshold
        overall_score >= min_score
    """
    output: List[Dict[str, Any]] = []

    for row in semantic_results:
        case_id = str(row.get("case_id", ""))

        judgment = llm_map.get(case_id)

        if not judgment:
            continue

        similarity = get_top1_similarity(row)
        score = judgment.get("overall_score")

        if similarity is None or score is None:
            continue

        if (
            similarity < similarity_threshold
            and score >= min_score
        ):
            top1 = get_top1_evidence(row) or {}

            output.append({
                "case_id": case_id,
                "similarity": similarity,
                "overall_score": score,
                "decision": judgment.get("decision"),
                "resolution_usefulness": judgment.get(
                    "resolution_usefulness"
                ),
                "grounding_value": judgment.get(
                    "grounding_value"
                ),
                "reason": judgment.get("reason", ""),
                "historical_case_id": top1.get(
                    "case_id"
                ),
                "resolution_signal": top1.get(
                    "resolution_signal"
                ),
                "historical_response": top1.get(
                    "apple_response",
                    ""
                ),
            })

    output.sort(
        key=lambda x: x["similarity"]
    )

    return output


# ---------------------------------------------------------------------
# PER-CASE COMPARISON TABLE
# ---------------------------------------------------------------------

def build_comparison_rows(
    tfidf_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    golden_rows: List[Dict[str, Any]],
    llm_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build one comparison row per golden case.
    """
    tfidf_map = {
        str(row.get("case_id")): row
        for row in tfidf_results
        if row.get("case_id")
    }

    semantic_map = {
        str(row.get("case_id")): row
        for row in semantic_results
        if row.get("case_id")
    }

    golden_map = {
        str(row.get("case_id")): row
        for row in golden_rows
        if row.get("case_id")
    }

    all_case_ids = sorted(
        set(golden_map)
        | set(tfidf_map)
        | set(semantic_map)
    )

    rows: List[Dict[str, Any]] = []

    for case_id in all_case_ids:
        golden = golden_map.get(case_id, {})
        tfidf = tfidf_map.get(case_id, {})
        semantic = semantic_map.get(case_id, {})
        judgment = llm_map.get(case_id, {})

        tfidf_top1 = get_top1_evidence(tfidf) or {}
        semantic_top1 = get_top1_evidence(semantic) or {}

        rows.append({
            "case_id": case_id,

            "gold_intent": golden.get(
                "annotator_intent",
                golden.get("gold_intent", "")
            ),

            "gold_state": golden.get(
                "conversation_state",
                ""
            ),

            "gold_action": golden.get(
                "expected_action",
                golden.get("gold_action", "")
            ),

            # ---------------------------------------------------------
            # TF-IDF
            # ---------------------------------------------------------

            "tfidf_top1_similarity": get_top1_similarity(
                tfidf
            ),

            "tfidf_resolved_top1": has_resolved_evidence(
                tfidf,
                1,
            ),

            "tfidf_resolved_top3": has_resolved_evidence(
                tfidf,
                3,
            ),

            "tfidf_resolved_top5": has_resolved_evidence(
                tfidf,
                5,
            ),

            "tfidf_top1_outcome": tfidf_top1.get(
                "resolution_signal",
                "",
            ),

            "tfidf_top1_case_id": tfidf_top1.get(
                "case_id",
                "",
            ),

            # ---------------------------------------------------------
            # Semantic
            # ---------------------------------------------------------

            "semantic_top1_similarity": get_top1_similarity(
                semantic
            ),

            "semantic_resolved_top1": has_resolved_evidence(
                semantic,
                1,
            ),

            "semantic_resolved_top3": has_resolved_evidence(
                semantic,
                3,
            ),

            "semantic_resolved_top5": has_resolved_evidence(
                semantic,
                5,
            ),

            "semantic_top1_outcome": semantic_top1.get(
                "resolution_signal",
                "",
            ),

            "semantic_top1_case_id": semantic_top1.get(
                "case_id",
                "",
            ),

            # ---------------------------------------------------------
            # LLM judge
            # ---------------------------------------------------------

            "llm_judged": bool(judgment),

            "llm_overall_score": judgment.get(
                "overall_score"
            ),

            "llm_decision": judgment.get(
                "decision",
                "",
            ),

            "llm_problem_relevance": judgment.get(
                "problem_relevance"
            ),

            "llm_resolution_usefulness": judgment.get(
                "resolution_usefulness"
            ),

            "llm_context_compatibility": judgment.get(
                "context_compatibility"
            ),

            "llm_grounding_value": judgment.get(
                "grounding_value"
            ),

            "llm_reason": judgment.get(
                "reason",
                "",
            ),
        })

    return rows


# ---------------------------------------------------------------------
# SERIALIZATION
# ---------------------------------------------------------------------

def clean_for_json(value: Any) -> Any:
    """
    Convert sets/tuples and other values into JSON-safe objects.
    """
    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, dict):
        return {
            key: clean_for_json(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            clean_for_json(item)
            for item in value
        ]

    return value


def write_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    """Write analysis JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            clean_for_json(data),
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    """Write per-case comparison CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------
# PRINTING
# ---------------------------------------------------------------------

def print_similarity_stats(
    name: str,
    stats: Dict[str, Optional[float]],
) -> None:
    """Print similarity statistics."""
    print(
        f"{name:<10} "
        f"mean={stats['mean']:.4f} "
        f"median={stats['median']:.4f} "
        f"P90={stats['p90']:.4f}"
        if stats["mean"] is not None
        else f"{name:<10} no similarity values"
    )


def print_outcome_distribution(
    name: str,
    distribution: Counter,
) -> None:
    """Print outcome counts."""
    print(f"{name}:")

    for outcome, count in distribution.most_common():
        print(
            f"  {outcome:<15} {count}"
        )


def print_qualitative_cases(
    title: str,
    cases: List[Dict[str, Any]],
) -> None:
    """Print qualitative examples."""
    print("-" * 72)
    print(title)
    print("-" * 72)

    if not cases:
        print("(none)")
        return

    for case in cases:
        print(
            f"{case['case_id']} | "
            f"sim={case['similarity']:.4f} | "
            f"score={case['overall_score']} | "
            f"{case['decision']}"
        )

        reason = case.get("reason", "").strip()

        if reason:
            print(
                f"  {reason}"
            )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("UNIFIED RETRIEVAL ANALYSIS")
    print("=" * 72)

    # -------------------------------------------------------------
    # Load inputs
    # -------------------------------------------------------------

    tfidf_results = load_jsonl(TFIDF_PATH)
    semantic_results = load_jsonl(SEMANTIC_PATH)
    llm_rows = load_jsonl(LLM_JUDGMENT_PATH)
    golden_rows = load_csv(GOLDEN_PATH)

    print(
        f"TF-IDF results       : {len(tfidf_results)}"
    )

    print(
        f"Semantic results     : {len(semantic_results)}"
    )

    print(
        f"Golden cases         : {len(golden_rows)}"
    )

    print(
        f"LLM judgments        : {len(llm_rows)}"
    )

    # -------------------------------------------------------------
    # LLM judge map
    # -------------------------------------------------------------

    llm_map = build_llm_judgment_map(
        llm_rows
    )

    # -------------------------------------------------------------
    # Similarity
    # -------------------------------------------------------------

    tfidf_similarity = similarity_stats(
        tfidf_results
    )

    semantic_similarity = similarity_stats(
        semantic_results
    )

    print()
    print("-" * 72)
    print("SIMILARITY")
    print("-" * 72)

    print_similarity_stats(
        "TF-IDF",
        tfidf_similarity,
    )

    print_similarity_stats(
        "Semantic",
        semantic_similarity,
    )

    # -------------------------------------------------------------
    # Resolved evidence coverage
    # -------------------------------------------------------------

    print()
    print("-" * 72)
    print("RESOLVED EVIDENCE COVERAGE")
    print("-" * 72)

    coverage: Dict[str, Dict[str, float]] = {}

    for k in (1, 3, 5):
        tfidf_cov = resolved_evidence_coverage(
            tfidf_results,
            k,
        )

        semantic_cov = resolved_evidence_coverage(
            semantic_results,
            k,
        )

        delta = semantic_cov - tfidf_cov

        coverage[f"top_{k}"] = {
            "tfidf_pct": tfidf_cov,
            "semantic_pct": semantic_cov,
            "delta_percentage_points": delta,
        }

        print(
            f"Top-{k}: "
            f"TF-IDF={tfidf_cov:.2f}% | "
            f"Semantic={semantic_cov:.2f}% | "
            f"Delta={delta:+.2f} pp"
        )

    # -------------------------------------------------------------
    # Outcome distributions
    # -------------------------------------------------------------

    print()
    print("-" * 72)
    print("RETRIEVED OUTCOME DISTRIBUTION")
    print("-" * 72)

    tfidf_outcomes = outcome_distribution(
        tfidf_results,
        5,
    )

    semantic_outcomes = outcome_distribution(
        semantic_results,
        5,
    )

    print_outcome_distribution(
        "TF-IDF Top-5 evidence",
        tfidf_outcomes,
    )

    print()

    print_outcome_distribution(
        "Semantic Top-5 evidence",
        semantic_outcomes,
    )

    # -------------------------------------------------------------
    # Leakage checks
    # -------------------------------------------------------------

    tfidf_leakage = count_leakage_violations(
        tfidf_results
    )

    semantic_leakage = count_leakage_violations(
        semantic_results
    )

    print()
    print("-" * 72)
    print("LEAKAGE CHECK")
    print("-" * 72)

    print(
        f"TF-IDF leakage violations   : "
        f"{tfidf_leakage}"
    )

    print(
        f"Semantic leakage violations : "
        f"{semantic_leakage}"
    )


    # -------------------------------------------------------------
    # LLM judge
    # -------------------------------------------------------------

    judge_summary = llm_judge_summary(
        llm_map
    )

    print()
    print("-" * 72)
    print("LLM-JUDGED SAMPLE")
    print("-" * 72)

    print(
        f"Judged cases : "
        f"{judge_summary['judged_cases']}"
    )

    print("Scores:")

    for score, count in sorted(
        judge_summary["scores"].items()
    ):
        print(
            f"  {score}/4"
            f"{' ' * 10}"
            f"{count}"
        )

    print("Decisions:")

    for decision, count in sorted(
        judge_summary["decisions"].items()
    ):
        print(
            f"  {decision:<20} {count}"
        )

    print("Dimension means:")

    for dimension, mean in (
        judge_summary["dimension_means"].items()
    ):
        if mean is None:
            print(
                f"  {dimension:<25} N/A"
            )
        else:
            print(
                f"  {dimension:<25} "
                f"{mean:.2f}/4"
            )

    # -------------------------------------------------------------
    # Qualitative analysis
    # -------------------------------------------------------------

    high_sim_poor = high_similarity_poor_evidence(
        semantic_results,
        llm_map,
        similarity_threshold=0.80,
        max_score=1,
    )

    low_sim_strong = lower_similarity_strong_evidence(
        semantic_results,
        llm_map,
        similarity_threshold=0.75,
        min_score=3,
    )

    print()
    print_qualitative_cases(
        "HIGH SIMILARITY + POOR EVIDENCE",
        high_sim_poor,
    )

    print()
    print_qualitative_cases(
        "LOWER SIMILARITY + STRONG EVIDENCE",
        low_sim_strong,
    )

    # -------------------------------------------------------------
    # Per-case comparison
    # -------------------------------------------------------------

    comparison_rows = build_comparison_rows(
        tfidf_results=tfidf_results,
        semantic_results=semantic_results,
        golden_rows=golden_rows,
        llm_map=llm_map,
    )

    # -------------------------------------------------------------
    # Unified JSON
    # -------------------------------------------------------------

    analysis = {
        "metadata": {
            "tfidf_results": len(tfidf_results),
            "semantic_results": len(semantic_results),
            "golden_cases": len(golden_rows),
            "llm_judgments": len(llm_rows),
            "llm_successfully_joined": len(llm_map),
        },

        "similarity": {
            "tfidf": tfidf_similarity,
            "semantic": semantic_similarity,
        },

        "resolved_evidence_coverage": coverage,

        "retrieved_outcome_distribution": {
            "tfidf_top5": dict(tfidf_outcomes),
            "semantic_top5": dict(semantic_outcomes),
        },

        "leakage": {
            "tfidf_violations": tfidf_leakage,
            "semantic_violations": semantic_leakage,
        },

      

        "llm_judge": {
            "summary": judge_summary,
            "high_similarity_poor_evidence": high_sim_poor,
            "lower_similarity_strong_evidence": low_sim_strong,
        },

        "comparison_rows": comparison_rows,
    }

    # -------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------

    write_json(
        OUTPUT_JSON,
        analysis,
    )

    write_csv(
        OUTPUT_CSV,
        comparison_rows,
    )

    # -------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------

    print()
    print("-" * 72)
    print("OUTPUT")
    print("-" * 72)

    print(
        OUTPUT_JSON
    )

    print(
        OUTPUT_CSV
    )

    print()
    print("-" * 72)
    print("VALIDATION")
    print("-" * 72)

    print(
        f"LLM judgments loaded : {len(llm_rows)}"
    )

    print(
        f"LLM judgments joined : {len(llm_map)}"
    )

    print(
        f"Comparison rows      : {len(comparison_rows)}"
    )

    if len(llm_rows) != len(llm_map):
        print(
            "[INFO] Some LLM rows were skipped during mapping "
            "because they lacked a valid case_id/judge object."
        )
    else:
        print(
            "[OK] All available LLM judgments joined successfully."
        )

    print()
    print(
        "[OK] Unified retrieval analysis completed."
    )


if __name__ == "__main__":
    main()

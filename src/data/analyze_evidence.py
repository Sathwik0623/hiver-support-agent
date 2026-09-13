"""
Analyze the historical AppleSupport evidence corpus.

Input:
    data/processed/historical_evidence.jsonl

Output:
    reports/evidence_analysis.md

Purpose:
    Validate whether the reconstructed historical evidence is suitable
    for retrieval before building the AI/retrieval layer.

The analysis covers:

    - total evidence records
    - message-count distribution
    - one-turn vs multi-turn cases
    - customer follow-up rate
    - resolution-signal distribution
    - response-length distribution
    - duplicate customer problems
    - useful UNKNOWN cases
    - longest conversations
    - random examples
    - basic corpus-quality warnings

Usage:
    python src/data/analyze_evidence.py

Custom input/output:
    python src/data/analyze_evidence.py \
        --input path/to/historical_evidence.jsonl \
        --output path/to/evidence_analysis.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_evidence.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "evidence_analysis.md"
)

RANDOM_SEED = 42
RANDOM_SAMPLE_SIZE = 10
USEFUL_UNKNOWN_SAMPLE_SIZE = 10
LONGEST_CASES_SIZE = 10


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_problem(text: str | None) -> str:
    """
    Normalize customer problems for duplicate analysis.

    This is intentionally conservative. We do not remove meaningful
    technical terms or aggressively stem words.
    """

    if not text:
        return ""

    text = text.lower().strip()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    # Remove excessive whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # Normalize common punctuation.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def truncate(
    text: str,
    max_length: int = 300,
) -> str:
    """
    Make examples readable in the Markdown report.
    """

    text = text.replace(
        "\n",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def percentile(
    values: list[int],
    p: float,
) -> float:
    """
    Calculate a percentile without requiring NumPy.
    """

    if not values:
        return 0.0

    values = sorted(values)

    position = (
        (len(values) - 1)
        * p
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = position - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


# ---------------------------------------------------------------------------
# Streaming random sampling
# ---------------------------------------------------------------------------

def reservoir_sample(
    current_sample: list[dict],
    record: dict,
    seen_count: int,
    sample_size: int,
    rng: random.Random,
) -> None:
    """
    Reservoir sampling.

    This lets us obtain a random sample from a large JSONL file without
    loading the entire corpus into memory.
    """

    if len(current_sample) < sample_size:

        current_sample.append(record)

        return

    replacement_index = rng.randint(
        0,
        seen_count - 1,
    )

    if replacement_index < sample_size:

        current_sample[
            replacement_index
        ] = record


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_evidence(
    input_path: Path,
) -> dict:
    """
    Analyze the complete evidence corpus.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Evidence file not found:\n{input_path}\n\n"
            "Run build_evidence.py first."
        )

    rng = random.Random(
        RANDOM_SEED
    )

    # -----------------------------------------------------------------------
    # Basic counters
    # -----------------------------------------------------------------------

    total_records = 0

    signal_counts = Counter()

    message_counts: list[int] = []

    customer_message_counts: list[int] = []

    apple_message_counts: list[int] = []

    response_lengths: list[int] = []

    problem_lengths: list[int] = []

    followup_lengths: list[int] = []

    # -----------------------------------------------------------------------
    # Conversation structure
    # -----------------------------------------------------------------------

    one_message_cases = 0

    one_customer_one_apple = 0

    multi_turn_cases = 0

    cases_with_followup = 0

    cases_without_followup = 0

    # -----------------------------------------------------------------------
    # Duplicate analysis
    # -----------------------------------------------------------------------

    normalized_problem_counts = Counter()

    problem_examples: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Samples
    # -----------------------------------------------------------------------

    random_sample: list[dict] = []

    useful_unknown_sample: list[dict] = []

    resolved_sample: list[dict] = []

    failed_sample: list[dict] = []

    # -----------------------------------------------------------------------
    # Longest cases
    # -----------------------------------------------------------------------

    longest_cases: list[dict] = []

    # -----------------------------------------------------------------------
    # Read corpus
    # -----------------------------------------------------------------------

    print("=" * 70)
    print("Historical Evidence Analysis")
    print("=" * 70)

    print(
        f"[INFO] Input: {input_path}"
    )

    print(
        "\n[INFO] Reading evidence corpus..."
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            total_records += 1

            # ----------------------------------------------------------------
            # Random sample
            # ----------------------------------------------------------------

            reservoir_sample(
                current_sample=random_sample,
                record=record,
                seen_count=total_records,
                sample_size=RANDOM_SAMPLE_SIZE,
                rng=rng,
            )

            # ----------------------------------------------------------------
            # Core fields
            # ----------------------------------------------------------------

            signal = record.get(
                "resolution_signal",
                "UNKNOWN",
            )

            signal_counts[
                signal
            ] += 1

            message_count = int(
                record.get(
                    "message_count",
                    0,
                )
                or 0
            )

            customer_count = int(
                record.get(
                    "customer_message_count",
                    0,
                )
                or 0
            )

            apple_count = int(
                record.get(
                    "apple_message_count",
                    0,
                )
                or 0
            )

            message_counts.append(
                message_count
            )

            customer_message_counts.append(
                customer_count
            )

            apple_message_counts.append(
                apple_count
            )

            # ----------------------------------------------------------------
            # Text lengths
            # ----------------------------------------------------------------

            problem = record.get(
                "customer_problem",
                "",
            ) or ""

            response = record.get(
                "apple_response",
                "",
            ) or ""

            followup = record.get(
                "follow_up",
                "",
            ) or ""

            problem_lengths.append(
                len(problem)
            )

            response_lengths.append(
                len(response)
            )

            followup_lengths.append(
                len(followup)
            )

            # ----------------------------------------------------------------
            # Conversation structure
            # ----------------------------------------------------------------

            if message_count <= 2:

                one_message_cases += (
                    1
                    if message_count == 1
                    else 0
                )

            if (
                customer_count == 1
                and apple_count == 1
            ):
                one_customer_one_apple += 1

            if message_count > 2:
                multi_turn_cases += 1

            if followup.strip():

                cases_with_followup += 1

            else:

                cases_without_followup += 1

            # ----------------------------------------------------------------
            # Duplicate customer problem analysis
            # ----------------------------------------------------------------

            normalized = normalize_problem(
                problem
            )

            if normalized:

                normalized_problem_counts[
                    normalized
                ] += 1

                if normalized not in problem_examples:

                    problem_examples[
                        normalized
                    ] = problem

            # ----------------------------------------------------------------
            # Useful UNKNOWN samples
            #
            # UNKNOWN does not mean useless. A historical response can still
            # be valuable for retrieval even when the outcome isn't observed.
            # ----------------------------------------------------------------

            if (
                signal == "UNKNOWN"
                and problem.strip()
                and response.strip()
                and message_count >= 2
            ):

                reservoir_sample(
                    current_sample=useful_unknown_sample,
                    record=record,
                    seen_count=total_records,
                    sample_size=USEFUL_UNKNOWN_SAMPLE_SIZE,
                    rng=rng,
                )

            # ----------------------------------------------------------------
            # Outcome samples
            # ----------------------------------------------------------------

            if signal == "RESOLVED":

                reservoir_sample(
                    current_sample=resolved_sample,
                    record=record,
                    seen_count=total_records,
                    sample_size=5,
                    rng=rng,
                )

            elif signal == "FAILED":

                reservoir_sample(
                    current_sample=failed_sample,
                    record=record,
                    seen_count=total_records,
                    sample_size=5,
                    rng=rng,
                )

            # ----------------------------------------------------------------
            # Longest cases
            # ----------------------------------------------------------------

            longest_cases.append(
                {
                    "case_id": record.get(
                        "case_id",
                        "",
                    ),
                    "message_count": message_count,
                    "customer_problem": problem,
                    "resolution_signal": signal,
                }
            )

            # Keep only a manageable number of longest records.
            longest_cases.sort(
                key=lambda item: item[
                    "message_count"
                ],
                reverse=True,
            )

            if len(longest_cases) > LONGEST_CASES_SIZE:

                longest_cases.pop()

            # ----------------------------------------------------------------
            # Progress
            # ----------------------------------------------------------------

            if total_records % 10_000 == 0:

                print(
                    f"[PROGRESS] "
                    f"{total_records:,} records analyzed"
                )

    print(
        f"\n[OK] Finished reading "
        f"{total_records:,} records."
    )

    # -----------------------------------------------------------------------
    # Duplicate statistics
    # -----------------------------------------------------------------------

    duplicate_problem_groups = {
        problem: count
        for problem, count
        in normalized_problem_counts.items()
        if count > 1
    }

    duplicate_problem_records = sum(
        count
        for count in duplicate_problem_groups.values()
    )

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------

    resolved = signal_counts[
        "RESOLVED"
    ]

    failed = signal_counts[
        "FAILED"
    ]

    in_progress = signal_counts[
        "IN_PROGRESS"
    ]

    escalated = signal_counts[
        "ESCALATED"
    ]

    unknown = signal_counts[
        "UNKNOWN"
    ]

    def percentage(
        value: int,
    ) -> float:

        if total_records == 0:
            return 0.0

        return (
            value
            / total_records
            * 100
        )

    summary = {
        "total_records": total_records,

        "resolved": resolved,
        "failed": failed,
        "in_progress": in_progress,
        "escalated": escalated,
        "unknown": unknown,

        "resolved_pct": percentage(
            resolved
        ),
        "failed_pct": percentage(
            failed
        ),
        "in_progress_pct": percentage(
            in_progress
        ),
        "escalated_pct": percentage(
            escalated
        ),
        "unknown_pct": percentage(
            unknown
        ),

        "multi_turn_cases": multi_turn_cases,
        "multi_turn_pct": percentage(
            multi_turn_cases
        ),

        "one_customer_one_apple": (
            one_customer_one_apple
        ),
        "one_customer_one_apple_pct": percentage(
            one_customer_one_apple
        ),

        "cases_with_followup": cases_with_followup,
        "cases_with_followup_pct": percentage(
            cases_with_followup
        ),

        "cases_without_followup": (
            cases_without_followup
        ),

        "duplicate_problem_groups": (
            len(duplicate_problem_groups)
        ),

        "duplicate_problem_records": (
            duplicate_problem_records
        ),
    }

    summary[
        "message_mean"
    ] = (
        statistics.mean(message_counts)
        if message_counts
        else 0
    )

    summary[
        "message_median"
    ] = (
        statistics.median(message_counts)
        if message_counts
        else 0
    )

    summary[
        "message_p90"
    ] = percentile(
        message_counts,
        0.90,
    )

    summary[
        "message_p95"
    ] = percentile(
        message_counts,
        0.95,
    )

    summary[
        "message_max"
    ] = (
        max(message_counts)
        if message_counts
        else 0
    )

    summary[
        "response_mean_chars"
    ] = (
        statistics.mean(response_lengths)
        if response_lengths
        else 0
    )

    summary[
        "response_median_chars"
    ] = (
        statistics.median(response_lengths)
        if response_lengths
        else 0
    )

    summary[
        "problem_mean_chars"
    ] = (
        statistics.mean(problem_lengths)
        if problem_lengths
        else 0
    )

    summary[
        "problem_median_chars"
    ] = (
        statistics.median(problem_lengths)
        if problem_lengths
        else 0
    )

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "CORPUS SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"Evidence records       : "
        f"{total_records:,}"
    )

    print(
        f"Multi-turn cases       : "
        f"{multi_turn_cases:,} "
        f"({percentage(multi_turn_cases):.2f}%)"
    )

    print(
        f"1 customer + 1 reply   : "
        f"{one_customer_one_apple:,} "
        f"({percentage(one_customer_one_apple):.2f}%)"
    )

    print(
        f"Customer follow-ups    : "
        f"{cases_with_followup:,} "
        f"({percentage(cases_with_followup):.2f}%)"
    )

    print(
        f"\nResolution signals:"
    )

    print(
        f"  RESOLVED             : "
        f"{resolved:,} "
        f"({percentage(resolved):.2f}%)"
    )

    print(
        f"  FAILED               : "
        f"{failed:,} "
        f"({percentage(failed):.2f}%)"
    )

    print(
        f"  IN_PROGRESS          : "
        f"{in_progress:,} "
        f"({percentage(in_progress):.2f}%)"
    )

    print(
        f"  ESCALATED            : "
        f"{escalated:,} "
        f"({percentage(escalated):.2f}%)"
    )

    print(
        f"  UNKNOWN              : "
        f"{unknown:,} "
        f"({percentage(unknown):.2f}%)"
    )

    print(
        f"\nMessage count:"
    )

    print(
        f"  Mean                 : "
        f"{summary['message_mean']:.2f}"
    )

    print(
        f"  Median               : "
        f"{summary['message_median']:.0f}"
    )

    print(
        f"  P90                  : "
        f"{summary['message_p90']:.0f}"
    )

    print(
        f"  P95                  : "
        f"{summary['message_p95']:.0f}"
    )

    print(
        f"  Maximum              : "
        f"{summary['message_max']}"
    )

    print(
        f"\nDuplicate normalized problems:"
    )

    print(
        f"  Duplicate groups     : "
        f"{len(duplicate_problem_groups):,}"
    )

    print(
        f"  Records involved     : "
        f"{duplicate_problem_records:,}"
    )

    return {
        "summary": summary,
        "signal_counts": signal_counts,
        "duplicate_problem_groups": (
            duplicate_problem_groups
        ),
        "problem_examples": problem_examples,
        "random_sample": random_sample,
        "useful_unknown_sample": (
            useful_unknown_sample
        ),
        "resolved_sample": resolved_sample,
        "failed_sample": failed_sample,
        "longest_cases": longest_cases,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(
    analysis: dict,
    output_path: Path,
) -> None:

    summary = analysis[
        "summary"
    ]

    signal_counts = analysis[
        "signal_counts"
    ]

    duplicate_groups = analysis[
        "duplicate_problem_groups"
    ]

    problem_examples = analysis[
        "problem_examples"
    ]

    random_sample = analysis[
        "random_sample"
    ]

    useful_unknown_sample = analysis[
        "useful_unknown_sample"
    ]

    resolved_sample = analysis[
        "resolved_sample"
    ]

    failed_sample = analysis[
        "failed_sample"
    ]

    longest_cases = analysis[
        "longest_cases"
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = []

    lines.append(
        "# AppleSupport Historical Evidence Analysis"
    )

    lines.append("")

    lines.append(
        "> Generated from `historical_evidence.jsonl`. "
        "This report is a data-quality analysis, not an evaluation "
        "of the final AI agent."
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # Executive summary
    # -----------------------------------------------------------------------

    lines.append(
        "## 1. Executive Summary"
    )

    lines.append("")

    lines.append(
        f"- **Evidence records:** "
        f"{summary['total_records']:,}"
    )

    lines.append(
        f"- **Multi-turn cases:** "
        f"{summary['multi_turn_cases']:,} "
        f"({summary['multi_turn_pct']:.2f}%)"
    )

    lines.append(
        f"- **Cases with customer follow-up:** "
        f"{summary['cases_with_followup']:,} "
        f"({summary['cases_with_followup_pct']:.2f}%)"
    )

    lines.append(
        f"- **Median messages per case:** "
        f"{summary['message_median']:.0f}"
    )

    lines.append(
        f"- **P90 messages per case:** "
        f"{summary['message_p90']:.0f}"
    )

    lines.append(
        f"- **Maximum messages in a case:** "
        f"{summary['message_max']}"
    )

    lines.append("")

    lines.append(
        "### Initial interpretation"
    )

    lines.append("")

    lines.append(
        "The corpus should be treated as historical support evidence "
        "rather than ground-truth resolved tickets. In particular, "
        "absence of a customer follow-up is not treated as successful "
        "resolution."
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # Resolution signals
    # -----------------------------------------------------------------------

    lines.append(
        "## 2. Resolution Signal Distribution"
    )

    lines.append("")

    lines.append(
        "| Signal | Count | Percentage |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for signal in [
        "RESOLVED",
        "FAILED",
        "IN_PROGRESS",
        "ESCALATED",
        "UNKNOWN",
    ]:

        count = signal_counts[
            signal
        ]

        pct = (
            count
            / summary["total_records"]
            * 100
            if summary["total_records"]
            else 0
        )

        lines.append(
            f"| {signal} | "
            f"{count:,} | "
            f"{pct:.2f}% |"
        )

    lines.append("")

    lines.append(
        "**Interpretation:** `UNKNOWN` is intentionally large because "
        "conversation termination does not prove successful resolution. "
        "These records can still be useful for learning historical "
        "support behavior."
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # Conversation structure
    # -----------------------------------------------------------------------

    lines.append(
        "## 3. Conversation Structure"
    )

    lines.append("")

    lines.append(
        "| Metric | Value |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Total cases | "
        f"{summary['total_records']:,} |"
    )

    lines.append(
        f"| 1 customer + 1 AppleSupport message | "
        f"{summary['one_customer_one_apple']:,} "
        f"({summary['one_customer_one_apple_pct']:.2f}%) |"
    )

    lines.append(
        f"| Multi-turn cases | "
        f"{summary['multi_turn_cases']:,} "
        f"({summary['multi_turn_pct']:.2f}%) |"
    )

    lines.append(
        f"| Cases with customer follow-up | "
        f"{summary['cases_with_followup']:,} "
        f"({summary['cases_with_followup_pct']:.2f}%) |"
    )

    lines.append(
        f"| Cases without customer follow-up | "
        f"{summary['cases_without_followup']:,} |"
    )

    lines.append("")

    lines.append(
        "### Message-count statistics"
    )

    lines.append("")

    lines.append(
        f"- Mean: `{summary['message_mean']:.2f}`"
    )

    lines.append(
        f"- Median: `{summary['message_median']:.0f}`"
    )

    lines.append(
        f"- P90: `{summary['message_p90']:.0f}`"
    )

    lines.append(
        f"- P95: `{summary['message_p95']:.0f}`"
    )

    lines.append(
        f"- Maximum: `{summary['message_max']}`"
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # Text lengths
    # -----------------------------------------------------------------------

    lines.append(
        "## 4. Text Lengths"
    )

    lines.append("")

    lines.append(
        "| Field | Mean chars | Median chars |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    lines.append(
        f"| Customer problem | "
        f"{summary['problem_mean_chars']:.1f} | "
        f"{summary['problem_median_chars']:.1f} |"
    )

    lines.append(
        f"| AppleSupport response | "
        f"{summary['response_mean_chars']:.1f} | "
        f"{summary['response_median_chars']:.1f} |"
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # Duplicate analysis
    # -----------------------------------------------------------------------

    lines.append(
        "## 5. Duplicate Customer Problems"
    )

    lines.append("")

    lines.append(
        f"After conservative text normalization, there are "
        f"**{len(duplicate_groups):,} repeated problem groups**, "
        f"involving **{sum(duplicate_groups.values()):,} records**."
    )

    lines.append("")

    lines.append(
        "This is not automatically a data-quality problem. Repeated "
        "customer problems can be valuable because they provide multiple "
        "historical examples of how AppleSupport handled similar issues."
    )

    lines.append("")

    top_duplicates = sorted(
        duplicate_groups.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    if top_duplicates:

        lines.append(
            "| Count | Example customer problem |"
        )

        lines.append(
            "|---:|---|"
        )

        for normalized, count in top_duplicates:

            example = problem_examples.get(
                normalized,
                normalized,
            )

            lines.append(
                f"| {count:,} | "
                f"{truncate(example, 180)} |"
            )

    else:

        lines.append(
            "No duplicate problem groups were detected."
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # Longest cases
    # -----------------------------------------------------------------------

    lines.append(
        "## 6. Longest Conversations"
    )

    lines.append("")

    lines.append(
        "| Case | Messages | Signal | Customer problem |"
    )

    lines.append(
        "|---|---:|---|---|"
    )

    for case in longest_cases:

        lines.append(
            f"| `{case['case_id']}` | "
            f"{case['message_count']} | "
            f"{case['resolution_signal']} | "
            f"{truncate(case['customer_problem'], 180)} |"
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # Random sample
    # -----------------------------------------------------------------------

    lines.append(
        "## 7. Random Corpus Sample"
    )

    lines.append("")

    for index, record in enumerate(
        random_sample,
        start=1,
    ):

        lines.append(
            f"### Example {index}"
        )

        lines.append("")

        lines.append(
            f"**Case:** `{record.get('case_id', '')}`  "
        )

        lines.append(
            f"**Signal:** "
            f"`{record.get('resolution_signal', 'UNKNOWN')}`"
        )

        lines.append("")

        lines.append(
            f"**Customer:** "
            f"{truncate(record.get('customer_problem', ''), 500)}"
        )

        lines.append("")

        lines.append(
            f"**AppleSupport:** "
            f"{truncate(record.get('apple_response', ''), 500)}"
        )

        followup = record.get(
            "follow_up",
            "",
        )

        if followup.strip():

            lines.append("")

            lines.append(
                f"**Follow-up:** "
                f"{truncate(followup, 500)}"
            )

        lines.append("")

    # -----------------------------------------------------------------------
    # UNKNOWN sample
    # -----------------------------------------------------------------------

    lines.append(
        "## 8. UNKNOWN Outcome — Useful Evidence Examples"
    )

    lines.append("")

    lines.append(
        "These examples demonstrate why `UNKNOWN` should not be "
        "interpreted as `USELESS`. The historical response may still "
        "provide valuable grounding even when the final outcome is not "
        "observable."
    )

    lines.append("")

    for index, record in enumerate(
        useful_unknown_sample,
        start=1,
    ):

        lines.append(
            f"### UNKNOWN Example {index}"
        )

        lines.append("")

        lines.append(
            f"**Case:** `{record.get('case_id', '')}`"
        )

        lines.append("")

        lines.append(
            f"**Customer:** "
            f"{truncate(record.get('customer_problem', ''), 500)}"
        )

        lines.append("")

        lines.append(
            f"**AppleSupport:** "
            f"{truncate(record.get('apple_response', ''), 500)}"
        )

        lines.append("")

    # -----------------------------------------------------------------------
    # Resolved examples
    # -----------------------------------------------------------------------

    lines.append(
        "## 9. Explicitly Resolved Examples"
    )

    lines.append("")

    for index, record in enumerate(
        resolved_sample,
        start=1,
    ):

        lines.append(
            f"### Resolved Example {index}"
        )

        lines.append("")

        lines.append(
            f"**Customer:** "
            f"{truncate(record.get('customer_problem', ''), 500)}"
        )

        lines.append("")

        lines.append(
            f"**AppleSupport:** "
            f"{truncate(record.get('apple_response', ''), 500)}"
        )

        lines.append("")

        lines.append(
            f"**Follow-up:** "
            f"{truncate(record.get('follow_up', ''), 500)}"
        )

        lines.append("")

    # -----------------------------------------------------------------------
    # Failed examples
    # -----------------------------------------------------------------------

    lines.append(
        "## 10. Failed Troubleshooting Examples"
    )

    lines.append("")

    for index, record in enumerate(
        failed_sample,
        start=1,
    ):

        lines.append(
            f"### Failed Example {index}"
        )

        lines.append("")

        lines.append(
            f"**Customer:** "
            f"{truncate(record.get('customer_problem', ''), 500)}"
        )

        lines.append("")

        lines.append(
            f"**AppleSupport:** "
            f"{truncate(record.get('apple_response', ''), 500)}"
        )

        lines.append("")

        lines.append(
            f"**Follow-up:** "
            f"{truncate(record.get('follow_up', ''), 500)}"
        )

        lines.append("")

    # -----------------------------------------------------------------------
    # Retrieval implications
    # -----------------------------------------------------------------------

    lines.append(
        "## 11. Retrieval Implications"
    )

    lines.append("")

    lines.append(
        "The corpus supports a retrieval-first architecture because "
        "each evidence record connects a customer problem with an "
        "historical AppleSupport response and, when observable, a "
        "conversation outcome."
    )

    lines.append("")

    lines.append(
        "However, retrieval should not filter exclusively for "
        "`RESOLVED` records. Doing so would discard a large amount of "
        "historical support behavior. Instead, the retrieval system "
        "should rank evidence using semantic similarity, intent/context "
        "compatibility, and evidence quality."
    )

    lines.append("")

    lines.append(
        "A future retrieval record should therefore distinguish:"
    )

    lines.append("")

    lines.append(
        "1. **Similarity** — how closely the historical problem matches."
    )

    lines.append(
        "2. **Context compatibility** — whether product/version/state "
        "information is compatible."
    )

    lines.append(
        "3. **Outcome confidence** — how strongly the historical "
        "conversation indicates success or failure."
    )

    lines.append(
        "4. **Response quality/usefulness** — whether the historical "
        "AppleSupport response contains actionable support behavior."
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # Quality warnings
    # -----------------------------------------------------------------------

    lines.append(
        "## 12. Quality Checks and Warnings"
    )

    lines.append("")

    warnings: list[str] = []

    if summary[
        "unknown_pct"
    ] > 80:

        warnings.append(
            "The majority of records have UNKNOWN outcomes. This is "
            "expected for Twitter conversations where customer success "
            "is often unobserved, but outcome should not be treated as "
            "ground truth."
        )

    if summary[
        "one_customer_one_apple_pct"
    ] > 70:

        warnings.append(
            "A large proportion of cases contain only one customer "
            "message and one AppleSupport response. Retrieval should "
            "therefore operate on both short examples and multi-turn "
            "conversation evidence."
        )

    if len(duplicate_groups) > 0:

        warnings.append(
            "Repeated customer problems are common. This is potentially "
            "useful for retrieval but requires duplicate-aware evaluation "
            "to avoid artificially optimistic results."
        )

    if summary[
        "multi_turn_pct"
    ] < 10:

        warnings.append(
            "The proportion of multi-turn cases is relatively low. "
            "Conversation-state modeling should therefore be evaluated "
            "separately from simple first-response retrieval."
        )

    if warnings:

        for warning in warnings:

            lines.append(
                f"- {warning}"
            )

    else:

        lines.append(
            "No major structural warnings were detected."
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # Conclusion
    # -----------------------------------------------------------------------

    lines.append(
        "## 13. Conclusion"
    )

    lines.append("")

    lines.append(
        "The evidence corpus is suitable as a historical retrieval "
        "source, provided that it is treated as observational support "
        "evidence rather than a fully labeled CRM dataset. The next "
        "engineering step is to build and evaluate a TF-IDF retrieval "
        "baseline against the held-out golden set."
    )

    lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"\n[OK] Report written to:\n{output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the AppleSupport historical evidence corpus."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to historical_evidence.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to evidence_analysis.md",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    try:

        analysis = analyze_evidence(
            input_path=args.input,
        )

        generate_report(
            analysis=analysis,
            output_path=args.output,
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "ANALYSIS COMPLETE"
        )

        print(
            "=" * 70
        )

        return 0

    except KeyboardInterrupt:

        print(
            "\n[ERROR] Analysis interrupted."
        )

        return 130

    except Exception as exc:

        print(
            f"\n[ERROR] {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())

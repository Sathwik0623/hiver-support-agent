"""
Build historical support evidence from reconstructed AppleSupport cases.

Input:
    data/processed/apple_cases.jsonl

Output:
    data/processed/historical_evidence.jsonl

Purpose:
    Convert reconstructed support conversations into retrieval-ready
    historical evidence.

Each evidence record captures:

    - customer problem
    - conversation context
    - AppleSupport response
    - customer follow-up
    - conservative resolution signal

Important:
    An AppleSupport response does NOT automatically mean the issue
    was resolved. Resolution is inferred only from observable
    conversation signals.

Usage:
    python src/data/build_evidence.py

Force rebuild:
    python src/data/build_evidence.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_cases.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_evidence.jsonl"
)


# ---------------------------------------------------------------------------
# Resolution signal vocabulary
# ---------------------------------------------------------------------------

RESOLVED = "RESOLVED"
FAILED = "FAILED"
IN_PROGRESS = "IN_PROGRESS"
ESCALATED = "ESCALATED"
UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str | None) -> str:
    """
    Normalize whitespace while preserving the original meaning.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse excessive spaces/tabs but preserve newlines.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def customer_messages(case: dict) -> list[dict]:
    """
    Return customer messages from a case.
    """

    return [
        message
        for message in case.get("messages", [])
        if message.get("sender") == "customer"
    ]


def apple_messages(case: dict) -> list[dict]:
    """
    Return AppleSupport messages from a case.
    """

    return [
        message
        for message in case.get("messages", [])
        if message.get("sender") == "apple_support"
    ]


def last_customer_message(case: dict) -> dict | None:
    messages = customer_messages(case)

    if not messages:
        return None

    return messages[-1]


def first_customer_message(case: dict) -> dict | None:
    messages = customer_messages(case)

    if not messages:
        return None

    return messages[0]


def first_apple_response(case: dict) -> dict | None:
    messages = apple_messages(case)

    if not messages:
        return None

    return messages[0]


def last_apple_response(case: dict) -> dict | None:
    messages = apple_messages(case)

    if not messages:
        return None

    return messages[-1]


# ---------------------------------------------------------------------------
# Resolution signal detection
# ---------------------------------------------------------------------------

RESOLUTION_PATTERNS = [
    # Strong positive signals.
    (
        RESOLVED,
        [
            r"\bthank(s| you)?\b",
            r"\bthat worked\b",
            r"\bworks now\b",
            r"\bworking now\b",
            r"\bfixed\b",
            r"\bproblem is solved\b",
            r"\bsolved\b",
            r"\bit works\b",
            r"\ball good\b",
            r"\bgot it working\b",
            r"\bissue is gone\b",
        ],
    ),

    # Strong negative signals.
    (
        FAILED,
        [
            r"\bstill (not|doesn'?t|won'?t|can'?t)\b",
            r"\bstill broken\b",
            r"\bstill not working\b",
            r"\bdidn'?t work\b",
            r"\bdoesn'?t work\b",
            r"\bnot working\b",
            r"\bwon'?t work\b",
            r"\bproblem persists\b",
            r"\bsame problem\b",
            r"\bsame issue\b",
            r"\bno luck\b",
        ],
    ),

    # Escalation / handoff signals.
    (
        ESCALATED,
        [
            r"\bescalat(e|ed|ing|ion)\b",
            r"\bspecialist\b",
            r"\bengineering team\b",
            r"\bengineer\b",
            r"\bcase\b.*\bnumber\b",
            r"\bdirect message\b",
            r"\bdm us\b",
            r"\bcontact support\b",
        ],
    ),
]


def detect_signal_from_text(
    text: str,
) -> str | None:
    """
    Detect a conservative resolution signal from one customer message.

    Priority:
        FAILED > RESOLVED > ESCALATED

    Failed troubleshooting is stronger evidence than a generic
    positive phrase elsewhere in the same message.
    """

    text = normalize_text(text).lower()

    if not text:
        return None

    # Failure takes precedence.
    for pattern in dict(
        RESOLUTION_PATTERNS
    )[FAILED]:

        if re.search(pattern, text):
            return FAILED

    for pattern in dict(
        RESOLUTION_PATTERNS
    )[RESOLVED]:

        if re.search(pattern, text):
            return RESOLVED

    for pattern in dict(
        RESOLUTION_PATTERNS
    )[ESCALATED]:

        if re.search(pattern, text):
            return ESCALATED

    return None


def infer_resolution_signal(
    case: dict,
) -> tuple[str, float, str]:
    """
    Infer the strongest observable resolution signal.

    Returns:
        signal
        confidence
        reason

    Confidence is deliberately conservative.

    We do not infer resolution merely because:
        - AppleSupport responded
        - the conversation ended
        - AppleSupport gave troubleshooting instructions
    """

    messages = case.get(
        "messages",
        [],
    )

    if not messages:
        return (
            UNKNOWN,
            0.99,
            "No messages available.",
        )

    customer = customer_messages(case)
    apple = apple_messages(case)

    if not apple:
        return (
            UNKNOWN,
            0.99,
            "No AppleSupport response available.",
        )

    # -----------------------------------------------------------------------
    # Examine customer follow-ups AFTER AppleSupport responses.
    # -----------------------------------------------------------------------

    signals: list[tuple[str, str]] = []

    apple_timestamps = [
        message.get("created_at")
        for message in apple
    ]

    last_apple_timestamp = (
        apple_timestamps[-1]
        if apple_timestamps
        else None
    )

    for message in customer:

        timestamp = message.get(
            "created_at"
        )

        if (
            last_apple_timestamp
            and timestamp
            and timestamp <= last_apple_timestamp
        ):
            continue

        text = message.get(
            "text",
            "",
        )

        signal = detect_signal_from_text(
            text
        )

        if signal:
            signals.append(
                (
                    signal,
                    normalize_text(text),
                )
            )

    # -----------------------------------------------------------------------
    # Strong explicit signals.
    # -----------------------------------------------------------------------

    if any(
        signal == FAILED
        for signal, _ in signals
    ):

        example = next(
            text
            for signal, text in signals
            if signal == FAILED
        )

        return (
            FAILED,
            0.90,
            "Customer follow-up indicates the suggested path did not work: "
            + example,
        )

    if any(
        signal == RESOLVED
        for signal, _ in signals
    ):

        example = next(
            text
            for signal, text in signals
            if signal == RESOLVED
        )

        return (
            RESOLVED,
            0.90,
            "Customer follow-up contains an explicit positive outcome: "
            + example,
        )

    if any(
        signal == ESCALATED
        for signal, _ in signals
    ):

        example = next(
            text
            for signal, text in signals
            if signal == ESCALATED
        )

        return (
            ESCALATED,
            0.85,
            "Conversation contains an observable escalation or handoff signal: "
            + example,
        )

    # -----------------------------------------------------------------------
    # If there is another customer message but no explicit outcome,
    # the conversation is still active/uncertain.
    # -----------------------------------------------------------------------

    followups = [
        message
        for message in customer
        if (
            not last_apple_timestamp
            or not message.get("created_at")
            or message.get("created_at") > last_apple_timestamp
        )
    ]

    if followups:
        return (
            IN_PROGRESS,
            0.70,
            "Customer continued the conversation after the latest "
            "AppleSupport response, but no explicit outcome was detected.",
        )

    # -----------------------------------------------------------------------
    # No follow-up.
    #
    # Important: silence is NOT proof of resolution.
    # -----------------------------------------------------------------------

    return (
        UNKNOWN,
        0.95,
        "No customer follow-up was observed; conversation ending is "
        "not treated as evidence of successful resolution.",
    )


# ---------------------------------------------------------------------------
# Build customer problem
# ---------------------------------------------------------------------------

def build_customer_problem(
    case: dict,
) -> str:
    """
    The initial customer message is the primary problem description.

    We do not use AppleSupport's interpretation as the customer's
    ground-truth problem.
    """

    first_customer = first_customer_message(
        case
    )

    if first_customer is None:
        return ""

    return normalize_text(
        first_customer.get(
            "text",
            "",
        )
    )


# ---------------------------------------------------------------------------
# Build conversation context
# ---------------------------------------------------------------------------

def build_conversation_context(
    case: dict,
) -> str:
    """
    Build a readable representation of the conversation before the
    final customer follow-up.

    This gives retrieval/generation access to troubleshooting context.
    """

    messages = case.get(
        "messages",
        [],
    )

    context_parts: list[str] = []

    for message in messages:

        sender = message.get(
            "sender",
            "unknown",
        )

        text = normalize_text(
            message.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        if sender == "customer":
            label = "Customer"

        elif sender == "apple_support":
            label = "AppleSupport"

        else:
            label = "Other"

        context_parts.append(
            f"{label}: {text}"
        )

    return "\n".join(
        context_parts
    )


# ---------------------------------------------------------------------------
# Build Apple response
# ---------------------------------------------------------------------------

def build_apple_response(
    case: dict,
) -> str:
    """
    Select the first AppleSupport response as the primary historical
    response.

    The complete conversation remains available in conversation_context.
    """

    response = first_apple_response(
        case
    )

    if response is None:
        return ""

    return normalize_text(
        response.get(
            "text",
            "",
        )
    )


# ---------------------------------------------------------------------------
# Build follow-up
# ---------------------------------------------------------------------------

def build_follow_up(
    case: dict,
) -> str:
    """
    Return customer messages after the first AppleSupport response.

    This is useful for determining whether the historical response
    appears to have worked.
    """

    messages = case.get(
        "messages",
        [],
    )

    apple_response = first_apple_response(
        case
    )

    if apple_response is None:
        return ""

    response_timestamp = (
        apple_response.get(
            "created_at"
        )
    )

    followups: list[str] = []

    for message in messages:

        if message.get("sender") != "customer":
            continue

        timestamp = message.get(
            "created_at"
        )

        if (
            response_timestamp
            and timestamp
            and timestamp <= response_timestamp
        ):
            continue

        text = normalize_text(
            message.get(
                "text",
                "",
            )
        )

        if text:
            followups.append(
                text
            )

    return "\n".join(
        followups
    )


# ---------------------------------------------------------------------------
# Evidence record
# ---------------------------------------------------------------------------

def build_evidence_record(
    case: dict,
) -> dict:
    """
    Convert one reconstructed case into one retrieval evidence record.
    """

    signal, confidence, reason = (
        infer_resolution_signal(
            case
        )
    )

    customer_problem = (
        build_customer_problem(
            case
        )
    )

    apple_response = (
        build_apple_response(
            case
        )
    )

    follow_up = (
        build_follow_up(
            case
        )
    )

    context = (
        build_conversation_context(
            case
        )
    )

    return {
        "evidence_id": (
            f"EVIDENCE_{case['case_id']}"
        ),

        "case_id": case["case_id"],

        "brand": case.get(
            "brand",
            "AppleSupport",
        ),

        "customer_id": case.get(
            "customer_id"
        ),

        "source_tweet_id": case.get(
            "source_tweet_id"
        ),

        "customer_problem": customer_problem,

        "conversation_context": context,

        "apple_response": apple_response,

        "follow_up": follow_up,

        "resolution_signal": signal,

        "resolution_confidence": confidence,

        "resolution_reason": reason,

        "message_count": case.get(
            "message_count",
            len(case.get("messages", [])),
        ),

        "customer_message_count": case.get(
            "customer_message_count",
        ),

        "apple_message_count": case.get(
            "apple_message_count",
        ),

        "created_at": case.get(
            "created_at"
        ),

        "last_activity_at": case.get(
            "last_activity_at"
        ),
    }


# ---------------------------------------------------------------------------
# Build evidence corpus
# ---------------------------------------------------------------------------

def build_evidence(
    input_path: Path,
    output_path: Path,
    force: bool = False,
) -> None:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}\n\n"
            "Run reconstruct_cases.py first."
        )

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists:\n{output_path}\n\n"
            "Use --force to rebuild it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Historical Evidence Builder")
    print("=" * 70)

    print(
        f"[INFO] Input : {input_path}"
    )

    print(
        f"[INFO] Output: {output_path}"
    )

    start = perf_counter()

    total_cases = 0

    signal_counts = {
        RESOLVED: 0,
        FAILED: 0,
        IN_PROGRESS: 0,
        ESCALATED: 0,
        UNKNOWN: 0,
    }

    empty_problem_count = 0
    empty_response_count = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for line_number, line in enumerate(
            input_file,
            start=1,
        ):

            if not line.strip():
                continue

            try:
                case = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            evidence = build_evidence_record(
                case
            )

            output_file.write(
                json.dumps(
                    evidence,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total_cases += 1

            signal = evidence[
                "resolution_signal"
            ]

            signal_counts[
                signal
            ] += 1

            if not evidence[
                "customer_problem"
            ]:
                empty_problem_count += 1

            if not evidence[
                "apple_response"
            ]:
                empty_response_count += 1

            if total_cases % 10_000 == 0:

                elapsed = (
                    perf_counter()
                    - start
                )

                print(
                    f"[PROGRESS] "
                    f"{total_cases:,} evidence records "
                    f"({elapsed:.1f}s)"
                )

    elapsed = (
        perf_counter()
        - start
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "EVIDENCE SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"Evidence records : "
        f"{total_cases:,}"
    )

    print(
        f"RESOLVED         : "
        f"{signal_counts[RESOLVED]:,}"
    )

    print(
        f"FAILED           : "
        f"{signal_counts[FAILED]:,}"
    )

    print(
        f"IN_PROGRESS      : "
        f"{signal_counts[IN_PROGRESS]:,}"
    )

    print(
        f"ESCALATED        : "
        f"{signal_counts[ESCALATED]:,}"
    )

    print(
        f"UNKNOWN          : "
        f"{signal_counts[UNKNOWN]:,}"
    )

    print(
        f"Empty problems   : "
        f"{empty_problem_count:,}"
    )

    print(
        f"Empty responses  : "
        f"{empty_response_count:,}"
    )

    print(
        f"Runtime          : "
        f"{elapsed:.1f}s"
    )

    print(
        "\n[OK] Historical evidence generated."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_evidence(
    output_path: Path,
) -> None:

    print(
        "\n[INFO] Validating evidence corpus..."
    )

    required_fields = {
        "evidence_id",
        "case_id",
        "brand",
        "customer_problem",
        "conversation_context",
        "apple_response",
        "follow_up",
        "resolution_signal",
        "resolution_confidence",
        "resolution_reason",
    }

    valid_signals = {
        RESOLVED,
        FAILED,
        IN_PROGRESS,
        ESCALATED,
        UNKNOWN,
    }

    evidence_ids: set[str] = set()

    count = 0

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            record = json.loads(
                line
            )

            missing = (
                required_fields
                - set(record.keys())
            )

            if missing:
                raise ValueError(
                    f"Line {line_number} "
                    f"missing fields: "
                    f"{sorted(missing)}"
                )

            evidence_id = record[
                "evidence_id"
            ]

            if evidence_id in evidence_ids:
                raise ValueError(
                    f"Duplicate evidence_id: "
                    f"{evidence_id}"
                )

            evidence_ids.add(
                evidence_id
            )

            if record[
                "resolution_signal"
            ] not in valid_signals:

                raise ValueError(
                    f"Invalid resolution signal "
                    f"on line {line_number}: "
                    f"{record['resolution_signal']}"
                )

            confidence = record[
                "resolution_confidence"
            ]

            if not (
                isinstance(
                    confidence,
                    (int, float),
                )
                and 0 <= confidence <= 1
            ):
                raise ValueError(
                    f"Invalid resolution confidence "
                    f"on line {line_number}: "
                    f"{confidence}"
                )

            count += 1

    print(
        f"[VALIDATION] Evidence records: "
        f"{count:,}"
    )

    print(
        "[OK] Evidence validation passed."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build retrieval-ready historical "
            "support evidence from AppleSupport cases."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input apple_cases.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output historical_evidence.jsonl",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    try:

        build_evidence(
            input_path=args.input,
            output_path=args.output,
            force=args.force,
        )

        validate_evidence(
            args.output
        )

        return 0

    except KeyboardInterrupt:

        print(
            "\n[ERROR] Evidence building interrupted."
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

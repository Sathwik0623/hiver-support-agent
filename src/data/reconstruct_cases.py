"""
Efficient AppleSupport case reconstruction from TWCS.

Case definition
---------------
A support case starts at a customer tweet that is directly answered
by AppleSupport.

After the anchor, we follow the conversation only when the sender is:

    - the same customer
    - AppleSupport

We intentionally do NOT treat the entire Twitter root as a support case.

Input:
    data/processed/twcs.sqlite

Output:
    data/processed/apple_cases.jsonl
    data/processed/apple_messages.jsonl

Usage:
    python src/data/reconstruct_cases.py

Force rebuild:
    python src/data/reconstruct_cases.py --force
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from time import perf_counter


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "twcs.sqlite"
)

DEFAULT_CASES_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_cases.jsonl"
)

DEFAULT_MESSAGES_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_messages.jsonl"
)

BRAND = "AppleSupport"


# ---------------------------------------------------------------------------
# SQLite configuration
# ---------------------------------------------------------------------------

def configure_connection(
    connection: sqlite3.Connection,
) -> None:
    """
    Configure SQLite for read-heavy reconstruction.
    """

    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA temp_store=MEMORY;")

    # Large enough cache for the reconstruction workload.
    connection.execute("PRAGMA cache_size=-262144;")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_text(text: str | None) -> str:
    if text is None:
        return ""

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def sender_type(
    author_id: str,
    customer_id: str,
) -> str:
    if author_id == BRAND:
        return "apple_support"

    if author_id == customer_id:
        return "customer"

    return "other"


def make_message(
    row: sqlite3.Row,
    customer_id: str,
) -> dict:

    return {
        "tweet_id": int(row["tweet_id"]),
        "sender": sender_type(
            row["author_id"],
            customer_id,
        ),
        "author_id": row["author_id"],
        "inbound": bool(row["inbound"]),
        "created_at": row["created_at"],
        "text": normalize_text(row["text"]),
        "in_response_to_tweet_id": (
            int(row["in_response_to_tweet_id"])
            if row["in_response_to_tweet_id"] is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Step 1: Find AppleSupport anchors
# ---------------------------------------------------------------------------

def get_anchors(
    connection: sqlite3.Connection,
):
    """
    Find unique customer tweets directly answered by AppleSupport.

    A customer tweet can have multiple AppleSupport replies, so DISTINCT
    prevents duplicate case anchors.
    """

    return connection.execute(
        """
        SELECT DISTINCT
            customer.tweet_id,
            customer.author_id,
            customer.inbound,
            customer.created_at,
            customer.text,
            customer.response_tweet_id,
            customer.in_response_to_tweet_id
        FROM tweets AS apple
        JOIN tweets AS customer
          ON customer.tweet_id = apple.in_response_to_tweet_id
        WHERE apple.author_id = ?
          AND apple.inbound = 0
          AND customer.inbound = 1
        ORDER BY customer.tweet_id
        """,
        (BRAND,),
    )

# ---------------------------------------------------------------------------
# Step 2: Build an in-memory lookup for AppleSupport + customers
# ---------------------------------------------------------------------------

def build_relevant_tweet_index(
    connection: sqlite3.Connection,
) -> dict[int, sqlite3.Row]:
    """
    Load only tweets that can participate in an AppleSupport case.

    We do NOT load the entire 2.8M-row dataset.

    We load:
        - AppleSupport tweets
        - inbound customer tweets

    This gives us enough information to reconstruct the relevant branches.
    """

    print(
        "[INFO] Building relevant tweet lookup..."
    )

    start = perf_counter()

    rows = connection.execute(
        """
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
            text,
            response_tweet_id,
            in_response_to_tweet_id
        FROM tweets
        WHERE author_id = ?
           OR inbound = 1
        """,
        (BRAND,),
    )

    tweet_index: dict[int, sqlite3.Row] = {}

    for row in rows:
        tweet_index[int(row["tweet_id"])] = row

    elapsed = perf_counter() - start

    print(
        f"[OK] Relevant tweet lookup built: "
        f"{len(tweet_index):,} rows "
        f"({elapsed:.1f}s)"
    )

    return tweet_index


# ---------------------------------------------------------------------------
# Step 3: Build parent → children index
# ---------------------------------------------------------------------------

def build_children_index(
    tweet_index: dict[int, sqlite3.Row],
) -> dict[int, list[int]]:
    """
    Build a parent → child lookup in memory.

    Only relevant tweets are included.
    """

    print(
        "[INFO] Building conversation-child index..."
    )

    start = perf_counter()

    children: dict[int, list[int]] = {}

    for tweet_id, row in tweet_index.items():

        parent_id = row["in_response_to_tweet_id"]

        if parent_id is None:
            continue

        parent_id = int(parent_id)

        children.setdefault(
            parent_id,
            [],
        ).append(tweet_id)

    elapsed = perf_counter() - start

    print(
        f"[OK] Child index built: "
        f"{len(children):,} parent nodes "
        f"({elapsed:.1f}s)"
    )

    return children


# ---------------------------------------------------------------------------
# Step 4: Reconstruct one case
# ---------------------------------------------------------------------------

def reconstruct_case(
    anchor: sqlite3.Row,
    tweet_index: dict[int, sqlite3.Row],
    children_index: dict[int, list[int]],
) -> dict | None:

    anchor_tweet_id = int(anchor["tweet_id"])

    customer_id = str(
        anchor["author_id"]
    )

    # -----------------------------------------------------------------------
    # Find the direct AppleSupport response.
    # -----------------------------------------------------------------------

    direct_replies = []

    for child_id in children_index.get(
        anchor_tweet_id,
        [],
    ):

        child = tweet_index.get(child_id)

        if child is None:
            continue

        if (
            child["author_id"] == BRAND
            and int(child["inbound"]) == 0
        ):
            direct_replies.append(child)

    if not direct_replies:
        return None

    # Deterministic choice if multiple Apple replies exist.
    direct_replies.sort(
        key=lambda row: (
            row["created_at"] or "",
            int(row["tweet_id"]),
        )
    )

    first_apple_reply = direct_replies[0]

    # -----------------------------------------------------------------------
    # Traverse only same customer + AppleSupport.
    # -----------------------------------------------------------------------

    queue = [
        anchor_tweet_id,
        int(first_apple_reply["tweet_id"]),
    ]

    visited: set[int] = set()

    messages: list[dict] = []

    while queue:

        tweet_id = queue.pop()

        if tweet_id in visited:
            continue

        visited.add(tweet_id)

        row = tweet_index.get(tweet_id)

        if row is None:
            continue

        author_id = row["author_id"]

        # Prevent unrelated users from entering the support case.
        if (
            author_id != BRAND
            and author_id != customer_id
        ):
            continue

        messages.append(
            make_message(
                row,
                customer_id,
            )
        )

        for child_id in children_index.get(
            tweet_id,
            [],
        ):

            child = tweet_index.get(child_id)

            if child is None:
                continue

            child_author = child["author_id"]

            if (
                child_author == BRAND
                or child_author == customer_id
            ):
                if child_id not in visited:
                    queue.append(child_id)

    if not messages:
        return None

    # -----------------------------------------------------------------------
    # Chronological order.
    # -----------------------------------------------------------------------

    messages.sort(
        key=lambda message: (
            message["created_at"] or "",
            message["tweet_id"],
        )
    )

    customer_messages = [
        message
        for message in messages
        if message["sender"] == "customer"
    ]

    apple_messages = [
        message
        for message in messages
        if message["sender"] == "apple_support"
    ]

    case_id = (
        f"APPLE_CASE_{anchor_tweet_id}_{customer_id}"
    )

    case = {
        "case_id": case_id,
        "brand": BRAND,
        "customer_id": customer_id,

        # The customer tweet directly answered by AppleSupport.
        "source_tweet_id": anchor_tweet_id,

        # We intentionally do not calculate the original public Twitter
        # root here. The root does not define our support case boundary.
        "original_root_tweet_id": None,

        "created_at": messages[0]["created_at"],
        "last_activity_at": messages[-1]["created_at"],

        "message_count": len(messages),
        "customer_message_count": len(customer_messages),
        "apple_message_count": len(apple_messages),

        "messages": messages,
    }

    return case


# ---------------------------------------------------------------------------
# Step 5: Write output
# ---------------------------------------------------------------------------

def write_case(
    file,
    case: dict,
) -> None:

    file.write(
        json.dumps(
            case,
            ensure_ascii=False,
        )
        + "\n"
    )


def write_message(
    file,
    case_id: str,
    message: dict,
) -> None:

    record = {
        "case_id": case_id,
        **message,
    }

    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_outputs(
    cases_path: Path,
    messages_path: Path,
) -> None:

    print(
        "\n[INFO] Validating generated outputs..."
    )

    case_count = 0
    message_count = 0

    case_ids: set[str] = set()

    # -----------------------------------------------------------------------
    # Validate cases
    # -----------------------------------------------------------------------

    with cases_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            case = json.loads(line)

            required_fields = {
                "case_id",
                "brand",
                "customer_id",
                "source_tweet_id",
                "messages",
            }

            missing = (
                required_fields
                - set(case.keys())
            )

            if missing:
                raise ValueError(
                    f"Case line {line_number} "
                    f"missing fields: {sorted(missing)}"
                )

            if case["brand"] != BRAND:
                raise ValueError(
                    f"Unexpected brand on line "
                    f"{line_number}: {case['brand']}"
                )

            if case["case_id"] in case_ids:
                raise ValueError(
                    f"Duplicate case_id: "
                    f"{case['case_id']}"
                )

            case_ids.add(
                case["case_id"]
            )

            if not case["messages"]:
                raise ValueError(
                    f"Case {case['case_id']} "
                    f"contains no messages."
                )

            case_count += 1

    # -----------------------------------------------------------------------
    # Validate flattened messages
    # -----------------------------------------------------------------------

    with messages_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            message = json.loads(line)

            required_fields = {
                "case_id",
                "tweet_id",
                "sender",
                "text",
            }

            missing = (
                required_fields
                - set(message.keys())
            )

            if missing:
                raise ValueError(
                    f"Message line {line_number} "
                    f"missing fields: {sorted(missing)}"
                )

            if message["sender"] not in {
                "customer",
                "apple_support",
            }:
                raise ValueError(
                    f"Unexpected sender on line "
                    f"{line_number}: "
                    f"{message['sender']}"
                )

            message_count += 1

    print(
        f"[VALIDATION] Cases    : {case_count:,}"
    )

    print(
        f"[VALIDATION] Messages : {message_count:,}"
    )

    print(
        "[OK] Output validation passed."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reconstruct_cases(
    db_path: Path,
    cases_output: Path,
    messages_output: Path,
    force: bool = False,
) -> None:

    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found:\n{db_path}\n\n"
            "Run build_index.py first."
        )

    if cases_output.exists() and not force:
        raise FileExistsError(
            f"Cases output already exists:\n"
            f"{cases_output}\n\n"
            "Use --force to rebuild it."
        )

    if messages_output.exists() and not force:
        raise FileExistsError(
            f"Messages output already exists:\n"
            f"{messages_output}\n\n"
            "Use --force to rebuild it."
        )

    cases_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    messages_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("AppleSupport Case Reconstruction")
    print("=" * 70)

    print(
        f"[INFO] Database       : {db_path}"
    )

    print(
        f"[INFO] Cases output   : {cases_output}"
    )

    print(
        f"[INFO] Messages output: {messages_output}"
    )

    overall_start = perf_counter()

    connection = sqlite3.connect(
        str(db_path)
    )

    connection.row_factory = sqlite3.Row

    try:

        configure_connection(
            connection
        )

        # -------------------------------------------------------------------
        # Database sanity check
        # -------------------------------------------------------------------

        apple_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM tweets
            WHERE author_id = ?
            """,
            (BRAND,),
        ).fetchone()[0]

        print(
            f"\n[INFO] AppleSupport tweets in DB: "
            f"{apple_count:,}"
        )

        # -------------------------------------------------------------------
        # Build relevant in-memory structures.
        # -------------------------------------------------------------------

        tweet_index = build_relevant_tweet_index(
            connection
        )

        children_index = build_children_index(
            tweet_index
        )

        # -------------------------------------------------------------------
        # Find anchors.
        # -------------------------------------------------------------------

        print("\n[INFO] Finding customer support anchors...")

        anchor_start = perf_counter()

        anchors = get_anchors(
            connection
        )

        anchor_elapsed = (
            perf_counter()
            - anchor_start
        )

        print(
            f"[OK] Anchor query executed "
            f"({anchor_elapsed:.1f}s)"
        )

        # -------------------------------------------------------------------
        # Stream reconstructed cases.
        # -------------------------------------------------------------------

        case_count = 0
        message_count = 0
        skipped_count = 0

        print(
            "\n[INFO] Reconstructing cases..."
        )

        with cases_output.open(
            "w",
            encoding="utf-8",
        ) as cases_file, messages_output.open(
            "w",
            encoding="utf-8",
        ) as messages_file:

            for anchor in anchors:

                case = reconstruct_case(
                    anchor,
                    tweet_index,
                    children_index,
                )

                if case is None:
                    skipped_count += 1
                    continue

                write_case(
                    cases_file,
                    case,
                )

                for message in case["messages"]:

                    write_message(
                        messages_file,
                        case["case_id"],
                        message,
                    )

                    message_count += 1

                case_count += 1

                if case_count % 5_000 == 0:

                    elapsed = (
                        perf_counter()
                        - overall_start
                    )

                    print(
                        f"[PROGRESS] "
                        f"Cases: {case_count:,} | "
                        f"Messages: {message_count:,} | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

        # -------------------------------------------------------------------
        # Summary.
        # -------------------------------------------------------------------

        elapsed = (
            perf_counter()
            - overall_start
        )

        print(
            "\n" + "-" * 70
        )

        print(
            "RECONSTRUCTION SUMMARY"
        )

        print(
            "-" * 70
        )

        print(
            f"AppleSupport tweets    : "
            f"{apple_count:,}"
        )

        print(
            f"Cases reconstructed    : "
            f"{case_count:,}"
        )

        print(
            f"Cases skipped          : "
            f"{skipped_count:,}"
        )

        print(
            f"Messages reconstructed : "
            f"{message_count:,}"
        )

        print(
            f"Runtime                : "
            f"{elapsed:.1f}s"
        )

        print(
            "\n[OK] Case reconstruction completed."
        )

        validate_outputs(
            cases_output,
            messages_output,
        )

    finally:
        connection.close()

    print(
        "\n" + "=" * 70
    )

    print(
        "RECONSTRUCTION COMPLETE"
    )

    print(
        "=" * 70
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Efficiently reconstruct AppleSupport "
            "support cases from TWCS."
        )
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to twcs.sqlite",
    )

    parser.add_argument(
        "--cases-output",
        type=Path,
        default=DEFAULT_CASES_OUTPUT,
        help="Output apple_cases.jsonl",
    )

    parser.add_argument(
        "--messages-output",
        type=Path,
        default=DEFAULT_MESSAGES_OUTPUT,
        help="Output apple_messages.jsonl",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    try:

        reconstruct_cases(
            db_path=args.db,
            cases_output=args.cases_output,
            messages_output=args.messages_output,
            force=args.force,
        )

        return 0

    except KeyboardInterrupt:

        print(
            "\n[ERROR] Reconstruction interrupted."
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

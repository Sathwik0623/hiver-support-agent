"""
Build a SQLite index from the TWCS customer-support Twitter dataset.

Input:
    data/raw/twcs.csv

Output:
    data/processed/twcs.sqlite

Usage:
    python src/data/build_index.py

Rebuild from scratch:
    python src/data/build_index.py --force

Custom paths:
    python src/data/build_index.py --csv path/to/twcs.csv --db path/to/twcs.sqlite
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path
from time import perf_counter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
DEFAULT_DB = PROJECT_ROOT / "data" / "processed" / "twcs.sqlite"

REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}

BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id INTEGER PRIMARY KEY,
    author_id TEXT NOT NULL,
    inbound INTEGER NOT NULL,
    created_at TEXT,
    text TEXT,
    response_tweet_id TEXT,
    in_response_to_tweet_id INTEGER
);
"""

CREATE_INDEXES_SQL = [
    """
    CREATE INDEX IF NOT EXISTS idx_tweets_parent
    ON tweets(in_response_to_tweet_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tweets_author
    ON tweets(author_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tweets_inbound
    ON tweets(inbound);
    """,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_inbound(value: str) -> int:
    """
    Convert TWCS inbound values into SQLite integers.

    True  -> 1
    False -> 0
    """
    value = value.strip().lower()

    if value == "true":
        return 1

    if value == "false":
        return 0

    raise ValueError(f"Unexpected inbound value: {value!r}")


def parse_optional_int(value: str | None) -> int | None:
    """
    Convert an optional numeric field into an integer.

    Empty strings become None.
    """
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    return int(value)


def validate_csv_header(csv_path: Path) -> list[str]:
    """
    Validate that the CSV contains all required TWCS columns.
    """
    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)
        header = next(reader, None)

    if header is None:
        raise ValueError("CSV file is empty.")

    missing = REQUIRED_COLUMNS - set(header)

    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return header


def configure_connection(connection: sqlite3.Connection) -> None:
    """
    Configure SQLite for faster bulk insertion.

    These settings are appropriate because this script is building
    a local derived index from an immutable source CSV.
    """
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA temp_store=MEMORY;")


def create_schema(connection: sqlite3.Connection) -> None:
    """
    Create the tweets table.
    """
    connection.execute(CREATE_TABLE_SQL)
    connection.commit()


def create_indexes(connection: sqlite3.Connection) -> None:
    """
    Create lookup indexes after data insertion.

    Creating indexes after bulk insertion is faster than maintaining
    them during every insert.
    """
    for sql in CREATE_INDEXES_SQL:
        connection.execute(sql)

    connection.commit()


# ---------------------------------------------------------------------------
# Main build process
# ---------------------------------------------------------------------------

def build_index(
    csv_path: Path,
    db_path: Path,
    force: bool = False,
) -> None:
    """
    Build the SQLite index from the TWCS CSV.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"TWCS CSV not found:\n{csv_path}"
        )

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if db_path.exists():

        if not force:
            raise FileExistsError(
                f"Database already exists:\n{db_path}\n\n"
                "Use --force if you intentionally want to rebuild it."
            )

        print(f"[INFO] Removing existing database: {db_path}")
        db_path.unlink()

        # SQLite WAL/SHM files can remain after an interrupted process.
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)

            if sidecar.exists():
                sidecar.unlink()

    print("=" * 70)
    print("TWCS → SQLite Index Builder")
    print("=" * 70)

    print(f"[INFO] Source CSV : {csv_path}")
    print(f"[INFO] Output DB  : {db_path}")

    print("\n[INFO] Validating CSV header...")

    header = validate_csv_header(csv_path)

    print("[OK] Required columns found.")

    # SQLite insert order.
    insert_sql = """
        INSERT INTO tweets (
            tweet_id,
            author_id,
            inbound,
            created_at,
            text,
            response_tweet_id,
            in_response_to_tweet_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    start_time = perf_counter()

    connection = sqlite3.connect(str(db_path))

    try:
        configure_connection(connection)
        create_schema(connection)

        print("[INFO] Starting CSV import...")

        inserted = 0
        batch = []

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            # Safety check against DictReader/header mismatch.
            if reader.fieldnames != header:
                print(
                    "[WARNING] CSV header changed between validation "
                    "and reading."
                )

            for row_number, row in enumerate(reader, start=2):

                try:
                    tweet_id = int(row["tweet_id"])

                    author_id = row["author_id"]

                    inbound = parse_inbound(row["inbound"])

                    created_at = row["created_at"]

                    text = row["text"]

                    response_tweet_id = row["response_tweet_id"]

                    in_response_to_tweet_id = parse_optional_int(
                        row["in_response_to_tweet_id"]
                    )

                    batch.append(
                        (
                            tweet_id,
                            author_id,
                            inbound,
                            created_at,
                            text,
                            response_tweet_id,
                            in_response_to_tweet_id,
                        )
                    )

                except Exception as exc:
                    raise ValueError(
                        f"Failed parsing CSV row {row_number}: {exc}"
                    ) from exc

                if len(batch) >= BATCH_SIZE:

                    connection.executemany(
                        insert_sql,
                        batch,
                    )

                    connection.commit()

                    inserted += len(batch)

                    batch.clear()

                    if inserted % 100_000 == 0:
                        elapsed = perf_counter() - start_time

                        print(
                            f"[PROGRESS] "
                            f"{inserted:,} rows inserted "
                            f"({elapsed:.1f}s)"
                        )

            # Insert final partial batch.
            if batch:

                connection.executemany(
                    insert_sql,
                    batch,
                )

                connection.commit()

                inserted += len(batch)

                batch.clear()

        print(
            f"\n[OK] Imported {inserted:,} tweets."
        )

        print("[INFO] Creating database indexes...")

        create_indexes(connection)

        print("[OK] Indexes created.")

        elapsed = perf_counter() - start_time

        print(
            f"\n[INFO] Data import completed in "
            f"{elapsed:.1f} seconds."
        )

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        print("\n[INFO] Running validation queries...")

        total_rows = connection.execute(
            "SELECT COUNT(*) FROM tweets"
        ).fetchone()[0]

        apple_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM tweets
            WHERE author_id = 'AppleSupport'
            """
        ).fetchone()[0]

        inbound_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM tweets
            WHERE inbound = 1
            """
        ).fetchone()[0]

        outbound_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM tweets
            WHERE inbound = 0
            """
        ).fetchone()[0]

        print(f"[VALIDATION] Total tweets      : {total_rows:,}")
        print(f"[VALIDATION] AppleSupport rows : {apple_rows:,}")
        print(f"[VALIDATION] Inbound tweets    : {inbound_rows:,}")
        print(f"[VALIDATION] Outbound tweets   : {outbound_rows:,}")

        if total_rows != inserted:
            raise RuntimeError(
                f"Row count mismatch: inserted={inserted}, "
                f"database={total_rows}"
            )

        print("\n[OK] Database validation passed.")

    finally:
        connection.close()

    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"SQLite database: {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Build a SQLite index from the TWCS dataset."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to twcs.csv (default: {DEFAULT_CSV})",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Output SQLite database (default: {DEFAULT_DB})",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild an existing SQLite database.",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    try:
        build_index(
            csv_path=args.csv,
            db_path=args.db,
            force=args.force,
        )

        return 0

    except KeyboardInterrupt:
        print("\n[ERROR] Build interrupted by user.")
        return 130

    except Exception as exc:
        print(
            f"\n[ERROR] {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())

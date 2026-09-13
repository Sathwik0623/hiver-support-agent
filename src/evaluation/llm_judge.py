import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "apple_support_golden_set.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_retrieval_results.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_llm_judgments.jsonl"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")


# ---------------------------------------------------------------------
# Structured judge output
# ---------------------------------------------------------------------

class JudgeResult(BaseModel):
    problem_relevance: int = Field(
        ge=0,
        le=4,
        description=(
            "How relevant the historical case is to the current "
            "customer problem. 0 = unrelated, 4 = highly relevant."
        ),
    )

    resolution_usefulness: int = Field(
        ge=0,
        le=4,
        description=(
            "How useful the historical response/resolution is for "
            "helping resolve the current problem. "
            "0 = useless, 4 = highly useful."
        ),
    )

    context_compatibility: int = Field(
        ge=0,
        le=4,
        description=(
            "How compatible the historical case context is with the "
            "current case. Consider product, symptoms, constraints, "
            "and conversation state."
        ),
    )

    grounding_value: int = Field(
        ge=0,
        le=4,
        description=(
            "How strongly the historical case can safely ground a "
            "response to the current customer. "
            "0 = should not be used, 4 = strong grounding evidence."
        ),
    )

    overall_score: int = Field(
        ge=0,
        le=4,
        description=(
            "Overall usefulness of this historical case as evidence "
            "for the current customer problem."
        ),
    )

    decision: Literal[
        "USE",
        "USE_WITH_CAUTION",
        "DO_NOT_USE",
    ] = Field(
        description=(
            "Whether the historical case should be used as evidence "
            "for answering the current customer."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation of the most important evidence "
            "supporting the judgment."
        )
    )


# ---------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are an expert evaluator for an AI customer-support retrieval system.

You are NOT generating a customer response.

Your job is to judge whether a retrieved historical support case is
useful evidence for the CURRENT customer problem.

Important rules:

1. Judge semantic and operational relevance, not just word overlap.
2. A case can be lexically similar but still have an incompatible
   resolution.
3. Do not assume that a historical response actually solved the issue
   merely because an agent replied.
4. Treat UNKNOWN resolution signals conservatively.
5. Do not invent facts that are absent from the provided evidence.
6. A historical case should only receive high grounding value when its
   response/resolution is plausibly applicable to the current problem.
7. If the current and historical cases have materially different
   causes, contexts, products, or troubleshooting states, reduce the
   score.
8. Consider whether using the historical response could mislead an
   automated support agent.
9. Be especially conservative when the issue involves account security,
   payments, data loss, hardware safety, or escalation-worthy situations.
"""


def build_prompt(current_case: dict, historical_case: dict) -> str:
    return f"""
{SYSTEM_INSTRUCTION}

CURRENT CUSTOMER CASE
---------------------
Customer problem:
{current_case.get("customer_message", "")}

Expected intent:
{current_case.get("annotator_intent", "")}

Conversation state:
{current_case.get("conversation_state", "")}

Expected action:
{current_case.get("expected_action", "")}

Risk flags:
{current_case.get("risk_flags", "")}


HISTORICAL SUPPORT CASE
-----------------------
Historical customer problem:
{historical_case.get("customer_problem", "")}

Historical conversation context:
{historical_case.get("conversation_context", "")}

Historical Apple response:
{historical_case.get("apple_response", "")}

Historical customer follow-up:
{historical_case.get("follow_up", "")}

Historical resolution signal:
{historical_case.get("resolution_signal", "")}

Historical resolution confidence:
{historical_case.get("resolution_confidence", "")}

Historical resolution reason:
{historical_case.get("resolution_reason", "")}


Evaluate the historical case as evidence for the CURRENT customer case.

Return only the requested structured evaluation.
"""


# ---------------------------------------------------------------------
# Single judgment
# ---------------------------------------------------------------------

def judge_case(current_case: dict, historical_case: dict) -> JudgeResult:
    prompt = build_prompt(current_case, historical_case)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": JudgeResult.model_json_schema(),
        },
    )

    return JudgeResult.model_validate_json(response.text)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_golden(path: Path) -> dict:
    import csv

    records = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            records[row["case_id"]] = row

    return records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Gemini judge model: {MODEL}")
    print(f"Results input: {RESULTS_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    retrieval_results = load_jsonl(RESULTS_PATH)
    golden = load_golden(GOLDEN_PATH)

    print(f"Retrieval results: {len(retrieval_results)}")
    print(f"Golden cases: {len(golden)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    processed.add(record["case_id"])

        print(f"Already judged: {len(processed)}")

    count = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as out:

        for result in retrieval_results:

            case_id = result["case_id"]

            if case_id in processed:
                continue

            current_case = golden.get(case_id)

            if not current_case:
                print(f"[WARN] Missing golden case: {case_id}")
                continue

            retrieved = result.get("retrieved", [])

            if not retrieved:
                print(f"[WARN] No retrieved evidence: {case_id}")
                continue

            # For the first pass, judge Top-1 only.
            historical_case = retrieved[0]

            try:
                judgment = judge_case(
                    current_case=current_case,
                    historical_case=historical_case,
                )

                output = {
                    "case_id": case_id,
                    "retrieval_rank": 1,
                    "historical_case_id": historical_case.get("case_id"),
                    "historical_evidence_id": historical_case.get(
                        "evidence_id"
                    ),
                    "retrieval_similarity": historical_case.get(
                        "similarity"
                    ),
                    "judge": judgment.model_dump(),
                    "judge_model": MODEL,
                }

                out.write(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                out.flush()

                count += 1

                print(
                    f"[{count}] {case_id} "
                    f"score={judgment.overall_score} "
                    f"decision={judgment.decision}"
                )

                # Small delay to avoid unnecessary burst traffic.
                time.sleep(0.5)

            except Exception as exc:
                print(
                    f"[ERROR] {case_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

                # Do not destroy previous results if a transient
                # API failure occurs.
                time.sleep(5)

    print()
    print("LLM judge run complete.")
    print(f"New judgments: {count}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

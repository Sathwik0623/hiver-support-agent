"""
Minimal end-to-end support-agent pipeline.

Flow:
SupportCase -> IssueAnalysis -> EvidenceAssessment -> AgentDecision
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .issue_analyzer import analyze_case
from .policy import decide_action
from .schemas import (
    AgentResponse,
    EvidenceAssessment,
    EvidenceDecision,
    Message,
    RetrievedEvidence,
    SupportCase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / "data" / "processed" / "historical_evidence.jsonl"


def _tokens(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def _lexical_similarity(query: str, candidate: str) -> float:
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)

    if not query_tokens or not candidate_tokens:
        return 0.0

    intersection = len(query_tokens & candidate_tokens)
    union = len(query_tokens | candidate_tokens)

    return intersection / union if union else 0.0


def load_evidence(limit: int = 50000) -> list[dict[str, Any]]:
    """
    Load a bounded evidence sample for the interactive demo.

    The full TF-IDF and semantic experiments remain separate and are
    not rerun by this pipeline.
    """
    if not EVIDENCE_PATH.exists():
        return []

    records: list[dict[str, Any]] = []

    with EVIDENCE_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            records.append(json.loads(line))

            if len(records) >= limit:
                break

    return records


def retrieve_evidence(
    customer_text: str,
    top_k: int = 3,
) -> list[RetrievedEvidence]:
    """
    Lightweight lexical retrieval for the runnable demo.

    Production retrieval is evaluated separately using TF-IDF and
    semantic embeddings.
    """
    records = load_evidence()

    ranked: list[tuple[float, dict[str, Any]]] = []

    for record in records:
        historical_text = " ".join(
            [
                str(record.get("customer_problem", "")),
                str(record.get("conversation_context", "")),
            ]
        )

        score = _lexical_similarity(customer_text, historical_text)
        ranked.append((score, record))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results: list[RetrievedEvidence] = []

    for score, record in ranked[:top_k]:
        results.append(
            RetrievedEvidence(
                evidence_id=str(record.get("evidence_id", "")),
                case_id=str(record.get("case_id", "")),
                similarity=max(0.0, min(1.0, score)),
                customer_problem=str(
                    record.get("customer_problem", "")
                ),
                conversation_context=str(
                    record.get("conversation_context", "")
                ),
                apple_response=str(
                    record.get("apple_response", "")
                ),
                follow_up=str(record.get("follow_up", "")),
                resolution_signal=str(
                    record.get("resolution_signal", "UNKNOWN")
                ),
                resolution_confidence=float(
                    record.get("resolution_confidence", 0.0)
                ),
                resolution_reason=str(
                    record.get("resolution_reason", "")
                ),
            )
        )

    return results


def assess_evidence(
    evidence: list[RetrievedEvidence],
) -> list[EvidenceAssessment]:
    """
    Conservative deterministic evidence gate.

    Only evidence with a resolved historical outcome and a usable
    response is eligible for automatic handling.
    """
    assessments: list[EvidenceAssessment] = []

    for item in evidence:
        resolved = item.resolution_signal == "RESOLVED"
        has_response = bool(item.apple_response.strip())
        similarity = item.similarity

        if resolved and has_response and similarity >= 0.20:
            relevance = min(1.0, similarity + 0.20)
            usefulness = 0.85
            compatibility = 0.75
            grounding = 0.80
            decision = EvidenceDecision.USE
            reason = (
                "Historical case is marked resolved, contains a response, "
                "and has sufficient lexical overlap."
            )
        else:
            relevance = similarity
            usefulness = 0.20 if not resolved else 0.55
            compatibility = 0.30
            grounding = 0.20
            decision = EvidenceDecision.DO_NOT_USE
            reason = (
                "Evidence did not pass the conservative resolved-case "
                "and similarity checks."
            )

        assessments.append(
            EvidenceAssessment(
                evidence_id=item.evidence_id,
                problem_relevance=relevance,
                resolution_usefulness=usefulness,
                context_compatibility=compatibility,
                grounding_value=grounding,
                already_tried=False,
                decision=decision,
                reason=reason,
            )
        )

    return assessments


def build_draft_response(
    decision_action: str,
    analysis: Any,
    evidence: list[RetrievedEvidence],
) -> str | None:
    """
    Generate a transparent demo response.

    This intentionally avoids pretending that an unvalidated historical
    response is safe to send.
    """
    if decision_action == "CLARIFY":
        return (
            "Thanks for contacting Apple Support. To help investigate this, "
            "could you please share the exact device model, iOS version, "
            "and the steps that lead to the issue?"
        )

    if decision_action == "ESCALATE":
        return (
            "Thanks for reaching out. This issue needs further review by "
	     "a support specialist. Your case has been flagged for human "
	     "assistance so that we do not recommend an unsuitable or "
	     "repeated troubleshooting step."
        )

    selected = [
        item
        for item in evidence
        if item.resolution_signal == "RESOLVED"
        and item.apple_response.strip()
    ]

    if not selected:
        return (
            "Thanks for contacting Apple Support. We need a little more "
            "information before recommending a reliable next step."
        )

    response = selected[0].apple_response.strip()

    return (
        "Thanks for contacting Apple Support. Based on a similar historical "
        "case, the following guidance may help:\n\n"
        f"{response}\n\n"
        "If this does not resolve the issue, please let us know what happened "
        "so a support specialist can review the case."
    )


def run_agent(
    customer_message: str,
    case_id: str = "DEMO_CASE_001",
) -> AgentResponse:
    case = SupportCase(
        case_id=case_id,
        customer_id="DEMO_CUSTOMER",
        brand="AppleSupport",
        messages=[
            Message(
                message_id=f"{case_id}_MESSAGE_001",
                sender="customer",
                text=customer_message,
            )
        ],
    )

    analysis = analyze_case(case)
    retrieved = retrieve_evidence(customer_message)
    assessments = assess_evidence(retrieved)
    decision = decide_action(analysis, assessments)

    draft = build_draft_response(
        decision.action.value,
        analysis,
        retrieved,
    )

    return AgentResponse(
        case_id=case_id,
        analysis=analysis,
        evidence=assessments,
        decision=decision,
        draft_response=draft,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Hiver support-agent demo."
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=(
            "My iPhone keeps freezing after the latest iOS update. "
            "What should I do?"
        ),
    )
    args = parser.parse_args()

    result = run_agent(args.message)

    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()

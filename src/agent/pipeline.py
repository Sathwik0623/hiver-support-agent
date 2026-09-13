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
    Intent,
    Message,
    RetrievedEvidence,
    SupportCase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / "data" / "processed" / "historical_evidence.jsonl"


def is_reusable_resolution(record: dict) -> bool:
    response = (record.get("apple_response") or "").lower()
    follow_up = (record.get("follow_up") or "").lower()

    acknowledgement_only = [
        "thank you",
        "thanks",
        "awesome",
        "you're welcome",
        "will do",
    ]

    if any(phrase in follow_up for phrase in acknowledgement_only):
        concrete_guidance = [
            "check",
            "go to",
            "open",
            "tap",
            "select",
            "contact",
            "update",
            "restart",
            "reset",
            "install",
            "enable",
            "disable",
            "article",
        ]

        if not any(phrase in response for phrase in concrete_guidance):
            return False

    routing_only = [
        "contact our",
        "contact support",
        "send us a dm",
        
        "join us in a dm",
        "join us in dm",
        "please join us in a dm",
        "in dm",
        "reach out to",
        "go to dm",
        "let's go to dm",
        "lets go to dm",
    ]

    if any(phrase in response for phrase in routing_only):
        return False

    return True

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


    if decision_action == "AUTO_HANDLE":

            
            usable_evidence = [
                item
                for item in evidence
                if item.apple_response.strip()
                and item.resolution_signal == "RESOLVED"
                and item.similarity >= 0.20
            ]

            if not usable_evidence:
                return (
                    "Thanks for contacting Apple Support. "
                    "Your issue appears to be resolved, or no further "
                    "verified troubleshooting is required."
                )

            best_evidence = max(
                usable_evidence,
                key=lambda item: item.similarity,
            )

            return (
                "Thanks for contacting Apple Support. "
                "Based on a similar previously resolved case, "
                "the following guidance may help:\n\n"
                f"{best_evidence.apple_response.strip()}\n\n"
                "If the issue continues, please let us know so a support "
                "specialist can review the case."
            )
    if decision_action == "CLARIFY":
        return (
            "Thanks for contacting Apple Support. To help investigate this, "
            "could you please share the exact device model, iOS version, "
            "and the steps that lead to the issue?"
        )

    if decision_action == "ESCALATE":
        intent = analysis.intent
        issue_details = analysis.entities.issue_details

        symptom = issue_details.get("symptom")

        if intent == Intent.DEVICE_HARDWARE:
            if symptom == "battery_or_charging_issue":
                return (
                    "Thanks for reaching out. We can help investigate the "
                    "battery drain issue. Could you please share your iPhone "
                    "model, current iOS version, battery health percentage, "
                    "when the battery drain started, and whether the device "
                    "becomes unusually warm? Please also let us know whether "
                    "the drain happens while using a specific app or even "
                    "when the phone is idle. Since we do not have a verified "
                    "historical fix for this exact issue, your case has been "
                    "flagged for human assistance."
                )

            if symptom == "display_issue":
                return (
                    "Thanks for reaching out. We can help investigate the "
                    "display issue. Could you please share your iPhone model, "
                    "current iOS version, when the screen went black, whether  "
                    "the device still makes sounds or vibrates, and whether "
                    "the screen responds to touch? Please also let us know "
                    "whether the issue started after a drop, liquid exposure, "
                    "or software update. Since we do not have a verified "
                    "historical fix for this exact issue, your case has been "
                    "flagged for human assistance."
                )

            

        if intent == "ACCOUNT_ICLOUD":
            account_issue = issue_details.get("account_issue")

            if account_issue == "forgotten_password":
                return (
                    "Thanks for reaching out. We can help with your "
                    "forgotten Apple Account password. Please use Apple's "
                    "official account-recovery process or contact a support "
                    "specialist for further assistance. For security, do not "
                    "share your password, verification codes, or other "
                    "sensitive account information. Your case has been "
                    "flagged for human assistance."
                )

            if account_issue == "region_change":
                return (
                    "Thanks for reaching out. We can help investigate the "
                    "Apple Account region-change issue. Could you please "
                    "share the exact error message and confirm whether you "
                    "have any remaining account balance, active subscriptions, "
                    "pending refunds, or Family Sharing membership? For "
                    "security, do not share your password or verification "
                    "codes. Your case has been flagged for human assistance."
                )

            return (
                "Thanks for reaching out. We can help investigate your "
                "Apple Account issue. Could you please describe the exact "
                "problem and share any error message? For security, do not "
                "share your password, verification codes, or other sensitive "
                "account information. Your case has been flagged for human "
                "assistance."
            )

        if intent == "DEVICE_PERFORMANCE":
            return (
                "Thanks for reaching out. We can help investigate the "
                "performance issue after the iOS update. Could you please "
                "share your iPhone model, current iOS version, whether the "
                "freezing happens in all apps or only one app, when the issue "
                "started, and whether you have already restarted the iPhone? "
                "Since we do not have a verified historical fix for this "
                "exact issue, your case has been flagged for human assistance."
            )

        return (
            "Thanks for reaching out. This issue needs further review by a "
            "support specialist. Could you please share the exact device "
            "model, software version, steps that lead to the issue, and any "
            "error message you see? Your case has been flagged for human "
            "assistance so that we do not recommend an unsuitable or "
            "repeated troubleshooting step."
        )


def run_agent(
    customer_message: str | list[str],
    case_id: str = "DEMO_CASE_001",
) -> AgentResponse:
    if isinstance(customer_message, str):
        customer_messages = [customer_message]
    else:
        customer_messages = customer_message

    case = SupportCase(
        case_id=case_id,
        customer_id="DEMO_CUSTOMER",
        brand="AppleSupport",
        messages=[
            Message(
                message_id=f"{case_id}_MESSAGE_{index:03d}",
                sender="customer",
                text=message,
            )
            for index, message in enumerate(customer_messages, start=1)
        ],
    )

    analysis = analyze_case(case)

    combined_customer_text = " ".join(customer_messages)

    retrieved = retrieve_evidence(combined_customer_text)
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

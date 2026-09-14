"""
Deterministic decision policy for the Hiver support agent.

The policy sits after issue analysis and evidence assessment.

The LLM may propose an interpretation, but high-risk decisions
are controlled by explicit rules.
"""

from __future__ import annotations

from typing import List

from .schemas import (
    AgentAction,
    AgentDecision,
    ConversationState,
    EvidenceAssessment,
    EvidenceDecision,
    IssueAnalysis,
    Intent,
    RiskFlag,
)


# ---------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------

MIN_INTENT_CONFIDENCE = 0.70

MIN_AUTO_EVIDENCE_RELEVANCE = 0.70
MIN_AUTO_EVIDENCE_RESOLUTION = 0.70
MIN_AUTO_EVIDENCE_CONTEXT = 0.70
MIN_AUTO_EVIDENCE_GROUNDING = 0.70

MIN_ACTION_CONFIDENCE = 0.75


# ---------------------------------------------------------------------
# HIGH-RISK FLAGS
# ---------------------------------------------------------------------

HIGH_RISK_FLAGS = {
    RiskFlag.ACCOUNT_COMPROMISE,
    RiskFlag.FRAUD,
    RiskFlag.PHISHING,
    RiskFlag.DATA_LOSS,
    RiskFlag.SAFETY_ISSUE,
    RiskFlag.PRIVACY_SENSITIVE,
    RiskFlag.PAYMENT_DISPUTE,
}


# ---------------------------------------------------------------------
# EVIDENCE HELPERS
# ---------------------------------------------------------------------

def is_strong_evidence(
    evidence: EvidenceAssessment,
) -> bool:
    """
    Determine whether evidence is strong enough for automatic
    handling.
    """

    if evidence.decision != EvidenceDecision.USE:
        return False

    if evidence.problem_relevance < MIN_AUTO_EVIDENCE_RELEVANCE:
        return False

    if evidence.resolution_usefulness < MIN_AUTO_EVIDENCE_RESOLUTION:
        return False

    if evidence.context_compatibility < MIN_AUTO_EVIDENCE_CONTEXT:
        return False

    if evidence.grounding_value < MIN_AUTO_EVIDENCE_GROUNDING:
        return False

    if evidence.already_tried:
        return False

    return True


def select_strong_evidence(
    evidence: List[EvidenceAssessment],
) -> List[EvidenceAssessment]:
    """
    Return evidence that passes all automatic-use checks.
    """

    return [
        item
        for item in evidence
        if is_strong_evidence(item)
    ]




def calculate_action_confidence(
    evidence: List[EvidenceAssessment],
) -> float:
    """
    Calculate action confidence from the strongest selected evidence.

    The confidence is bounded so that automatic handling never appears
    artificially certain.
    """

    if not evidence:
        return 0.0

    strongest_evidence = max(
        evidence,
        key=lambda item: (
            item.problem_relevance
            + item.resolution_usefulness
            + item.context_compatibility
            + item.grounding_value
        ),
    )

    average_score = (
        strongest_evidence.problem_relevance
        + strongest_evidence.resolution_usefulness
        + strongest_evidence.context_compatibility
        + strongest_evidence.grounding_value
    ) / 4

    return round(
        min(max(average_score, MIN_ACTION_CONFIDENCE), 0.95),
        2,
    )

# ---------------------------------------------------------------------
# POLICY
# ---------------------------------------------------------------------

def decide_action(
    analysis: IssueAnalysis,
    evidence: List[EvidenceAssessment],
) -> AgentDecision:
    """
    Decide whether the case should be automatically handled,
    clarified, or escalated.

    Priority order:

        1. High-risk situations
        2. Failed troubleshooting
        3. Resolved cases
        4. Low-confidence / unclear cases
        5. Missing evidence
        6. Strong evidence -> AUTO_HANDLE
    """

    # -------------------------------------------------------------
    # 1. High-risk cases
    # -------------------------------------------------------------

    high_risk = [
        flag
        for flag in analysis.risk_flags
        if flag in HIGH_RISK_FLAGS
    ]

    if high_risk:
        flags = ", ".join(
            flag.value
            for flag in high_risk
        )

        return AgentDecision(
            action=AgentAction.ESCALATE,
            reason=(
                "High-risk support condition detected: "
                f"{flags}. Human review is required."
            ),
            action_confidence=0.98,
            requires_human=True,
            selected_evidence_ids=[],
        )

    # -------------------------------------------------------------
    # 2. Failed troubleshooting
    # -------------------------------------------------------------

    if (
        analysis.conversation_state
        == ConversationState.FAILED_TROUBLESHOOTING
    ):
        return AgentDecision(
            action=AgentAction.ESCALATE,
            reason=(
                "Previous troubleshooting has already failed. "
                "Further automated troubleshooting may repeat "
                "unsuccessful steps."
            ),
            action_confidence=0.95,
            requires_human=True,
            selected_evidence_ids=[],
        )

    # -------------------------------------------------------------
    # 3. Already resolved
    # -------------------------------------------------------------

    if (
        analysis.conversation_state
        == ConversationState.RESOLVED
    ):
        return AgentDecision(
            action=AgentAction.AUTO_HANDLE,
            reason=(
                "The conversation is already marked as resolved; "
                "no further escalation is required."
            ),
            action_confidence=0.95,
            requires_human=False,
            selected_evidence_ids=[],
        )

    # -------------------------------------------------------------
    # 4. Low intent confidence
    # -------------------------------------------------------------

    if analysis.intent_confidence < MIN_INTENT_CONFIDENCE:
        return AgentDecision(
            action=AgentAction.CLARIFY,
            reason=(
                "The customer's issue cannot be classified "
                "with sufficient confidence. Clarification is "
                "required before recommending a resolution."
            ),
            action_confidence=0.80,
            requires_human=False,
            selected_evidence_ids=[],
        )

    # -------------------------------------------------------------
    # 5. OTHER_UNCLEAR
    # -------------------------------------------------------------

    if analysis.intent == Intent.OTHER_UNCLEAR:
        return AgentDecision(
            action=AgentAction.CLARIFY,
            reason=(
                "The issue is ambiguous or outside the supported "
                "intent taxonomy. Clarification is required."
            ),
            action_confidence=0.85,
            requires_human=False,
            selected_evidence_ids=[],
        )

    # -------------------------------------------------------------
    # 6. Select strong evidence
    # -------------------------------------------------------------

    strong_evidence = select_strong_evidence(evidence)

    if not strong_evidence:
        return AgentDecision(
            action=AgentAction.ESCALATE,
            reason=(
                "No historical evidence passed the relevance, "
		"resolution usefulness, context compatibility, and grounding checks "
		"required for automatic handling."
            ),
            action_confidence=0.90,
            requires_human=True,
            selected_evidence_ids=[],
        )

    # -------------------------------------------------------------
    # 7. AUTO_HANDLE
    # -------------------------------------------------------------

    selected_ids = [
        item.evidence_id
        for item in strong_evidence
    ]

    return AgentDecision(
        action=AgentAction.AUTO_HANDLE,
        reason=(
            "High-confidence issue classification with compatible "
            "historical resolution evidence and no high-risk or "
            "failed-troubleshooting conditions."
        ),
        action_confidence=calculate_action_confidence(
    strong_evidence
),
        requires_human=False,
        selected_evidence_ids=selected_ids,
    )

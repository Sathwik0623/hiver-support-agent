from src.agent.policy import decide_action
from src.agent.schemas import (
    AgentAction,
    CaseEntities,
    ConversationState,
    EvidenceAssessment,
    EvidenceDecision,
    Intent,
    IssueAnalysis,
    RiskFlag,
)


def make_analysis(
    intent=Intent.DEVICE_PERFORMANCE,
    state=ConversationState.TROUBLESHOOTING,
    confidence=0.90,
    risk_flags=None,
):
    return IssueAnalysis(
        intent=intent,
        conversation_state=state,
        intent_confidence=confidence,
        entities=CaseEntities(),
        risk_flags=risk_flags or [],
    )


def make_good_evidence():
    return EvidenceAssessment(
        evidence_id="EVIDENCE_001",
        problem_relevance=0.90,
        resolution_usefulness=0.90,
        context_compatibility=0.90,
        grounding_value=0.90,
        already_tried=False,
        decision=EvidenceDecision.USE,
        reason="Directly relevant resolved historical case.",
    )


def test_high_quality_case_auto_handles():
    analysis = make_analysis()

    decision = decide_action(
        analysis,
        [make_good_evidence()],
    )

    assert decision.action == AgentAction.AUTO_HANDLE
    assert decision.requires_human is False


def test_high_risk_case_escalates():
    analysis = make_analysis(
        risk_flags=[RiskFlag.ACCOUNT_COMPROMISE]
    )

    decision = decide_action(
        analysis,
        [make_good_evidence()],
    )

    assert decision.action == AgentAction.ESCALATE
    assert decision.requires_human is True


def test_failed_troubleshooting_escalates():
    analysis = make_analysis(
        state=ConversationState.FAILED_TROUBLESHOOTING
    )

    decision = decide_action(
        analysis,
        [make_good_evidence()],
    )

    assert decision.action == AgentAction.ESCALATE


def test_low_confidence_clarifies():
    analysis = make_analysis(
        confidence=0.50
    )

    decision = decide_action(
        analysis,
        [make_good_evidence()],
    )

    assert decision.action == AgentAction.CLARIFY


def test_other_unclear_clarifies():
    analysis = make_analysis(
        intent=Intent.OTHER_UNCLEAR
    )

    decision = decide_action(
        analysis,
        [make_good_evidence()],
    )

    assert decision.action == AgentAction.CLARIFY


def test_weak_evidence_escalates():
    analysis = make_analysis()

    weak_evidence = EvidenceAssessment(
        evidence_id="EVIDENCE_002",
        problem_relevance=0.90,
        resolution_usefulness=0.30,
        context_compatibility=0.90,
        grounding_value=0.30,
        already_tried=False,
        decision=EvidenceDecision.USE_WITH_CAUTION,
        reason="Relevant but insufficient resolution evidence.",
    )

    decision = decide_action(
        analysis,
        [weak_evidence],
    )

    assert decision.action == AgentAction.ESCALATE


def test_already_tried_evidence_is_not_auto_used():
    analysis = make_analysis()

    evidence = EvidenceAssessment(
        evidence_id="EVIDENCE_003",
        problem_relevance=0.95,
        resolution_usefulness=0.95,
        context_compatibility=0.95,
        grounding_value=0.95,
        already_tried=True,
        decision=EvidenceDecision.USE,
        reason="Customer already attempted this recommendation.",
    )

    decision = decide_action(
        analysis,
        [evidence],
    )

    assert decision.action == AgentAction.ESCALATE


def test_auto_handle_confidence_is_derived_from_evidence():
    evidence = [
        EvidenceAssessment(
            evidence_id="strong-1",
            problem_relevance=0.95,
            resolution_usefulness=0.90,
            context_compatibility=0.85,
            grounding_value=0.90,
            already_tried=False,
            decision=EvidenceDecision.USE,
            reason="Strong compatible historical resolution.",
        )
    ]

    analysis = IssueAnalysis(
        intent=Intent.DEVICE_PERFORMANCE,
        conversation_state=ConversationState.TROUBLESHOOTING,
        intent_confidence=0.95,
    )

    decision = decide_action(analysis, evidence)

    assert decision.action == AgentAction.AUTO_HANDLE
    assert decision.requires_human is False
    assert decision.action_confidence == 0.90

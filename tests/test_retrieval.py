from src.agent.pipeline import (
    _lexical_similarity,
    assess_evidence,
    is_reusable_resolution,
)
from src.agent.schemas import (
    EvidenceDecision,
    RetrievedEvidence,
)


def make_evidence(
    *,
    similarity: float = 0.80,
    resolution_signal: str = "RESOLVED",
    apple_response: str = "Please restart your iPhone and check again.",
    follow_up: str = "It works now, thank you.",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="TEST_EVIDENCE_001",
        case_id="TEST_CASE_001",
        similarity=similarity,
        customer_problem="My iPhone keeps freezing.",
        conversation_context="Customer reported freezing.",
        apple_response=apple_response,
        follow_up=follow_up,
        resolution_signal=resolution_signal,
        resolution_confidence=0.95,
        resolution_reason="Customer confirmed the issue was resolved.",
    )


def test_lexical_similarity_identical_text():
    score = _lexical_similarity(
        "iPhone freezing",
        "iPhone freezing",
    )

    assert score == 1.0


def test_lexical_similarity_unrelated_text():
    score = _lexical_similarity(
        "iPhone freezing",
        "MacBook battery charging",
    )

    assert score == 0.0


def test_routing_only_resolution_is_not_reusable():
    record = {
        "apple_response": (
            "Let's take a closer look. "
            "Please join us in a DM."
        ),
        "follow_up": "",
    }

    assert is_reusable_resolution(record) is False


def test_concrete_resolution_is_reusable():
    record = {
        "apple_response": (
            "Please restart your iPhone and check whether "
            "the issue continues."
        ),
        "follow_up": "It works now, thank you.",
    }

    assert is_reusable_resolution(record) is True


def test_resolved_evidence_passes_assessment():
    evidence = assess_evidence([make_evidence()])

    assert len(evidence) == 1
    assert evidence[0].decision == EvidenceDecision.USE
    assert evidence[0].problem_relevance >= 0.70


def test_unresolved_evidence_is_rejected():
    evidence = assess_evidence(
        [
            make_evidence(
                resolution_signal="UNKNOWN",
            )
        ]
    )

    assert len(evidence) == 1
    assert evidence[0].decision == EvidenceDecision.DO_NOT_USE


def test_low_similarity_evidence_is_rejected():
    evidence = assess_evidence(
        [
            make_evidence(similarity=0.10),
        ]
    )

    assert len(evidence) == 1
    assert evidence[0].decision == EvidenceDecision.DO_NOT_USE
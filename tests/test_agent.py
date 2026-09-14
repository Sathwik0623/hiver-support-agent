from src.agent.pipeline import run_agent
from src.agent.schemas import AgentAction, ConversationState, Intent


def test_run_agent_returns_complete_response():
    result = run_agent(
        "My iPhone keeps freezing after the latest iOS update."
    )

    assert result.case_id == "DEMO_CASE_001"
    assert result.analysis.intent == Intent.DEVICE_PERFORMANCE
    assert result.analysis.conversation_state == ConversationState.TROUBLESHOOTING
    assert result.decision.action in {
        AgentAction.AUTO_HANDLE,
        AgentAction.ESCALATE,
        AgentAction.CLARIFY,
    }


def test_run_agent_handles_battery_issue():
    result = run_agent(
        "My iPhone battery is draining very quickly."
    )

    assert result.analysis.intent.value == "DEVICE_HARDWARE"
    assert result.analysis.entities.issue_details.get(
        "symptom"
    ) == "battery_or_charging_issue"


def test_run_agent_handles_account_security_issue():
    result = run_agent(
        "Someone accessed my Apple Account without permission."
    )

    assert result.analysis.intent.value == "ACCOUNT_ICLOUD"
    assert result.decision.action == AgentAction.ESCALATE
    assert result.decision.requires_human is True


def test_run_agent_handles_unclear_message():
    result = run_agent("Hello, I need help.")

    assert result.decision.action == AgentAction.CLARIFY

def test_run_agent_handles_failed_multi_message_troubleshooting():
    result = run_agent(
        [
            "My iPhone keeps freezing.",
            "I restarted it but it still freezes.",
        ]
    )

    assert result.analysis.intent == Intent.DEVICE_PERFORMANCE
    assert (
        result.analysis.conversation_state
        == ConversationState.FAILED_TROUBLESHOOTING
    )
    assert result.decision.action == AgentAction.ESCALATE
    assert result.decision.requires_human is True


def test_run_agent_handles_resolved_conversation():
    result = run_agent(
        [
            "My iPhone was freezing.",
            "I restarted it and the issue is now resolved.",
        ]
    )

    assert result.analysis.conversation_state == ConversationState.RESOLVED
    assert result.decision.action == AgentAction.AUTO_HANDLE
    assert result.decision.requires_human is False


def test_run_agent_escalates_high_risk_case():
    result = run_agent(
        "Someone accessed my Apple Account without permission."
    )

    assert result.analysis.risk_flags
    assert result.decision.action == AgentAction.ESCALATE
    assert result.decision.requires_human is True


def test_run_agent_generates_response_for_escalated_case():
    result = run_agent(
        "My iPhone keeps freezing after the latest iOS update."
    )

    assert result.draft_response is not None
    assert "human assistance" in result.draft_response.lower()
    assert "iPhone model" in result.draft_response
    assert "iOS version" in result.draft_response


def test_run_agent_generates_response_for_resolved_case():
    result = run_agent(
        [
            "My iPhone was freezing.",
            "I restarted it and the issue is now resolved.",
        ]
    )

    assert result.draft_response is not None
    assert "resolved" in result.draft_response.lower()
    assert "troubleshooting" in result.draft_response.lower()



def test_run_agent_does_not_auto_handle_weak_evidence():
    result = run_agent(
        [
            "My iPhone keeps freezing.",
            "I restarted it but it still freezes.",
        ]
    )

    assert result.decision.action != AgentAction.AUTO_HANDLE
    assert result.decision.action in {
        AgentAction.ESCALATE,
        AgentAction.CLARIFY,
    }
    assert result.decision.requires_human is True


def test_run_agent_handles_empty_message():
    result = run_agent("")

    assert result.decision.action == AgentAction.CLARIFY
    assert result.draft_response is not None


def test_run_agent_handles_whitespace_only_message():
    result = run_agent("   ")

    assert result.decision.action == AgentAction.CLARIFY
    assert result.draft_response is not None


def test_run_agent_handles_empty_message_list():
    result = run_agent([])

    assert result.decision.action == AgentAction.CLARIFY
    assert result.draft_response is not None


def test_run_agent_filters_empty_messages():
    result = run_agent(
        [
            "",
            "   ",
            "My iPhone keeps freezing.",
        ]
    )

    assert result.analysis.intent == Intent.DEVICE_PERFORMANCE


def test_run_agent_handles_mixed_empty_messages():
    result = run_agent(
        [
            "",
            "   ",
            "My iPhone battery is draining very quickly.",
            "",
        ]
    )

    assert result.analysis.intent == Intent.DEVICE_HARDWARE
    assert result.analysis.entities.issue_details.get(
        "symptom"
    ) == "battery_or_charging_issue"


def test_run_agent_handles_list_with_only_empty_messages():
    result = run_agent(["", "   ", "\n"])

    assert result.decision.action == AgentAction.CLARIFY
    assert result.draft_response is not None


def test_run_agent_handles_very_long_message():
    long_message = (
        "My iPhone keeps freezing after the latest iOS update. "
        * 500
    )

    result = run_agent(long_message)

    assert result.analysis.intent == Intent.DEVICE_PERFORMANCE
    assert result.decision.action in {
        AgentAction.AUTO_HANDLE,
        AgentAction.ESCALATE,
        AgentAction.CLARIFY,
    }


def test_run_agent_rejects_unsupported_input_type():
    import pytest

    with pytest.raises((TypeError, ValueError)):
        run_agent(12345)

def test_run_agent_escalation_response_has_correct_spacing():
    result = run_agent("My iPhone keeps freezing.")

    assert result.draft_response is not None
    assert "currentiOS version" not in result.draft_response
    assert "current  iOS version" not in result.draft_response
    assert "current iOS version" in result.draft_response

def test_account_compromise_response_is_risk_specific():
    result = run_agent("My account may have been compromised.")

    assert result.decision.action == AgentAction.ESCALATE
    assert result.decision.requires_human is True
    assert result.draft_response is not None
    assert "account compromise" in result.draft_response.lower()
    assert "password" in result.draft_response.lower()
    assert "verification codes" in result.draft_response.lower()

def test_data_loss_response_is_risk_specific():
    result = run_agent("I lost all my data.")

    assert result.analysis.data_loss is True
    assert "data_loss" in result.analysis.risk_flags
    assert result.decision.action == AgentAction.ESCALATE
    assert result.decision.requires_human is True
    assert result.draft_response is not None
    assert "data loss" in result.draft_response.lower()
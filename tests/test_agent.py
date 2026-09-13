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
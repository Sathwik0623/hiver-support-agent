import pytest

from src.agent.pipeline import run_agent


@pytest.mark.parametrize(
    ("message", "expected_action", "expected_risk"),
    [
        (
            "My account may have been compromised.",
            "ESCALATE",
            "account_compromise",
        ),
        (
            "I lost all my data.",
            "ESCALATE",
            "data_loss",
        ),
        (
            "My iPhone keeps freezing.",
            "ESCALATE",
            None,
        ),
        (
            "I cannot log in to my account.",
            "ESCALATE",
            None,
        ),
    ],
)
def test_high_value_customer_cases(
    message,
    expected_action,
    expected_risk,
):
    result = run_agent(message)

    assert result.decision.action == expected_action

    if expected_risk is not None:
        assert expected_risk in result.analysis.risk_flags
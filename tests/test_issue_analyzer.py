from src.agent.issue_analyzer import analyze_case
from src.agent.schemas import (
    Intent,
    SupportCase,
    Message,
)


def make_case(message: str) -> SupportCase:
    return SupportCase(
        case_id="TEST_CASE_001",
        customer_id="TEST_CUSTOMER",
        brand="AppleSupport",
        messages=[
            Message(
                message_id="TEST_MESSAGE_001",
                sender="customer",
                text=message,
            )
        ],
    )


def test_black_screen_issue():
    case = make_case("My iPhone screen went completely black")

    result = analyze_case(case)

    assert result.intent == Intent.DEVICE_HARDWARE
    assert result.intent_confidence >= 0.90
    assert result.entities.device == "iPhone"
    assert result.entities.issue_details["symptom"] == "display_issue"
    assert "display issue" in result.observed_symptoms


def test_battery_issue():
    case = make_case("My iPhone battery drains very quickly")

    result = analyze_case(case)

    assert result.intent == Intent.DEVICE_HARDWARE
    assert result.entities.issue_details["symptom"] == (
        "battery_or_charging_issue"
    )


def test_software_update_issue():
    case = make_case(
        "My iPhone keeps freezing after the latest iOS update"
    )

    result = analyze_case(case)

    assert result.intent == Intent.DEVICE_PERFORMANCE


def test_account_password_issue():
    case = make_case("I forgot my Apple Account password")

    result = analyze_case(case)

    assert result.intent == Intent.ACCOUNT_ICLOUD
    assert result.entities.issue_details["account_issue"] == (
        "forgotten_password"
    )


def test_unknown_issue():
    case = make_case("I need help with something")

    result = analyze_case(case)

    assert result.intent == Intent.OTHER_UNCLEAR
    assert 0.0 <= result.intent_confidence <= 1.0
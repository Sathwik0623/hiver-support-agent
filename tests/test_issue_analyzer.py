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

def test_account_compromised_issue():
    case = make_case("Someone hacked my Apple Account")

    result = analyze_case(case)

    assert result.intent == Intent.ACCOUNT_ICLOUD
    assert result.entities.issue_details["security"] == (
        "customer_reported_compromise"
    )


def test_unknown_issue():
    case = make_case("I need help with something")

    result = analyze_case(case)

    assert result.intent == Intent.OTHER_UNCLEAR
    assert 0.0 <= result.intent_confidence <= 1.0


def test_data_loss_issue():
    case = make_case("My photos disappeared from my iPhone")

    result = analyze_case(case)

    assert result.entities.issue_details["data_loss"] == (
        "customer_reported"
    )

def test_battery_details():
    case = make_case("My iPhone battery is draining very quickly")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "battery_or_charging_issue"
    )
    assert result.entities.issue_details["battery"] == (
        "customer_reported"
    )

def test_charging_issue():
    case = make_case("My iPhone is not charging")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "battery_or_charging_issue"
    )
    assert result.entities.issue_details["charging"] == (
        "customer_reported"
    )

def test_connectivity_issue():
    case = make_case("My iPhone cannot connect to Wi-Fi")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "connectivity_issue"
    )

def test_hardware_issue():
    case = make_case("My iPhone camera is physically damaged")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "hardware_issue"
    )

def test_input_display_issue():
    case = make_case("My iPhone touchscreen is not responding")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "input_or_display_issue"
    )

def test_backup_restore_issue():
    case = make_case("I cannot restore my iPhone from an iCloud backup")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "backup_or_restore_issue"
    )


def test_security_issue():
    case = make_case("I think someone accessed my Apple Account")

    result = analyze_case(case)

    assert result.entities.issue_details["security"] == (
        "customer_reported_compromise"
    )


def test_multiple_issue_details():
    case = make_case(
        "My iPhone battery is draining quickly and it is not charging"
    )

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "battery_or_charging_issue"
    )
    assert result.entities.issue_details["battery"] == (
        "customer_reported"
    )
    assert result.entities.issue_details["charging"] == (
        "customer_reported"
    )


def test_display_issue_variation():
    case = make_case("The display is black on my iPhone")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "display_issue"
    )


def test_wifi_connectivity_issue():
    case = make_case("My iPhone Wi-Fi is not working")

    result = analyze_case(case)

    assert result.entities.issue_details["symptom"] == (
        "connectivity_issue"
    )
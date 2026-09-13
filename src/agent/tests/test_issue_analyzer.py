from src.agent.issue_analyzer import analyze_case
from src.agent.schemas import (
    SupportCase,
    Message,
    Intent,
    ConversationState,
    RiskFlag,
)


def make_case(text: str) -> SupportCase:
    """
    Build a minimal support case for unit testing.
    """

    return SupportCase(
        case_id="TEST_CASE",
        customer_id="CUSTOMER_1",
        brand="AppleSupport",
        created_at="2026-01-01T00:00:00Z",
        messages=[
            Message(
                message_id="1",
                sender="customer",
                timestamp="2026-01-01T00:00:00Z",
                text=text,
                attachments=[],
            )
        ],
    )


def test_software_update():
    result = analyze_case(
        make_case("My iPhone update keeps failing to install.")
    )

    assert result.intent == Intent.SOFTWARE_UPDATE
    assert result.intent_confidence >= 0.70


def test_device_performance_after_update():
    result = analyze_case(
        make_case("My iPhone keeps freezing after the latest update.")
    )

    assert result.intent == Intent.DEVICE_PERFORMANCE
    assert result.intent_confidence >= 0.70


def test_account_compromise_is_high_risk():
    result = analyze_case(
        make_case("I think someone hacked my Apple ID.")
    )

    assert result.intent == Intent.ACCOUNT_ICLOUD
    assert RiskFlag.ACCOUNT_COMPROMISE in result.risk_flags


def test_unauthorized_purchase():
    result = analyze_case(
        make_case("There is an unauthorized purchase on my account.")
    )

    assert result.intent == Intent.PURCHASE_BILLING
    assert RiskFlag.FRAUD in result.risk_flags


def test_backup_restore():
    result = analyze_case(
        make_case("I restored my iPhone but my photos are missing.")
    )

    assert result.intent == Intent.BACKUP_RESTORE
    assert result.backup_restore_related is True
    assert result.data_loss is True


def test_failed_troubleshooting():
    result = analyze_case(
        make_case("I already tried restarting it and nothing worked.")
    )

    assert result.conversation_state == ConversationState.FAILED_TROUBLESHOOTING


def test_information_request():
    result = analyze_case(
        make_case("How do I connect my iPhone to Wi-Fi?")
    )

    assert result.conversation_state == ConversationState.INFORMATION_REQUEST


def test_device_model_and_ios_version_extraction():
    result = analyze_case(
        make_case("My iPhone 15 Pro on iOS 18.5 is freezing.")
    )

    assert result.entities.device == "iPhone 15 Pro"
    assert result.entities.os_version == "iOS 18.5"

def test_generic_battery_mention_is_not_battery_failure():
    case = make_case("I bought a phone with good battery life")

    result = analyze_case(case)

    assert result.intent != Intent.DEVICE_HARDWARE


def test_generic_update_mention_is_not_update_failure():
    case = make_case("What is included in the latest iOS update?")

    result = analyze_case(case)

    assert result.intent == Intent.SOFTWARE_UPDATE
    assert result.intent_confidence < 0.90


def test_security_terms_are_not_triggered_by_normal_account_usage():
    case = make_case("I signed in to my Apple Account successfully")

    result = analyze_case(case)

    assert result.entities.issue_details.get("security") is None



def test_multi_message_conversation_combines_context():
    case = SupportCase(
        case_id="TEST_CASE_002",
        customer_id="TEST_CUSTOMER",
        brand="AppleSupport",
        messages=[
            Message(
                message_id="TEST_MESSAGE_001",
                sender="customer",
                text="My iPhone keeps freezing",
            ),
            Message(
                message_id="TEST_MESSAGE_002",
                sender="customer",
                text="I restarted it but the problem continues",
            ),
        ],
    )

    result = analyze_case(case)

    assert result.intent == Intent.DEVICE_PERFORMANCE
    assert "freezing" in result.observed_symptoms
    assert result.conversation_state == ConversationState.FAILED_TROUBLESHOOTING


def test_non_customer_messages_are_ignored():
    case = SupportCase(
        case_id="TEST_CASE_003",
        customer_id="TEST_CUSTOMER",
        brand="AppleSupport",
        messages=[
            Message(
                message_id="TEST_MESSAGE_001",
                sender="agent",
                text="The customer reports a battery issue",
            ),
            Message(
                message_id="TEST_MESSAGE_002",
                sender="customer",
                text="I cannot sign in to my Apple Account",
            ),
        ],
    )

    result = analyze_case(case)

    assert result.intent == Intent.ACCOUNT_ICLOUD
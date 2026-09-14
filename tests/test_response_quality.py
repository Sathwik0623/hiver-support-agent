import pytest

from src.agent.pipeline import run_agent


@pytest.mark.parametrize(
    ("message", "required_phrases"),
    [
        (
            "My account may have been compromised.",
            [
                "support specialist",
                "password",
                "verification codes",
            ],
        ),
        (
            "I lost all my data.",
            [
                "data loss",
                "support specialist",
                "avoid",
            ],
        ),
        (
            "My iPhone keeps freezing.",
            [
                "human assistance",
            ],
        ),
    ],
)
def test_draft_response_contains_safe_and_relevant_guidance(
    message,
    required_phrases,
):
    result = run_agent(message)

    draft = result.draft_response.lower()

    for phrase in required_phrases:
        assert phrase.lower() in draft
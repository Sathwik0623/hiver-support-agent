from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.pipeline import run_agent


EVALUATION_CASES = [
    "My account may have been compromised.",
    "I lost all my data.",
    "My iPhone keeps freezing.",
    "I cannot log in to my account.",
    "The issue is fixed now, thank you.",
]


def main() -> None:
    for index, message in enumerate(EVALUATION_CASES, start=1):
        result = run_agent(message)

        print("=" * 80)
        print(f"CASE {index}")
        print(f"Input: {message}")
        print(f"Intent: {result.analysis.intent}")
        print(f"Risk flags: {result.analysis.risk_flags}")
        print(f"Action: {result.decision.action}")
        print(f"Confidence: {result.decision.action_confidence}")
        print(f"Requires human: {result.decision.requires_human}")
        print(f"Draft response: {result.draft_response}")


if __name__ == "__main__":
    main()
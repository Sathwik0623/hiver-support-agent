import json
import subprocess
import sys


def run_cli(message: str):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.agent.pipeline",
            message,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    return result


def test_cli_returns_valid_json():
    result = run_cli("My iPhone keeps freezing.")

    assert result.returncode == 0
    assert result.stderr == ""

    payload = json.loads(result.stdout)

    assert payload["case_id"] == "DEMO_CASE_001"
    assert "analysis" in payload
    assert "decision" in payload
    assert "draft_response" in payload


def test_cli_handles_high_risk_issue():
    result = run_cli("I think someone accessed my account.")

    assert result.returncode == 0

    payload = json.loads(result.stdout)

    assert payload["decision"]["action"] == "ESCALATE"
    assert payload["decision"]["requires_human"] is True


def test_cli_handles_empty_message():
    result = run_cli("")

    assert result.returncode == 0

    payload = json.loads(result.stdout)

    assert "analysis" in payload
    assert "draft_response" in payload


def test_cli_preserves_response_spacing():
    result = run_cli("My iPhone keeps freezing.")

    assert result.returncode == 0

    payload = json.loads(result.stdout)
    draft_response = payload["draft_response"]

    assert "currentiOS version" not in draft_response
    assert "current  iOS version" not in draft_response
    assert "current iOS version" in draft_response
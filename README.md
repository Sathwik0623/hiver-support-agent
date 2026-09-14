# Hiver Support Agent

An AI-assisted customer-support agent for AppleSupport conversations from the Twitter Customer Support dataset.

The project combines historical case retrieval, issue analysis, evidence validation, and deterministic policy decisions to produce a draft response or escalate a case for human assistance.

## 1. Overview

The agent:

1. Reconstructs customer-support cases from tweet relationships.
2. Classifies the customer's primary intent.
3. Extracts useful entities, symptoms, claims, and risk signals.
4. Retrieves similar historical cases.
5. Validates whether retrieved evidence is safe to reuse.
6. Automatically handles, asks for clarification, or escalates the case.
7. Generates a draft response or an escalation reason.

The system is deliberately conservative:

- Similarity alone is not treated as proof that a historical response is appropriate.
- Unknown historical outcomes are not treated as confirmed resolutions.
- Routing-only responses are not treated as troubleshooting solutions.
- High-risk cases are escalated.
- Weak or conflicting evidence is not automatically reused.

## 2. Dataset and Case Reconstruction

The selected brand was `AppleSupport`.

| Metric | Value |
|---|---:|
| Total tweets | 2,811,774 |
| AppleSupport tweets | 106,860 |
| Reconstructed cases | 106,623 |
| Reconstructed messages | 290,796 |
| Historical resolved cases | 3,116 |
| Historical unknown outcomes | 95,548 |
| Multi-turn cases | 30,629 |

A case is reconstructed by following the tweet-response graph around AppleSupport replies and grouping messages belonging to the same customer-support interaction.

An important limitation is that the absence of a customer follow-up does not prove resolution. Therefore, unresolved or unknown outcomes are retained rather than incorrectly labelled as resolved.

## 3. Architecture

```text
Customer message
       |
       v
Case reconstruction
       |
       v
Issue analysis
       |
       +--> intent
       +--> entities
       +--> symptoms
       +--> conversation state
       +--> risk flags
       |
       v
Historical evidence retrieval
       |
       v
Evidence-quality assessment
       |
       v
Deterministic policy
       |
       +--> AUTO_HANDLE
       +--> CLARIFY
       +--> ESCALATE
       |
       v
Draft response or escalation reason
````

## 4. Project Structure

```text
hiver-support-agent/
|-- data/
|   `-- historical support data
|-- src/
|   `-- agent/
|       |-- analyzer.py
|       |-- pipeline.py
|       |-- policy.py
|       |-- retrieval.py
|       |-- schemas.py
|       `-- ...
|-- tests/
|   `-- test_agent.py
|-- requirements.txt
|-- README.md
`-- pyproject.toml
```

The exact files may vary as the project evolves.

## 5. Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/Sathwik0623/hiver-support-agent.git
cd hiver-support-agent
```

Create a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 6. Running the Agent

Run the agent from the project root:

```bash
python -m src.agent.pipeline "My iPhone keeps freezing."
```

Example input:

```text
My iPhone keeps freezing.
```

The agent analyzes the issue, retrieves relevant historical evidence, applies the deterministic policy, and prints the resulting decision and draft response.

You can also provide a multi-message conversation:

```bash
python -m src.agent.pipeline "My iPhone keeps freezing." "I already restarted it but the issue continues."
```

## 7. Running Tests

Run the complete test suite:

```bash
python -m pytest -q
```

Run only the agent tests:

```bash
python -m pytest tests/test_agent.py -q
```

The test suite covers:

* Basic pipeline execution
* Intent classification
* Device-performance issues
* Account-security issues
* Unclear customer messages
* Failed troubleshooting
* Resolved conversations
* High-risk escalation
* Weak evidence handling
* Empty and whitespace-only input
* Multiple messages
* Invalid input types
* Long messages
* Response-formatting regressions

The current test suite contains 63 passing tests.

## 8. Decision Policy

The policy produces one of three actions.

### `AUTO_HANDLE`

Used only when the evidence is sufficiently strong, relevant, resolved, and safe to reuse.

### `CLARIFY`

Used when the customer message is incomplete or additional information is required.

### `ESCALATE`

Used when:

* The issue is high risk.
* Previous troubleshooting failed.
* Evidence is weak or conflicting.
* No verified historical resolution exists.
* The case requires human investigation.

The policy is deterministic and is evaluated after issue analysis, retrieval, and evidence assessment.

## 9. Safety and Reliability Limitations

This project is a proof of concept and should not be used as an unsupervised production support system.

Important limitations include:

* Historical tweet data may be incomplete or noisy.
* A missing follow-up does not prove that an issue was resolved.
* Lexical similarity does not guarantee semantic equivalence.
* The system does not independently verify real-time device state.
* The generated response is a draft, not a guaranteed solution.
* Some cases require account access, internal tools, or human investigation.
* The current retrieval approach is not a full semantic-vector retrieval system.
* The system does not directly send messages to customers.

Human review remains necessary for uncertain, sensitive, or high-risk cases.

## 10. Logging

The pipeline records important execution stages through Python logging, including:

* Pipeline start
* Issue analysis
* Policy decision
* Unexpected failures

For more detailed logs, configure the Python logging level before running the application.

## 11. Current Status

The project currently supports:

* Single-message input
* Multi-message conversations
* Issue classification
* Historical evidence retrieval
* Evidence validation
* Deterministic action selection
* Draft response generation
* Human escalation
* Automated tests for core behavior

## 12. Future Improvements

Possible future improvements include:

* Semantic embeddings and vector search
* Better conversation-state detection
* More advanced entity extraction
* Evaluation datasets with manually verified labels
* Confidence calibration
* Structured observability and metrics
* API deployment using FastAPI
* Containerization with Docker
* Human-review dashboards
* Integration with a real customer-support platform
* Stronger privacy and data-redaction controls

````

## Evaluation

The project includes a representative evaluation script covering security,
data-loss, device-performance, account-access, and resolved-issue scenarios.

Run the evaluation from the project root:

```powershell
python scripts\evaluate_agent.py
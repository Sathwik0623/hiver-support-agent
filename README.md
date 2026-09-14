# Hiver Support Agent

An AI-assisted customer-support agent for AppleSupport conversations from the Twitter Customer Support dataset.

The project combines historical case reconstruction, issue analysis, evidence retrieval, evidence validation, and deterministic policy decisions to generate a draft response or escalate a case for human assistance.

> **Project status:** Proof of concept  
> **Test status:** 79 tests passing  
> **Primary language:** Python

---

## Overview

The Hiver Support Agent analyzes customer-support conversations and determines the safest next action.

The agent can:

1. Reconstruct customer-support cases from tweet relationships.
2. Classify the customer's primary and secondary intents.
3. Extract entities, symptoms, customer claims, and risk signals.
4. Retrieve similar historical support cases.
5. Evaluate whether retrieved evidence is relevant and safe to reuse.
6. Detect troubleshooting, resolved, unclear, and failed-troubleshooting states.
7. Select one of three actions:
   - `AUTO_HANDLE`
   - `CLARIFY`
   - `ESCALATE`
8. Generate a draft response or an explanation for human escalation.

The system is intentionally conservative. Similarity alone is not treated as proof that a historical response is suitable for the current customer.

---

## Key Design Principles

### Conservative evidence reuse

Historical evidence is reused only when it passes multiple checks:

- Problem relevance
- Resolution usefulness
- Context compatibility
- Grounding value
- Historical resolution status
- Whether the suggested troubleshooting step was already attempted

### Deterministic decision policy

The final action is selected by deterministic policy logic rather than relying entirely on an unconstrained language-model response.

### Human-in-the-loop safety

The system escalates cases when:

- The issue is high risk.
- Previous troubleshooting has failed.
- Evidence is weak or conflicting.
- No verified historical resolution exists.
- The issue may involve account access, fraud, privacy, payment disputes, or data loss.

### Transparent decisions

The output includes:

- Issue analysis
- Retrieved evidence
- Evidence-quality scores
- Selected action
- Action confidence
- Human-review requirement
- Draft response or escalation reason

---

## Architecture

```text
Customer message
       |
       v
Case reconstruction
       |
       v
Issue analysis
       |
       +--> Primary intent
       +--> Secondary intent
       +--> Entities
       +--> Symptoms
       +--> Customer claims
       +--> Conversation state
       +--> Risk flags
       |
       v
Historical evidence retrieval
       |
       v
Evidence-quality assessment
       |
       +--> Problem relevance
       +--> Resolution usefulness
       +--> Context compatibility
       +--> Grounding value
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
```

---

## Dataset and Case Reconstruction

The selected brand is `AppleSupport`.

| Metric | Value |
|---|---:|
| Total tweets | 2,811,774 |
| AppleSupport tweets | 106,860 |
| Reconstructed cases | 106,623 |
| Reconstructed messages | 290,796 |
| Historical resolved cases | 3,116 |
| Historical unknown outcomes | 95,548 |
| Multi-turn cases | 30,629 |

A support case is reconstructed by following the tweet-response graph around AppleSupport replies and grouping messages that belong to the same customer-support interaction.

An important limitation is that the absence of a customer follow-up does not prove that the issue was resolved. Unknown or unresolved outcomes are therefore retained instead of being incorrectly labelled as successful resolutions.

---

## Project Structure

```text
hiver-support-agent/
|
|-- app/
|   `-- demo.py
|
|-- baselines/
|   |-- semantic_baseline.py
|   |-- tfidf_baseline.py
|   `-- ...
|
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- golden/
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_retrieval_analysis.ipynb
|   `-- 03_error_analysis.ipynb
|
|-- reports/
|   `-- evaluation and analysis reports
|
|-- scripts/
|   `-- evaluate_agent.py
|
|-- src/
|   |-- agent/
|   |   |-- pipeline.py
|   |   |-- policy.py
|   |   |-- schemas.py
|   |   `-- tests/
|   |-- cache/
|   |-- classification/
|   |-- data/
|   |-- decision/
|   |-- evaluation/
|   |-- generation/
|   |-- retrieval/
|   `-- taxonomy/
|
|-- tests/
|   `-- integration and regression tests
|
|-- requirements.txt
`-- README.md
```

The project separates core agent logic, data processing, retrieval, evaluation, experiments, reports, and automated tests.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sathwik0623/hiver-support-agent.git
cd hiver-support-agent
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Agent

### Run the main pipeline

From the project root:

```bash
python -m src.agent.pipeline "My iPhone keeps freezing."
```

Example input:

```text
My iPhone keeps freezing.
```

The agent analyzes the issue, retrieves relevant historical evidence, applies the deterministic policy, and prints the resulting decision and draft response.

### Run a multi-message conversation

```bash
python -m src.agent.pipeline \
  "My iPhone keeps freezing." \
  "I already restarted it but the issue continues."
```

### Run the CLI demo

The project also includes a lightweight demo entry point:

```bash
python -m app.demo "My iPhone freezes after the latest iOS update"
```

The CLI output includes:

- Detected intent
- Conversation state
- Extracted symptoms
- Risk flags
- Retrieved historical evidence
- Evidence-quality scores
- Selected action
- Action confidence
- Human-review requirement
- Draft response

Example decision:

```json
{
  "action": "ESCALATE",
  "action_confidence": 0.9,
  "requires_human": true
}
```

The demo uses a conservative policy. Weak or unverified historical evidence is not automatically reused.

---

## Decision Policy

The policy produces one of three actions.

### `AUTO_HANDLE`

Used only when:

- The issue is sufficiently understood.
- Historical evidence is strong and relevant.
- The evidence represents a verified resolution.
- The context is compatible.
- The suggested action has not already been attempted.
- No high-risk condition is present.

### `CLARIFY`

Used when:

- The customer message is incomplete.
- The intent is unclear.
- Required device or account information is missing.
- Additional troubleshooting details are needed before proceeding.

### `ESCALATE`

Used when:

- The issue is high risk.
- Previous troubleshooting has failed.
- Evidence is weak or conflicting.
- No verified historical resolution exists.
- The issue requires human investigation.
- The case involves account compromise, fraud, phishing, privacy, payment disputes, or possible data loss.

The policy is evaluated after issue analysis, retrieval, and evidence assessment.

---

## Supported Intent Categories

The agent supports intent classification across several customer-support categories, including:

- Software updates
- Device performance
- Applications and services
- Device hardware
- Connectivity
- Apple Account issues
- Purchases and billing
- Input and display
- Backup and restore
- Other or unclear issues

The analysis also extracts:

- Device information
- Operating-system version
- Application information
- Product information
- Region
- Issue details
- Observed symptoms
- Customer claims
- Confirmed causes
- Working hypotheses
- Risk flags

---

## Evidence Evaluation

Each retrieved historical case is evaluated using multiple evidence dimensions.

| Evidence Dimension | Description |
|---|---|
| Problem relevance | How closely the historical case matches the current issue |
| Resolution usefulness | Whether the historical response contains a useful resolution |
| Context compatibility | Whether the historical context matches the current case |
| Grounding value | Whether the evidence supports the proposed response |
| Already tried | Whether the customer has already attempted the suggested action |

Evidence is rejected when it fails the required quality thresholds or does not represent a verified resolution.

This prevents the system from treating a merely similar conversation as a reliable troubleshooting solution.

---

## Running Tests

Run the complete test suite:

```bash
python -m pytest -q
```

Run the integration and regression tests:

```bash
python -m pytest tests -q
```

Run the agent policy tests:

```bash
python -m pytest src/agent/tests/test_policy.py -q
```

The test suite covers:

- Basic pipeline execution
- Intent classification
- Device-performance issues
- Account-security issues
- Unclear customer messages
- Failed troubleshooting
- Resolved conversations
- High-risk escalation
- Weak evidence handling
- Empty input
- Whitespace-only input
- Multiple messages
- Invalid input types
- Long messages
- Response-formatting regressions
- Action-confidence calculation
- Policy decision behavior

Current validation result:

```text
79 passed
```

---

## Evaluation

The project includes a representative evaluation script covering security, data-loss, device-performance, account-access, and resolved-issue scenarios.

Run the evaluation from the project root:

```powershell
python scripts\evaluate_agent.py
```

### Representative Results

| Scenario | Predicted Action | Human Review |
|---|---|---|
| Possible account compromise | `ESCALATE` | Yes |
| Reported data loss | `ESCALATE` | Yes |
| iPhone freezing | `ESCALATE` | Yes |
| Unable to log in | `ESCALATE` | Yes |
| Issue already resolved | `AUTO_HANDLE` | No |

The evaluation confirms that high-risk and unresolved cases are routed to human support, while a clearly resolved issue can be handled automatically.

---

## Logging

The pipeline records important execution stages through Python logging, including:

- Pipeline start
- Issue analysis
- Evidence retrieval
- Evidence assessment
- Policy decision
- Unexpected failures

For more detailed logs, configure the Python logging level before running the application.

---

## Current Status

The project currently supports:

- Single-message input
- Multi-message conversations
- Issue classification
- Entity and symptom extraction
- Historical case reconstruction
- Historical evidence retrieval
- Evidence validation
- Conversation-state detection
- Risk-flag detection
- Deterministic action selection
- Action-confidence calculation
- Draft response generation
- Human escalation
- Structured JSON output
- Automated regression tests

---

## Safety and Reliability Limitations

This project is a proof of concept and should not be used as an unsupervised production support system.

Important limitations include:

- Historical tweet data may be incomplete or noisy.
- A missing customer follow-up does not prove that an issue was resolved.
- Lexical similarity does not guarantee semantic equivalence.
- The system does not independently verify real-time device state.
- Generated responses are drafts, not guaranteed solutions.
- Some cases require account access, internal tools, or human investigation.
- The current retrieval approach is not a full semantic-vector retrieval system.
- The system does not directly send messages to customers.
- Confidence values are policy-based estimates and are not calibrated probabilities.
- The dataset may contain outdated troubleshooting information.
- The project does not replace trained customer-support personnel.

Human review remains necessary for uncertain, sensitive, or high-risk cases.

---

## Future Improvements

Possible future improvements include:

- Semantic embeddings and vector search
- Improved conversation-state detection
- More advanced entity extraction
- Manually verified evaluation datasets
- Confidence calibration
- Structured observability and metrics
- FastAPI deployment
- Docker containerization
- Human-review dashboards
- Integration with a real customer-support platform
- Stronger privacy and data-redaction controls
- Better response-quality evaluation
- Improved multilingual support
- Retrieval reranking
- Feedback-driven policy refinement

---

## Technology Stack

- Python
- Pydantic
- Pytest
- Natural-language processing
- Historical case retrieval
- Deterministic policy rules
- Python logging
- JSON-based structured outputs

---

## Example Workflow

Given the input:

```text
My iPhone freezes after the latest iOS update.
```

The agent may produce the following analysis:

```json
{
  "intent": "DEVICE_PERFORMANCE",
  "secondary_intent": "SOFTWARE_UPDATE",
  "conversation_state": "TROUBLESHOOTING",
  "observed_symptoms": [
    "freezing"
  ],
  "risk_flags": []
}
```

If no historical evidence passes the required quality checks, the policy selects:

```json
{
  "action": "ESCALATE",
  "requires_human": true
}
```

The system then generates a draft response requesting additional information and explaining that human assistance is required.

---

## Repository

GitHub repository:

https://github.com/Sathwik0623/hiver-support-agent

---

## Author

**Sathwik Kothapalli**

Software Engineer —  at Cisco  
Python • SQL • HTML • CSS • DSA • AI-assisted Automation

GitHub:  
https://github.com/Sathwik0623

LinkedIn:  
https://www.linkedin.com/in/kothapallisathwik
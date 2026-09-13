"""
Pydantic schemas for the Hiver support agent.

These schemas define the structured contract between:
    case context
    issue analysis
    retrieval/evidence
    decision policy
    response generation
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------
# INTENTS
# ---------------------------------------------------------------------

class Intent(str, Enum):
    SOFTWARE_UPDATE = "SOFTWARE_UPDATE"
    DEVICE_PERFORMANCE = "DEVICE_PERFORMANCE"
    APP_SERVICE = "APP_SERVICE"
    DEVICE_HARDWARE = "DEVICE_HARDWARE"
    CONNECTIVITY = "CONNECTIVITY"
    ACCOUNT_ICLOUD = "ACCOUNT_ICLOUD"
    PURCHASE_BILLING = "PURCHASE_BILLING"
    INPUT_DISPLAY = "INPUT_DISPLAY"
    OTHER_UNCLEAR = "OTHER_UNCLEAR"
    BACKUP_RESTORE = "BACKUP_RESTORE"


# ---------------------------------------------------------------------
# CONVERSATION STATE
# ---------------------------------------------------------------------

class ConversationState(str, Enum):
    TROUBLESHOOTING = "TROUBLESHOOTING"
    FAILED_TROUBLESHOOTING = "FAILED_TROUBLESHOOTING"
    INFORMATION_REQUEST = "INFORMATION_REQUEST"
    RESOLVED = "RESOLVED"


# ---------------------------------------------------------------------
# AGENT ACTION
# ---------------------------------------------------------------------

class AgentAction(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    CLARIFY = "CLARIFY"
    ESCALATE = "ESCALATE"


# ---------------------------------------------------------------------
# RISK FLAGS
# ---------------------------------------------------------------------

class RiskFlag(str, Enum):
    ACCOUNT_COMPROMISE = "account_compromise"
    FRAUD = "fraud"
    PHISHING = "phishing"
    DATA_LOSS = "data_loss"
    SAFETY_ISSUE = "safety_issue"
    PRIVACY_SENSITIVE = "privacy_sensitive"
    PAYMENT_DISPUTE = "payment_dispute"


# ---------------------------------------------------------------------
# MESSAGE
# ---------------------------------------------------------------------

class Message(BaseModel):
    """
    One message inside a support case.
    """

    model_config = ConfigDict(extra="forbid")

    message_id: str
    sender: str
    timestamp: Optional[str] = None
    text: str = Field(min_length=1)
    attachments: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# CASE
# ---------------------------------------------------------------------

class SupportCase(BaseModel):
    """
    Reconstructed support case.

    A case contains the full conversation context needed by the agent.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    customer_id: str
    brand: str = "AppleSupport"
    created_at: Optional[str] = None
    messages: List[Message] = Field(min_length=1)


# ---------------------------------------------------------------------
# ENTITIES
# ---------------------------------------------------------------------

class CaseEntities(BaseModel):
    """
    Structured entities extracted from the customer conversation.
    """

    model_config = ConfigDict(extra="forbid")

    device: Optional[str] = None
    os_version: Optional[str] = None
    app: Optional[str] = None
    product: Optional[str] = None
    region: Optional[str] = None
    issue_details: Dict[str, str] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------
# ISSUE ANALYSIS
# ---------------------------------------------------------------------

class IssueAnalysis(BaseModel):
    """
    Structured understanding of the customer's issue.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent

    secondary_intent: Optional[Intent] = None

    conversation_state: ConversationState

    intent_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    entities: CaseEntities = Field(
        default_factory=CaseEntities
    )

    observed_symptoms: List[str] = Field(
        default_factory=list
    )

    customer_claims: List[str] = Field(
        default_factory=list
    )

    confirmed_causes: List[str] = Field(
        default_factory=list
    )

    working_hypotheses: List[str] = Field(
        default_factory=list
    )

    risk_flags: List[RiskFlag] = Field(
        default_factory=list
    )

    backup_restore_related: bool = False

    data_loss: bool = False


# ---------------------------------------------------------------------
# RETRIEVED EVIDENCE
# ---------------------------------------------------------------------

class RetrievedEvidence(BaseModel):
    """
    One historical support case retrieved from the evidence corpus.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    case_id: str
    similarity: float = Field(
        ge=0.0,
        le=1.0,
    )

    customer_problem: str
    conversation_context: str = ""
    apple_response: str

    follow_up: str = ""

    resolution_signal: str = "UNKNOWN"
    resolution_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    resolution_reason: str = ""


# ---------------------------------------------------------------------
# EVIDENCE ASSESSMENT
# ---------------------------------------------------------------------

class EvidenceDecision(str, Enum):
    USE = "USE"
    USE_WITH_CAUTION = "USE_WITH_CAUTION"
    DO_NOT_USE = "DO_NOT_USE"


class EvidenceAssessment(BaseModel):
    """
    Evaluation of whether retrieved evidence is suitable for
    grounding a response.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str

    problem_relevance: float = Field(
        ge=0.0,
        le=1.0,
    )

    resolution_usefulness: float = Field(
        ge=0.0,
        le=1.0,
    )

    context_compatibility: float = Field(
        ge=0.0,
        le=1.0,
    )

    grounding_value: float = Field(
        ge=0.0,
        le=1.0,
    )

    already_tried: bool = False

    decision: EvidenceDecision

    reason: str


# ---------------------------------------------------------------------
# AGENT DECISION
# ---------------------------------------------------------------------

class AgentDecision(BaseModel):
    """
    Final policy decision.
    """

    model_config = ConfigDict(extra="forbid")

    action: AgentAction

    reason: str

    action_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    requires_human: bool

    selected_evidence_ids: List[str] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------
# AGENT RESPONSE
# ---------------------------------------------------------------------

class AgentResponse(BaseModel):
    """
    Final response returned by the support agent.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str

    analysis: IssueAnalysis

    evidence: List[EvidenceAssessment] = Field(
        default_factory=list
    )

    decision: AgentDecision

    draft_response: Optional[str] = None

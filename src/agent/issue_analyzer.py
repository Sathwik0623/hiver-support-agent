"""
Deterministic issue analyzer for the Hiver support agent.

Responsibilities:
1. Classify the customer's primary support intent.
2. Estimate intent confidence.
3. Determine the conversation state.
4. Extract lightweight entities.
5. Extract observable symptoms and customer claims.
6. Detect high-risk signals.
7. Preserve important backup/restore and data-loss metadata.

Design principle:
- This module is intentionally deterministic.
- Customer statements are treated as observations/claims, not confirmed
  root causes.
- Classification is based on the object/problem being supported rather
  than merely the words used.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from src.agent.schemas import (
    CaseEntities,
    ConversationState,
    Intent,
    IssueAnalysis,
    Message,
    RiskFlag,
    SupportCase,
)


# ---------------------------------------------------------------------------
# Keyword groups
# ---------------------------------------------------------------------------

# General update mentions are intentionally broad.
# They are checked only after stronger issue-specific signals.
SOFTWARE_UPDATE_TERMS = (
    "update",
    "updating",
    "software update",
    "ios update",
    "ios upgrade",
    "upgrade",
    "install update",
    "installing update",
    "update failed",
    "update fails",
    "update won't install",
    "update wont install",
    "update cannot install",
    "update can't install",
    "update cant install",
)

# IMPORTANT:
# Merely mentioning "backup" should NOT classify a case as BACKUP_RESTORE.
# These terms represent an actual backup/restore/migration problem.
BACKUP_RESTORE_TERMS = (
    "restore",
    "restored",
    "restoring",
    "restore backup",
    "restore from backup",
    "restoring backup",
    "backup failed",
    "backup fails",
    "backup won't complete",
    "backup wont complete",
    "backup can't complete",
    "backup cant complete",
    "backup cannot complete",
    "unable to backup",
    "unable to back up",
    "couldn't back up",
    "could not back up",
    "can't back up",
    "cant back up",
    "missing after restore",
    "lost after restore",
    "lost data after restore",
    "recover data",
    "recover my data",
    "recover photos",
    "restore my photos",
    "restore my data",
    "data migration",
    "migration failed",
    "migration issue",
    "migrate data",
    "migrate my data",
    "transfer data",
    "transferred data",
)

ACCOUNT_TERMS = (
    "apple id",
    "apple account",
    "icloud",
    "sign in",
    "sign-in",
    "signin",
    "login",
    "log in",
    "logged out",
    "password",
    "account locked",
    "locked account",
    "account recovery",
    "verification code",
    "two factor",
    "2fa",
)



# ---------------------------------------------------------------------------
# Battery and charging terms
# ---------------------------------------------------------------------------

BATTERY_TERMS = (
    "battery",
    "battery drain",
    "battery draining",
    "battery dies",
    "battery life",
    "battery health",
    "battery percentage",
    "battery level",
    "battery capacity",
    "battery drops",
    "battery dropping",
    "loses battery",
    "losing battery",
    "drains battery",
    "draining battery",
    "empty within",
    "does not hold a charge",
    "doesn't hold a charge",
    "not holding a charge",
    "charge does not last",
    "charge doesn't last",
    "charge not lasting",
    "not lasting throughout the day",
    "shuts down with battery",
    "shuts down even though",
    "turns off with battery",
    "battery drops quickly",
    "battery percentage drops",
    "battery percentage is dropping",
    "battery percentage is not increasing",
    "battery percentage not increasing",
    "battery health dropped",
    "battery health has dropped",
    "battery health degraded",
    "battery health declined",
)

CHARGING_TERMS = (
    "not charging",
    "does not charge",
    "doesn't charge",
    "wont charge",
    "won't charge",
    "will not charge",
    "cannot charge",
    "can't charge",
    "charging slowly",
    "charges slowly",
    "charge slowly",
    "slow charging",
    "charging is slow",
    "charging very slowly",
    "charges very slowly",
    "charge very slowly",
    "not charging properly",
    "charging issue",
    "charging problem",
    "connected to the charger",
    "connected to charger",
    "plugged in",
    "plugged into charger",
    "battery percentage is not increasing",
    "battery percentage not increasing",
    "stops charging",
    "stopped charging",
    "charging stops",
    "charging keeps stopping",
    "keeps disconnecting from charging",
    "disconnecting from charging",
    "charger disconnects",
    "charging cable",
    "charging cable is connected",
    "charging port",
    "charge at 80 percent",
    "charging at 80 percent",
    "stops at 80 percent",
    "stuck at 80 percent",
    "charger",
    "power adapter",
)

BATTERY_SAFETY_TERMS = (
    "battery swollen",
    "swollen battery",
    "battery swelling",
    "battery is swelling",
    "phone is extremely hot",
    "phone gets extremely hot",
    "very hot while charging",
    "burning smell while charging",
    "smoke while charging",
    "sparks while charging",
)

CONNECTIVITY_TERMS = (
    "wifi",
    "wi-fi",
    "wireless",
    "internet",
    "cellular",
    "mobile data",
    "network",
    "bluetooth",
    "hotspot",
    "signal",
    "no service",
    "can't connect",
    "cannot connect",
    "cant connect",
    "connection",
    "disconnecting",
    "disconnects",
    "connectivity",
    "calls",
    "call drops",
)

BILLING_TERMS = (
    "billing",
    "bill",
    "payment",
    "purchase",
    "purchased",
    "transaction",
    "refund",
    "subscription",
    "subscription charge",
    "unauthorized purchase",
    "unauthorised purchase",
    "unauthorized charge",
    "unauthorised charge",
    "fraud",
    "fraudulent",
    "money taken",
    "charged for",
    "charged me",
    "payment declined",
    "payment failed",
    "payment method",
    "credit card",
    "debit card",
)

HARDWARE_TERMS = (
    "broken",
    "cracked",
    "physical damage",
    "damaged",
    "hardware",
    "screen cracked",
    "display cracked",
    "camera broken",
    "speaker broken",
    "microphone broken",
    "button broken",
    "port broken",
    "charging port",
    "power button",
    "volume button",
    "home button",
    "physical button",
    "touchscreen damaged",
    "screen damage",
    "battery swollen",
    "swollen battery",
)

PERFORMANCE_TERMS = (
    "freezing",
    "frozen",
    "freeze",
    "freezes",
    "slow",
    "slowness",
    "lag",
    "lagging",
    "lags",
    "crashing",
    "crash",
    "crashes",
    "restarting",
    "restart",
    "restarts",
    "random restart",
    "randomly restarts",
    "overheating",
    "overheats",
    "overheated",
    "performance",
    "unresponsive",
    "not responding",
    "hangs",
    "hanging",
)

# Avoid generic "type", "touch", "screen", "display", etc.
# Those words occur frequently as context and create false positives.
INPUT_DISPLAY_TERMS = (
    "keyboard",
    "keyboard not working",
    "keyboard doesn't work",
    "keyboard does not work",
    "keyboard not responding",
    "typing issue",
    "typing problem",
    "can't type",
    "cant type",
    "cannot type",
    "unable to type",
    "autocorrect",
    "auto correct",
    "input issue",
    "input problem",
    "touch screen",
    "touchscreen",
    "touch screen not responding",
    "touchscreen not responding",
    "screen not responding",
    "screen doesn't respond",
    "screen does not respond",
    "display issue",
    "display problem",
    "screen issue",
    "screen problem",
    "characters rendering",
    "character rendering",
    "font rendering",
    "text rendering",
    "text appears",
    "text display",
    "notification display",
    "notifications not showing",
    "icons not showing",
    "icons missing",
    "interface issue",
    "ui issue",
)

# Specific services/apps plus actual app/service failure expressions.
# Generic "app", "apps", "notes", "photos", etc. are intentionally removed.
APP_SERVICE_TERMS = (
    "apple music",
    "music app",
    "app store",
    "imessage",
    "i message",
    "facetime",
    "face time",
    "apple pay",
    "notes app",
    "safari",
    "mail app",
    "calendar app",
    "photos app",
    "itunes",
    "third party app",
    "third-party app",
    "application not working",
    "app not working",
    "app doesn't work",
    "app does not work",
    "app not responding",
    "app won't open",
    "app wont open",
    "app crashes",
    "app crash",
    "service not working",
    "service doesn't work",
    "service does not work",
)

# Information-seeking / status-seeking language.
# These are intentionally broader than only "how do I".
INFORMATION_TERMS = (
    "how do i",
    "how can i",
    "can i",
    "is there a way",
    "where can i",
    "what is",
    "what are",
    "which",
    "do i need",
    "does it support",
    "is it possible",
    "can you tell me",
    "tell me how",

    # Natural information/status requests
    "is this normal",
    "is that normal",
    "is it normal",
    "why is this happening",
    "why does this happen",
    "why did this happen",
    "what happened",
    "what's happening",
    "whats happening",
    "any idea why",
    "any idea what",
    "is apple aware",
    "are you aware",
    "is apple looking",
    "are you looking into",
    "when will this be fixed",
    "when will this get fixed",
    "when is this going to be fixed",
    "any update on",
    "any information on",
    "looking for information",
    "need information",
    "want to know",
    "wondering if",
    "just wondering",
    "please explain",
    "explain why",
    "why does",
    "why is",
    "why are",
)

FAILED_TROUBLESHOOTING_TERMS = (
    "already tried",
    "tried restarting",
    "tried rebooting",
    "restarted it",
    "rebooted it",
    "nothing worked",
    "didn't work",
    "did not work",
    "still not working",
    "still doesn't work",
    "still does not work",
    "no luck",
    "same problem",
    "problem persists",
    "issue persists",
    "still broken",
    "still failing",
    "tried everything",
    "keeps happening",
    "continues to happen",
    "still happens",
    "still happening",
)

RESOLVED_TERMS = (
    "it works now",
    "working now",
    "works now",
    "fixed now",
    "problem solved",
    "issue solved",
    "resolved",
    "all good now",
    "that fixed it",
    "this fixed it",
    "thank you it works",
    "thanks it works",
    "thanks that worked",
    "it worked",
)

DATA_LOSS_TERMS = (
    "data loss",
    "lost my data",
    "lost data",
    "missing data",
    "photos are missing",
    "missing photos",
    "lost photos",
    "photos disappeared",
    "photos have disappeared",
    "files are missing",
    "missing files",
    "lost files",
    "contacts are missing",
    "missing contacts",
    "messages are missing",
    "missing messages",
    "notes are missing",
    "missing notes",
)

ACCOUNT_COMPROMISE_TERMS = (
    "hacked",
    "hack",
    "someone accessed my account",
    "someone has access to my account",
    "someone got into my account",
    "account compromised",
    "account has been compromised",
    "someone logged into my account",
    "someone logged in",
    "unauthorized access",
    "unauthorised access",
)

FRAUD_TERMS = (
    "fraud",
    "fraudulent",
    "unauthorized purchase",
    "unauthorised purchase",
    "unauthorized charge",
    "unauthorised charge",
    "charge i don't recognize",
    "charge i do not recognize",
    "purchase i don't recognize",
    "purchase i do not recognize",
    "didn't make this purchase",
    "did not make this purchase",
)

SAFETY_TERMS = (
    "swollen battery",
    "battery swollen",
    "smoke",
    "smoking",
    "burning smell",
    "burning",
    "overheating dangerously",
    "very hot",
    "exploded",
    "explosion",
    "sparks",
    "electric shock",
)

PRIVACY_TERMS = (
    "privacy",
    "private information",
    "personal information",
    "personal data",
    "privacy concern",
    "privacy issue",
)

PAYMENT_DISPUTE_TERMS = (
    "charge i don't recognize",
    "charge i do not recognize",
    "unauthorized purchase",
    "unauthorised purchase",
    "unauthorized charge",
    "unauthorised charge",
    "fraud",
    "fraudulent",
    "refund",
    "dispute",
    "dispute a charge",
)

PHISHING_TERMS = (
    "phishing",
    "phishing email",
    "phishing message",
    "scam email",
    "scam message",
    "suspicious email",
    "suspicious link",
    "fake apple email",
    "fake apple message",
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """
    Normalize text for deterministic matching.

    Lowercasing is intentional because classification rules are
    case-insensitive.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_any(text: str, terms: Tuple[str, ...]) -> bool:
    """Return True when at least one term occurs in text."""
    return any(term in text for term in terms)


def _contains_any_regex(text: str, patterns: Tuple[str, ...]) -> bool:
    """Return True when at least one regex pattern matches."""
    return any(re.search(pattern, text, re.I) for pattern in patterns)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _extract_entities(text: str) -> dict:
    """
    Extract lightweight routing entities.

    Extracted entities describe what the customer mentioned.
    They are NOT treated as confirmed root causes.
    """
    

    entities = {
        "device": None,
        "os_version": None,
        "app": None,
        "product": None,
        "region": None,
        "issue_details": {},
    }


    # ---------------------------------------------------------
    # iPhone model
    # ---------------------------------------------------------

    iphone_match = re.search(
        r"\biphone(?:\s+(\d+)(?:\s+(pro|max|plus|mini))?)?\b",
        text,
        re.I,
    )

    if iphone_match:
        model_number = iphone_match.group(1)
        variant = iphone_match.group(2)

        if model_number:
            if variant:
                variant = variant.capitalize()
                entities["device"] = f"iPhone {model_number} {variant}"
            else:
                entities["device"] = f"iPhone {model_number}"
        else:
            entities["device"] = "iPhone"

    # ---------------------------------------------------------
    # iPad model
    # ---------------------------------------------------------

    ipad_match = re.search(
        r"\bipad(?:\s+(pro|air|mini))?\b",
        text,
        re.I,
    )

    if ipad_match:
        suffix = ipad_match.group(1)

        entities["device"] = (
            "iPad"
            if not suffix
            else f"iPad {suffix.capitalize()}"
        )

    # ---------------------------------------------------------
    # Mac
    # ---------------------------------------------------------

    mac_match = re.search(
        r"\b(macbook(?:\s+(?:air|pro))?|imac|mac mini|mac studio)\b",
        text,
        re.I,
    )

    if mac_match:
        entities["device"] = mac_match.group(1).title()

    # ---------------------------------------------------------
    # iOS version
    # ---------------------------------------------------------

    ios_match = re.search(
        r"\bios\s+(\d+(?:\.\d+)*)\b",
        text,
        re.I,
    )

    if ios_match:
        entities["os_version"] = f"iOS {ios_match.group(1)}"
    elif re.search(
        r"\b(?:latest|recent|new)\s+ios(?:\s+update|\s+version)?\b",
        text,
        re.I,
    ):
        entities["os_version"] = "latest iOS"

    # ---------------------------------------------------------
    # macOS version
    # ---------------------------------------------------------

    macos_match = re.search(
        r"\bmacos\s+([A-Za-z0-9.]+)\b",
        text,
        re.I,
    )

    if macos_match:
        entities["os_version"] = f"macOS {macos_match.group(1)}"

    # ---------------------------------------------------------
    # Common Apple apps/services
    # ---------------------------------------------------------

    app_patterns = {
        "Apple Music": (
            "apple music",
        ),
        "App Store": (
            "app store",
        ),
        "iMessage": (
            "imessage",
            "i message",
        ),
        "FaceTime": (
            "facetime",
            "face time",
        ),
        "Apple Pay": (
            "apple pay",
        ),
        "Notes": (
            "notes app",
        ),
        "Safari": (
            "safari",
        ),
        "Mail": (
            "mail app",
        ),
        "Calendar": (
            "calendar app",
        ),
        "Photos": (
            "photos app",
        ),
    }

    for app_name, patterns in app_patterns.items():
        if _contains_any(text, patterns):
            entities["app"] = app_name
            break

    # ---------------------------------------------------------
    # Product context
    # ---------------------------------------------------------

    if _contains_any(
        text,
        (
            "iphone",
            "ipad",
            "ipod",
        ),
    ):
        entities["product"] = "iPhone/iPad ecosystem"

    if _contains_any(
        text,
        (
            "apple id",
            "apple account",
            "icloud",
        ),
    ):
        entities["product"] = "Apple Account/iCloud"

    # ---------------------------------------------------------
    # Issue details
    # ---------------------------------------------------------

    issue_details: Dict[str, str] = {}





    # Apple Account/iCloud issue details
    if (
    "password" in text
    and any(word in text for word in ("forgot", "forgotten", "lost"))
):
        issue_details["account_issue"] = "forgotten_password"

    if _contains_any(text, PERFORMANCE_TERMS):
        issue_details["symptom"] = "performance_issue"

    # Display/black-screen symptom extraction
    if any(
        phrase in text
        for phrase in (
            "black screen",
            "screen is black",
            "screen completely black",
            "screen went completely black",
            "display is black",
            "screen went black",
            "screen turns black",
            "screen turned black",
        )
    ):
        issue_details["symptom"] = "display_issue"

    if _contains_any(text, CONNECTIVITY_TERMS):
        issue_details["symptom"] = "connectivity_issue"

    if _contains_any(text, HARDWARE_TERMS):
        issue_details["symptom"] = "hardware_issue"

    if _contains_any(text, INPUT_DISPLAY_TERMS):
        issue_details["symptom"] = "input_or_display_issue"

    if _contains_any(text, BACKUP_RESTORE_TERMS):
        issue_details["symptom"] = "backup_or_restore_issue"

    if _contains_any(text, DATA_LOSS_TERMS):
        issue_details["data_loss"] = "customer_reported"

    if _contains_any(text, ACCOUNT_COMPROMISE_TERMS):
        issue_details["security"] = "customer_reported_compromise"

    # Battery and charging issues
    if _contains_any(text, BATTERY_TERMS):
        issue_details["symptom"] = "battery_or_charging_issue"
        issue_details["battery"] = "customer_reported"

    if _contains_any(text, CHARGING_TERMS):
        issue_details["symptom"] = "battery_or_charging_issue"
        issue_details["charging"] = "customer_reported"

    # Add issue_details only after all checks are completed
    if issue_details:
        entities["issue_details"] = issue_details

    return entities

# ---------------------------------------------------------------------------
# Risk detection
# ---------------------------------------------------------------------------

def _detect_risk_flags(text: str) -> List[RiskFlag]:
    """
    Detect safety/security/payment/privacy signals.

    These flags are deliberately conservative because the policy layer
    can use them to prefer escalation over autonomous handling.
    """

    flags: List[RiskFlag] = []

    if _contains_any(text, ACCOUNT_COMPROMISE_TERMS):
        flags.append(RiskFlag.ACCOUNT_COMPROMISE)

    if _contains_any(text, FRAUD_TERMS):
        flags.append(RiskFlag.FRAUD)

    if _contains_any(text, SAFETY_TERMS):
        flags.append(RiskFlag.SAFETY_ISSUE)

    if _contains_any(text, PRIVACY_TERMS):
        flags.append(RiskFlag.PRIVACY_SENSITIVE)

    if _contains_any(text, PAYMENT_DISPUTE_TERMS):
        flags.append(RiskFlag.PAYMENT_DISPUTE)

    if _contains_any(text, PHISHING_TERMS):
        flags.append(RiskFlag.PHISHING)

    if _contains_any(text, DATA_LOSS_TERMS):
        flags.append(RiskFlag.DATA_LOSS)

    return list(dict.fromkeys(flags))


# ---------------------------------------------------------------------------
# Symptom / claim extraction
# ---------------------------------------------------------------------------

def _extract_observed_symptoms(text: str) -> List[str]:
    """
    Extract symptoms explicitly described by the customer.

    These are observations, not diagnoses.
    """

    symptoms: List[str] = []

    symptom_patterns = {
        "freezing": (
            "freezing",
            "frozen",
            "freeze",
            "freezes",
        ),
        "crashing": (
            "crashing",
            "crash",
            "crashes",
        ),
        "restarting": (
            "restarting",
            "random restart",
            "randomly restarts",
        ),
        "overheating": (
            "overheating",
            "overheats",
            "overheated",
        ),
        "slowness": (
            "slow",
            "slowness",
            "lag",
            "lagging",
        ),
        "battery drain": (
            "battery drain",
            "battery draining",
            "battery dies",
        ),
        "connectivity failure": (
            "can't connect",
            "cannot connect",
            "cant connect",
            "connection",
            "disconnecting",
            "no service",
        ),
        "keyboard/input issue": (
            "keyboard",
            "typing",
            "autocorrect",
            "auto correct",
        ),
        "display issue": (
            "display",
            "screen",
            "characters",
            "rendering",
            "black screen",
            "screen is black",
            "screen completely black",
            "display is black",
            "screen went black",
            "screen turns black",
            "screen turned black",
        ),
        "account access issue": (
            "can't sign in",
            "cannot sign in",
            "cant sign in",
            "unable to sign in",
            "locked account",
        ),
        "backup/restore issue": (
            "restore",
            "restoring",
            "migration",
            "migrate",
            "backup failed",
            "backup won't complete",
            "unable to backup",
        ),

                "battery drain": (
            "battery drain",
            "battery draining",
            "battery dies",
            "battery life",
            "loses battery",
            "losing battery",
            "drains battery",
            "draining battery",
            "empty within",
            "does not hold a charge",
            "doesn't hold a charge",
            "not holding a charge",
            "not lasting throughout the day",
            "battery percentage drops",
            "battery drops quickly",
        ),

        "battery health issue": (
            "battery health",
            "battery health dropped",
            "battery health has dropped",
            "battery health degraded",
            "battery health declined",
        ),

        "charging failure": (
            "not charging",
            "does not charge",
            "doesn't charge",
            "wont charge",
            "won't charge",
            "cannot charge",
            "can't charge",
            "not charging properly",
            "battery percentage is not increasing",
            "battery percentage not increasing",
            "stops charging",
            "stopped charging",
            "charging stops",
            "stuck at 80 percent",
            "stops at 80 percent",
        ),

        "slow charging": (
            "charging slowly",
            "charges slowly",
            "slow charging",
            "charging is slow",
            "charge very slowly",
        ),

        "charging disconnection": (
            "keeps disconnecting from charging",
            "disconnecting from charging",
            "charger disconnects",
            "charging cable keeps disconnecting",
        ),
    }

    for symptom, patterns in symptom_patterns.items():
        if _contains_any(text, patterns):
            symptoms.append(symptom)

    return symptoms


def _extract_customer_claims(text: str) -> List[str]:
    """
    Capture important customer assertions.

    Claims are intentionally not treated as confirmed causes.
    """

    claims: List[str] = []

    if _contains_any(text, ACCOUNT_COMPROMISE_TERMS):
        claims.append(
            "customer_reports_account_compromise"
        )

    if _contains_any(text, FRAUD_TERMS):
        claims.append(
            "customer_reports_unauthorized_transaction"
        )

    if _contains_any(text, DATA_LOSS_TERMS):
        claims.append(
            "customer_reports_data_loss"
        )

    if _contains_any(text, SAFETY_TERMS):
        claims.append(
            "customer_reports_safety_issue"
        )

    return claims


# ---------------------------------------------------------------------------
# Intent classification helpers
# ---------------------------------------------------------------------------

def _has_explicit_update_failure(text: str) -> bool:
    """
    Detect an actual failed/repeated update operation.

    A generic update mention is not enough.
    """

    update_failure_terms = (
        "update failed",
        "update fails",
        "update keeps failing",
        "update won't install",
        "update wont install",
        "update can't install",
        "update cant install",
        "update cannot install",
        "fails to install",
        "failed to install",
        "stuck installing",
        "stuck on update",
        "update loop",
        "keeps asking me to update",
        "keeps asking for the update",
        "asks me to update again",
        "asks to update again",
        "update again",
        "updated and then asks to update",
        "update installed but",
        "update didn't complete",
        "update did not complete",
        "update never completes",
        "update won't finish",
        "update wont finish",
    )

    return _contains_any(text, update_failure_terms)


def _has_meaningful_backup_restore_issue(text: str) -> bool:
    """
    Determine whether backup/restore is actually the primary problem.

    This intentionally avoids treating contextual mentions such as
    "backup phone" or "backup my photos" as a support issue.
    """

    strong_terms = (
        "restore",
        "restored",
        "restoring",
        "restore backup",
        "restore from backup",
        "restoring backup",
        "backup failed",
        "backup fails",
        "backup won't complete",
        "backup wont complete",
        "backup can't complete",
        "backup cant complete",
        "backup cannot complete",
        "unable to backup",
        "unable to back up",
        "couldn't back up",
        "could not back up",
        "can't back up",
        "cant back up",
        "missing after restore",
        "lost after restore",
        "lost data after restore",
        "recover data",
        "recover my data",
        "recover photos",
        "restore my photos",
        "restore my data",
        "data migration",
        "migration failed",
        "migration issue",
        "migrate data",
        "migrate my data",
        "transfer data",
        "transferred data",
    )

    return _contains_any(text, strong_terms)


def _has_account_access_issue(text: str) -> bool:
    """Detect meaningful Apple Account/iCloud access problems."""

    account_access_terms = (
        "apple id",
        "apple account",
        "icloud",
        "can't sign in",
        "cannot sign in",
        "cant sign in",
        "unable to sign in",
        "can't log in",
        "cannot log in",
        "cant log in",
        "unable to log in",
        "account locked",
        "locked account",
        "account recovery",
        "forgot my password",
        "forgot password",
        "lost my password",
        "verification code",
        "two factor",
        "2fa",
    )

    return _contains_any(text, account_access_terms)


def _has_security_issue(text: str) -> bool:
    """Detect explicit account compromise/phishing/fraud signals."""

    return (
        _contains_any(text, ACCOUNT_COMPROMISE_TERMS)
        or _contains_any(text, PHISHING_TERMS)
        or _contains_any(text, FRAUD_TERMS)
    )


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _classify_intent(text: str) -> Tuple[Intent, float]:
    """
    Classify the customer's primary support intent.

    Battery and charging problems are routed to DEVICE_HARDWARE
    because the current Intent enum has no dedicated battery intent.
    """

    # 1. Explicit software update failure
    if _has_explicit_update_failure(text):
        return Intent.SOFTWARE_UPDATE, 0.94

    # 2. Backup and restore
    if _has_meaningful_backup_restore_issue(text):
        if _has_account_access_issue(text):
            return Intent.ACCOUNT_ICLOUD, 0.88

        return Intent.BACKUP_RESTORE, 0.90

    # 3. Security and account access
    if _has_security_issue(text):
        if _has_account_access_issue(text):
            return Intent.ACCOUNT_ICLOUD, 0.93

    if _has_account_access_issue(text):
        return Intent.ACCOUNT_ICLOUD, 0.90

    # 4. Explicit black-screen/display failure
    if any(
        phrase in text
        for phrase in (
            "black screen",
            "screen is black",
            "screen completely black",
            "screen went completely black",
            "display is black",
            "screen went black",
            "screen turns black",
            "screen turned black",
        )
    ):
        return Intent.DEVICE_HARDWARE, 0.92

    # 4. Battery and charging
    # Must be checked before billing, connectivity, and performance.
    if (
        _contains_any(text, BATTERY_TERMS)
        or _contains_any(text, CHARGING_TERMS)
    ):
        return Intent.DEVICE_HARDWARE, 0.92

    # 5. Physical hardware
    if _contains_any(text, HARDWARE_TERMS):
        return Intent.DEVICE_HARDWARE, 0.88

    # 6. Device performance
    if _contains_any(text, PERFORMANCE_TERMS):
        return Intent.DEVICE_PERFORMANCE, 0.90

    # 7. Connectivity
    if _contains_any(text, CONNECTIVITY_TERMS):
        return Intent.CONNECTIVITY, 0.86

    # 8. Billing and purchases
    if _contains_any(text, BILLING_TERMS):
        return Intent.PURCHASE_BILLING, 0.87

    # 9. App and service
    if _contains_any(text, APP_SERVICE_TERMS):
        return Intent.APP_SERVICE, 0.82

    # 10. Input and display
    if _contains_any(text, INPUT_DISPLAY_TERMS):
        return Intent.INPUT_DISPLAY, 0.84

    # 11. General software update information
    if _contains_any(text, SOFTWARE_UPDATE_TERMS):
        return Intent.SOFTWARE_UPDATE, 0.78

    # 12. Unknown
    return Intent.OTHER_UNCLEAR, 0.40


# ---------------------------------------------------------------------------
# Conversation state classification
# ---------------------------------------------------------------------------

def _classify_state(text: str) -> ConversationState:
    """
    Determine the current conversation state.

    Priority:
        FAILED_TROUBLESHOOTING
        RESOLVED
        INFORMATION_REQUEST
        TROUBLESHOOTING

    Failed troubleshooting wins because it represents a stronger
    state transition than a generic question.
    """

    # ---------------------------------------------------------
    # 1. Failed troubleshooting
    # ---------------------------------------------------------

    if _contains_any(
        text,
        FAILED_TROUBLESHOOTING_TERMS,
    ):
        return ConversationState.FAILED_TROUBLESHOOTING

    # ---------------------------------------------------------
    # 2. Resolved
    # ---------------------------------------------------------

    if _contains_any(
        text,
        RESOLVED_TERMS,
    ):
        return ConversationState.RESOLVED

    # ---------------------------------------------------------
    # 3. Explicit information/status request
    # ---------------------------------------------------------

    if _contains_any(
        text,
        INFORMATION_TERMS,
    ):
        return ConversationState.INFORMATION_REQUEST

    # ---------------------------------------------------------
    # 4. Default
    # ---------------------------------------------------------

    return ConversationState.TROUBLESHOOTING


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def analyze_case(case: SupportCase) -> IssueAnalysis:
    """
    Analyze a support case.

    The complete customer conversation is used rather than only the latest
    customer message. This allows the analyzer to see context such as:

        Customer: My phone keeps freezing.
        Customer: I restarted it and it still freezes.

    which should be treated differently from a fresh troubleshooting case.
    """

    customer_messages = [
        message.text
        for message in case.messages
        if message.sender == "customer"
    ]

    text = _normalise(
        " ".join(
            message
            for message in customer_messages
            if message
        )
    )




        
    # ---------------------------------------------------------
    # Empty case
    # ---------------------------------------------------------

    if not text:
        return IssueAnalysis(
            intent=Intent.OTHER_UNCLEAR,
            conversation_state=ConversationState.INFORMATION_REQUEST,
            intent_confidence=0.40,
            entities=CaseEntities(),
            observed_symptoms=[],
            customer_claims=[],
            confirmed_causes=[],
            working_hypotheses=[],
            risk_flags=[],
            backup_restore_related=False,
            data_loss=False,
        )

    # ---------------------------------------------------------
    # Core classification
    # ---------------------------------------------------------

    intent, confidence = _classify_intent(text)

    state = _classify_state(text)

    # ---------------------------------------------------------
    # Metadata extraction
    # ---------------------------------------------------------

    raw_entities = _extract_entities(text)

    entities = CaseEntities(
        **{
            key: value
            for key, value in raw_entities.items()
            if value is not None
        }
    )

    risk_flags = _detect_risk_flags(text)

    observed_symptoms = _extract_observed_symptoms(text)

    customer_claims = _extract_customer_claims(text)

    backup_restore_related = _has_meaningful_backup_restore_issue(text)

    data_loss = (
        RiskFlag.DATA_LOSS in risk_flags
        or _contains_any(text, DATA_LOSS_TERMS)
    )

    # ---------------------------------------------------------
    # Secondary intent
    #
    # Keep this conservative. The primary intent remains the issue
    # that should drive routing.
    # ---------------------------------------------------------

    secondary_intent = None

    intent_candidates: List[Intent] = []

    # Update is secondary only when it is genuinely mentioned as
    # contextual information and isn't the primary issue.
    if (
        intent != Intent.SOFTWARE_UPDATE
        and _contains_any(text, SOFTWARE_UPDATE_TERMS)
    ):
        intent_candidates.append(Intent.SOFTWARE_UPDATE)

    if (
        intent != Intent.BACKUP_RESTORE
        and _has_meaningful_backup_restore_issue(text)
    ):
        intent_candidates.append(Intent.BACKUP_RESTORE)

    if (
        intent != Intent.CONNECTIVITY
        and _contains_any(text, CONNECTIVITY_TERMS)
    ):
        intent_candidates.append(Intent.CONNECTIVITY)

    if (
        intent != Intent.ACCOUNT_ICLOUD
        and _contains_any(text, ACCOUNT_TERMS)
    ):
        intent_candidates.append(Intent.ACCOUNT_ICLOUD)

    if (
        intent != Intent.PURCHASE_BILLING
        and _contains_any(text, BILLING_TERMS)
    ):
        intent_candidates.append(Intent.PURCHASE_BILLING)

    if (
        intent != Intent.APP_SERVICE
        and _contains_any(text, APP_SERVICE_TERMS)
    ):
        intent_candidates.append(Intent.APP_SERVICE)

    if (
        intent != Intent.DEVICE_PERFORMANCE
        and _contains_any(text, PERFORMANCE_TERMS)
    ):
        intent_candidates.append(Intent.DEVICE_PERFORMANCE)

    if (
        intent != Intent.DEVICE_HARDWARE
        and _contains_any(text, HARDWARE_TERMS)
    ):
        intent_candidates.append(Intent.DEVICE_HARDWARE)

    if (
        intent != Intent.INPUT_DISPLAY
        and _contains_any(text, INPUT_DISPLAY_TERMS)
    ):
        intent_candidates.append(Intent.INPUT_DISPLAY)

    if intent_candidates:
        secondary_intent = intent_candidates[0]

    # ---------------------------------------------------------
    # Preserve customer statements as claims.
    #
    # Do not fabricate confirmed causes.
    # ---------------------------------------------------------

    confirmed_causes: List[str] = []

    working_hypotheses: List[str] = []

    # Example: update is temporal context, not necessarily the cause.
    if (
        intent == Intent.DEVICE_PERFORMANCE
        and _contains_any(text, SOFTWARE_UPDATE_TERMS)
    ):
        working_hypotheses.append(
            "performance_issue_reported_after_software_update"
        )

    # ---------------------------------------------------------
    # Return structured analysis
    # ---------------------------------------------------------

    return IssueAnalysis(
        intent=intent,
        secondary_intent=secondary_intent,
        conversation_state=state,
        intent_confidence=confidence,
        entities=entities,
        observed_symptoms=observed_symptoms,
        customer_claims=customer_claims,
        confirmed_causes=confirmed_causes,
        working_hypotheses=working_hypotheses,
        risk_flags=risk_flags,
        backup_restore_related=backup_restore_related,
        data_loss=data_loss,
    )

"""
Evaluate the deterministic issue analyzer against the locked golden set.

Metrics:
- Intent accuracy
- Intent macro / weighted F1
- Per-intent precision / recall / F1
- Conversation-state accuracy
- Per-state precision / recall / F1
- Risk-flag precision / recall / F1
- Confusion matrices
- Misclassified examples

The golden set is evaluation-only. It is never used to modify the analyzer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.agent.issue_analyzer import analyze_case
from src.agent.schemas import (
    ConversationState,
    Message,
    SupportCase,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GOLDEN_PATH = (
    PROJECT_ROOT / "data" / "golden" / "apple_support_golden_set.csv"
)

DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "apple_support_cases.jsonl"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "issue_analyzer_evaluation.json"
)

DEFAULT_ERRORS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "issue_analyzer_errors.jsonl"
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def _load_golden_set(path: Path) -> List[dict]:
    """Load the locked golden annotations."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(
            f"Golden set is empty: {path}"
        )

    required_columns = {
        "case_id",
        "customer_message",
        "customer_followup",
        "annotator_intent",
        "conversation_state",
        "risk_flags",
        "backup_restore_related",
        "data_loss",
    }

    missing = required_columns - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Golden set is missing columns: {sorted(missing)}"
        )

    return rows


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def _build_case_from_golden_row(row: dict) -> SupportCase:
    """
    Build a SupportCase directly from the golden-set customer conversation.

    This is used when a separately reconstructed-case file is unavailable.

    We intentionally use only customer text for the analyzer because the
    analyzer is supposed to understand the customer's problem, not infer
    intent from Apple's response.
    """

    messages = []

    customer_message = (
        row.get("customer_message") or ""
    ).strip()

    customer_followup = (
        row.get("customer_followup") or ""
    ).strip()

    if customer_message:
        messages.append(
            Message(
                message_id=f"{row['case_id']}_CUSTOMER_1",
                sender="customer",
                timestamp="2026-01-01T00:00:00Z",
                text=customer_message,
                attachments=[],
            )
        )

    if customer_followup:
        messages.append(
            Message(
                message_id=f"{row['case_id']}_CUSTOMER_2",
                sender="customer",
                timestamp="2026-01-01T00:01:00Z",
                text=customer_followup,
                attachments=[],
            )
        )

    return SupportCase(
        case_id=row["case_id"],
        customer_id=f"GOLDEN_{row['case_id']}",
        brand="AppleSupport",
        created_at="2026-01-01T00:00:00Z",
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Optional reconstructed-case loader
# ---------------------------------------------------------------------------

def _load_reconstructed_cases(
    path: Path,
) -> Dict[str, SupportCase]:
    """
    Load reconstructed cases if a JSONL case file exists.

    The loader accepts the project's expected SupportCase-compatible
    structure and skips malformed records rather than silently fabricating
    data.
    """

    if not path.exists():
        return {}

    cases: Dict[str, SupportCase] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw = json.loads(line)
                case = SupportCase.model_validate(raw)
                cases[case.case_id] = case
            except Exception as exc:
                print(
                    f"[WARN] Could not parse case on line "
                    f"{line_number}: {exc}"
                )

    return cases


# ---------------------------------------------------------------------------
# Risk flag parsing
# ---------------------------------------------------------------------------

def _parse_risk_flags(value: str) -> Set[str]:
    """
    Parse the golden-set risk_flags field.

    Expected format is usually:
        ["account_compromise", "fraud"]

    Empty values become an empty set.
    """

    if not value:
        return set()

    value = value.strip()

    if not value:
        return set()

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return {
                str(item).strip()
                for item in parsed
                if str(item).strip()
            }

    except json.JSONDecodeError:
        pass

    # Conservative fallback for malformed annotation serialization.
    value = value.strip("[]")
    value = value.replace('"', "")
    value = value.replace("'", "")

    if not value.strip():
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


# ---------------------------------------------------------------------------
# Risk flag normalization
# ---------------------------------------------------------------------------

def _normalize_predicted_risk_flag(flag) -> str:
    """
    Convert a RiskFlag enum/string into the same representation used by
    the golden set.
    """

    value = getattr(flag, "value", flag)

    return str(value).strip()


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> dict:
    """Calculate robust classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def _risk_metrics(
    golden_flags: List[Set[str]],
    predicted_flags: List[Set[str]],
) -> dict:
    """Calculate micro-averaged multilabel risk metrics."""

    all_labels = sorted(
        set().union(*golden_flags, *predicted_flags)
    )

    if not all_labels:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "per_flag": {},
        }

    true_positive = Counter()
    false_positive = Counter()
    false_negative = Counter()

    for true_set, pred_set in zip(
        golden_flags,
        predicted_flags,
    ):
        for label in all_labels:
            if label in true_set and label in pred_set:
                true_positive[label] += 1

            elif label not in true_set and label in pred_set:
                false_positive[label] += 1

            elif label in true_set and label not in pred_set:
                false_negative[label] += 1

    total_tp = sum(true_positive.values())
    total_fp = sum(false_positive.values())
    total_fn = sum(false_negative.values())

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )

    micro_f1 = (
        2
        * micro_precision
        * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    per_flag = {}

    for label in all_labels:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_flag[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_flag": per_flag,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    golden_path: Path,
    cases_path: Path,
    output_path: Path,
    errors_path: Path,
) -> dict:

    golden_rows = _load_golden_set(golden_path)

    reconstructed_cases = _load_reconstructed_cases(
        cases_path
    )

    print(
        f"Golden records       : {len(golden_rows)}"
    )

    print(
        f"Reconstructed cases  : {len(reconstructed_cases)}"
    )

    if len(golden_rows) != 200:
        print(
            f"[WARN] Expected 200 golden cases, "
            f"found {len(golden_rows)}"
        )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    y_true_intent = []
    y_pred_intent = []

    y_true_state = []
    y_pred_state = []

    true_risk_flags = []
    pred_risk_flags = []

    errors = []

    confidence_values = []

    for row in golden_rows:

        case_id = row["case_id"]

        case = reconstructed_cases.get(case_id)

        if case is None:
            case = _build_case_from_golden_row(row)

        result = analyze_case(case)

        true_intent = (
            row["annotator_intent"]
            .strip()
        )

        pred_intent = (
            result.intent.value
        )

        true_state = (
            row["conversation_state"]
            .strip()
        )

        pred_state = (
            result.conversation_state.value
        )

        golden_risks = _parse_risk_flags(
            row.get("risk_flags", "")
        )

        predicted_risks = {
            _normalize_predicted_risk_flag(flag)
            for flag in result.risk_flags
        }

        y_true_intent.append(true_intent)
        y_pred_intent.append(pred_intent)

        y_true_state.append(true_state)
        y_pred_state.append(pred_state)

        true_risk_flags.append(golden_risks)
        pred_risk_flags.append(predicted_risks)

        confidence_values.append(
            float(result.intent_confidence)
        )

        if (
            true_intent != pred_intent
            or true_state != pred_state
        ):
            errors.append(
                {
                    "case_id": case_id,
                    "customer_message": row.get(
                        "customer_message",
                        "",
                    ),
                    "customer_followup": row.get(
                        "customer_followup",
                        "",
                    ),
                    "gold_intent": true_intent,
                    "pred_intent": pred_intent,
                    "gold_state": true_state,
                    "pred_state": pred_state,
                    "intent_confidence": result.intent_confidence,
                    "predicted_risk_flags": sorted(
                        predicted_risks
                    ),
                    "gold_risk_flags": sorted(
                        golden_risks
                    ),
                }
            )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    intent_labels = [
        "SOFTWARE_UPDATE",
        "DEVICE_PERFORMANCE",
        "APP_SERVICE",
        "DEVICE_HARDWARE",
        "CONNECTIVITY",
        "ACCOUNT_ICLOUD",
        "PURCHASE_BILLING",
        "INPUT_DISPLAY",
        "OTHER_UNCLEAR",
        "BACKUP_RESTORE",
    ]

    state_labels = [
        "TROUBLESHOOTING",
        "FAILED_TROUBLESHOOTING",
        "INFORMATION_REQUEST",
        "RESOLVED",
    ]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    intent_metrics = _classification_metrics(
        y_true_intent,
        y_pred_intent,
        intent_labels,
    )

    state_metrics = _classification_metrics(
        y_true_state,
        y_pred_state,
        state_labels,
    )

    risk_metrics = _risk_metrics(
        true_risk_flags,
        pred_risk_flags,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    result = {
        "evaluation": {
            "golden_set_path": str(
                golden_path.relative_to(PROJECT_ROOT)
                if golden_path.is_relative_to(PROJECT_ROOT)
                else golden_path
            ),
            "cases_path": str(
                cases_path.relative_to(PROJECT_ROOT)
                if cases_path.is_relative_to(PROJECT_ROOT)
                else cases_path
            ),
            "records_evaluated": len(golden_rows),
            "misclassified_records": len(errors),
        },
        "intent": intent_metrics,
        "conversation_state": state_metrics,
        "risk_flags": risk_metrics,
        "confidence": {
            "mean": (
                sum(confidence_values)
                / len(confidence_values)
                if confidence_values
                else 0.0
            ),
            "min": min(confidence_values)
            if confidence_values
            else 0.0,
            "max": max(confidence_values)
            if confidence_values
            else 0.0,
        },
    }

    # ---------------------------------------------------------
    # Write evaluation report
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Write error analysis
    # ---------------------------------------------------------

    errors_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with errors_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for error in errors:
            f.write(
                json.dumps(
                    error,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ISSUE ANALYZER EVALUATION")
    print("=" * 70)

    print()
    print("INTENT")
    print("-" * 70)

    print(
        f"Accuracy       : "
        f"{intent_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{intent_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{intent_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{intent_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{intent_metrics['weighted_f1']:.4f}"
    )

    print()
    print("CONVERSATION STATE")
    print("-" * 70)

    print(
        f"Accuracy       : "
        f"{state_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{state_metrics['macro_f1']:.4f}"
    )

    print()
    print("RISK FLAGS")
    print("-" * 70)

    print(
        f"Micro Precision: "
        f"{risk_metrics['micro_precision']:.4f}"
    )

    print(
        f"Micro Recall   : "
        f"{risk_metrics['micro_recall']:.4f}"
    )

    print(
        f"Micro F1       : "
        f"{risk_metrics['micro_f1']:.4f}"
    )

    print()
    print("CONFIDENCE")
    print("-" * 70)

    print(
        f"Mean           : "
        f"{result['confidence']['mean']:.4f}"
    )

    print(
        f"Min            : "
        f"{result['confidence']['min']:.4f}"
    )

    print(
        f"Max            : "
        f"{result['confidence']['max']:.4f}"
    )

    print()
    print("ERRORS")
    print("-" * 70)

    print(
        f"Misclassified  : "
        f"{len(errors)} / {len(golden_rows)}"
    )

    print()
    print(f"Evaluation written to:")
    print(f"  {output_path}")

    print()
    print(f"Error analysis written to:")
    print(f"  {errors_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the deterministic issue analyzer "
            "against the AppleSupport golden set."
        )
    )

    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to the locked golden CSV.",
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=(
            "Optional reconstructed SupportCase JSONL. "
            "If absent, cases are built from the golden customer text."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON evaluation report.",
    )

    parser.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_ERRORS_PATH,
        help="Output JSONL error-analysis file.",
    )

    args = parser.parse_args()

    evaluate(
        golden_path=args.golden,
        cases_path=args.cases,
        output_path=args.output,
        errors_path=args.errors,
    )


if __name__ == "__main__":
    main()

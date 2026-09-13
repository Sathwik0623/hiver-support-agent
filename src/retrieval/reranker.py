from typing import Any


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def rerank_evidence(
    query: str,
    retrieved: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved evidence using semantic similarity and
    conversation-level evidence quality.

    Semantic similarity remains the dominant signal.
    Original similarity, rank, and resolution labels are preserved.
    """

    positive_phrases = (
        "it worked",
        "that worked",
        "works now",
        "working now",
        "got it back",
        "problem solved",
        "issue resolved",
        "fixed now",
        "all good now",
        "it's working",
        "it is working",
        "was able to get it back",
        "was able to restore",
        "successfully downloaded",
    )

    negative_phrases = (
        "didn't work",
        "did not work",
        "doesn't work",
        "does not work",
        "still not",
        "still doesn't",
        "still nothing",
        "nothing happens",
        "unable to",
        "no luck",
        "not fixed",
        "not working",
        "isn't working",
        "is not working",
        "cannot play",
        "can't play",
        "can't download",
        "cannot download",
    )

    clarification_phrases = (
        "what's happening",
        "what is happening",
        "which steps",
        "what version",
        "which version",
        "what device",
        "which device",
        "can you provide",
        "please provide",
        "have you tried",
    )

    scored = []

    for item in retrieved:
        result = dict(item)

        similarity = float(item.get("similarity", 0.0) or 0.0)
        response = _text(item.get("apple_response"))
        follow_up = _text(item.get("follow_up"))
        context = _text(item.get("conversation_context"))
        resolution_signal = _text(item.get("resolution_signal"))

        combined_text = f"{context} {response} {follow_up}"

        has_positive_signal = _has_any(
            combined_text,
            positive_phrases,
        )

        has_negative_signal = _has_any(
            combined_text,
            negative_phrases,
        )

        clarification_only = _has_any(
            response,
            clarification_phrases,
        )

        # Keep semantic similarity as the main ranking signal.
        score = similarity
        reasons = []

        quality_adjustment = 0.0

        if response:
            quality_adjustment += 0.04
            reasons.append("response_available")

        if resolution_signal == "resolved":
            quality_adjustment += 0.10
            reasons.append("resolved_signal")

        elif resolution_signal == "failed":
            quality_adjustment -= 0.10
            reasons.append("failed_signal")

        elif resolution_signal == "in_progress":
            quality_adjustment -= 0.02
            reasons.append("in_progress_signal")

        else:
            quality_adjustment -= 0.03
            reasons.append("unknown_resolution_signal")

        if has_positive_signal:
            quality_adjustment += 0.06
            reasons.append("positive_outcome_signal")

        if has_negative_signal:
            quality_adjustment -= 0.08
            reasons.append("negative_outcome_signal")

        if follow_up:
            quality_adjustment += 0.02
            reasons.append("follow_up_available")

        if clarification_only:
            quality_adjustment -= 0.04
            reasons.append("clarification_only_response")

        score += quality_adjustment

        if has_negative_signal:
            inferred_quality = "weak"
        elif has_positive_signal and response:
            inferred_quality = "strong"
        elif response:
            inferred_quality = "moderate"
        else:
            inferred_quality = "weak"

        result["original_rank"] = item.get("rank")
        result["base_similarity"] = similarity
        result["quality_score"] = round(quality_adjustment, 6)
        result["final_score"] = round(score, 6)
        result["inferred_evidence_quality"] = inferred_quality
        result["rerank_reasons"] = reasons

        scored.append(result)

    if scored:
        best_similarity = max(
            item["base_similarity"]
            for item in scored
        )

        similarity_window = 0.05

        eligible = [
            item
            for item in scored
            if item["base_similarity"]
            >= best_similarity - similarity_window
        ]

        ineligible = [
            item
            for item in scored
            if item["base_similarity"]
            < best_similarity - similarity_window
        ]

        eligible.sort(
            key=lambda item: (
                item["final_score"],
                item["base_similarity"],
            ),
            reverse=True,
        )

        ineligible.sort(
            key=lambda item: item["base_similarity"],
            reverse=True,
        )

        scored = eligible + ineligible

        for rank, item in enumerate(scored[:top_k], start=1):
            item["rank"] = rank

        return scored[:top_k]
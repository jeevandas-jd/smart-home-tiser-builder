"""
Validator tools for the TISER pipeline.

Fixes vs original:
  B5 - validate_answer: was reading goal["current_value"] (never present).
       Now walks targets[*].asserts[*].value matching the real eval schema.
  B6 - validate_no_hallucinated_entities: stop-word filter + only checks
       identifier-shaped tokens (contain _ or digit) so normal prose is not
       flagged. Also safely handles an empty valid_entities set.
"""

# src/tools.py

import re

REQUIRED_TAGS = ["reasoning", "timeline", "reflection", "answer"]

# Common words that are fine in any reasoning trace.
_STOP_WORDS = {
    "the", "and", "for", "are", "was", "has", "have", "that", "this",
    "with", "from", "will", "been", "they", "them", "then", "than",
    "when", "what", "which", "also", "both", "each", "its", "into",
    "not", "can", "but", "out", "one", "two", "all", "any", "more",
    "most", "some", "such", "only", "over", "very", "just", "does",
    "room", "mode", "state", "time", "load", "dryer", "washer",
    "start", "pause", "running", "stopped", "finish", "ready",
    "before", "after", "while", "since", "until", "about", "being",
    "should", "would", "could", "their", "there", "these", "those",
    "between", "because", "however", "therefore", "although",
}


# ── Structural ────────────────────────────────────────────────────────────────

def extract_tag_content(text, tag):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def validate_structure(llm_output):
    for tag in REQUIRED_TAGS:
        if f"<{tag}>" not in llm_output:
            return False, f"Missing opening tag: <{tag}>"
        if f"</{tag}>" not in llm_output:
            return False, f"Missing closing tag: </{tag}>"
    return True, "Structure valid"


# ── Answer ────────────────────────────────────────────────────────────────────

def validate_answer(extracted_answer, ground_truth_goals):
    """
    FIX B5: original code read goal["current_value"] which never exists.
    Now correctly walks targets[*].asserts[*] and also handles infeasibility
    episodes where the answer should flag a conflict rather than a value.
    """
    if not ground_truth_goals:
        return True, "No goals to validate"

    answer_lower = extracted_answer.strip().lower()

    # Collect expected values / descriptions from the real schema
    expected_values = []
    for goal in ground_truth_goals:
        for target in goal.get("targets", []):
            for assertion in target.get("asserts", []):
                val = assertion.get("value")
                desc = assertion.get("description", "")
                if val is not None:
                    expected_values.append(str(val).lower())
                if desc:
                    expected_values.append(desc.lower())

    if not expected_values:
        return True, "Goals present but contain no assert values"

    # Infeasibility episodes: answer must mention the conflict
    infeasibility_keywords = [
        "infeasible", "not possible", "cannot", "conflict",
        "impossible", "contradiction", "incompatible",
    ]
    if any(kw in answer_lower for kw in infeasibility_keywords):
        return True, "Answer correctly identifies infeasibility"

    # Feasibility episodes: at least one expected value must appear
    for ev in expected_values:
        if ev in answer_lower:
            return True, f"Answer references expected value: '{ev}'"

    return (
        False,
        (
            f"Answer does not reference any expected goal value. "
            f"Expected one of: {expected_values[:6]} | "
            f"Got: '{extracted_answer[:120]}'"
        ),
    )


# ── Hallucination ─────────────────────────────────────────────────────────────

def validate_no_hallucinated_entities(reasoning_text, valid_entities):
    """
    FIX B6: original flagged every English word because valid_entities was
    always empty. Now:
      - only checks identifier-shaped tokens (contain _ or a digit)
      - skips tokens < 5 chars and stop-words
      - skips entirely when valid_entities is empty
    """
    if not valid_entities:
        return []   # nothing to check against

    reasoning_lower = reasoning_text.lower()
    hallucinated = []

    for token in re.findall(r"\b[a-z][a-z0-9_]+\b", reasoning_lower):
        if len(token) < 5:
            continue
        if token in _STOP_WORDS:
            continue
        if "_" not in token and not any(c.isdigit() for c in token):
            continue  # plain English adjective/noun — skip
        if token not in valid_entities:
            hallucinated.append(token)

    return list(set(hallucinated))


# ── Reflection quality ────────────────────────────────────────────────────────

def validate_reflection_quality(reflection_text):
    banned_patterns = [
        "my reasoning is correct",
        "i am confident",
        "the answer is correct",
        "i verified the answer",
    ]
    reflection_lower = reflection_text.lower()
    for pattern in banned_patterns:
        if pattern in reflection_lower:
            return False, f"Low-quality reflection pattern: '{pattern}'"
    return True, "Reflection quality acceptable"


# ── Master validator ──────────────────────────────────────────────────────────

def verify_tiser_output(llm_output, ground_truth_goals, valid_entities):
    # 1. Structure
    valid, msg = validate_structure(llm_output)
    if not valid:
        return False, msg

    # 2. Extract
    reasoning  = extract_tag_content(llm_output, "reasoning")
    reflection = extract_tag_content(llm_output, "reflection")
    answer     = extract_tag_content(llm_output, "answer")

    # 3. Answer
    valid, msg = validate_answer(answer or "", ground_truth_goals)
    if not valid:
        return False, msg

    # 4. Reflection
    valid, msg = validate_reflection_quality(reflection or "")
    if not valid:
        return False, msg

    # 5. Hallucination
    if valid_entities:
        hallucinated = validate_no_hallucinated_entities(
            reasoning or "", valid_entities
        )
        if hallucinated:
            return False, f"Hallucinated identifiers: {hallucinated[:10]}"

    return True, "Validation Passed"

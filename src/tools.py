"""
The Validator Agent uses this module to verify 
that the generated text strictly matches our rules and doesn't hallucinate structural parameters.
"""

# src/tools.py
import re
from difflib import SequenceMatcher


REQUIRED_TAGS = [
    "reasoning",
    "timeline",
    "reflection",
    "answer"
]


def extract_tag_content(text, tag):

    pattern = rf"<{tag}>(.*?)</{tag}>"

    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    return match.group(1).strip()


def validate_structure(llm_output):

    for tag in REQUIRED_TAGS:

        if f"<{tag}>" not in llm_output:
            return False, f"Missing opening tag: <{tag}>"

        if f"</{tag}>" not in llm_output:
            return False, f"Missing closing tag: </{tag}>"

    return True, "Structure valid"


def validate_answer(
    extracted_answer,
    ground_truth_goals
):

    for goal in ground_truth_goals:

        expected = str(
            goal.get("current_value", "")
        ).strip().lower()

        actual = extracted_answer.strip().lower()

        if expected != actual:
            return False, (
                f"Answer mismatch | "
                f"Expected: {expected} | "
                f"Found: {actual}"
            )

    return True, "Answer valid"


def validate_no_hallucinated_entities(
    reasoning_text,
    valid_entities
):

    reasoning_lower = reasoning_text.lower()

    hallucinated = []

    for token in re.findall(r'\b[a-zA-Z_]+\b', reasoning_lower):

        if (
            token not in valid_entities
            and len(token) > 3
        ):
            hallucinated.append(token)

    hallucinated = list(set(hallucinated))

    return hallucinated


def validate_reflection_quality(
    reflection_text
):

    banned_patterns = [
        "my reasoning is correct",
        "i am confident",
        "the answer is correct",
        "i verified the answer"
    ]

    reflection_lower = reflection_text.lower()

    for pattern in banned_patterns:

        if pattern in reflection_lower:
            return False, (
                f"Low-quality reflection pattern: {pattern}"
            )

    return True, "Reflection quality acceptable"


def verify_tiser_output(
    llm_output,
    ground_truth_goals,
    valid_entities
):

    # --------------------------------------------------
    # 1. Structural Validation
    # --------------------------------------------------

    valid, msg = validate_structure(llm_output)

    if not valid:
        return False, msg

    # --------------------------------------------------
    # 2. Extract Sections
    # --------------------------------------------------

    reasoning = extract_tag_content(
        llm_output,
        "reasoning"
    )

    timeline = extract_tag_content(
        llm_output,
        "timeline"
    )

    reflection = extract_tag_content(
        llm_output,
        "reflection"
    )

    answer = extract_tag_content(
        llm_output,
        "answer"
    )

    # --------------------------------------------------
    # 3. Answer Validation
    # --------------------------------------------------

    valid, msg = validate_answer(
        answer,
        ground_truth_goals
    )

    if not valid:
        return False, msg

    # --------------------------------------------------
    # 4. Reflection Validation
    # --------------------------------------------------

    valid, msg = validate_reflection_quality(
        reflection
    )

    if not valid:
        return False, msg

    # --------------------------------------------------
    # 5. Hallucination Detection
    # --------------------------------------------------

    hallucinated = validate_no_hallucinated_entities(
        reasoning,
        valid_entities
    )

    if hallucinated:
        return False, (
            f"Hallucinated entities detected: "
            f"{hallucinated[:10]}"
        )

    return True, "Validation Passed"

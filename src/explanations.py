from typing import List, Tuple, Dict


def classify_survival_level(survival_probability: float) -> str:
    """
    Convert a survival probability into a qualitative label.
    """
    if survival_probability >= 0.80:
        return "very favorable"
    elif survival_probability >= 0.65:
        return "favorable"
    elif survival_probability >= 0.45:
        return "intermediate"
    elif survival_probability >= 0.25:
        return "concerning"
    return "high risk"


def summarize_top_factors(top_effects: List[Tuple[str, float]], max_items: int = 5) -> Dict[str, List[str]]:
    """
    Split SHAP-like contributions into positive and negative groups.
    Input format example:
        [("Risk Group = High Risk", -0.18), ("HLA Match = 10/10", +0.12)]
    """
    positive = []
    negative = []

    for name, value in top_effects[:max_items]:
        if value > 0:
            positive.append(name)
        elif value < 0:
            negative.append(name)

    return {
        "positive": positive,
        "negative": negative
    }


def generate_doctor_explanation(
    survival_probability: float,
    death_probability: float,
    risk_label: str,
    top_effects: List[Tuple[str, float]]
) -> str:
    """
    Detailed, medically oriented explanation for doctors.
    """
    summary = summarize_top_factors(top_effects)
    tone = classify_survival_level(survival_probability)

    text = []
    text.append(
        f"This prediction suggests a **{survival_probability:.1%} estimated probability of survival** "
        f"and a **{death_probability:.1%} estimated probability of death**, corresponding to a "
        f"**{risk_label.lower()}** profile."
    )

    text.append(
        f"Overall, the model classifies this case as **{tone}** based on the combination of "
        f"recipient characteristics, donor profile, transplant compatibility, and graft-related parameters."
    )

    if summary["positive"]:
        text.append(
            "**Factors supporting survival in this case:** "
            + ", ".join(summary["positive"]) + "."
        )

    if summary["negative"]:
        text.append(
            "**Factors reducing the predicted survival probability:** "
            + ", ".join(summary["negative"]) + "."
        )

    text.append(
        "This output should be interpreted as a probabilistic decision-support signal rather than a "
        "deterministic prognosis. Clinical judgment remains essential, especially in the context of "
        "small-sample pediatric transplant datasets and possible calibration limits."
    )

    return "\n\n".join(text)


def generate_nurse_explanation(
    survival_probability: float,
    death_probability: float,
    risk_label: str,
    top_effects: List[Tuple[str, float]]
) -> str:
    """
    Practical clinical explanation for nurses.
    """
    summary = summarize_top_factors(top_effects)

    text = []
    text.append(
        f"The model estimates a **{survival_probability:.1%} chance of survival** "
        f"and places this patient in a **{risk_label.lower()}** category."
    )

    text.append(
        "This means the patient may require closer monitoring depending on the clinical context, "
        "especially when several negative transplant or compatibility factors are present."
    )

    if summary["positive"]:
        text.append(
            "**Helpful factors in this prediction:** "
            + ", ".join(summary["positive"]) + "."
        )

    if summary["negative"]:
        text.append(
            "**Factors that may require attention:** "
            + ", ".join(summary["negative"]) + "."
        )

    text.append(
        "This result is not a replacement for medical decision-making, but it may help highlight "
        "which clinical elements are pushing the prediction toward a better or worse outcome."
    )

    return "\n\n".join(text)


def generate_public_explanation(
    survival_probability: float,
    death_probability: float,
    risk_label: str,
    top_effects: List[Tuple[str, float]]
) -> str:
    """
    Simplified explanation for non-medical users.
    """
    summary = summarize_top_factors(top_effects)

    text = []
    text.append(
        f"The system estimates a **{survival_probability:.1%} probability of survival** for this case."
    )

    text.append(
        f"In simple terms, this result corresponds to a **{risk_label.lower()}** situation."
    )

    if summary["positive"]:
        text.append(
            "Some elements in the medical data seem to support a better outcome, such as: "
            + ", ".join(summary["positive"]) + "."
        )

    if summary["negative"]:
        text.append(
            "Other elements seem to make the situation more difficult, such as: "
            + ", ".join(summary["negative"]) + "."
        )

    text.append(
        "This prediction is only a computer estimate based on medical data. It cannot replace "
        "a doctor’s evaluation, and it should always be discussed with a healthcare professional."
    )

    return "\n\n".join(text)


def generate_role_based_explanation(
    role: str,
    survival_probability: float,
    death_probability: float,
    risk_label: str,
    top_effects: List[Tuple[str, float]]
) -> str:
    """
    Main router that returns the right explanation depending on user role.
    """
    role = role.strip().lower()

    if role == "doctor":
        return generate_doctor_explanation(
            survival_probability, death_probability, risk_label, top_effects
        )

    if role == "nurse":
        return generate_nurse_explanation(
            survival_probability, death_probability, risk_label, top_effects
        )

    return generate_public_explanation(
        survival_probability, death_probability, risk_label, top_effects
       )
    

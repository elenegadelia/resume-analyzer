"""
explainer.py — Generate plain-English explanations of match results.

Numbers alone aren't enough. A good tool tells the user what to do next.
This module converts scores and keyword lists into human-readable feedback.
"""


def score_label(score: float) -> str:
    """Return a descriptive label for the match score."""
    if score >= 80:
        return "Excellent Match"
    elif score >= 60:
        return "Good Match"
    elif score >= 40:
        return "Moderate Match"
    elif score >= 20:
        return "Weak Match"
    else:
        return "Poor Match"


def score_color(score: float) -> str:
    """Return a color string for UI display based on score."""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "blue"
    elif score >= 40:
        return "orange"
    else:
        return "red"


def generate_explanation(score: float, matched: list[str], missing: list[str]) -> str:
    """
    Generate a clear, actionable plain-English explanation of the analysis.

    This is what makes the tool feel intelligent and useful — not just numbers,
    but context that helps the user understand and act.
    """
    label = score_label(score)
    n_matched = len(matched)
    n_missing = len(missing)

    lines = []

    # Opening summary
    lines.append(f"**Overall Assessment: {label} ({score:.1f}%)**\n")

    # Score explanation
    if score >= 80:
        lines.append(
            "Your resume aligns very strongly with this job description. "
            "The language, skills, and experience you highlight closely match "
            "what the employer is looking for."
        )
    elif score >= 60:
        lines.append(
            "Your resume is a solid match for this role. You cover many of the "
            "key requirements, though there are a few gaps worth addressing before applying."
        )
    elif score >= 40:
        lines.append(
            "Your resume partially matches this job. There is meaningful overlap, "
            "but a significant number of important keywords and skills are missing. "
            "Consider tailoring your resume more specifically to this role."
        )
    elif score >= 20:
        lines.append(
            "Your resume has limited alignment with this job description. "
            "You may have relevant experience, but the terminology and keywords "
            "you use don't closely match what this employer is looking for."
        )
    else:
        lines.append(
            "Your resume does not appear well-matched to this job description. "
            "This could mean the role is outside your current experience, or your "
            "resume needs significant rewriting to reflect the right skills and language."
        )

    lines.append("")

    # Matched keywords feedback
    if n_matched > 0:
        lines.append(
            f"**Strengths ({n_matched} matched keywords):** "
            f"Your resume already includes key terms the employer is looking for, such as: "
            f"*{', '.join(matched[:8])}{'...' if n_matched > 8 else ''}*."
        )
    else:
        lines.append(
            "**Strengths:** No strong keyword matches were found. "
            "Your resume may use different terminology than the job description."
        )

    lines.append("")

    # Missing keywords feedback
    if n_missing > 0:
        top_missing = missing[:8]
        lines.append(
            f"**Gaps to Address ({n_missing} missing keywords):** "
            f"Consider adding these terms to your resume if they reflect your actual experience: "
            f"*{', '.join(top_missing)}{'...' if n_missing > 8 else ''}*."
        )
        lines.append("")
        lines.append(
            "> **Tip:** Don't just paste keywords in. Integrate them naturally into your "
            "bullet points and descriptions. ATS systems and human reviewers both appreciate "
            "context over keyword stuffing."
        )
    else:
        lines.append(
            "**Gaps:** No major gaps detected — your resume covers the key terms well."
        )

    return "\n".join(lines)


def generate_improvement_suggestions(missing: list[str], score: float) -> list[str]:
    """
    Generate a short list of concrete improvement actions.
    """
    suggestions = []

    if score < 60:
        suggestions.append("Rewrite your resume summary to mirror the job description's language.")
        suggestions.append("Add a dedicated 'Skills' section listing relevant technologies and tools.")

    if missing:
        suggestions.append(
            f"Incorporate these missing keywords into your experience bullets: "
            f"{', '.join(missing[:5])}."
        )

    suggestions.append("Quantify your achievements (e.g., 'Improved model accuracy by 15%').")
    suggestions.append("Use the same job title or role name that appears in the job description.")

    return suggestions

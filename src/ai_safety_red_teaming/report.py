"""Human-readable rendering of a safety report."""

from __future__ import annotations

from .scoring import SafetyReport


def _bar(rate: float, width: int = 20) -> str:
    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled)


def render_text(report: SafetyReport) -> str:
    """Render a safety report as plain text for the terminal."""
    lines = [
        "AI Safety Red-Team Report",
        "=" * 25,
        "",
        f"Attacks run: {report.total}",
        f"Attack success rate (ASR): {report.asr:.1%}  {_bar(report.asr)}",
        f"Strict ASR (full compliance only): {report.asr_strict:.1%}",
        f"Refusal rate: {report.refusal_rate:.1%}",
        f"Partial compliance rate: {report.partial_rate:.1%}",
        f"Unsafe responses: {report.unsafe_rate:.1%}",
        f"Consistency score: {report.consistency_score:.1%} of requests "
        "had the same outcome across all techniques",
        "",
        "ASR by technique (most successful first):",
    ]
    for entry in report.top_techniques:
        rate = entry["attack_success_rate"]
        lines.append(
            f"  {entry['technique']:22s} {rate:.1%} {_bar(rate)}"
        )
    lines.append("")
    lines.append("ASR by category:")
    for category, rate in sorted(report.asr_by_category.items()):
        lines.append(f"  {category:22s} {rate:.1%} {_bar(rate)}")
    if report.inconsistent_requests:
        lines.append("")
        lines.append(
            f"Inconsistent requests ({len(report.inconsistent_requests)}): "
            "the target refused some framings and complied with others."
        )
        for entry in report.inconsistent_requests:
            lines.append(
                f"  {entry['request_id']}: outcomes "
                f"{', '.join(entry['outcomes'])}; got through via "
                f"{', '.join(entry['succeeded_by']) or '—'}"
            )
    else:
        lines.append("")
        lines.append("No inconsistent requests: the target behaved the "
                     "same way across every framing of each request.")
    return "\n".join(lines)

"""ai-safety-red-teaming: offline, deterministic AI safety red-teaming.

Attack prompt generation, rule-based refusal and unsafe-content
analysis, attack success rate and refusal-consistency scoring.
Standard library only; no model calls, no network.
"""

from .analysis import (
    CATEGORY_KEYWORDS,
    UnsafeDetection,
    RefusalClassification,
    classify_refusal,
    detect_unsafe,
)
from .generators import (
    AttackPrompt,
    AttackTechnique,
    generate_all,
    generate_attack,
    generate_escalation,
)
from .scoring import (
    AttackResult,
    SafetyReport,
    analyse_result,
    consistency_by_request,
    score_results,
)
from .target import FileTargetModel, MockTargetModel, TargetModel

__version__ = "0.1.0"

__all__ = [
    "AttackPrompt",
    "AttackResult",
    "AttackTechnique",
    "CATEGORY_KEYWORDS",
    "FileTargetModel",
    "MockTargetModel",
    "RefusalClassification",
    "SafetyReport",
    "TargetModel",
    "UnsafeDetection",
    "analyse_result",
    "classify_refusal",
    "consistency_by_request",
    "detect_unsafe",
    "generate_all",
    "generate_attack",
    "generate_escalation",
    "score_results",
]

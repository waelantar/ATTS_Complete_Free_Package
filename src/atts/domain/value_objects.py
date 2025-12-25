"""
Value Objects - Immutable domain objects with validation.

These represent concepts that are defined by their values rather than identity.
They are immutable and self-validating.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class ComputeMode(Enum):
    """
    Section 2.3.2: Compute Allocation Policy modes.

    Each mode represents a different compute budget:
    - DIRECT: Minimal tokens, single-step solutions
    - THINKING: Moderate tokens, step-by-step reasoning
    - DEEP: Maximum tokens, full verification chain
    """
    DIRECT = "direct"
    THINKING = "thinking"
    DEEP = "deep"

    @property
    def description(self) -> str:
        """Human-readable description of the mode."""
        descriptions = {
            "direct": "Fast single-step solution (low compute)",
            "thinking": "Multi-step reasoning (medium compute)",
            "deep": "Full verification chain (high compute)",
        }
        return descriptions.get(self.value, "Unknown mode")

    @property
    def expected_tokens(self) -> int:
        """Expected token usage for this mode."""
        tokens = {"direct": 150, "thinking": 500, "deep": 1000}
        return tokens.get(self.value, 500)

    def can_escalate(self) -> bool:
        """Check if mode can be escalated further."""
        return self != ComputeMode.DEEP

    def next_mode(self) -> Optional["ComputeMode"]:
        """Get the next escalation mode."""
        if self == ComputeMode.DIRECT:
            return ComputeMode.THINKING
        elif self == ComputeMode.THINKING:
            return ComputeMode.DEEP
        return None


@dataclass(frozen=True)
class DifficultyEstimate:
    """
    Section 2.3.1: Difficulty Estimation result.

    Attributes:
        value: The estimated difficulty (1-10)
        uncertainty: Variance from Pass@k sampling
        samples: Individual estimates from each sample
        recommended_mode: Mode recommended based on difficulty
    """
    value: int
    uncertainty: float
    samples: tuple  # Immutable list of individual estimates
    recommended_mode: ComputeMode

    @property
    def confidence(self) -> float:
        """Confidence score (inverse of uncertainty, normalized)."""
        return max(0.0, min(1.0, 1.0 - self.uncertainty / 5.0))

    @property
    def explanation(self) -> str:
        """Human-readable explanation of the estimate."""
        conf_label = "high" if self.confidence > 0.7 else "moderate" if self.confidence > 0.4 else "low"
        return (
            f"Difficulty: {self.value}/10 (confidence: {conf_label}). "
            f"Samples: {list(self.samples)}. "
            f"Recommended mode: {self.recommended_mode.value}."
        )


@dataclass(frozen=True)
class RubricScores:
    """
    Section 2.1.2: USVA Generalized Verification Rubrics.

    Attributes:
        lc: Logical Coherence - Do steps follow logically?
        fc: Factual Correctness - Are calculations correct?
        cm: Completeness - Are all aspects addressed?
        ga: Goal Alignment - Is there a clear final answer?
    """
    lc: float  # Logical Coherence
    fc: float  # Factual Correctness
    cm: float  # Completeness
    ga: float  # Goal Alignment

    def __post_init__(self):
        """Validate scores are in [0, 1]."""
        for name, value in [("LC", self.lc), ("FC", self.fc),
                           ("CM", self.cm), ("GA", self.ga)]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

    @property
    def overall(self) -> float:
        """Overall score (average of all rubrics)."""
        return (self.lc + self.fc + self.cm + self.ga) / 4.0

    @property
    def weakest_rubric(self) -> tuple:
        """Return the weakest rubric (name, score)."""
        rubrics = {"LC": self.lc, "FC": self.fc, "CM": self.cm, "GA": self.ga}
        weakest = min(rubrics.items(), key=lambda x: x[1])
        return weakest

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"LC": self.lc, "FC": self.fc, "CM": self.cm, "GA": self.ga}

    @property
    def explanation(self) -> str:
        """Human-readable explanation of rubric scores."""
        rubric_names = {
            "LC": "Logical Coherence",
            "FC": "Factual Correctness",
            "CM": "Completeness",
            "GA": "Goal Alignment",
        }
        lines = [f"Verification Rubric Scores (Overall: {self.overall:.2f}):"]
        for key, full_name in rubric_names.items():
            score = getattr(self, key.lower())
            status = "PASS" if score >= 0.7 else "WARN" if score >= 0.5 else "FAIL"
            lines.append(f"  {key} ({full_name}): {score:.2f} [{status}]")
        weakest_name, weakest_score = self.weakest_rubric
        lines.append(f"  Weakest area: {rubric_names[weakest_name]} ({weakest_score:.2f})")
        return "\n".join(lines)


@dataclass(frozen=True)
class VerificationResult:
    """
    Section 2.1: Complete verification result.

    Combines rubric scores with overall assessment.
    """
    rubric_scores: RubricScores
    overall_score: float
    raw_response: str = ""
    reasoning: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if verification passed minimum threshold."""
        return self.overall_score >= 0.6

    @property
    def needs_escalation(self) -> bool:
        """Check if escalation is recommended."""
        return self.overall_score < 0.8


@dataclass
class DecisionTrace:
    """
    Explainability: Full decision trace for a workflow step.

    Provides transparency into why each decision was made.
    """
    step: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    decision: str = ""
    reasoning: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    alternatives_considered: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "confidence": self.confidence,
            "alternatives_considered": self.alternatives_considered,
        }

    @property
    def summary(self) -> str:
        """One-line summary of the decision."""
        return f"[{self.step}] {self.decision} (confidence: {self.confidence:.0%})"

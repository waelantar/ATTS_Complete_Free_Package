"""
Domain Entities - Core business objects for ATTS.

These entities represent the fundamental concepts in the ATTS workflow.
They are pure data structures with no external dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class DifficultyLabel(Enum):
    """Problem difficulty categories."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Problem:
    """
    A math problem to be solved.

    Attributes:
        id: Unique identifier
        problem: The problem statement text
        answer: The correct answer
        difficulty_label: Ground truth difficulty (easy/medium/hard)
        metadata: Additional problem metadata
    """
    id: str
    problem: str
    answer: str
    difficulty_label: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def difficulty_numeric(self) -> int:
        """Convert label to numeric value for comparison."""
        mapping = {"easy": 3, "medium": 5, "hard": 8}
        return mapping.get(self.difficulty_label, 5)


@dataclass
class Solution:
    """
    A generated solution with metadata.

    Attributes:
        text: The solution text
        tokens_used: Number of tokens consumed
        mode: Compute mode used (direct/thinking/deep)
        generation_time: Time taken to generate
    """
    text: str
    tokens_used: int = 0
    mode: str = "direct"
    generation_time: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Check if solution is empty or invalid."""
        return not self.text or len(self.text.strip()) < 3


@dataclass
class RefinementStep:
    """
    One iteration of dialectical refinement.

    Section 1.2: Dialectical Nature of Advanced Reasoning

    Attributes:
        iteration: The refinement iteration number
        critique: The critique generated
        critique_valid: Whether meta-verification passed
        action: What action was taken (refined/stopped)
        reason: Why the action was taken
        improved_solution: The refined solution (if refined)
        tokens_used: Tokens consumed in this step
    """
    iteration: int
    critique: str
    critique_valid: bool
    action: str  # "refined" or "stopped"
    reason: str
    improved_solution: Optional[str] = None
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "critique": self.critique,
            "critique_valid": self.critique_valid,
            "action": self.action,
            "reason": self.reason,
            "improved_solution": self.improved_solution,
            "tokens_used": self.tokens_used,
        }


@dataclass
class WorkflowResult:
    """
    Complete result of ATTS workflow execution.

    Appendix A: Full ATTS Workflow Result

    Contains all information needed for:
    - Accuracy evaluation
    - Token efficiency analysis
    - Decision explainability
    - Pareto frontier computation
    """
    # Problem info
    problem_id: str
    true_difficulty: str

    # Difficulty estimation (Section 2.3.1)
    predicted_difficulty: int
    difficulty_uncertainty: float

    # Mode selection (Section 2.3.2)
    initial_mode: str
    final_mode: str
    escalated: bool
    escalation_path: List[str]

    # Verification (Section 2.1)
    verification_score: float
    rubric_scores: Dict[str, float]

    # Refinement (Section 2.4)
    refinement_history: List[RefinementStep]

    # Results
    solution: str
    tokens_used: int
    correct: bool

    # Timing
    total_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Decision trace for explainability
    decision_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.problem_id,
            "true_difficulty": self.true_difficulty,
            "predicted_difficulty": self.predicted_difficulty,
            "difficulty_uncertainty": self.difficulty_uncertainty,
            "initial_mode": self.initial_mode,
            "final_mode": self.final_mode,
            "escalated": self.escalated,
            "escalation_path": self.escalation_path,
            "verification_score": self.verification_score,
            "rubric_scores": self.rubric_scores,
            "refinement_history": [r.to_dict() for r in self.refinement_history],
            "solution": self.solution,
            "tokens": self.tokens_used,
            "correct": self.correct,
            "total_time": self.total_time,
            "timestamp": self.timestamp,
            "decision_trace": self.decision_trace,
        }

    @property
    def refinement_iterations(self) -> int:
        """Number of refinement iterations performed."""
        return len(self.refinement_history)

    @property
    def mode_efficiency_label(self) -> str:
        """Human-readable efficiency label."""
        if self.final_mode == "direct":
            return "Efficient (minimal compute)"
        elif self.final_mode == "thinking":
            return "Balanced (moderate compute)"
        else:
            return "Thorough (maximum compute)"

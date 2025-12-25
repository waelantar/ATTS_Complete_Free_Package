"""
Decision Explainer - Provides human-readable explanations for ATTS decisions.

Implements explainable AI features for transparency and debugging.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..domain.entities import WorkflowResult


@dataclass
class ExplanationBlock:
    """A block of explanation text with metadata."""
    title: str
    content: str
    confidence: float = 1.0
    importance: str = "normal"  # low, normal, high, critical


class DecisionExplainer:
    """
    Generate human-readable explanations for ATTS workflow decisions.

    Provides:
    - Step-by-step decision breakdown
    - Confidence indicators
    - Alternative paths not taken
    - Recommendations for improvement
    """

    def explain_result(self, result: WorkflowResult) -> List[ExplanationBlock]:
        """
        Generate a full explanation for a workflow result.

        Args:
            result: The workflow result to explain

        Returns:
            List of explanation blocks
        """
        blocks = []

        # Overview
        blocks.append(self._explain_overview(result))

        # Difficulty estimation
        blocks.append(self._explain_difficulty(result))

        # Mode selection and escalation
        blocks.append(self._explain_mode_selection(result))

        # Verification
        blocks.append(self._explain_verification(result))

        # Refinement (if applicable)
        if result.refinement_history:
            blocks.append(self._explain_refinement(result))

        # Final outcome
        blocks.append(self._explain_outcome(result))

        return blocks

    def _explain_overview(self, result: WorkflowResult) -> ExplanationBlock:
        """Explain the overall workflow execution."""
        status = "CORRECT" if result.correct else "INCORRECT"
        efficiency = self._calculate_efficiency_label(result)

        return ExplanationBlock(
            title="Workflow Overview",
            content=(
                f"Problem {result.problem_id} ({result.true_difficulty} difficulty)\n"
                f"Result: {status} | Tokens: {result.tokens_used} | Time: {result.total_time:.2f}s\n"
                f"Efficiency: {efficiency}\n"
                f"Path: {' -> '.join(result.escalation_path)}"
            ),
            importance="high" if not result.correct else "normal",
        )

    def _explain_difficulty(self, result: WorkflowResult) -> ExplanationBlock:
        """Explain difficulty estimation."""
        accuracy = "accurate" if self._difficulty_matches(result) else "inaccurate"

        return ExplanationBlock(
            title="Difficulty Estimation (Section 2.3.1)",
            content=(
                f"Predicted: {result.predicted_difficulty}/10 "
                f"(uncertainty: {result.difficulty_uncertainty:.2f})\n"
                f"Actual: {result.true_difficulty} "
                f"({self._difficulty_to_numeric(result.true_difficulty)}/10)\n"
                f"Assessment: {accuracy.upper()}\n"
                f"Impact: {'None - correct mode selected' if accuracy == 'accurate' else 'May have affected mode selection'}"
            ),
            confidence=1.0 - result.difficulty_uncertainty / 5.0,
        )

    def _explain_mode_selection(self, result: WorkflowResult) -> ExplanationBlock:
        """Explain mode selection and any escalation."""
        if result.escalated:
            escalation_text = (
                f"Escalation: YES ({result.initial_mode} -> {result.final_mode})\n"
                f"Reason: Verification score below threshold\n"
                f"Path: {' -> '.join(result.escalation_path)}"
            )
            importance = "high"
        else:
            escalation_text = (
                f"Escalation: NO\n"
                f"Mode used: {result.final_mode}\n"
                f"Reason: Initial verification met quality threshold"
            )
            importance = "normal"

        return ExplanationBlock(
            title="Mode Selection (Section 2.3.2)",
            content=(
                f"Initial mode: {result.initial_mode} "
                f"(based on difficulty {result.predicted_difficulty})\n"
                f"{escalation_text}"
            ),
            importance=importance,
        )

    def _explain_verification(self, result: WorkflowResult) -> ExplanationBlock:
        """Explain USVA verification results."""
        rubric_text = "\n".join(
            f"  {k}: {v:.2f} {'[OK]' if v >= 0.7 else '[WEAK]' if v >= 0.5 else '[FAIL]'}"
            for k, v in result.rubric_scores.items()
        )

        quality = (
            "HIGH" if result.verification_score >= 0.8 else
            "ACCEPTABLE" if result.verification_score >= 0.6 else
            "LOW"
        )

        return ExplanationBlock(
            title="Verification (Section 2.1)",
            content=(
                f"Overall Score: {result.verification_score:.2f} ({quality})\n"
                f"USVA Rubric Breakdown:\n{rubric_text}"
            ),
            confidence=result.verification_score,
            importance="critical" if quality == "LOW" else "normal",
        )

    def _explain_refinement(self, result: WorkflowResult) -> ExplanationBlock:
        """Explain dialectical refinement process."""
        iterations = len(result.refinement_history)
        refined_count = sum(1 for r in result.refinement_history if r.action == "refined")
        hallucinated = sum(1 for r in result.refinement_history if not r.critique_valid)

        refinement_details = []
        for step in result.refinement_history:
            status = "Applied" if step.action == "refined" else "Stopped"
            validity = "Valid" if step.critique_valid else "Hallucinated"
            refinement_details.append(
                f"  Iteration {step.iteration + 1}: {status} ({validity} critique)"
            )

        return ExplanationBlock(
            title="Dialectical Refinement (Section 1.2, 2.4)",
            content=(
                f"Iterations: {iterations}\n"
                f"Refinements applied: {refined_count}\n"
                f"Hallucinated critiques filtered: {hallucinated}\n"
                f"Details:\n" + "\n".join(refinement_details)
            ),
            importance="high" if refined_count > 0 else "low",
        )

    def _explain_outcome(self, result: WorkflowResult) -> ExplanationBlock:
        """Explain the final outcome and recommendations."""
        if result.correct:
            recommendation = "No action needed - solution verified correct."
        else:
            # Analyze what might have gone wrong
            issues = []
            if result.verification_score < 0.6:
                issues.append("Low verification score suggests quality issues")
            if not result.escalated and result.true_difficulty == "hard":
                issues.append("Hard problem may have benefited from escalation")
            if result.difficulty_uncertainty > 2.0:
                issues.append("High difficulty uncertainty affected mode selection")

            recommendation = (
                "Potential issues:\n" + "\n".join(f"  - {i}" for i in issues)
                if issues else "Analysis inconclusive - manual review recommended"
            )

        return ExplanationBlock(
            title="Outcome Analysis",
            content=(
                f"Final Answer: {'CORRECT' if result.correct else 'INCORRECT'}\n"
                f"Tokens Used: {result.tokens_used}\n"
                f"Time: {result.total_time:.2f}s\n"
                f"\n{recommendation}"
            ),
            importance="critical" if not result.correct else "normal",
        )

    def _calculate_efficiency_label(self, result: WorkflowResult) -> str:
        """Calculate efficiency label based on mode and tokens."""
        if result.final_mode == "direct" and result.tokens_used < 200:
            return "OPTIMAL (minimal compute for easy problem)"
        elif result.final_mode == "thinking" and result.tokens_used < 600:
            return "EFFICIENT (balanced compute)"
        elif result.final_mode == "deep":
            if result.refinement_history:
                return "THOROUGH (full verification with refinement)"
            return "THOROUGH (full verification chain)"
        return "STANDARD"

    def _difficulty_matches(self, result: WorkflowResult) -> bool:
        """Check if predicted difficulty roughly matches actual."""
        actual_numeric = self._difficulty_to_numeric(result.true_difficulty)
        diff = abs(result.predicted_difficulty - actual_numeric)
        return diff <= 2

    def _difficulty_to_numeric(self, label: str) -> int:
        """Convert difficulty label to numeric value."""
        mapping = {"easy": 3, "medium": 5, "hard": 8}
        return mapping.get(label.lower(), 5)

    def summarize_for_log(self, result: WorkflowResult) -> str:
        """Generate a one-line summary for logging."""
        status = "OK" if result.correct else "ERR"
        return (
            f"[{status}] P{result.problem_id}: "
            f"d={result.predicted_difficulty} "
            f"m={result.final_mode} "
            f"v={result.verification_score:.2f} "
            f"t={result.tokens_used}"
        )

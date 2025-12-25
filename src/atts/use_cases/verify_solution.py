"""
Solution Verifier - USVA implementation with meta-verification.

Implements:
- Section 2.1: Unified Self-Verification Architecture
- Section 2.1.2: Generalized Verification Rubrics
- Section 2.1.3: Integrated Meta-Verification
"""

import re
from typing import Tuple

import numpy as np

from ..ports.model_caller import IModelCaller
from ..ports.config_loader import IConfigLoader
from ..domain.value_objects import RubricScores, VerificationResult, DecisionTrace


class SolutionVerifier:
    """
    Section 2.1: Unified Self-Verification Architecture (USVA)

    Evaluates solutions using:
    - LC: Logical Coherence
    - FC: Factual Correctness
    - CM: Completeness
    - GA: Goal Alignment

    Also implements meta-verification (Section 2.1.3) to detect
    hallucinated critiques.
    """

    def __init__(self, model: IModelCaller, config: IConfigLoader):
        """
        Initialize solution verifier.

        Args:
            model: Language model for verification
            config: Configuration loader
        """
        self._model = model
        self._config = config
        self._usva_prompt = config.get_prompt("verification.usva")
        self._meta_prompt = config.get_prompt("verification.meta_verification")

    def verify(self, problem_text: str, solution_text: str) -> VerificationResult:
        """
        Verify a solution using USVA rubrics.

        Args:
            problem_text: The original problem
            solution_text: The solution to verify

        Returns:
            VerificationResult with rubric scores and overall assessment
        """
        if not solution_text or len(solution_text.strip()) < 3:
            return VerificationResult(
                rubric_scores=RubricScores(lc=0.0, fc=0.0, cm=0.0, ga=0.0),
                overall_score=0.0,
                reasoning="Solution is empty or too short.",
            )

        # Generate verification
        prompt = self._usva_prompt.template.format(
            problem=problem_text,
            solution=solution_text,
        )

        response = self._model.generate(
            prompt=prompt,
            max_tokens=self._usva_prompt.max_tokens,
            temperature=0.3,  # Lower temperature for consistent evaluation
        )

        # Parse rubric scores
        rubric_scores = self._parse_rubric_scores(response.text)

        # Parse overall score
        overall_score = self._parse_overall_score(response.text, rubric_scores)

        # Extract reasoning
        reasoning = self._extract_reasoning(response.text)

        return VerificationResult(
            rubric_scores=rubric_scores,
            overall_score=overall_score,
            raw_response=response.text,
            reasoning=reasoning,
        )

    def meta_verify_critique(
        self,
        problem_text: str,
        solution_text: str,
        critique: str,
    ) -> Tuple[bool, str]:
        """
        Section 2.1.3: Integrated Meta-Verification

        Check if a critique is valid or hallucinated.

        Args:
            problem_text: The original problem
            solution_text: The solution being critiqued
            critique: The critique to verify

        Returns:
            Tuple of (is_valid, reason)
        """
        # Skip meta-verification for empty or "no issues" critiques
        if not critique or len(critique.strip()) < 10:
            return True, "Critique is empty or trivial"

        if "no issues" in critique.lower():
            return True, "Critique indicates no issues"

        # Generate meta-verification
        prompt = self._meta_prompt.template.format(
            problem=problem_text,
            solution=solution_text,
            critique=critique,
        )

        response = self._model.generate(
            prompt=prompt,
            max_tokens=self._meta_prompt.max_tokens,
            temperature=0.1,  # Very low temperature for binary decision
        )

        # Parse response
        response_lower = response.text.lower()

        if "valid" in response_lower and "hallucinated" not in response_lower:
            return True, "Meta-verification confirmed critique as valid"
        elif "hallucinated" in response_lower:
            return False, "Meta-verification detected hallucinated critique"
        else:
            # Default to valid if unclear
            return True, "Meta-verification inconclusive, assuming valid"

    def _parse_rubric_scores(self, response: str) -> RubricScores:
        """Parse individual rubric scores from response."""
        scores = {}

        for rubric in ["LC", "FC", "CM", "GA"]:
            match = re.search(rf'{rubric}:\s*([\d.]+)', response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    scores[rubric.lower()] = max(0.0, min(1.0, score))
                except ValueError:
                    scores[rubric.lower()] = 0.5
            else:
                scores[rubric.lower()] = 0.5

        return RubricScores(**scores)

    def _parse_overall_score(self, response: str, rubric_scores: RubricScores) -> float:
        """Parse overall score from response or compute from rubrics."""
        match = re.search(r'Overall:\s*([\d.]+)', response, re.IGNORECASE)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass

        # Fallback to average of rubrics
        return rubric_scores.overall

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning explanation from response."""
        match = re.search(r'Reasoning:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def create_trace(
        self,
        problem_text: str,
        solution_text: str,
        result: VerificationResult,
    ) -> DecisionTrace:
        """Create an explainability trace for verification."""
        weakest_name, weakest_score = result.rubric_scores.weakest_rubric

        return DecisionTrace(
            step="verification",
            decision=f"Score={result.overall_score:.2f}, Valid={result.is_valid}",
            reasoning=(
                f"USVA rubric evaluation: {result.rubric_scores.to_dict()}. "
                f"Weakest area: {weakest_name} ({weakest_score:.2f}). "
                f"{'Needs escalation' if result.needs_escalation else 'Acceptable quality'}. "
                f"Model reasoning: {result.reasoning[:100]}..."
            ),
            inputs={
                "solution_length": len(solution_text),
            },
            outputs={
                "overall_score": round(result.overall_score, 2),
                "rubrics": result.rubric_scores.to_dict(),
                "is_valid": result.is_valid,
                "needs_escalation": result.needs_escalation,
            },
            confidence=result.overall_score,
            alternatives_considered=[
                f"Escalate if score < 0.80",
                f"Refine if score < 0.60",
            ],
        )

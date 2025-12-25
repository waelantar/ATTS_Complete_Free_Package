"""
Difficulty Estimator - Section 2.3.1: Pass@k-inspired difficulty estimation.

Estimates problem difficulty using multiple samples and uncertainty measurement.
"""

import re
from typing import List

import numpy as np

from ..ports.model_caller import IModelCaller
from ..ports.config_loader import IConfigLoader
from ..domain.value_objects import DifficultyEstimate, ComputeMode, DecisionTrace


class DifficultyEstimator:
    """
    Section 2.3.1: Difficulty Estimation (Pass@k inspired)

    d(P) = 1 - Pass@k(P) / k

    We approximate by taking multiple estimates and measuring variance.
    High variance = uncertain = likely harder problem.
    """

    def __init__(self, model: IModelCaller, config: IConfigLoader):
        """
        Initialize difficulty estimator.

        Args:
            model: Language model for generating estimates
            config: Configuration loader
        """
        self._model = model
        self._config = config
        self._prompt_config = config.get_prompt("difficulty_estimation")
        self._thresholds = config.load_thresholds()

    def estimate(self, problem_text: str, k: int = None) -> DifficultyEstimate:
        """
        Estimate difficulty of a problem using Pass@k sampling.

        Args:
            problem_text: The problem statement
            k: Number of samples (default from config)

        Returns:
            DifficultyEstimate with value, uncertainty, and recommended mode
        """
        if k is None:
            k = self._thresholds.passk_k

        # Collect k estimates
        samples = []
        for _ in range(k):
            estimate = self._estimate_single(problem_text)
            samples.append(estimate)

        # Compute statistics
        avg_difficulty = int(np.mean(samples))
        uncertainty = float(np.std(samples))

        # Adjust difficulty based on uncertainty (uncertain = harder)
        adjusted_difficulty = min(10, avg_difficulty + int(uncertainty))

        # Determine recommended mode based on difficulty
        recommended_mode = self._select_mode(adjusted_difficulty)

        return DifficultyEstimate(
            value=adjusted_difficulty,
            uncertainty=uncertainty,
            samples=tuple(samples),
            recommended_mode=recommended_mode,
        )

    def _estimate_single(self, problem_text: str) -> int:
        """Get a single difficulty estimate from the model."""
        prompt = self._prompt_config.template.format(problem=problem_text)
        response = self._model.generate(
            prompt=prompt,
            max_tokens=self._prompt_config.max_tokens,
            temperature=0.7,  # Some variance for diversity
        )

        # Parse the response for a number
        try:
            numbers = re.findall(r'\d+', response.text)
            if numbers:
                return max(1, min(10, int(numbers[0])))
        except Exception:
            pass

        # Default to medium difficulty if parsing fails
        return 5

    def _select_mode(self, difficulty: int) -> ComputeMode:
        """
        Section 2.3.2: Compute Allocation Policy

        Select mode based on difficulty thresholds.
        """
        if difficulty < self._thresholds.direct_threshold:
            return ComputeMode.DIRECT
        elif difficulty < self._thresholds.thinking_threshold:
            return ComputeMode.THINKING
        return ComputeMode.DEEP

    def create_trace(
        self,
        problem_text: str,
        estimate: DifficultyEstimate,
    ) -> DecisionTrace:
        """Create an explainability trace for the difficulty estimation."""
        return DecisionTrace(
            step="difficulty_estimation",
            decision=f"Difficulty={estimate.value}, Mode={estimate.recommended_mode.value}",
            reasoning=(
                f"Sampled {len(estimate.samples)} estimates: {list(estimate.samples)}. "
                f"Mean={np.mean(estimate.samples):.1f}, StdDev={estimate.uncertainty:.2f}. "
                f"Uncertainty adjustment: +{int(estimate.uncertainty)}. "
                f"Mode selected based on thresholds: "
                f"direct<{self._thresholds.direct_threshold}, "
                f"thinking<{self._thresholds.thinking_threshold}."
            ),
            inputs={"problem_preview": problem_text[:100] + "..."},
            outputs={
                "difficulty": estimate.value,
                "uncertainty": round(estimate.uncertainty, 2),
                "mode": estimate.recommended_mode.value,
            },
            confidence=estimate.confidence,
            alternatives_considered=[
                f"direct (if difficulty < {self._thresholds.direct_threshold})",
                f"thinking (if difficulty < {self._thresholds.thinking_threshold})",
                "deep (otherwise)",
            ],
        )

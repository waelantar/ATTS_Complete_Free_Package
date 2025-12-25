"""
Test cases for ATTS domain layer.

Run with: pytest tests/test_domain.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atts.domain.entities import Problem, Solution, WorkflowResult, RefinementStep
from atts.domain.value_objects import (
    ComputeMode,
    DifficultyEstimate,
    RubricScores,
    VerificationResult,
    DecisionTrace,
)
from atts.domain.exceptions import ATTSError, ModelError, ConfigError


class TestProblem:
    """Tests for Problem entity."""

    def test_create_problem(self):
        problem = Problem(
            id="1",
            problem="What is 2+2?",
            answer="4",
            difficulty_label="easy",
        )
        assert problem.id == "1"
        assert problem.problem == "What is 2+2?"
        assert problem.answer == "4"
        assert problem.difficulty_label == "easy"

    def test_difficulty_numeric(self):
        easy = Problem(id="1", problem="", answer="", difficulty_label="easy")
        medium = Problem(id="2", problem="", answer="", difficulty_label="medium")
        hard = Problem(id="3", problem="", answer="", difficulty_label="hard")

        assert easy.difficulty_numeric == 3
        assert medium.difficulty_numeric == 5
        assert hard.difficulty_numeric == 8


class TestSolution:
    """Tests for Solution entity."""

    def test_create_solution(self):
        solution = Solution(
            text="The answer is 4",
            tokens_used=10,
            mode="direct",
        )
        assert solution.text == "The answer is 4"
        assert solution.tokens_used == 10
        assert solution.mode == "direct"

    def test_is_empty(self):
        empty = Solution(text="", tokens_used=0)
        not_empty = Solution(text="Answer", tokens_used=5)

        assert empty.is_empty
        assert not not_empty.is_empty


class TestComputeMode:
    """Tests for ComputeMode value object."""

    def test_mode_properties(self):
        direct = ComputeMode.DIRECT
        thinking = ComputeMode.THINKING
        deep = ComputeMode.DEEP

        assert direct.can_escalate()
        assert thinking.can_escalate()
        assert not deep.can_escalate()

        assert direct.next_mode() == ComputeMode.THINKING
        assert thinking.next_mode() == ComputeMode.DEEP
        assert deep.next_mode() is None

    def test_expected_tokens(self):
        assert ComputeMode.DIRECT.expected_tokens == 150
        assert ComputeMode.THINKING.expected_tokens == 500
        assert ComputeMode.DEEP.expected_tokens == 1000


class TestRubricScores:
    """Tests for RubricScores value object."""

    def test_create_rubric_scores(self):
        scores = RubricScores(lc=0.8, fc=0.9, cm=0.7, ga=0.85)

        assert scores.lc == 0.8
        assert scores.fc == 0.9
        assert scores.cm == 0.7
        assert scores.ga == 0.85

    def test_overall_score(self):
        scores = RubricScores(lc=0.8, fc=0.8, cm=0.8, ga=0.8)
        assert scores.overall == 0.8

    def test_weakest_rubric(self):
        scores = RubricScores(lc=0.9, fc=0.6, cm=0.8, ga=0.85)
        name, value = scores.weakest_rubric

        assert name == "FC"
        assert value == 0.6

    def test_validation(self):
        with pytest.raises(ValueError):
            RubricScores(lc=1.5, fc=0.5, cm=0.5, ga=0.5)

        with pytest.raises(ValueError):
            RubricScores(lc=-0.1, fc=0.5, cm=0.5, ga=0.5)


class TestDifficultyEstimate:
    """Tests for DifficultyEstimate value object."""

    def test_create_estimate(self):
        estimate = DifficultyEstimate(
            value=5,
            uncertainty=1.2,
            samples=(4, 5, 6),
            recommended_mode=ComputeMode.THINKING,
        )

        assert estimate.value == 5
        assert estimate.uncertainty == 1.2
        assert estimate.samples == (4, 5, 6)
        assert estimate.recommended_mode == ComputeMode.THINKING

    def test_confidence(self):
        low_uncertainty = DifficultyEstimate(
            value=5, uncertainty=0.5, samples=(5,), recommended_mode=ComputeMode.THINKING
        )
        high_uncertainty = DifficultyEstimate(
            value=5, uncertainty=3.0, samples=(3, 5, 7), recommended_mode=ComputeMode.THINKING
        )

        assert low_uncertainty.confidence > high_uncertainty.confidence


class TestExceptions:
    """Tests for custom exceptions."""

    def test_atts_error(self):
        error = ATTSError("Test error", {"key": "value"})
        assert "Test error" in str(error)
        assert error.details == {"key": "value"}

    def test_model_error(self):
        error = ModelError("Connection failed", {"host": "localhost"})
        assert "Connection failed" in str(error)
        assert error.details["host"] == "localhost"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Analysis Reporter - Comprehensive analysis and reporting of ATTS experiments.

Implements Section 3: Theoretical Analysis including Pareto frontier computation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..domain.entities import WorkflowResult
from .workflow_visualizer import WorkflowVisualizer


@dataclass
class ExperimentAnalysis:
    """Complete analysis of an ATTS experiment."""
    # Accuracy metrics
    atts_accuracy: float
    baseline_accuracy: float
    accuracy_diff: float

    # Token metrics
    atts_avg_tokens: float
    baseline_avg_tokens: float
    token_savings: float

    # Mode distribution
    mode_distribution: Dict[str, int]
    escalation_rate: float

    # Verification metrics
    avg_verification_score: float
    avg_rubric_scores: Dict[str, float]

    # Refinement metrics
    avg_refinement_iterations: float

    # Difficulty estimation
    difficulty_mae: float
    avg_uncertainty: float

    # Pareto analysis (Section 3.2)
    atts_efficiency: float
    baseline_efficiency: float
    is_pareto_improvement: bool
    efficiency_gain: float


class AnalysisReporter:
    """
    Section 3: Theoretical Analysis

    Computes:
    - Accuracy-efficiency trade-offs
    - Pareto frontier analysis
    - Per-difficulty breakdowns
    - Hypothesis validation
    """

    def __init__(self, visualizer: Optional[WorkflowVisualizer] = None):
        """
        Initialize analysis reporter.

        Args:
            visualizer: Optional visualizer for output
        """
        self._visualizer = visualizer or WorkflowVisualizer()

    def analyze(
        self,
        atts_results: List[WorkflowResult],
        baseline_results: List[WorkflowResult],
    ) -> ExperimentAnalysis:
        """
        Perform comprehensive analysis of experiment results.

        Args:
            atts_results: Results from ATTS workflow
            baseline_results: Results from baseline (always deep) workflow

        Returns:
            ExperimentAnalysis with all metrics
        """
        # Convert to dicts for easier analysis
        atts_dicts = [r.to_dict() for r in atts_results]
        baseline_dicts = [r.to_dict() for r in baseline_results]

        # Accuracy
        atts_accuracy = np.mean([r["correct"] for r in atts_dicts])
        baseline_accuracy = np.mean([r["correct"] for r in baseline_dicts])

        # Tokens
        atts_tokens = np.mean([r["tokens"] for r in atts_dicts])
        baseline_tokens = np.mean([r["tokens"] for r in baseline_dicts])
        token_savings = (baseline_tokens - atts_tokens) / baseline_tokens if baseline_tokens > 0 else 0

        # Mode distribution
        mode_dist = {}
        for r in atts_dicts:
            mode = r["final_mode"]
            mode_dist[mode] = mode_dist.get(mode, 0) + 1

        # Escalation rate
        escalation_rate = np.mean([r["escalated"] for r in atts_dicts])

        # Verification
        avg_verification = np.mean([r["verification_score"] for r in atts_dicts])

        # Rubric averages
        rubric_scores = {"LC": [], "FC": [], "CM": [], "GA": []}
        for r in atts_dicts:
            for k in rubric_scores:
                rubric_scores[k].append(r["rubric_scores"].get(k, 0.5))
        avg_rubrics = {k: np.mean(v) for k, v in rubric_scores.items()}

        # Refinement
        avg_refinement = np.mean([len(r["refinement_history"]) for r in atts_dicts])

        # Difficulty estimation
        difficulty_errors = []
        uncertainties = []
        for r in atts_dicts:
            true_numeric = {"easy": 3, "medium": 5, "hard": 8}.get(r["true_difficulty"], 5)
            difficulty_errors.append(abs(r["predicted_difficulty"] - true_numeric))
            uncertainties.append(r["difficulty_uncertainty"])

        difficulty_mae = np.mean(difficulty_errors)
        avg_uncertainty = np.mean(uncertainties)

        # Pareto analysis (Section 3.2)
        atts_efficiency = atts_accuracy / atts_tokens if atts_tokens > 0 else 0
        baseline_efficiency = baseline_accuracy / baseline_tokens if baseline_tokens > 0 else 0
        efficiency_gain = (
            (atts_efficiency - baseline_efficiency) / baseline_efficiency
            if baseline_efficiency > 0 else 0
        )

        # Pareto improvement: saves tokens AND maintains accuracy
        is_pareto = bool(token_savings > 0.2 and (baseline_accuracy - atts_accuracy) < 0.05)

        return ExperimentAnalysis(
            atts_accuracy=atts_accuracy,
            baseline_accuracy=baseline_accuracy,
            accuracy_diff=atts_accuracy - baseline_accuracy,
            atts_avg_tokens=atts_tokens,
            baseline_avg_tokens=baseline_tokens,
            token_savings=token_savings,
            mode_distribution=mode_dist,
            escalation_rate=escalation_rate,
            avg_verification_score=avg_verification,
            avg_rubric_scores=avg_rubrics,
            avg_refinement_iterations=avg_refinement,
            difficulty_mae=difficulty_mae,
            avg_uncertainty=avg_uncertainty,
            atts_efficiency=atts_efficiency,
            baseline_efficiency=baseline_efficiency,
            is_pareto_improvement=is_pareto,
            efficiency_gain=efficiency_gain,
        )

    def print_report(
        self,
        analysis: ExperimentAnalysis,
        atts_results: List[WorkflowResult],
        baseline_results: List[WorkflowResult],
    ):
        """Print a comprehensive analysis report."""
        v = self._visualizer

        v.print_header("COMPREHENSIVE RESULTS")

        v.print_status(
            f"Baseline: {analysis.baseline_accuracy * 100:.1f}% accuracy, "
            f"{analysis.baseline_avg_tokens:.0f} avg tokens"
        )
        v.print_status(
            f"ATTS:     {analysis.atts_accuracy * 100:.1f}% accuracy, "
            f"{analysis.atts_avg_tokens:.0f} avg tokens"
        )
        v.print_status(f"\nToken Savings: {analysis.token_savings * 100:.1f}%", style="bold green")

        # Mode distribution
        v.print_status(f"\nMode Distribution: {analysis.mode_distribution}")
        v.print_status(f"Escalation Rate: {analysis.escalation_rate * 100:.1f}%")
        v.print_status(f"Avg Refinement Iterations: {analysis.avg_refinement_iterations:.2f}")

        # Difficulty estimation
        v.print_status(f"\nDifficulty Estimation MAE: {analysis.difficulty_mae:.2f}")
        v.print_status(f"Avg Difficulty Uncertainty: {analysis.avg_uncertainty:.2f}")

        # USVA rubrics
        v.print_status("\nUSVA Rubric Scores:")
        for rubric, score in analysis.avg_rubric_scores.items():
            v.print_status(f"   {rubric}: {score:.2f}")

        # Per-difficulty breakdown
        v.print_status("\nPerformance by Difficulty:")
        self._print_difficulty_breakdown(atts_results, baseline_results)

        # Pareto analysis
        v.print_header("PARETO FRONTIER ANALYSIS (Section 3.2)")
        v.print_status(f"ATTS Efficiency Ratio:     {analysis.atts_efficiency:.6f}")
        v.print_status(f"Baseline Efficiency Ratio: {analysis.baseline_efficiency:.6f}")
        v.print_status(f"Efficiency Gain:           {analysis.efficiency_gain * 100:+.1f}%")
        v.print_status(f"Token Savings:             {analysis.token_savings * 100:.1f}%")
        v.print_status(f"Accuracy Cost:             {-analysis.accuracy_diff * 100:.1f}%")

        if analysis.is_pareto_improvement:
            v.print_status("\nPareto Improvement: YES", style="bold green")
        else:
            v.print_status("\nPareto Improvement: NO", style="yellow")

        # Hypothesis validation
        v.print_header("HYPOTHESIS VALIDATION")
        if analysis.token_savings > 0.20 and analysis.accuracy_diff > -0.05:
            v.print_status("HYPOTHESIS SUPPORTED!", style="bold green")
            v.print_status("  - Token savings > 20%")
            v.print_status("  - Accuracy within 5% of baseline")
            if analysis.is_pareto_improvement:
                v.print_status("  - Pareto improvement achieved")
        else:
            v.print_status("Mixed results - hypothesis partially supported", style="yellow")

    def _print_difficulty_breakdown(
        self,
        atts_results: List[WorkflowResult],
        baseline_results: List[WorkflowResult],
    ):
        """Print per-difficulty analysis."""
        v = self._visualizer

        for diff in ["easy", "medium", "hard"]:
            atts_subset = [r for r in atts_results if r.true_difficulty == diff]
            baseline_subset = [r for r in baseline_results if r.true_difficulty == diff]

            if not atts_subset:
                continue

            atts_acc = np.mean([r.correct for r in atts_subset]) * 100
            baseline_acc = np.mean([r.correct for r in baseline_subset]) * 100
            atts_tokens = np.mean([r.tokens_used for r in atts_subset])
            baseline_tokens = np.mean([r.tokens_used for r in baseline_subset])
            savings = (1 - atts_tokens / baseline_tokens) * 100 if baseline_tokens > 0 else 0

            v.print_status(
                f"  {diff.capitalize()}: ATTS={atts_acc:.0f}% / Baseline={baseline_acc:.0f}% | "
                f"Tokens: {atts_tokens:.0f} vs {baseline_tokens:.0f} ({savings:+.1f}%)"
            )

    def export_to_dict(self, analysis: ExperimentAnalysis) -> Dict[str, Any]:
        """Export analysis to a dictionary for JSON serialization."""
        return {
            "accuracy": {
                "atts": analysis.atts_accuracy,
                "baseline": analysis.baseline_accuracy,
                "difference": analysis.accuracy_diff,
            },
            "tokens": {
                "atts_avg": analysis.atts_avg_tokens,
                "baseline_avg": analysis.baseline_avg_tokens,
                "savings": analysis.token_savings,
            },
            "mode_distribution": analysis.mode_distribution,
            "escalation_rate": analysis.escalation_rate,
            "verification": {
                "avg_score": analysis.avg_verification_score,
                "rubrics": analysis.avg_rubric_scores,
            },
            "refinement": {
                "avg_iterations": analysis.avg_refinement_iterations,
            },
            "difficulty_estimation": {
                "mae": analysis.difficulty_mae,
                "avg_uncertainty": analysis.avg_uncertainty,
            },
            "pareto": {
                "atts_efficiency": analysis.atts_efficiency,
                "baseline_efficiency": analysis.baseline_efficiency,
                "is_improvement": analysis.is_pareto_improvement,
                "efficiency_gain": analysis.efficiency_gain,
            },
        }

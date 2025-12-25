"""
ATTS Workflow - Main Orchestration

This module implements the complete ATTS pipeline, orchestrating all stages
of the adaptive test-time scaling process.

Algorithm Overview:
    Given a problem P and model M, ATTS executes:

    1. ESTIMATE_DIFFICULTY(P, k):
       - Sample k difficulty estimates from M
       - Compute mean difficulty d and variance σ²
       - Adjust: d_adj = min(10, d + σ) [uncertainty increases difficulty]

    2. SELECT_MODE(d_adj):
       - If d_adj < τ_direct: mode = DIRECT (minimal compute)
       - If d_adj < τ_thinking: mode = THINKING (moderate compute)
       - Else: mode = DEEP (maximum compute)

    3. GENERATE_SOLUTION(P, mode):
       - Use mode-specific prompt template
       - Token budget scales with mode complexity

    4. VERIFY_SOLUTION(P, solution):
       - Apply USVA rubrics: LC, FC, CM, GA ∈ [0,1]
       - Compute overall score v = mean(LC, FC, CM, GA)

    5. ESCALATE_IF_NEEDED(v, mode):
       - If v < τ_escalation and mode.can_escalate():
         - mode = mode.next_mode()
         - Re-execute stages 3-4

    6. REFINE_IF_DEEP(mode, solution):
       - If mode = DEEP and refinement enabled:
         - For i in 1..max_iterations:
           - critique = GENERATE_CRITIQUE(P, solution)
           - valid = META_VERIFY(P, solution, critique)
           - If not valid or "no issues": break
           - solution = REFINE(P, solution, critique)

Theoretical Foundation:
    The algorithm implements compute-optimal scaling where compute budget
    is allocated proportionally to problem difficulty. This achieves
    Pareto efficiency: reducing compute on easy problems while maintaining
    quality on hard problems.

Key Equations:
    - Difficulty: d(P) ≈ 1 - Pass@k(P)/k (approximated via variance)
    - Mode selection: argmin_m {cost(m) | quality(m,P) ≥ τ}
    - Pareto improvement: Δtokens > 20% AND Δaccuracy < 5%

Experimental Metrics:
    - Accuracy: % problems answered correctly
    - Token Efficiency: (baseline_tokens - atts_tokens) / baseline_tokens
    - Escalation Rate: % problems that required mode escalation
    - Refinement Effectiveness: accuracy gain from refinement iterations
"""

import re
import time
from typing import List, Optional

from ..ports.model_caller import IModelCaller
from ..ports.config_loader import IConfigLoader
from ..domain.entities import Problem, Solution, WorkflowResult, RefinementStep
from ..domain.value_objects import ComputeMode, DecisionTrace
from .estimate_difficulty import DifficultyEstimator
from .solve_problem import ProblemSolver
from .verify_solution import SolutionVerifier
from .dialectical_refinement import DialecticalRefiner


class ATTSWorkflow:
    """
    Main ATTS workflow orchestrator.

    This class implements the complete adaptive test-time scaling pipeline,
    coordinating difficulty estimation, mode selection, solution generation,
    verification, escalation, and refinement.

    Attributes:
        _model: Language model adapter for generation
        _config: Configuration loader for thresholds and prompts
        _enable_escalation: Whether to allow mode escalation on low scores
        _enable_refinement: Whether to run dialectical refinement in deep mode

    Example:
        >>> workflow = ATTSWorkflow(model, config, enable_refinement=True)
        >>> result = workflow.execute(problem)
        >>> print(f"Accuracy: {result.correct}, Tokens: {result.tokens_used}")
    """

    def __init__(
        self,
        model: IModelCaller,
        config: IConfigLoader,
        enable_escalation: bool = True,
        enable_refinement: bool = False,
    ):
        self._model = model
        self._config = config
        self._thresholds = config.load_thresholds()
        self._enable_escalation = enable_escalation
        self._enable_refinement = enable_refinement

        # Initialize component use cases
        self._difficulty_estimator = DifficultyEstimator(model, config)
        self._solver = ProblemSolver(model, config)
        self._verifier = SolutionVerifier(model, config)
        self._refiner = DialecticalRefiner(model, config, self._verifier)

    def execute(self, problem: Problem, passk_k: int = None) -> WorkflowResult:
        """
        Execute the complete ATTS workflow for a single problem.

        This method implements the full 6-stage pipeline:
        1. Difficulty estimation with uncertainty quantification
        2. Adaptive mode selection based on difficulty
        3. Solution generation with mode-appropriate prompting
        4. USVA verification with rubric scoring
        5. Conditional escalation if quality threshold not met
        6. Dialectical refinement for deep mode (if enabled)

        Args:
            problem: The math problem to solve
            passk_k: Number of samples for Pass@k estimation (default from config)

        Returns:
            WorkflowResult containing solution, metrics, and decision trace

        Note:
            The decision_trace field provides full explainability for each
            stage, useful for analyzing model behavior and debugging.
        """
        start_time = time.time()
        total_tokens = 0
        decision_trace: List[DecisionTrace] = []

        # Stage 1: Difficulty Estimation
        # Uses Pass@k-inspired sampling to estimate difficulty with uncertainty
        difficulty_estimate = self._difficulty_estimator.estimate(
            problem.problem,
            k=passk_k or self._thresholds.passk_k,
        )
        decision_trace.append(
            self._difficulty_estimator.create_trace(problem.problem, difficulty_estimate)
        )

        # Stage 2: Mode Selection
        # Thresholds: direct < τ_d, thinking < τ_t, else deep
        initial_mode = difficulty_estimate.recommended_mode
        current_mode = initial_mode
        escalation_path = [current_mode.value]

        # Stage 3: Solution Generation
        # Token budget scales with mode: direct=150, thinking=500, deep=1000
        solution = self._solver.solve(problem.problem, current_mode)
        total_tokens += solution.tokens_used
        decision_trace.append(
            self._solver.create_trace(problem.problem, current_mode, solution)
        )

        # Stage 4: USVA Verification
        # Computes rubric scores: LC, FC, CM, GA and overall average
        verification_result = self._verifier.verify(problem.problem, solution.text)
        decision_trace.append(
            self._verifier.create_trace(problem.problem, solution.text, verification_result)
        )

        # Stage 5: Conditional Escalation
        # Escalate if score < threshold and higher mode available
        escalated = False
        if self._enable_escalation and self._should_escalate(verification_result.overall_score):
            next_mode = current_mode.next_mode()
            if next_mode:
                escalated = True
                current_mode = next_mode
                escalation_path.append(current_mode.value)

                # Re-solve with escalated mode
                solution = self._solver.solve(problem.problem, current_mode)
                total_tokens += solution.tokens_used

                # Re-verify after escalation
                verification_result = self._verifier.verify(problem.problem, solution.text)

                decision_trace.append(DecisionTrace(
                    step="escalation",
                    decision=f"Escalated from {initial_mode.value} to {current_mode.value}",
                    reasoning=(
                        f"Verification score {verification_result.overall_score:.2f} < "
                        f"threshold {self._thresholds.escalation_threshold}. "
                        f"Escalated to higher compute mode for quality improvement."
                    ),
                    inputs={"previous_score": verification_result.overall_score},
                    outputs={"new_mode": current_mode.value},
                    confidence=0.8,
                ))

        # Stage 6: Dialectical Refinement (deep mode only)
        # Implements critic-refiner loop with meta-verification
        refinement_history: List[RefinementStep] = []
        if self._enable_refinement and current_mode == ComputeMode.DEEP:
            solution_text, refine_tokens, refinement_history = self._refiner.refine(
                problem.problem,
                solution.text,
            )
            total_tokens += refine_tokens

            solution = Solution(
                text=solution_text,
                tokens_used=solution.tokens_used + refine_tokens,
                mode=current_mode.value,
            )

            # Final verification after refinement
            verification_result = self._verifier.verify(problem.problem, solution.text)

            decision_trace.append(
                self._refiner.create_trace(problem.problem, refinement_history)
            )

        # Evaluate correctness against ground truth
        correct = self._check_answer(solution.text, problem.answer)

        return WorkflowResult(
            problem_id=problem.id,
            true_difficulty=problem.difficulty_label,
            predicted_difficulty=difficulty_estimate.value,
            difficulty_uncertainty=difficulty_estimate.uncertainty,
            initial_mode=initial_mode.value,
            final_mode=current_mode.value,
            escalated=escalated,
            escalation_path=escalation_path,
            verification_score=verification_result.overall_score,
            rubric_scores=verification_result.rubric_scores.to_dict(),
            refinement_history=refinement_history,
            solution=solution.text,
            tokens_used=total_tokens,
            correct=correct,
            total_time=time.time() - start_time,
            decision_trace=[dt.to_dict() for dt in decision_trace],
        )

    def _should_escalate(self, verification_score: float) -> bool:
        """
        Determine if mode escalation is warranted.

        Escalation triggers:
        1. Score < ascot_trigger (0.60): Very low quality, immediate escalation
        2. Score < escalation_threshold (0.80): Below acceptable quality
        """
        return (
            verification_score < self._thresholds.ascot_trigger or
            verification_score < self._thresholds.escalation_threshold
        )

    def _check_answer(self, response: str, correct_answer: str) -> bool:
        """
        Check if model response contains the correct answer.

        Uses two matching strategies:
        1. String containment (case-insensitive, whitespace-normalized)
        2. Numeric matching with tolerance (|predicted - actual| < 0.01)
        """
        response_lower = response.lower().replace(" ", "")
        answer_lower = correct_answer.lower().replace(" ", "")

        # Direct string match
        if answer_lower in response_lower:
            return True

        # Numeric match with tolerance
        try:
            response_nums = re.findall(r'-?\d+\.?\d*', response)
            answer_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
            if answer_nums:
                target = float(answer_nums[0])
                for num in response_nums:
                    if abs(float(num) - target) < 0.01:
                        return True
        except (ValueError, IndexError):
            pass

        return False


class BaselineWorkflow:
    """
    Baseline workflow for comparison (always uses deep mode).

    This serves as the control condition in experiments. By always using
    the maximum compute mode, it establishes the upper bound on accuracy
    and the baseline token usage that ATTS aims to improve upon.

    Experimental Hypothesis:
        ATTS achieves >20% token savings compared to baseline while
        maintaining accuracy within 5% of baseline performance.
    """

    def __init__(
        self,
        model: IModelCaller,
        config: IConfigLoader,
        enable_refinement: bool = False,
    ):
        self._model = model
        self._config = config
        self._enable_refinement = enable_refinement
        self._solver = ProblemSolver(model, config)
        self._verifier = SolutionVerifier(model, config)
        self._refiner = DialecticalRefiner(model, config, self._verifier)
        self._thresholds = config.load_thresholds()

    def execute(self, problem: Problem) -> WorkflowResult:
        """Execute baseline workflow (always deep mode, no adaptation)."""
        start_time = time.time()
        total_tokens = 0

        # Always use maximum compute mode
        mode = ComputeMode.DEEP
        solution = self._solver.solve(problem.problem, mode)
        total_tokens += solution.tokens_used

        # Optional refinement for fair comparison
        refinement_history: List[RefinementStep] = []
        if self._enable_refinement:
            solution_text, refine_tokens, refinement_history = self._refiner.refine(
                problem.problem,
                solution.text,
            )
            total_tokens += refine_tokens
            solution = Solution(
                text=solution_text,
                tokens_used=total_tokens,
                mode=mode.value,
            )

        correct = self._check_answer(solution.text, problem.answer)

        return WorkflowResult(
            problem_id=problem.id,
            true_difficulty=problem.difficulty_label,
            predicted_difficulty=8,  # Assumed hard (baseline doesn't estimate)
            difficulty_uncertainty=0.0,
            initial_mode=mode.value,
            final_mode=mode.value,
            escalated=False,
            escalation_path=[mode.value],
            verification_score=0.0,
            rubric_scores={},
            refinement_history=refinement_history,
            solution=solution.text,
            tokens_used=total_tokens,
            correct=correct,
            total_time=time.time() - start_time,
            decision_trace=[],
        )

    def _check_answer(self, response: str, correct_answer: str) -> bool:
        """Check if model response contains the correct answer."""
        response_lower = response.lower().replace(" ", "")
        answer_lower = correct_answer.lower().replace(" ", "")

        if answer_lower in response_lower:
            return True

        try:
            response_nums = re.findall(r'-?\d+\.?\d*', response)
            answer_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
            if answer_nums:
                target = float(answer_nums[0])
                for num in response_nums:
                    if abs(float(num) - target) < 0.01:
                        return True
        except (ValueError, IndexError):
            pass

        return False

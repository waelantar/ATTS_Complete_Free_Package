"""
Dialectical Refinement - Implements critic-refiner loop.

Section 1.2: The Dialectical Nature of Advanced Reasoning
Section 2.4: Distilled Verification Knowledge
"""

from typing import List, Tuple

from ..ports.model_caller import IModelCaller
from ..ports.config_loader import IConfigLoader
from ..domain.entities import RefinementStep
from ..domain.value_objects import DecisionTrace


class DialecticalRefiner:
    """
    Section 1.2: The Dialectical Nature of Advanced Reasoning
    Section 2.4: Distilled Verification Knowledge

    Implements the full dialectical loop:
    1. Generator/Proposer: Create solution
    2. Verifier/Critic: Identify issues
    3. Meta-Verifier: Validate critique
    4. Refiner/Synthesizer: Improve solution
    """

    def __init__(
        self,
        model: IModelCaller,
        config: IConfigLoader,
        verifier,  # SolutionVerifier - avoid circular import
    ):
        """
        Initialize dialectical refiner.

        Args:
            model: Language model for critique/refinement
            config: Configuration loader
            verifier: SolutionVerifier for meta-verification
        """
        self._model = model
        self._config = config
        self._verifier = verifier
        self._critique_prompt = config.get_prompt("dialectical.critique")
        self._refine_prompt = config.get_prompt("dialectical.refine")
        self._thresholds = config.load_thresholds()

    def refine(
        self,
        problem_text: str,
        initial_solution: str,
        max_iterations: int = None,
    ) -> Tuple[str, int, List[RefinementStep]]:
        """
        Perform dialectical refinement on a solution.

        Args:
            problem_text: The original problem
            initial_solution: The solution to refine
            max_iterations: Maximum refinement cycles

        Returns:
            Tuple of (final_solution, total_tokens, refinement_history)
        """
        if max_iterations is None:
            max_iterations = self._thresholds.max_refinement_iterations

        solution = initial_solution
        total_tokens = 0
        history: List[RefinementStep] = []

        for iteration in range(max_iterations):
            # Stage 1: Critic - Identify issues
            critique, critique_tokens = self._generate_critique(problem_text, solution)
            total_tokens += critique_tokens

            # Stage 2: Meta-Verification - Is critique valid?
            critique_valid, meta_reason = self._verifier.meta_verify_critique(
                problem_text, solution, critique
            )

            # Stage 3: Check if we should stop
            should_stop, stop_reason = self._should_stop_refinement(
                critique, critique_valid
            )

            if should_stop:
                history.append(RefinementStep(
                    iteration=iteration,
                    critique=critique,
                    critique_valid=critique_valid,
                    action="stopped",
                    reason=stop_reason,
                    tokens_used=critique_tokens,
                ))
                break

            # Stage 4: Refiner - Synthesize improved solution
            improved_solution, refine_tokens = self._generate_refinement(
                problem_text, solution, critique
            )
            total_tokens += refine_tokens

            history.append(RefinementStep(
                iteration=iteration,
                critique=critique,
                critique_valid=critique_valid,
                action="refined",
                reason="Valid critique addressed",
                improved_solution=improved_solution,
                tokens_used=critique_tokens + refine_tokens,
            ))

            solution = improved_solution

        return solution, total_tokens, history

    def _generate_critique(self, problem_text: str, solution: str) -> Tuple[str, int]:
        """Generate a critique of the solution."""
        prompt = self._critique_prompt.template.format(
            problem=problem_text,
            solution=solution,
        )

        response = self._model.generate(
            prompt=prompt,
            max_tokens=self._critique_prompt.max_tokens,
            temperature=0.5,
        )

        return response.text, response.tokens_used

    def _generate_refinement(
        self,
        problem_text: str,
        solution: str,
        critique: str,
    ) -> Tuple[str, int]:
        """Generate an improved solution based on critique."""
        prompt = self._refine_prompt.template.format(
            problem=problem_text,
            solution=solution,
            critique=critique,
        )

        response = self._model.generate(
            prompt=prompt,
            max_tokens=self._refine_prompt.max_tokens,
            temperature=0.7,
        )

        return response.text, response.tokens_used

    def _should_stop_refinement(
        self,
        critique: str,
        critique_valid: bool,
    ) -> Tuple[bool, str]:
        """Determine if refinement should stop."""
        # Stop if critique is invalid (hallucinated)
        if not critique_valid:
            return True, "Critique was hallucinated/invalid"

        # Stop if no issues found
        if "no issues" in critique.lower():
            return True, "No issues identified"

        # Stop if critique is too short (nothing substantial)
        if len(critique.strip()) < 20:
            return True, "Critique too brief to be meaningful"

        return False, ""

    def create_trace(
        self,
        problem_text: str,
        history: List[RefinementStep],
    ) -> DecisionTrace:
        """Create an explainability trace for refinement."""
        if not history:
            return DecisionTrace(
                step="refinement",
                decision="No refinement performed",
                reasoning="Refinement was not triggered (not in deep mode or disabled)",
                confidence=1.0,
            )

        total_iterations = len(history)
        refined_count = sum(1 for h in history if h.action == "refined")
        stopped_reasons = [h.reason for h in history if h.action == "stopped"]

        return DecisionTrace(
            step="refinement",
            decision=f"{refined_count} refinements in {total_iterations} iterations",
            reasoning=(
                f"Dialectical loop executed {total_iterations} times. "
                f"Refined solution {refined_count} times. "
                f"Stop reasons: {stopped_reasons}. "
                f"Meta-verification filtered {sum(1 for h in history if not h.critique_valid)} hallucinated critiques."
            ),
            inputs={
                "max_iterations": self._thresholds.max_refinement_iterations,
            },
            outputs={
                "total_iterations": total_iterations,
                "refinements_applied": refined_count,
                "final_action": history[-1].action if history else "none",
            },
            confidence=0.8 if refined_count > 0 else 1.0,
            alternatives_considered=[
                "Could continue refining if more issues found",
                "Early exit if verification score > 0.85",
            ],
        )

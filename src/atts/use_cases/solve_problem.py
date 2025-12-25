"""
Problem Solver - Generate solutions using appropriate compute mode.

Implements Section 2.3.2: Compute Allocation Policy.
"""

from ..ports.model_caller import IModelCaller
from ..ports.config_loader import IConfigLoader
from ..domain.entities import Solution
from ..domain.value_objects import ComputeMode, DecisionTrace


class ProblemSolver:
    """
    Generate solutions using the appropriate compute mode.

    Section 2.3.2: Compute Allocation Policy

    Modes:
    - DIRECT: Fast, single-step solutions for easy problems
    - THINKING: Multi-step reasoning for medium problems
    - DEEP: Full verification chain for hard problems
    """

    def __init__(self, model: IModelCaller, config: IConfigLoader):
        """
        Initialize problem solver.

        Args:
            model: Language model for generating solutions
            config: Configuration loader
        """
        self._model = model
        self._config = config

    def solve(self, problem_text: str, mode: ComputeMode) -> Solution:
        """
        Generate a solution for the problem using the specified mode.

        Args:
            problem_text: The problem statement
            mode: Compute mode to use

        Returns:
            Solution with generated text and metadata
        """
        # Get the appropriate prompt for the mode
        prompt_config = self._config.get_prompt(mode.value)

        # Format the prompt with the problem
        prompt = prompt_config.template.format(problem=problem_text)

        # Generate the solution
        response = self._model.generate(
            prompt=prompt,
            max_tokens=prompt_config.max_tokens,
            temperature=0.7,
        )

        return Solution(
            text=response.text,
            tokens_used=response.tokens_used,
            mode=mode.value,
            generation_time=response.generation_time,
        )

    def create_trace(
        self,
        problem_text: str,
        mode: ComputeMode,
        solution: Solution,
    ) -> DecisionTrace:
        """Create an explainability trace for solution generation."""
        return DecisionTrace(
            step="solution_generation",
            decision=f"Generated solution using {mode.value} mode",
            reasoning=(
                f"Mode '{mode.value}' selected based on difficulty. "
                f"{mode.description}. "
                f"Used {solution.tokens_used} tokens in {solution.generation_time:.2f}s."
            ),
            inputs={
                "mode": mode.value,
                "expected_tokens": mode.expected_tokens,
            },
            outputs={
                "tokens_used": solution.tokens_used,
                "generation_time": round(solution.generation_time, 2),
                "solution_length": len(solution.text),
            },
            confidence=1.0,  # Generation itself is deterministic
            alternatives_considered=[
                f"Could have used {m.value} mode ({m.expected_tokens} tokens)"
                for m in ComputeMode
                if m != mode
            ],
        )

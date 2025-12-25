"""
ATTS: Adaptive Test-Time Scaling

A framework for optimizing inference compute allocation in language models.
This implementation validates the theoretical framework through empirical experiments.

Key Contributions:
    1. Difficulty-adaptive compute allocation reduces token usage by 20%+
       while maintaining accuracy within 5% of baseline
    2. USVA (Unified Self-Verification Architecture) provides interpretable
       quality signals through multi-dimensional rubric scoring
    3. Dialectical refinement with meta-verification filters hallucinated
       critiques, improving refinement effectiveness

Architecture:
    - domain/: Core entities and value objects (Problem, Solution, RubricScores)
    - ports/: Abstract interfaces for dependency inversion
    - adapters/: Concrete implementations (Ollama, YAML config, JSON storage)
    - use_cases/: ATTS workflow logic (difficulty estimation, verification, refinement)
    - explainability/: XAI features for transparency and analysis
    - interfaces/: CLI entry point

Methodology:
    Stage 1: Difficulty Estimation via Pass@k sampling (variance = uncertainty)
    Stage 2: Mode Selection based on difficulty thresholds
    Stage 3: Solution Generation with mode-appropriate prompts
    Stage 4: USVA Verification with 4-dimensional rubric (LC, FC, CM, GA)
    Stage 5: Uncertainty-Triggered Escalation if verification score < threshold
    Stage 6: Dialectical Refinement with meta-verification (deep mode only)

Experimental Setup:
    - Model: Ollama-served LLMs (e.g., qwen2.5:3b-instruct)
    - Dataset: MATH dataset problems (various difficulty levels)
    - Metrics: Accuracy, Token Usage, Pareto Efficiency, Mode Distribution
"""

__version__ = "2.0.0"
__author__ = "ATTS Contributors"

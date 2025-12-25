#!/usr/bin/env python3
"""
ATTS Experiment Runner - Clean Architecture Version

This is the main entry point for running ATTS experiments.
It's a drop-in replacement for the old atts_experiment_local.py.

Usage:
    python run_atts.py --model qwen2.5:3b-instruct --max-problems 25
    python run_atts.py --quick-test --verbose
    python run_atts.py --enable-refinement --max-problems 100

Paper Sections Validated:
    Section 1.2: Dialectical Nature of Advanced Reasoning
    Section 2.1: Unified Self-Verification Architecture (USVA)
    Section 2.1.2: Generalized Verification Rubrics
    Section 2.1.3: Integrated Meta-Verification
    Section 2.3: Adaptive Test-Time Scaling
    Section 2.3.1: Difficulty Estimation (Pass@k)
    Section 2.3.2: Compute Allocation Policy
    Section 2.3.3: Uncertainty-Triggered Escalation
    Section 2.4: Distilled Verification
    Section 3: Theoretical Analysis (Pareto frontier)
    Section 4.1: Simulation Protocol
    Appendix A: Full ATTS Workflow
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from atts.interfaces.cli import main

if __name__ == "__main__":
    main()

"""
ATTS Command-Line Interface

Usage:
    python -m atts.interfaces.cli --model qwen2.5:3b-instruct --max-problems 25

Or via entry point:
    atts --model qwen2.5:3b-instruct --max-problems 25
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from ..adapters.ollama_adapter import OllamaAdapter
from ..adapters.yaml_config_loader import YamlConfigLoader
from ..adapters.json_repository import JsonResultRepository, JsonProblemRepository
from ..use_cases.atts_workflow import ATTSWorkflow, BaselineWorkflow
from ..domain.entities import Problem, WorkflowResult
from ..domain.exceptions import ATTSError, ModelError, ConfigError
from ..explainability.workflow_visualizer import WorkflowVisualizer
from ..explainability.decision_explainer import DecisionExplainer
from ..explainability.analysis_reporter import AnalysisReporter

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="ATTS - Adaptive Test-Time Scaling Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with 5 problems
    python -m atts.interfaces.cli --quick-test

    # Run 25 problems with refinement enabled
    python -m atts.interfaces.cli --max-problems 25 --enable-refinement

    # Compare with baseline
    python -m atts.interfaces.cli --max-problems 50

Paper Sections Validated:
    Section 1.2: Dialectical Nature of Advanced Reasoning
    Section 2.1: Unified Self-Verification Architecture (USVA)
    Section 2.3: Adaptive Test-Time Scaling
    Section 3: Theoretical Analysis (Pareto frontier)
    Appendix A: Full ATTS Workflow
        """,
    )

    # Model settings
    parser.add_argument(
        "--model", "-m",
        default="qwen2.5:3b-instruct",
        help="Ollama model name (default: qwen2.5:3b-instruct)"
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset", "-d",
        default="data/math_problems.json",
        help="Path to dataset file"
    )
    parser.add_argument(
        "--max-problems", "-n",
        type=int,
        default=None,
        help="Maximum number of problems to process"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run on first 5 problems only"
    )

    # ATTS settings
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Disable mode escalation"
    )
    parser.add_argument(
        "--enable-refinement",
        action="store_true",
        help="Enable dialectical refinement (uses more tokens)"
    )
    parser.add_argument(
        "--passk-k",
        type=int,
        default=2,
        help="Number of samples for Pass@k difficulty estimation (default: 2)"
    )

    # Experiment settings
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline comparison"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline (always deep mode)"
    )

    # Output settings
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output filename"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each problem"
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich console formatting"
    )

    # Config override
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to config directory"
    )

    return parser


def run_experiment(
    problems: List[Problem],
    atts_workflow: ATTSWorkflow,
    baseline_workflow: BaselineWorkflow,
    result_repo: JsonResultRepository,
    visualizer: WorkflowVisualizer,
    explainer: DecisionExplainer,
    args: argparse.Namespace,
) -> tuple:
    """Run the ATTS experiment."""
    atts_results: List[WorkflowResult] = []
    baseline_results: List[WorkflowResult] = []
    thresholds = atts_workflow._thresholds

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"checkpoint_atts_{timestamp}"

    visualizer.print_header("ATTS EXPERIMENT")
    visualizer.print_status(f"Problems: {len(problems)}")
    visualizer.print_status(f"Escalation: {'Disabled' if args.no_escalation else 'Enabled'}")
    visualizer.print_status(f"Refinement: {'Enabled' if args.enable_refinement else 'Disabled'}")
    visualizer.print_status(f"Pass@k: {args.passk_k}")
    visualizer.print_status("")

    # Run ATTS
    if not args.baseline_only:
        visualizer.print_status("Running ATTS workflow...", style="bold blue")

        iterator = tqdm(problems, desc="ATTS") if TQDM_AVAILABLE else problems
        for idx, problem in enumerate(iterator, 1):
            try:
                result = atts_workflow.execute(problem, passk_k=args.passk_k)
                atts_results.append(result)

                if args.verbose:
                    visualizer.show_result_panel(result)

                # Checkpoint
                if idx % thresholds.checkpoint_interval == 0:
                    result_repo.save_checkpoint(atts_results, checkpoint_name)
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Checkpoint saved ({idx}/{len(problems)})")

                # Safety break
                if idx % thresholds.break_interval == 0 and idx < len(problems):
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Safety break ({thresholds.break_duration}s)...")
                    time.sleep(thresholds.break_duration)

            except Exception as e:
                if TQDM_AVAILABLE:
                    tqdm.write(f"Error on problem {problem.id}: {e}")
                continue

    # Run Baseline
    if not args.skip_baseline:
        visualizer.print_status("\nRunning Baseline workflow...", style="bold blue")

        iterator = tqdm(problems, desc="Baseline") if TQDM_AVAILABLE else problems
        for idx, problem in enumerate(iterator, 1):
            try:
                result = baseline_workflow.execute(problem)
                baseline_results.append(result)

                # Safety break
                if idx % thresholds.break_interval == 0 and idx < len(problems):
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Safety break ({thresholds.break_duration}s)...")
                    time.sleep(thresholds.break_duration)

            except Exception as e:
                if TQDM_AVAILABLE:
                    tqdm.write(f"Error on problem {problem.id}: {e}")
                continue

    # Clean up checkpoint
    result_repo.delete_checkpoint(checkpoint_name)

    return atts_results, baseline_results


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Initialize visualizer
    visualizer = WorkflowVisualizer(use_rich=not args.no_rich)
    explainer = DecisionExplainer()
    reporter = AnalysisReporter(visualizer)

    visualizer.print_header("ATTS - Adaptive Test-Time Scaling")
    visualizer.print_status(f"Model: {args.model}")
    visualizer.print_status("Paper Sections Validated: 12+")
    visualizer.print_status("Safety Features: Checkpointing, Auto-breaks, Error recovery")

    # Initialize config
    try:
        config_dir = Path(args.config_dir) if args.config_dir else None
        config = YamlConfigLoader(config_dir)
        visualizer.print_status("Config loaded", style="green")
    except ConfigError as e:
        visualizer.print_status(f"Config error: {e}", style="red")
        sys.exit(1)

    # Initialize Ollama
    try:
        model = OllamaAdapter(
            host=args.ollama_host,
            model_name=args.model,
        )
        if not model.is_available():
            visualizer.print_status(
                f"Model {args.model} not available. Pull it with: ollama pull {args.model}",
                style="red"
            )
            sys.exit(1)
        visualizer.print_status("Ollama connected", style="green")
    except ModelError as e:
        visualizer.print_status(f"Ollama error: {e}", style="red")
        visualizer.print_status("Make sure Ollama is running: docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama")
        sys.exit(1)

    # Load problems
    try:
        problem_repo = JsonProblemRepository()
        max_problems = 5 if args.quick_test else args.max_problems
        problems = problem_repo.load_problems(args.dataset, max_problems)
        visualizer.print_status(f"Loaded {len(problems)} problems", style="green")
    except Exception as e:
        visualizer.print_status(f"Dataset error: {e}", style="red")
        visualizer.print_status("Convert MATH dataset: python convert_math_dataset.py --size 100")
        sys.exit(1)

    # Warn about large datasets
    if len(problems) > 100:
        visualizer.print_status(
            f"\nLarge dataset: {len(problems)} problems",
            style="yellow"
        )
        visualizer.print_status(f"Estimated runtime: {len(problems) * 0.5 / 60:.1f} minutes")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            visualizer.print_status("Cancelled")
            sys.exit(0)

    # Initialize workflows
    result_repo = JsonResultRepository()
    atts_workflow = ATTSWorkflow(
        model=model,
        config=config,
        enable_escalation=not args.no_escalation,
        enable_refinement=args.enable_refinement,
    )
    baseline_workflow = BaselineWorkflow(
        model=model,
        config=config,
        enable_refinement=args.enable_refinement,
    )

    # Run experiment
    try:
        atts_results, baseline_results = run_experiment(
            problems=problems,
            atts_workflow=atts_workflow,
            baseline_workflow=baseline_workflow,
            result_repo=result_repo,
            visualizer=visualizer,
            explainer=explainer,
            args=args,
        )
    except KeyboardInterrupt:
        visualizer.print_status("\nInterrupted by user", style="yellow")
        sys.exit(0)

    # Analyze results
    if atts_results and baseline_results:
        analysis = reporter.analyze(atts_results, baseline_results)
        reporter.print_report(analysis, atts_results, baseline_results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        filename = f"comprehensive_results_{timestamp}{suffix}.json"

        result_repo.save_results(
            atts_results,
            filename,
            metadata={
                "model": args.model,
                "dataset_size": len(problems),
                "escalation_enabled": not args.no_escalation,
                "refinement_enabled": args.enable_refinement,
                "passk_k": args.passk_k,
                "analysis": reporter.export_to_dict(analysis),
            }
        )
        visualizer.print_status(f"\nResults saved: results/{filename}", style="green")

        # Show summary table
        if args.verbose:
            visualizer.show_summary_table(atts_results)

    elif atts_results:
        visualizer.print_status("\nATTS Results (no baseline comparison):")
        accuracy = sum(r.correct for r in atts_results) / len(atts_results)
        avg_tokens = sum(r.tokens_used for r in atts_results) / len(atts_results)
        visualizer.print_status(f"Accuracy: {accuracy * 100:.1f}%")
        visualizer.print_status(f"Avg Tokens: {avg_tokens:.0f}")

    # Show model statistics
    model_info = model.get_model_info()
    visualizer.print_status("\nModel Statistics:")
    visualizer.print_status(f"  Total API calls: {model_info['statistics']['total_calls']}")
    visualizer.print_status(f"  Total tokens: {model_info['statistics']['total_tokens']}")
    visualizer.print_status(f"  Total time: {model_info['statistics']['total_time']:.1f}s")


if __name__ == "__main__":
    main()

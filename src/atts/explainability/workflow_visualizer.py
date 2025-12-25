"""
Workflow Visualizer - Rich console visualization of ATTS workflow.

Uses the 'rich' library for beautiful terminal output.
"""

from typing import List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.tree import Tree
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..domain.entities import WorkflowResult


class WorkflowVisualizer:
    """
    Rich console visualization for ATTS workflow.

    Features:
    - Decision tree visualization
    - Progress bars for batch processing
    - Color-coded status indicators
    - Live updating panels
    """

    def __init__(self, use_rich: bool = True):
        """
        Initialize visualizer.

        Args:
            use_rich: Whether to use rich formatting (falls back to plain text)
        """
        self._use_rich = use_rich and RICH_AVAILABLE
        if self._use_rich:
            self._console = Console()
        else:
            self._console = None

    def show_workflow_tree(self, result: WorkflowResult):
        """Display the workflow as a decision tree."""
        if not self._use_rich:
            self._show_workflow_plain(result)
            return

        tree = Tree(
            f"[bold blue]ATTS Workflow - Problem {result.problem_id}[/bold blue]"
        )

        # Difficulty estimation
        diff_node = tree.add(
            f"[yellow]Difficulty Estimation[/yellow]: "
            f"{result.predicted_difficulty}/10 "
            f"(uncertainty: {result.difficulty_uncertainty:.2f})"
        )
        diff_node.add(f"True difficulty: {result.true_difficulty}")

        # Mode selection
        mode_node = tree.add(
            f"[cyan]Mode Selection[/cyan]: {result.initial_mode}"
        )

        # Escalation
        if result.escalated:
            esc_node = mode_node.add(
                f"[red]ESCALATED[/red] -> {result.final_mode}"
            )
            esc_node.add(f"Path: {' -> '.join(result.escalation_path)}")
        else:
            mode_node.add("[green]No escalation needed[/green]")

        # Verification
        score_color = (
            "green" if result.verification_score >= 0.8 else
            "yellow" if result.verification_score >= 0.6 else
            "red"
        )
        verify_node = tree.add(
            f"[{score_color}]Verification[/{score_color}]: "
            f"{result.verification_score:.2f}"
        )
        for rubric, score in result.rubric_scores.items():
            r_color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            verify_node.add(f"[{r_color}]{rubric}: {score:.2f}[/{r_color}]")

        # Refinement
        if result.refinement_history:
            refine_node = tree.add(
                f"[magenta]Refinement[/magenta]: "
                f"{len(result.refinement_history)} iterations"
            )
            for step in result.refinement_history:
                status = "[green]Applied[/green]" if step.action == "refined" else "[dim]Stopped[/dim]"
                valid = "[green]Valid[/green]" if step.critique_valid else "[red]Hallucinated[/red]"
                refine_node.add(f"Iteration {step.iteration + 1}: {status} ({valid})")

        # Outcome
        outcome_color = "green" if result.correct else "red"
        outcome_node = tree.add(
            f"[bold {outcome_color}]Outcome[/bold {outcome_color}]: "
            f"{'CORRECT' if result.correct else 'INCORRECT'}"
        )
        outcome_node.add(f"Tokens: {result.tokens_used}")
        outcome_node.add(f"Time: {result.total_time:.2f}s")

        self._console.print(tree)

    def show_result_panel(self, result: WorkflowResult):
        """Display a summary panel for a result."""
        if not self._use_rich:
            self._show_result_plain(result)
            return

        status_color = "green" if result.correct else "red"
        status_text = "CORRECT" if result.correct else "INCORRECT"

        content = Text()
        content.append(f"Problem: {result.problem_id}\n", style="bold")
        content.append(f"Status: ", style="dim")
        content.append(f"{status_text}\n", style=f"bold {status_color}")
        content.append(f"Difficulty: {result.predicted_difficulty}/10 ", style="yellow")
        content.append(f"(actual: {result.true_difficulty})\n", style="dim")
        content.append(f"Mode: {result.initial_mode}", style="cyan")
        if result.escalated:
            content.append(f" -> {result.final_mode}", style="red")
        content.append(f"\nVerification: {result.verification_score:.2f}\n", style="blue")
        content.append(f"Tokens: {result.tokens_used} | Time: {result.total_time:.2f}s", style="dim")

        panel = Panel(
            content,
            title=f"[bold]Problem {result.problem_id}[/bold]",
            border_style=status_color,
        )
        self._console.print(panel)

    def show_summary_table(self, results: List[WorkflowResult]):
        """Display a summary table of all results."""
        if not self._use_rich:
            self._show_summary_plain(results)
            return

        table = Table(
            title="ATTS Experiment Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
        )

        table.add_column("ID", style="dim", width=6)
        table.add_column("Diff", justify="center", width=8)
        table.add_column("Mode", justify="center", width=10)
        table.add_column("Esc", justify="center", width=4)
        table.add_column("Score", justify="center", width=8)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Status", justify="center", width=8)

        for r in results:
            status_style = "green" if r.correct else "red"
            status_text = "OK" if r.correct else "ERR"
            esc_text = "Yes" if r.escalated else "-"

            table.add_row(
                str(r.problem_id),
                f"{r.predicted_difficulty}/10",
                r.final_mode,
                esc_text,
                f"{r.verification_score:.2f}",
                str(r.tokens_used),
                Text(status_text, style=status_style),
            )

        self._console.print(table)

    def create_progress(self, total: int, description: str = "Processing") -> "Progress":
        """Create a progress bar for batch processing."""
        if not self._use_rich:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self._console,
        )

    def print_status(self, message: str, style: str = ""):
        """Print a status message."""
        if self._use_rich:
            if style:
                self._console.print(f"[{style}]{message}[/{style}]")
            else:
                self._console.print(message)
        else:
            print(message)

    def print_header(self, title: str):
        """Print a section header."""
        if self._use_rich:
            self._console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
            self._console.print(f"[bold blue]{title.center(60)}[/bold blue]")
            self._console.print(f"[bold blue]{'=' * 60}[/bold blue]\n")
        else:
            print(f"\n{'=' * 60}")
            print(title.center(60))
            print(f"{'=' * 60}\n")

    # Plain text fallbacks
    def _show_workflow_plain(self, result: WorkflowResult):
        """Plain text workflow display."""
        print(f"\nATTS Workflow - Problem {result.problem_id}")
        print("-" * 40)
        print(f"Difficulty: {result.predicted_difficulty}/10 (actual: {result.true_difficulty})")
        print(f"Mode: {result.initial_mode}", end="")
        if result.escalated:
            print(f" -> {result.final_mode} (escalated)")
        else:
            print()
        print(f"Verification: {result.verification_score:.2f}")
        print(f"  Rubrics: {result.rubric_scores}")
        if result.refinement_history:
            print(f"Refinement: {len(result.refinement_history)} iterations")
        print(f"Result: {'CORRECT' if result.correct else 'INCORRECT'}")
        print(f"Tokens: {result.tokens_used} | Time: {result.total_time:.2f}s")

    def _show_result_plain(self, result: WorkflowResult):
        """Plain text result display."""
        status = "OK" if result.correct else "ERR"
        print(f"[{status}] Problem {result.problem_id}: "
              f"d={result.predicted_difficulty} m={result.final_mode} "
              f"v={result.verification_score:.2f} t={result.tokens_used}")

    def _show_summary_plain(self, results: List[WorkflowResult]):
        """Plain text summary display."""
        print("\nATTS Experiment Results")
        print("-" * 60)
        print(f"{'ID':>6} {'Diff':>8} {'Mode':>10} {'Esc':>4} {'Score':>8} {'Tokens':>8} {'Status':>8}")
        print("-" * 60)
        for r in results:
            esc = "Yes" if r.escalated else "-"
            status = "OK" if r.correct else "ERR"
            print(f"{r.problem_id:>6} {r.predicted_difficulty:>6}/10 {r.final_mode:>10} "
                  f"{esc:>4} {r.verification_score:>8.2f} {r.tokens_used:>8} {status:>8}")

"""
JSON Repository - Implementation of repository interfaces for JSON files.

Handles saving/loading experiment results and problems as JSON.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..ports.repository import IResultRepository, IProblemRepository
from ..domain.entities import Problem, WorkflowResult
from ..domain.exceptions import DataError


class JsonResultRepository(IResultRepository):
    """
    Repository for saving/loading experiment results as JSON.

    Features:
    - Checkpoint support for crash recovery
    - Automatic timestamping
    - Pretty-printed output
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize JSON result repository.

        Args:
            results_dir: Directory for results (default: ./results)
        """
        if results_dir is None:
            results_dir = Path.cwd() / "results"

        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        results: List[WorkflowResult],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save experiment results to JSON file."""
        filepath = self._results_dir / filename

        output = {
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return filepath

    def load_results(self, filename: str) -> List[Dict[str, Any]]:
        """Load experiment results from JSON file."""
        filepath = self._results_dir / filename

        if not filepath.exists():
            raise DataError(
                f"Results file not found: {filepath}",
                {"filename": filename}
            )

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both old format (list) and new format (dict with results key)
        if isinstance(data, list):
            return data
        return data.get("results", [])

    def save_checkpoint(
        self,
        results: List[WorkflowResult],
        checkpoint_name: str,
    ) -> Path:
        """Save a checkpoint for crash recovery."""
        filepath = self._results_dir / f"{checkpoint_name}.json"

        checkpoint = {
            "checkpoint": True,
            "timestamp": datetime.now().isoformat(),
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        return filepath

    def load_checkpoint(self, checkpoint_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load a checkpoint if it exists."""
        filepath = self._results_dir / f"{checkpoint_name}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("results", [])
        except (json.JSONDecodeError, IOError):
            return None

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a checkpoint file."""
        filepath = self._results_dir / f"{checkpoint_name}.json"

        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def list_results(self) -> List[str]:
        """List all result files in the results directory."""
        return [
            f.name for f in self._results_dir.glob("*.json")
            if not f.name.startswith("checkpoint_")
        ]


class JsonProblemRepository(IProblemRepository):
    """
    Repository for loading problems from JSON datasets.

    Supports the format:
    {
        "problems": [
            {"id": "1", "problem": "...", "answer": "...", "difficulty_label": "easy"}
        ]
    }
    """

    def load_problems(
        self,
        dataset_path: str,
        max_problems: Optional[int] = None,
    ) -> List[Problem]:
        """Load problems from a JSON dataset file."""
        filepath = Path(dataset_path)

        if not filepath.exists():
            raise DataError(
                f"Dataset not found: {filepath}",
                {
                    "path": str(filepath),
                    "hint": "Run: python convert_math_dataset.py --size 100"
                }
            )

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataError(
                f"Invalid JSON in dataset: {str(e)}",
                {"path": str(filepath)}
            )

        # Support both formats: {"problems": [...]} or [...]
        if isinstance(data, dict):
            raw_problems = data.get("problems", [])
        else:
            raw_problems = data

        # Apply limit
        if max_problems is not None:
            raw_problems = raw_problems[:max_problems]

        # Convert to Problem objects
        problems = []
        for i, p in enumerate(raw_problems):
            problem = Problem(
                id=str(p.get("id", i + 1)),
                problem=p.get("problem", ""),
                answer=str(p.get("answer", "")),
                difficulty_label=p.get("difficulty_label", "medium"),
                metadata=p.get("metadata", {}),
            )
            problems.append(problem)

        return problems

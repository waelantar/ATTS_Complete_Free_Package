"""
Repository Port - Abstract interface for data persistence.

This port defines the contract for saving and loading experiment results.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..domain.entities import Problem, WorkflowResult


class IResultRepository(ABC):
    """
    Abstract interface for persisting experiment results.

    Implementations:
    - JsonRepository: Save/load as JSON files
    - (Future) CsvRepository: Save as CSV
    - (Future) SqliteRepository: Save to SQLite database
    """

    @abstractmethod
    def save_results(
        self,
        results: List[WorkflowResult],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save experiment results.

        Args:
            results: List of WorkflowResult objects
            filename: Output filename (without path)
            metadata: Optional experiment metadata

        Returns:
            Path to the saved file
        """
        pass

    @abstractmethod
    def load_results(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load experiment results.

        Args:
            filename: The filename to load

        Returns:
            List of result dictionaries
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        results: List[WorkflowResult],
        checkpoint_name: str,
    ) -> Path:
        """
        Save a checkpoint for crash recovery.

        Args:
            results: Current results to checkpoint
            checkpoint_name: Name for the checkpoint file

        Returns:
            Path to the checkpoint file
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load a checkpoint if it exists.

        Args:
            checkpoint_name: Name of the checkpoint file

        Returns:
            List of results or None if no checkpoint exists
        """
        pass

    @abstractmethod
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete a checkpoint file.

        Args:
            checkpoint_name: Name of the checkpoint file

        Returns:
            True if deleted, False if not found
        """
        pass


class IProblemRepository(ABC):
    """Abstract interface for loading problems."""

    @abstractmethod
    def load_problems(
        self,
        dataset_path: str,
        max_problems: Optional[int] = None,
    ) -> List[Problem]:
        """
        Load problems from a dataset.

        Args:
            dataset_path: Path to the dataset file
            max_problems: Optional limit on number of problems

        Returns:
            List of Problem objects
        """
        pass

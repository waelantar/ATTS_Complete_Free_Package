"""
Config Loader Port - Abstract interface for configuration loading.

This port defines the contract for loading configuration from any source.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PromptConfig:
    """Configuration for a single prompt template."""
    template: str
    max_tokens: int
    description: str = ""
    placeholders: tuple = ()


@dataclass
class ThresholdsConfig:
    """Configuration for ATTS thresholds."""
    # Difficulty thresholds
    direct_threshold: int = 4
    thinking_threshold: int = 7
    passk_k: int = 2

    # Escalation thresholds
    escalation_threshold: float = 0.80
    ascot_trigger: float = 0.60

    # Refinement settings
    max_refinement_iterations: int = 2
    early_exit_score: float = 0.85

    # Meta-verification
    meta_verification_threshold: float = 0.7

    # Safety settings
    checkpoint_interval: int = 10
    break_interval: int = 25
    break_duration: int = 5


class IConfigLoader(ABC):
    """
    Abstract interface for loading configuration.

    Implementations:
    - YamlConfigLoader: Load from YAML files
    - (Future) EnvConfigLoader: Load from environment
    - (Future) DictConfigLoader: Load from dictionary (testing)
    """

    @abstractmethod
    def load_prompts(self) -> Dict[str, PromptConfig]:
        """
        Load all prompt templates.

        Returns:
            Dictionary mapping prompt names to PromptConfig objects
        """
        pass

    @abstractmethod
    def load_thresholds(self) -> ThresholdsConfig:
        """
        Load threshold configuration.

        Returns:
            ThresholdsConfig object with all thresholds
        """
        pass

    @abstractmethod
    def get_prompt(self, name: str) -> PromptConfig:
        """
        Get a specific prompt by name.

        Args:
            name: Prompt name (e.g., "difficulty_estimation", "direct")

        Returns:
            PromptConfig for the requested prompt
        """
        pass

    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a general setting by key path.

        Args:
            key: Dot-separated key path (e.g., "ollama.host")
            default: Default value if not found

        Returns:
            The setting value or default
        """
        pass

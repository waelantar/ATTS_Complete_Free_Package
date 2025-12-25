"""
Model Caller Port - Abstract interface for language model interaction.

This port defines the contract for any LLM backend (Ollama, OpenAI, etc.)
Following hexagonal architecture, this allows swapping implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ModelResponse:
    """Response from a model call with metadata."""
    text: str
    tokens_used: int
    model_name: str
    generation_time: float = 0.0
    raw_response: Optional[dict] = None

    @property
    def is_empty(self) -> bool:
        """Check if response is empty."""
        return not self.text or len(self.text.strip()) < 1


class IModelCaller(ABC):
    """
    Abstract interface for language model calls.

    Implementations:
    - OllamaAdapter: Local Ollama models
    - (Future) OpenAIAdapter: OpenAI API
    - (Future) MockAdapter: Testing
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            ModelResponse with generated text and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and ready."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the current model."""
        pass

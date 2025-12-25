"""
Ollama Adapter - Implementation of IModelCaller for local Ollama models.

This adapter communicates with the Ollama API for local LLM inference.
"""

import time
from typing import Optional

try:
    import ollama
except ImportError:
    ollama = None

from ..ports.model_caller import IModelCaller, ModelResponse
from ..domain.exceptions import ModelError


class OllamaAdapter(IModelCaller):
    """
    Adapter for Ollama local language models.

    Usage:
        adapter = OllamaAdapter(
            host="http://localhost:11434",
            model_name="qwen2.5:3b-instruct"
        )
        response = adapter.generate("Solve: 2+2=")
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model_name: str = "qwen2.5:3b-instruct",
        default_temperature: float = 0.7,
    ):
        """
        Initialize Ollama adapter.

        Args:
            host: Ollama server URL
            model_name: Name of the model to use
            default_temperature: Default sampling temperature
        """
        if ollama is None:
            raise ModelError(
                "Ollama package not installed",
                {"hint": "Install with: pip install ollama"}
            )

        self._host = host
        self._model_name = model_name
        self._default_temperature = default_temperature
        self._client = ollama.Client(host=host)

        # Track usage statistics
        self._total_calls = 0
        self._total_tokens = 0
        self._total_time = 0.0

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """
        Generate text from a prompt using Ollama.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)

        Returns:
            ModelResponse with generated text and metadata

        Raises:
            ModelError: If Ollama call fails
        """
        if temperature is None:
            temperature = self._default_temperature

        start_time = time.time()

        try:
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            )

            generation_time = time.time() - start_time
            text = response.get("response", "")
            tokens = response.get("eval_count", len(text.split()))

            # Update statistics
            self._total_calls += 1
            self._total_tokens += tokens
            self._total_time += generation_time

            return ModelResponse(
                text=text,
                tokens_used=tokens,
                model_name=self._model_name,
                generation_time=generation_time,
                raw_response=response,
            )

        except Exception as e:
            raise ModelError(
                f"Ollama generation failed: {str(e)}",
                {
                    "model": self._model_name,
                    "host": self._host,
                    "prompt_length": len(prompt),
                }
            )

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            models = self._client.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            # Check if our model (or a variant) is available
            return any(self._model_name in name for name in model_names)
        except Exception:
            return False

    def get_model_info(self) -> dict:
        """Get information about the current model and usage."""
        return {
            "model_name": self._model_name,
            "host": self._host,
            "temperature": self._default_temperature,
            "statistics": {
                "total_calls": self._total_calls,
                "total_tokens": self._total_tokens,
                "total_time": round(self._total_time, 2),
                "avg_tokens_per_call": (
                    round(self._total_tokens / self._total_calls, 1)
                    if self._total_calls > 0 else 0
                ),
                "avg_time_per_call": (
                    round(self._total_time / self._total_calls, 2)
                    if self._total_calls > 0 else 0
                ),
            }
        }

    def reset_statistics(self):
        """Reset usage statistics."""
        self._total_calls = 0
        self._total_tokens = 0
        self._total_time = 0.0

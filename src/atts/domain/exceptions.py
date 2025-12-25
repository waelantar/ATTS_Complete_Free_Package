"""
Domain Exceptions - Custom exceptions for ATTS.

These exceptions provide clear, actionable error messages.
"""


class ATTSError(Exception):
    """Base exception for all ATTS errors."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ModelError(ATTSError):
    """
    Error communicating with the language model.

    Raised when:
    - Ollama is not running
    - Model is not available
    - API call times out
    - Response is malformed
    """
    pass


class VerificationError(ATTSError):
    """
    Error during solution verification.

    Raised when:
    - Verification prompt fails
    - Rubric scores cannot be parsed
    - Meta-verification fails unexpectedly
    """
    pass


class ConfigError(ATTSError):
    """
    Error in configuration.

    Raised when:
    - Config file not found
    - Invalid YAML syntax
    - Missing required config keys
    - Invalid threshold values
    """
    pass


class WorkflowError(ATTSError):
    """
    Error in workflow execution.

    Raised when:
    - Escalation limit reached without success
    - Refinement loop exceeds maximum iterations
    - Critical step fails
    """
    pass


class DataError(ATTSError):
    """
    Error in data handling.

    Raised when:
    - Dataset file not found
    - Invalid problem format
    - Answer format mismatch
    """
    pass

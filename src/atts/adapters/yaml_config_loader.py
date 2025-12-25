"""
YAML Config Loader - Implementation of IConfigLoader for YAML files.

Loads configuration from YAML files in the config/ directory.
"""

from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

from ..ports.config_loader import IConfigLoader, PromptConfig, ThresholdsConfig
from ..domain.exceptions import ConfigError


class YamlConfigLoader(IConfigLoader):
    """
    Load configuration from YAML files.

    Expected file structure:
        config/
            prompts.yaml      - Prompt templates
            thresholds.yaml   - Threshold values
            settings.yaml     - General settings
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize YAML config loader.

        Args:
            config_dir: Path to config directory (default: ./config)
        """
        if yaml is None:
            raise ConfigError(
                "PyYAML package not installed",
                {"hint": "Install with: pip install PyYAML"}
            )

        if config_dir is None:
            # Try to find config relative to package or current dir
            package_dir = Path(__file__).parent.parent.parent.parent
            config_dir = package_dir / "config"
            if not config_dir.exists():
                config_dir = Path.cwd() / "config"

        self._config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}

        if not self._config_dir.exists():
            raise ConfigError(
                f"Config directory not found: {self._config_dir}",
                {"hint": "Create config/ directory with prompts.yaml and thresholds.yaml"}
            )

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load and cache a YAML file."""
        if filename in self._cache:
            return self._cache[filename]

        filepath = self._config_dir / filename
        if not filepath.exists():
            raise ConfigError(
                f"Config file not found: {filepath}",
                {"filename": filename}
            )

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._cache[filename] = data
            return data
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML in {filename}: {str(e)}",
                {"filename": filename}
            )

    def load_prompts(self) -> Dict[str, PromptConfig]:
        """Load all prompt templates from prompts.yaml."""
        data = self._load_yaml("prompts.yaml")
        prompts = {}

        # Flatten nested structure
        def extract_prompts(d: Dict, prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict) and "template" in value:
                    prompts[full_key] = PromptConfig(
                        template=value["template"],
                        max_tokens=value.get("max_tokens", 500),
                        description=value.get("description", ""),
                    )
                elif isinstance(value, dict):
                    extract_prompts(value, full_key)

        extract_prompts(data)
        return prompts

    def load_thresholds(self) -> ThresholdsConfig:
        """Load threshold configuration from thresholds.yaml."""
        data = self._load_yaml("thresholds.yaml")

        return ThresholdsConfig(
            # Difficulty
            direct_threshold=data.get("difficulty", {}).get("direct_threshold", 4),
            thinking_threshold=data.get("difficulty", {}).get("thinking_threshold", 7),
            passk_k=data.get("difficulty", {}).get("passk_k", 2),
            # Escalation
            escalation_threshold=data.get("escalation", {}).get("threshold", 0.80),
            ascot_trigger=data.get("escalation", {}).get("ascot_trigger", 0.60),
            # Refinement
            max_refinement_iterations=data.get("refinement", {}).get("max_iterations", 2),
            early_exit_score=data.get("refinement", {}).get("early_exit_score", 0.85),
            # Meta-verification
            meta_verification_threshold=data.get("meta_verification", {}).get("threshold", 0.7),
            # Safety
            checkpoint_interval=data.get("safety", {}).get("checkpoint_interval", 10),
            break_interval=data.get("safety", {}).get("break_interval", 25),
            break_duration=data.get("safety", {}).get("break_duration", 5),
        )

    def get_prompt(self, name: str) -> PromptConfig:
        """Get a specific prompt by name."""
        prompts = self.load_prompts()

        if name in prompts:
            return prompts[name]

        # Try common prefixes
        for prefix in ["solution_modes", "verification", "dialectical"]:
            full_key = f"{prefix}.{name}"
            if full_key in prompts:
                return prompts[full_key]

        raise ConfigError(
            f"Prompt not found: {name}",
            {"available": list(prompts.keys())}
        )

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a general setting by dot-separated key path."""
        # Load settings.yaml
        data = self._load_yaml("settings.yaml")

        # Navigate the key path
        parts = key.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def reload(self):
        """Clear cache and reload all configs."""
        self._cache.clear()

"""Ports layer - Abstract interfaces defining contracts."""

from .model_caller import IModelCaller
from .config_loader import IConfigLoader
from .repository import IResultRepository

__all__ = ["IModelCaller", "IConfigLoader", "IResultRepository"]

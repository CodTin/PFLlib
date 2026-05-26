"""PFLlib - Personalized Federated Learning Library."""

__version__ = "0.1.0"

from pfllib.config import ExperimentConfig
from pfllib.registry import (
    get_algorithm,
    get_model_builder,
    list_algorithms,
    list_models,
    register_algorithm,
    register_model,
)

__all__ = [
    "ExperimentConfig",
    "get_algorithm",
    "get_model_builder",
    "list_algorithms",
    "list_models",
    "register_algorithm",
    "register_model",
]

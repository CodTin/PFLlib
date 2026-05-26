from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    pass

_ALGORITHM_REGISTRY: dict[str, tuple[Type, Type, bool]] = {}
_MODEL_REGISTRY: dict[str, Type] = {}


def register_algorithm(name: str, client_cls: Type, uses_head_split: bool = False):
    def decorator(server_cls: Type) -> Type:
        _ALGORITHM_REGISTRY[name] = (server_cls, client_cls, uses_head_split)
        return server_cls

    return decorator


def register_model(name: str):
    def decorator(model_cls: Type) -> Type:
        _MODEL_REGISTRY[name] = model_cls
        return model_cls

    return decorator


def get_algorithm(name: str) -> tuple[Type, Type, bool]:
    if name not in _ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_ALGORITHM_REGISTRY.keys())}")
    return _ALGORITHM_REGISTRY[name]


def get_model_builder(name: str) -> Type:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHM_REGISTRY.keys())


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())

"""Agent registry — name -> factory.

`@register_agent("name")` registers an `Agent` subclass under a string name;
`build_agent("name", **kwargs)` instantiates one. The eval harness uses this to
build tournament matrices declaratively (issue #007).

The registry intentionally lives in `tichu_ml` rather than `tichu_engine` so
that the engine remains free of ML-side imports.
"""

from typing import Callable

from tichu_ml.agent import Agent


AGENT_REGISTRY: dict[str, Callable[..., Agent]] = {}


def register_agent(name: str) -> Callable[[type[Agent]], type[Agent]]:
    def decorator(cls: type[Agent]) -> type[Agent]:
        if name in AGENT_REGISTRY:
            raise ValueError(f"agent name {name!r} is already registered")
        AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def build_agent(name: str, **kwargs) -> Agent:
    if name not in AGENT_REGISTRY:
        raise KeyError(f"unknown agent name: {name!r}")
    return AGENT_REGISTRY[name](**kwargs)

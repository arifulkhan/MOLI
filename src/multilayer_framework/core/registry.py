from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class MethodEntry:
    name: str
    factory: Callable[..., Any]
    description: str


class MethodRegistry:
    def __init__(self):
        self._methods: Dict[str, MethodEntry] = {}

    def register(self, name: str, factory: Callable[..., Any], description: str = ""):
        self._methods[name] = MethodEntry(name=name, factory=factory, description=description)

    def create(self, name: str, *args, **kwargs):
        entry = self._methods.get(name)
        if not entry:
            raise KeyError(f"Unknown method: {name}")
        return entry.factory(*args, **kwargs)

    def list_methods(self):
        return sorted(self._methods.keys())

"""CPU, GPU and memory resource management."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Resources:
    cpu_cores: int = 0
    gpus: int = 0
    memory_gib: int = 0

    def __post_init__(self) -> None:
        for name in ("cpu_cores", "gpus", "memory_gib"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer"
                )

    def __add__(self, other: Resources) -> Resources:
        return Resources(
            self.cpu_cores + other.cpu_cores,
            self.gpus + other.gpus,
            self.memory_gib + other.memory_gib,
        )

    def __sub__(self, other: Resources) -> Resources:
        return Resources(
            self.cpu_cores - other.cpu_cores,
            self.gpus - other.gpus,
            self.memory_gib - other.memory_gib,
        )

    def fits_in(self, capacity: Resources) -> bool:
        return (
            self.cpu_cores <= capacity.cpu_cores
            and self.gpus <= capacity.gpus
            and self.memory_gib <= capacity.memory_gib
        )

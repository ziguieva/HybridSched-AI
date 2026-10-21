"""HPC job representation."""

from dataclasses import dataclass
from hybridsched.resources import Resources


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    arrival_s: int
    requested_walltime_s: int
    resources: Resources
    priority: int = 0
    deadline_s: int | None = None

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id cannot be empty")

        if type(self.arrival_s) is not int or self.arrival_s < 0:
            raise ValueError(
                "arrival_s must be a non-negative integer"
            )

        if (
            type(self.requested_walltime_s) is not int
            or self.requested_walltime_s <= 0
        ):
            raise ValueError(
                "requested_walltime_s must be positive"
            )

        if type(self.priority) is not int or self.priority < 0:
            raise ValueError(
                "priority must be a non-negative integer"
            )

        if self.deadline_s is not None and (
            type(self.deadline_s) is not int
            or self.deadline_s < self.arrival_s
        ):
            raise ValueError(
                "deadline_s must be at or after arrival_s"
            )

        if not isinstance(self.resources, Resources):
            raise TypeError(
                "resources must be a Resources instance"
            )

        if (
            self.resources.cpu_cores < 1
            or self.resources.memory_gib < 1
        ):
            raise ValueError(
                "jobs need at least one CPU core "
                "and one GiB of memory"
            )

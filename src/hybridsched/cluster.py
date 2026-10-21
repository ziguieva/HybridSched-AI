"""HPC cluster resource allocation."""

from dataclasses import dataclass, field

from hybridsched.job import Job
from hybridsched.resources import Resources


@dataclass(slots=True)
class Cluster:
    name: str
    capacity: Resources

    _running: dict[str, Job] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _used: Resources = field(
        default_factory=Resources,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "cluster name cannot be empty"
            )

        if not isinstance(self.capacity, Resources):
            raise TypeError(
                "capacity must be a Resources instance"
            )

        if (
            self.capacity.cpu_cores < 1
            or self.capacity.memory_gib < 1
        ):
            raise ValueError(
                "cluster needs CPU and memory capacity"
            )

    @property
    def used(self) -> Resources:
        return self._used

    @property
    def available(self) -> Resources:
        return self.capacity - self._used

    @property
    def running_job_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._running))

    def can_allocate(self, job: Job) -> bool:
        return (
            job.job_id not in self._running
            and job.resources.fits_in(self.available)
        )

    def allocate(self, job: Job) -> None:
        if job.job_id in self._running:
            raise ValueError(
                f"job {job.job_id!r} is already allocated"
            )

        if not job.resources.fits_in(self.available):
            raise ValueError(
                f"insufficient resources for {job.job_id!r}"
            )

        self._running[job.job_id] = job
        self._used = self._used + job.resources

    def release(self, job_id: str) -> None:
        if job_id not in self._running:
            raise KeyError(
                f"job {job_id!r} is not allocated"
            )

        job = self._running.pop(job_id)
        self._used = self._used - job.resources

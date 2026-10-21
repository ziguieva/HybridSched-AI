"""Hybrid HPC infrastructure and cluster specifications."""

from dataclasses import dataclass

from hybridsched.resources import Resources


@dataclass(frozen=True, slots=True)
class ClusterSpec:
    name: str
    provider: str
    capacity: Resources

    speed_num: int = 1
    speed_den: int = 1

    gpu_cost_per_s: int = 0
    cpu_cost_per_s: int = 0
    startup_s: int = 0

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Cluster name cannot be empty")

        if not self.provider.strip():
            raise ValueError("Provider cannot be empty")

        if self.capacity.cpu_cores <= 0:
            raise ValueError("Cluster requires CPU capacity")

        if self.capacity.memory_gib <= 0:
            raise ValueError("Cluster requires memory capacity")

        if self.speed_num <= 0 or self.speed_den <= 0:
            raise ValueError("Speed ratio must be positive")

        if min(
            self.gpu_cost_per_s,
            self.cpu_cost_per_s,
            self.startup_s,
        ) < 0:
            raise ValueError("Costs and startup must be nonnegative")

    def execution_time(self, requested_s: int) -> int:
        """Compute illustrative cluster-dependent duration."""

        if requested_s <= 0:
            raise ValueError("Duration must be positive")

        scaled = (
            requested_s * self.speed_den
            + self.speed_num - 1
        ) // self.speed_num

        return scaled + self.startup_s

    def execution_cost(self, resources: Resources, duration_s: int) -> int:
        """Return synthetic cost units, not actual currency."""

        if duration_s < 0:
            raise ValueError("Duration cannot be negative")

        return duration_s * (
            resources.gpus * self.gpu_cost_per_s
            + resources.cpu_cores * self.cpu_cost_per_s
        )


def default_infrastructure() -> list[ClusterSpec]:
    """Illustrative local, HPC and cloud infrastructure."""

    return [
        ClusterSpec(
            name="local-gpu",
            provider="ON_PREMISE",
            capacity=Resources(
                cpu_cores=16,
                gpus=4,
                memory_gib=64,
            ),
            speed_num=1,
            speed_den=1,
            gpu_cost_per_s=1,
            cpu_cost_per_s=0,
            startup_s=0,
        ),
        ClusterSpec(
            name="hpc-cluster",
            provider="HPC",
            capacity=Resources(
                cpu_cores=32,
                gpus=8,
                memory_gib=128,
            ),
            speed_num=5,
            speed_den=4,
            gpu_cost_per_s=2,
            cpu_cost_per_s=0,
            startup_s=20,
        ),
        ClusterSpec(
            name="gcp-gpu",
            provider="GCP_SIMULATED",
            capacity=Resources(
                cpu_cores=24,
                gpus=6,
                memory_gib=96,
            ),
            speed_num=3,
            speed_den=2,
            gpu_cost_per_s=5,
            cpu_cost_per_s=1,
            startup_s=60,
        ),
    ]


def main():
    print("=" * 78)
    print("HybridSched-AI | HYBRID HPC INFRASTRUCTURE")
    print("=" * 78)

    for cluster in default_infrastructure():
        print(f"\nCluster: {cluster.name}")
        print(f"Provider: {cluster.provider}")
        print(f"CPU cores: {cluster.capacity.cpu_cores}")
        print(f"GPUs: {cluster.capacity.gpus}")
        print(f"RAM: {cluster.capacity.memory_gib} GiB")
        print(f"900s job duration: {cluster.execution_time(900)}s")
        print(f"GPU cost rate: {cluster.gpu_cost_per_s} units/s")

    print("\nInfrastructure initialized successfully.")


if __name__ == "__main__":
    main()

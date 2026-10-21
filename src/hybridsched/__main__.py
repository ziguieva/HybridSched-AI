"""Command-line entry point for HybridSched-AI."""

from hybridsched import __version__


def main() -> None:
    print("=" * 54)
    print("HybridSched-AI")
    print("Learning-Based Scheduling for Hybrid Cloud & HPC")
    print(f"Version: {__version__}")
    print("Status: Development environment ready")
    print("=" * 54)


if __name__ == "__main__":
    main()

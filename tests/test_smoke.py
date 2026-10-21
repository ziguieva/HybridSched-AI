"""Initial smoke tests."""

from hybridsched import __version__
from hybridsched.__main__ import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_main(capsys) -> None:
    main()
    output = capsys.readouterr().out
    assert "HybridSched-AI" in output
    assert "Development environment ready" in output

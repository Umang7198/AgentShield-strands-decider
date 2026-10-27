import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_help() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "app.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "frontend" in result.stdout.lower()
    assert "--fake-decider" in result.stdout

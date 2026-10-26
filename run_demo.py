"""Run the six research figure-generation scripts."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.call(
        [sys.executable, "-u", str(root / "demo" / "run_all.py")],
        cwd=str(root / "demo"), env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())

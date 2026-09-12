"""Thin console-script shim so ``pip install -e .`` exposes ``edcl-predict``
as a real CLI command (instead of only ``python scripts/predict.py``).

Kept as a shim rather than moving scripts/predict.py wholesale: the script
is documented/tested at its existing path and this avoids churn to
existing tests/docs mid-repo-freeze. Registered via
``[project.scripts]`` in pyproject.toml.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if not (scripts_dir / "predict.py").exists():
        raise SystemExit(
            "edcl-predict: scripts/predict.py not found next to the "
            "installed package (expected at "
            f"{scripts_dir}). This console script only works from an "
            "editable install (`pip install -e .`) inside a checkout of "
            "the EDCL repo; use `python scripts/predict.py` otherwise."
        )
    sys.path.insert(0, str(scripts_dir))
    from predict import main as _predict_main  # type: ignore

    return _predict_main()


if __name__ == "__main__":
    raise SystemExit(main())

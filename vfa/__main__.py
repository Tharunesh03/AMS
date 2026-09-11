"""``python -m vfa`` — thin wrapper so the documented module entry point works.

All logic lives in :mod:`vfa.cli`; keeping it out of this file means the same ``main``
can be reused as a console-script entry point (see ``[project.scripts]`` in pyproject.toml).
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

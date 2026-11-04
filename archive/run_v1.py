"""
Run the frozen v1 suite without editing it.

`v1_suite.py` is a pytest suite that imports from a package called `rck`,
which is what `core/` was named when the suite was written. The file is
evidence and must not be changed, so this shim aliases the old package name
onto the current one and then hands the file to pytest.

The claim this file exists to let a reader check is the one in README.md:
the suite passes, and passing is exactly the problem.

    python archive/run_v1.py
"""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core                                                    # noqa: E402

sys.modules["rck"] = core
for sub in ("abstraction", "kl", "reporting", "scm", "sheaf"):
    sys.modules[f"rck.{sub}"] = importlib.import_module(f"core.{sub}")

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([os.path.join(ROOT, "archive", "v1_suite.py"),
                                  "-q", "--no-header", "-p", "no:cacheprovider"]))

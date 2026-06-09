"""Put the repo root on sys.path so `tests/scripts/*` can `import scripts.<module>`.

These tests import the pure helpers out of the runnable scripts under `scripts/`.
That package lives at the repo root, which is not on the path under every pytest
invocation (`pytest` vs `python -m pytest`). Inserting it here is invocation-robust.
NOTE: there is deliberately no `tests/scripts/__init__.py` — packaging this dir would
create a `scripts` test-package that shadows the real repo-root `scripts` package.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

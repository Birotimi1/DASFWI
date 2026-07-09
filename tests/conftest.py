import sys
from pathlib import Path

# Make `import ADFWI` and `import das`/`forge`/`inversion` work regardless of
# where pytest is launched from. ADFWI is not pip-installed; its repo root
# (CODES/ADFWI/) must be on sys.path.
_CODES = Path(__file__).resolve().parents[2]
for _p in (str(_CODES / "ADFWI"), str(Path(__file__).resolve().parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sys
from pathlib import Path

# Make `import ADFWI` and `import das`/`forge`/`inversion` work regardless of
# where pytest is launched from. ADFWI is not pip-installed; its repo root
# (CODES/ADFWI/) must be on sys.path.
_REPO = Path(__file__).resolve().parents[1]
_CODES = Path(__file__).resolve().parents[2]
# Prefer a sibling ADFWI checkout (the cluster layout); fall back to the repo's
# own ADFWI_local mirror, because a standalone clone has no sibling. Same
# resolution order as hpc/*/common.py's ADFWI_ROOT, so tests and drivers agree.
_adfwi = _CODES / "ADFWI"
if not (_adfwi / "ADFWI").is_dir():
    _adfwi = _REPO / "ADFWI_local"
for _p in (str(_adfwi), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

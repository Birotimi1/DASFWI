"""Utah FORGE well 58-32 wireline logs -- the ONLY ground truth the field has.

Schlumberger DSI run, `...ME-ESW1_Run1_DSI Sonic.las`. Park validate their VM0
and VM3 against exactly this: "a wireline sonic log from 656 to 2307 m in
depth". The file's own header says STRT 2150.5 ft, STOP 7569.0 ft = 655.5 to
2307.0 m -- matching to the metre, so this is the same log.

>>> IT ALSO CARRIES SHEAR, WHICH PARK DO NOT USE. <<<
    DTCO  compressional slowness -> Vp
    DTSM  SHEAR slowness         -> Vs
    PR    Poisson's ratio (logged directly, so our sqrt(3) assumption is
          CHECKABLE rather than assumed)
    GR    gamma ray -- lithology, i.e. the zone I/II/III boundaries
Park invert acoustic Vp only and name elastic Vs as future work. A Vs log is
therefore ground truth for the part of our claim nobody has validated at this
site.

GEOMETRY, measured not assumed: 58-32 is a THIRD well, not one of the two DAS
wells. Projected onto our 2-D section it sits at along-section x = +328 m
(inside the -1547..+1412 m span) and only 68 m OFF-line -- under half a
wavelength at 10 Hz, so the 2-D approximation is sound there.

>>> VALIDATION ONLY. <<< Never an inversion constraint, never a tuning target.
Our mandate is Vp and Vs from DAS strain rate ALONE; tuning to this log would
destroy the transferability claim that is the entire point. See
inversion/field_acceptance.py, whose thresholds were fixed before any field run.
"""

# --- import bootstrap: depend on NO environment ----------------------------- #
# `python forge/preflight.py` puts forge/ on sys.path, NOT the repo root, so
# ADFWI/forge/inversion are unimportable unless PYTHONPATH happens to be set.
# It was set in every shell I tested in and NOT in the user's cluster shell, so
# this died there with a bare ModuleNotFoundError. Resolve from __file__
# instead: walk up to the directory holding forge/ and inversion/, and add the
# tracked ADFWI package next to it.
import sys as _sys
from pathlib import Path as _Path

for _r in _Path(__file__).resolve().parents:
    if (_r / "forge").is_dir() and (_r / "inversion").is_dir():
        for _p in (_r, _r / "ADFWI_local"):
            if (_p / "ADFWI").is_dir() and str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
        if str(_r) not in _sys.path:
            _sys.path.insert(0, str(_r))
        break
# ---------------------------------------------------------------------------- #
import os
from pathlib import Path

import numpy as np

FT_M = 0.3048
#: Park's stated sonic interval, for cross-checking whatever we load
PARK_RANGE_M = (656.0, 2307.0)
#: 58-32 on our 2-D section, from the LAS header lat/lon (UTM 12N) vs the
#: 78A-32 wellhead and the section's PCA axis. Recomputed by `section_position`.
X_ON_SECTION_M, OFF_SECTION_M = 328.0, 68.0


def _default_log_path():
    root = Path(os.environ.get("FORGE_DAS_DIR", "."))
    hits = sorted(root.glob("*DSI*Sonic*.las")) or sorted(root.glob("*.las"))
    return hits[0] if hits else None


def read_las(path=None):
    """Minimal LAS 2.0 reader -> (curves dict, header dict).

    Deliberately dependency-free: `lasio` is not in env.yml and a 200-line
    parser is not worth a new dependency on the cluster. NULL values become NaN.
    """
    path = Path(path) if path else _default_log_path()
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"no LAS found (looked in $FORGE_DAS_DIR): {path}")
    names, header, rows, sec = [], {}, [], None
    null = -999.25
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("~"):
                sec = s[1:2].upper()
                continue
            if sec in ("W", "P") and ":" in s:
                k = s.split(".")[0].strip()
                v = s.split(":")[0].split(".", 1)[-1].strip()
                header[k] = v
                if k == "NULL":
                    try:
                        null = float(v.split()[-1])
                    except ValueError:
                        pass
            elif sec == "C" and "." in s:
                names.append(s.split(".")[0].strip())
            elif sec == "A":
                rows.append(s.split())
    if not rows:
        raise ValueError(f"{path}: no ~A data section")
    a = np.array([[float(x) for x in r] for r in rows if len(r) == len(names)])
    a[np.isclose(a, null)] = np.nan
    return {n: a[:, i] for i, n in enumerate(names)}, header


def slowness_to_velocity(dt_us_per_ft):
    """us/ft -> m/s. Zero/NaN slowness returns NaN, never inf."""
    dt = np.asarray(dt_us_per_ft, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = FT_M * 1e6 / dt
    return np.where(np.isfinite(v) & (dt > 0), v, np.nan)


def load_58_32(path=None):
    """Vp, Vs, Poisson and gamma from the 58-32 DSI log, in METRES and m/s.

    Returns a dict with `z_m` (measured depth), `vp`, `vs`, `pr`, `gr`, plus
    `deviation_m` from the ED/ND departures -- because a deviated well means
    measured depth is NOT TVD, and placing the log at the wrong depth would
    invalidate every comparison made against it.
    """
    c, h = read_las(path)
    z = np.asarray(c["DEPT"], float) * FT_M
    out = dict(z_m=z, header=h,
               vp=slowness_to_velocity(c.get("DTCO")),
               vs=slowness_to_velocity(c["DTSM"]) if "DTSM" in c else None,
               pr=np.asarray(c["PR"], float) if "PR" in c else None,
               gr=np.asarray(c["GR_EDTC"], float) if "GR_EDTC" in c else None)
    if "ED" in c and "ND" in c:
        dev = np.hypot(np.asarray(c["ED"], float),
                       np.asarray(c["ND"], float)) * FT_M
        out["deviation_m"] = dev
        out["max_deviation_m"] = float(np.nanmax(dev))
        # TVD from measured depth, given the horizontal departure. For a nearly
        # vertical well this is a small correction, but "nearly" has to be
        # MEASURED -- Park's figure plots the log along the well, not at MD.
        out["tvd_m"] = np.sqrt(np.maximum(z ** 2 - dev ** 2, 0.0))
        # What matters is the MD->TVD CORRECTION against the grid cell, not the
        # raw departure. 58-32 departs 56 m over 2307 m -- 1.4 degrees -- which
        # is a 0.7 m depth correction, i.e. 0.07 of a 10 m cell. Reporting the
        # departure alone made a vertical well look deviated.
        out["tvd_correction_m"] = float(np.nanmax(z - out["tvd_m"]))
    return out


def poisson_check(vp, vs):
    """Measured Vp/Vs against the sqrt(3) Poisson-solid assumption we use.

    `starting_model.vs_from_vp` and the elastic synthetic both assume
    Vp/Vs = sqrt(3) ~ 1.732. This log can confirm or refute that AT THIS SITE,
    which is worth knowing before it propagates into every elastic result.
    """
    vp, vs = np.asarray(vp, float), np.asarray(vs, float)
    m = np.isfinite(vp) & np.isfinite(vs) & (vs > 0)
    if not m.any():
        return dict(n=0)
    r = vp[m] / vs[m]
    return dict(n=int(m.sum()), median_ratio=float(np.median(r)),
                p10=float(np.percentile(r, 10)),
                p90=float(np.percentile(r, 90)),
                sqrt3=float(np.sqrt(3.0)),
                assumption_ok=bool(abs(np.median(r) - np.sqrt(3.0)) < 0.15))


def section_position(lat=None, lon=None, das_dir=None):
    """(along, off) section position of 58-32, in metres, MEASURED not assumed.

    The off-section distance is the number that decides whether a 2-D
    comparison is legitimate at all: if the well sits far off the line, the
    section does not pass through the rock the log samples.
    """
    from pyproj import Transformer
    from forge.field_loader import read_shot_geometry, project_to_2d, DAS_VSP_DIR
    lat = 38.500562 if lat is None else lat
    lon = -112.88703 if lon is None else lon
    e, n = Transformer.from_crs("EPSG:4326", "EPSG:32612",
                                always_xy=True).transform(lon, lat)
    g = read_shot_geometry(Path(das_dir or DAS_VSP_DIR) / "78A-32", n_shots=None)
    p = project_to_2d(g["src_xyz"], g["rcv_xyz"])
    d = np.array([e, n]) - g["rcv_xyz"][:, :2].mean(axis=0)
    ax = p["axis"]
    return float(d @ ax), float(d @ np.array([-ax[1], ax[0]]))

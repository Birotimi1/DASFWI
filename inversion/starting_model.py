"""Transferable starting models for DAS strain-rate FWI (ROUTE B + Vs seed).

The project's mandate is an EXPLORATORY capability: recover Vp and Vs with no
sonic logs, no check-shots, no region-calibrated priors - deployable anywhere.
That rules out picking-plus-eikonal (first-break picking is the fragile step on
DAS: broadside-blind channels, and S picks buried in the P coda) and rules out
empirical Vp/Vs trends such as Castagna's mudrock line, which is calibrated on
water-saturated clastics and is simply wrong for a crystalline geothermal
reservoir like FORGE.

ROUTE B (this module) instead measures the kinematics with the WAVE EQUATION:
a cross-correlation traveltime misfit on the full strain-rate waveforms, started
from a data-independent 1-D linear v(z). No picking anywhere.

Why this is exact for DAS, and better than pick+eikonal: the E3 gauge operator
is applied to the SYNTHETICS as well as living in the observed data, so both
sides are shaped identically and the cross-correlation time shift carries ZERO
operator-induced kinematic bias. A picked observed traveltime compared against a
geometric eikonal solver has no such guarantee (the solver knows nothing about
gauge averaging).

Vs seed: Vp/sqrt(3) is the ISOTROPIC PHYSICS default (Poisson solid, nu = 0.25,
lambda = mu) - a universal "no information" prior rather than a basin-specific
fit, and close to observed crystalline values (Vp/Vs ~ 1.70-1.75). It is only a
BOOTSTRAP: the S kinematics refine it. Caveat worth remembering - in a ~1 km
sedimentary cover with true Vp/Vs ~ 2.2 the sqrt(3) seed is ~200 ms off in
one-way S time, which EXCEEDS T/2 = 167 ms at 3 Hz, i.e. the S wavefield can be
cycle-skipped at the starting frequency. That is exactly why the Vs-release stage
starts with a robust (lambda = 1) objective; see inversion/adaptive_misfit.py.
"""
import numpy as np

SQRT3 = float(np.sqrt(3.0))


# --------------------------------------------------------------------------- #
# data-independent starting models
# --------------------------------------------------------------------------- #
def linear_vz(nz, nx, v_top, v_bottom, water_rows=0, v_water=None):
    """1-D linear v(z), constant laterally - the data-independent Route B start.

    Uses NO information from the true model beyond the two end velocities, which
    in the field come from the water/near-surface velocity and a bulk gradient
    guess. Optionally pins a water layer.
    """
    prof = np.linspace(float(v_top), float(v_bottom), int(nz))
    v = np.repeat(prof[:, None], int(nx), axis=1)
    if water_rows > 0:
        v[:water_rows] = float(v_water if v_water is not None else v_top)
    return v


def vs_from_vp(vp, ratio=SQRT3, depth_ratio=None, dz=None):
    """Vs seed from Vp.

    Args:
        vp: (nz, nx) P velocity.
        ratio: constant Vp/Vs (default sqrt(3), the Poisson-solid physics prior).
        depth_ratio: optional [(z_top_m, ratio), ...] for a lithology-graded seed
            (e.g. a higher Vp/Vs in sedimentary cover, ~1.73 in basement).
            Entries apply from their z downward; requires `dz`.
        dz: vertical grid spacing (m), needed only with depth_ratio.

    NOTE this is a BOOTSTRAP. Route B refines Vs from the S kinematics, so the
    final model is not bound to the assumed ratio.
    """
    vp = np.asarray(vp, dtype=float)
    if depth_ratio:
        if dz is None:
            raise ValueError("depth_ratio requires dz")
        r = np.full(vp.shape[0], float(ratio))
        for z_top, rr in sorted(depth_ratio, key=lambda t: t[0]):
            r[int(round(float(z_top) / float(dz))):] = float(rr)
        return vp / r[:, None]
    return vp / float(ratio)


def clip_to_bounds(v, vmin, vmax):
    return np.clip(np.asarray(v, dtype=float), float(vmin), float(vmax))


def smooth_model(v, sigma):
    """Gaussian smoothing - a traveltime starter must stay long-wavelength."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(v, dtype=float), float(sigma), mode="nearest")


def poisson_clamp(vp, vs, min_vp_vs=1.5):
    """Enforce vs <= vp / min_vp_vs (below sqrt(2) the elastic scheme diverges)."""
    return np.minimum(np.asarray(vs, dtype=float),
                      np.asarray(vp, dtype=float) / float(min_vp_vs))

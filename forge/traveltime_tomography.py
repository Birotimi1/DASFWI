"""First-break traveltime tomography -> data-driven FWI starting model.

FWI needs a starting model in the right basin of attraction (autograd doesn't
relax that; the misfit/multiscale machinery does). For a walkaway VSP a much
better starting model than a blind 1-D gradient can be built directly from the
data: the direct-arrival first-break times constrain the vertical velocity
profile (the classic check-shot / VSP interval-velocity method).

Pipeline:
  1. pick_first_breaks   - STA/LTA first-arrival picking, per trace.
  2. vsp_checkshot_velocity - straight-ray deskew of a near-offset shot's
       first-break curve to vertical time, then v(z) = dz/dt from a smoothed,
       monotonic t(z).
  3. build_starting_model - interpolate v(z) onto the grid, extend, tile
       across x, clip -> a 2-D vp starting model for FWI.

This is a STARTING model (smooth, long-wavelength), not the answer; FWI refines
it. It removes the need to guess a gradient for an unknown site while still not
eliminating the need for a starting model.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# 1. first-break picking (STA/LTA)
# --------------------------------------------------------------------------- #
def pick_first_breaks(gather, dt, sta_s=0.01, lta_s=0.05, threshold=3.0,
                      min_time_s=0.0):
    """First-arrival sample time per trace via the STA/LTA ratio.

    Args:
        gather: [nt, C] traces (time first).
        dt: sample interval [s].
        sta_s, lta_s: short / long window lengths [s].
        threshold: STA/LTA ratio that declares an arrival.
        min_time_s: ignore picks before this time (mutes the direct-from-t0
            ramp / pre-trigger noise).

    Returns:
        picks [C] first-arrival times [s]; NaN where no trace energy crosses
        the threshold.
    """
    g = np.asarray(gather, dtype=np.float64)
    nt, C = g.shape
    sta_n = max(1, int(round(sta_s / dt)))
    lta_n = max(sta_n + 1, int(round(lta_s / dt)))
    e = g ** 2
    # cumulative-sum moving sums (causal, trailing windows). Divide by the
    # ACTUAL window length, not the nominal n -- at the start the trailing
    # window is truncated, and dividing by n would pin the ratio to
    # lta_n/sta_n (a false trigger at sample 0).
    cs = np.concatenate([np.zeros((1, C)), np.cumsum(e, axis=0)], axis=0)
    idx = np.arange(nt)
    sta_len = (idx + 1) - np.maximum(idx + 1 - sta_n, 0)
    lta_len = (idx + 1) - np.maximum(idx + 1 - lta_n, 0)
    sta = (cs[idx + 1] - cs[np.maximum(idx + 1 - sta_n, 0)]) / sta_len[:, None]
    lta = (cs[idx + 1] - cs[np.maximum(idx + 1 - lta_n, 0)]) / lta_len[:, None]
    ratio = sta / (lta + 1e-30)
    # mute until the LTA window is full (ratio unreliable during warm-up) and
    # before the requested minimum time
    i0 = max(int(round(min_time_s / dt)), lta_n)
    ratio[:i0, :] = 0.0
    picks = np.full(C, np.nan)
    for c in range(C):
        hits = np.nonzero(ratio[:, c] >= threshold)[0]
        if hits.size:
            picks[c] = hits[0] * dt
    return picks


# --------------------------------------------------------------------------- #
# 2. VSP check-shot 1-D velocity from first breaks
# --------------------------------------------------------------------------- #
def vsp_checkshot_velocity(pick_times, z_rcv, x_offset=0.0, smooth_n=5,
                           v_bounds=(1400.0, 6500.0), surface_anchor=True):
    """1-D interval velocity v(z) from a near-offset VSP first-break curve.

    surface_anchor: a downhole fiber gives NO information above its shallowest
    channel. Extending the shallowest INTERVAL velocity upward is wrong (it can
    make the whole overburden unphysically slow). Instead, prepend a z=0 point
    at the AVERAGE velocity to the shallowest receiver (z_top / t_vert_top),
    which the first break does constrain. This is the physically correct shallow
    extrapolation for a deep VSP.

    Straight-ray deskew to vertical: for a receiver at depth z and a source at
    horizontal offset x, the straight-ray path length is sqrt(z^2 + x^2), so
    the equivalent VERTICAL time is t_vert = t * z / sqrt(z^2 + x^2). Then
    v(z) = dz/dt_vert from a smoothed, monotonic t_vert(z). Accurate for deep
    receivers (z >> x); shallow, wide-offset picks are the approximation's
    weak point (documented).

    Args:
        pick_times [C], z_rcv [C]: first-break times [s] and receiver depths [m]
            (NaN picks are dropped).
        x_offset: source-to-well horizontal offset [m].
        smooth_n: moving-average window (samples) on t(z) before differencing.
        v_bounds: clip the interval velocity to this range.

    Returns:
        (z_sorted [K], v_of_z [K]): depths and interval velocities.
    """
    z = np.asarray(z_rcv, float)
    t = np.asarray(pick_times, float)
    ok = np.isfinite(t) & np.isfinite(z) & (z > 0)
    z, t = z[ok], t[ok]
    order = np.argsort(z)
    z, t = z[order], t[order]
    if z.size < 3:
        raise ValueError("need >= 3 valid first-break picks")

    t_vert = t * z / np.sqrt(z ** 2 + x_offset ** 2)
    # smooth, then enforce STRICTLY increasing vertical time so dz/dt is finite
    # everywhere (equal consecutive times -> inf/NaN velocity otherwise). Ties
    # are broken by the minimum increment consistent with the upper velocity
    # bound (dt >= dz / v_max), which also caps the implied interval velocity.
    if smooth_n > 1:
        k = np.ones(smooth_n) / smooth_n
        t_vert = np.convolve(t_vert, k, mode="same")
    dz = np.diff(z)
    t_min_inc = dz / v_bounds[1]                   # smallest physical dt per step
    t_fixed = np.empty_like(t_vert)
    t_fixed[0] = t_vert[0]
    for i in range(1, len(t_vert)):
        t_fixed[i] = max(t_vert[i], t_fixed[i - 1] + t_min_inc[i - 1])
    v = np.gradient(z, t_fixed)                     # dz/dt, guaranteed finite
    v = np.clip(np.nan_to_num(v, nan=v_bounds[0],
                              posinf=v_bounds[1], neginf=v_bounds[0]),
                *v_bounds)
    if surface_anchor and t_fixed[0] > 0:
        v_avg = float(np.clip(z[0] / t_fixed[0], *v_bounds))   # avg to fiber top
        z = np.concatenate([[0.0], z])
        v = np.concatenate([[v_avg], v])
    return z, v


# --------------------------------------------------------------------------- #
# 3. assemble a 2-D starting model
# --------------------------------------------------------------------------- #
def build_starting_model(z_prof, v_prof, nz, nx, dz, smooth_nodes=4,
                         v_bounds=(1400.0, 6500.0)):
    """Interpolate a 1-D v(z) onto the grid and tile across x.

    Depths outside the profile are held at the nearest profile value
    (constant extension). Returns vp [nz, nx] float64.
    """
    z_nodes = np.arange(nz) * dz
    v_col = np.interp(z_nodes, z_prof, v_prof,
                      left=v_prof[0], right=v_prof[-1])
    if smooth_nodes > 1:
        k = np.ones(smooth_nodes) / smooth_nodes
        v_col = np.convolve(v_col, k, mode="same")
    v_col = np.clip(v_col, *v_bounds)
    return np.tile(v_col[:, None], (1, nx))


def starting_model_from_gathers(gathers, dt, z_rcv, x_offset, nz, nx, dz,
                                sta_s=0.01, lta_s=0.05, threshold=3.0,
                                min_time_s=0.0, v_bounds=(1400.0, 6500.0)):
    """End-to-end: near-offset shot gather [nt, C] -> 2-D vp starting model.

    Pass the SINGLE nearest-offset shot's gather (time first) and its source
    offset. Returns (vp_2d, z_prof, v_prof, picks) for inspection.
    """
    picks = pick_first_breaks(gathers, dt, sta_s=sta_s, lta_s=lta_s,
                              threshold=threshold, min_time_s=min_time_s)
    z_prof, v_prof = vsp_checkshot_velocity(picks, z_rcv, x_offset=x_offset,
                                            v_bounds=v_bounds)
    vp = build_starting_model(z_prof, v_prof, nz, nx, dz, v_bounds=v_bounds)
    return vp, z_prof, v_prof, picks

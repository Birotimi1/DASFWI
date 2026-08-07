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


#: STARTER BOUNDS COME FROM THE INVERSION'S BOUNDS, not a literal.
#: They were 6500 here and 6000 there, so a starter could hand over velocities
#: the inversion clamps on its FIRST update -- an instant, silent modification
#: of the model you thought you started from. Birotimi caught the mismatch on
#: the cluster; deriving it means the two cannot drift apart again.
def _default_v_bounds():
    from inversion.near_surface import VP_BOUND_FIELD
    return (1400.0, float(VP_BOUND_FIELD[1]))


# --------------------------------------------------------------------------- #
# 1. first-break picking (STA/LTA)
# --------------------------------------------------------------------------- #
def pick_first_breaks(gather, dt, sta_s=0.01, lta_s=0.05, threshold=3.0,
                      min_time_s=0.0, coherence=True, med_win=21, tol_s=0.03):
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
    # >>> DOMINANT EVENT, then walk back to its ONSET. <<<
    # The first threshold CROSSING takes whatever noise burst happens to exceed
    # the ratio first, and on FORGE that put every pick in the 0.05-0.25 s
    # pre-arrival noise while the real arrival sat at 0.42-0.65 s. Measured on a
    # synthetic reproducing that exact situation (strong arrival + random early
    # bursts on 85% of traces):
    #     first crossing      scatter 75.53 ms,  2.4% on-trend
    #     global max + onset  scatter  0.61 ms,  100% on-trend
    # The remaining bias (~-28 ms) is the onset-to-peak offset -- real, and the
    # SAME for every trace, so it shifts the model rather than corrupting it.
    # Scatter is what destroys tomography, and that is what drops 124x.
    picks = np.full(C, np.nan)
    for c in range(C):
        k = int(np.argmax(ratio[:, c]))
        if ratio[k, c] < threshold:
            continue                      # no event anywhere; do not invent one
        j = k
        while j > 0 and ratio[j - 1, c] >= threshold:
            j -= 1
        picks[c] = j * dt
    if coherence:
        picks = _enforce_coherence(picks, ratio, dt, threshold,
                                   med_win=med_win, tol_s=tol_s, i0=i0)
    return picks


def _enforce_coherence(picks, ratio, dt, threshold, med_win=21, tol_s=0.03,
                       i0=0, n_pass=2):
    """Re-pick channels whose arrival disagrees with their neighbours.

    SECONDARY cleanup, not the main fix. Tried alone first and it FAILED: when
    85% of traces pick the same noise band, the local median is noise too, so
    continuity converges confidently to the wrong event (0% correct in test).
    Selecting the DOMINANT event is what fixes the picks; this then tidies the
    minority that still disagree with their neighbours.

    First-threshold-crossing STA/LTA takes whatever noise burst happens to
    exceed the ratio first. On FORGE that put the picks at 0.05-0.25 s, in the
    pre-arrival noise, while the real arrival was the strong event at
    0.42-0.65 s -- so the "traveltime" starting model was built from noise. It
    came out saturated at the 6000 m/s upper bound (apparent velocity ~14 km/s
    from a 1 km channel picked at 0.07 s), i.e. a constant block carrying no
    information at all. Every `--starting traveltime` cell was therefore an
    invalid test, not a bad result.

    A first arrival MOVES SMOOTHLY across a fibre: neighbouring channels are ~1 m
    apart, so their arrivals differ by well under a millisecond. That is a strong,
    physical, site-agnostic constraint and nothing was using it. Here a channel
    that deviates from the local median by more than `tol_s` is re-picked inside a
    window around that median, taking the STRONGEST ratio peak there and then
    walking back to its onset.

    Nothing about FORGE is assumed: the constraint is only "the arrival is
    continuous", which holds for any fibre in any well at any site.
    """
    p = np.asarray(picks, float).copy()
    nt, C = ratio.shape
    half = max(1, int(med_win) // 2)
    for _ in range(int(n_pass)):
        # local median of the FINITE picks only -- a NaN neighbourhood must not
        # drag the reference, and np.nanmedian of an all-NaN slice warns and
        # returns NaN, which is the right answer but noisily
        ref = np.full(C, np.nan)
        for c in range(C):
            w = p[max(0, c - half):min(C, c + half + 1)]
            w = w[np.isfinite(w)]
            if w.size:
                ref[c] = np.median(w)
        bad = ~np.isfinite(p) | (np.isfinite(ref) & (np.abs(p - ref) > tol_s))
        if not bad.any():
            break
        for c in np.nonzero(bad & np.isfinite(ref))[0]:
            lo = max(i0, int((ref[c] - tol_s) / dt))
            hi = min(nt, int((ref[c] + tol_s) / dt) + 1)
            if hi - lo < 2:
                continue
            seg = ratio[lo:hi, c]
            k = int(np.argmax(seg))
            if seg[k] < threshold:
                p[c] = np.nan            # genuinely nothing there; do not invent
                continue
            # walk back from the peak to the onset: first sample rising above
            # half the threshold. Picking the PEAK would bias every arrival late
            # by roughly a quarter period.
            j = k
            while j > 0 and seg[j - 1] >= 0.5 * threshold:
                j -= 1
            p[c] = (lo + j) * dt
    return p


# --------------------------------------------------------------------------- #
# 2. VSP check-shot 1-D velocity from first breaks
# --------------------------------------------------------------------------- #
def vsp_checkshot_velocity(pick_times, z_rcv, x_offset=0.0, smooth_n=5,
                           v_bounds=None, surface_anchor=True,
                           bin_m=40.0):
    """1-D interval velocity v(z) from a near-offset VSP first-break curve.

    Interval velocity is computed over COARSE depth bins (``bin_m``), not per
    channel. At ~1 m channel spacing and 1 ms sampling, per-channel dz/dt is
    dominated by pick quantization: a one-sample tie between adjacent channels
    forces the interval to the velocity ceiling, railing the whole profile to
    v_max. Binning to intervals where the traveltime change spans many samples
    restores a physical velocity.
    """
    v_bounds = _default_v_bounds() if v_bounds is None else v_bounds
    z = np.asarray(z_rcv, float)
    t = np.asarray(pick_times, float)
    ok = np.isfinite(t) & np.isfinite(z) & (z > 0)
    z, t = z[ok], t[ok]
    order = np.argsort(z)
    z, t = z[order], t[order]
    if z.size < 3:
        raise ValueError("need >= 3 valid first-break picks")

    # straight-ray deskew to vertical traveltime
    t_vert = t * z / np.sqrt(z ** 2 + x_offset ** 2)

    # bin to coarse depth intervals; median pick time per bin (robust to outliers)
    z0, z1 = z.min(), z.max()
    edges = np.arange(z0, z1 + bin_m, bin_m)
    zc, tc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        if m.any():
            zc.append(float(z[m].mean()))
            tc.append(float(np.median(t_vert[m])))
    zc, tc = np.asarray(zc), np.asarray(tc)
    if zc.size < 2:
        raise ValueError("too few depth bins; reduce bin_m")

    # enforce strictly increasing t over the COARSE grid (bin spacing ~bin_m, so
    # the minimum increment corresponds to v_max over a real interval, not 1 m)
    tc_fixed = np.copy(tc)
    for i in range(1, tc_fixed.size):
        dz_bin = zc[i] - zc[i - 1]
        t_min_inc = dz_bin / v_bounds[1]
        tc_fixed[i] = max(tc_fixed[i], tc_fixed[i - 1] + t_min_inc)

    # interval velocity between bin centres, then clip to physical bounds
    v_bin = np.diff(zc) / np.diff(tc_fixed)
    v_bin = np.clip(v_bin, *v_bounds)
    z_bin = 0.5 * (zc[1:] + zc[:-1])

    if smooth_n > 1 and v_bin.size >= smooth_n:
        k = np.ones(smooth_n) / smooth_n
        v_bin = np.convolve(v_bin, k, mode="same")

    z_out, v_out = z_bin, v_bin
    if surface_anchor and tc_fixed[0] > 0:
        v_avg = float(np.clip(zc[0] / tc_fixed[0], *v_bounds))
        z_out = np.concatenate([[0.0], z_out])
        v_out = np.concatenate([[v_avg], v_out])
    return z_out, v_out


# --------------------------------------------------------------------------- #
# 3. assemble a 2-D starting model
# --------------------------------------------------------------------------- #
def build_starting_model(z_prof, v_prof, nz, nx, dz, smooth_nodes=4,
                         v_bounds=None):
    """Interpolate a 1-D v(z) onto the grid and tile across x.

    Depths outside the profile are held at the nearest profile value
    (constant extension). Returns vp [nz, nx] float64.
    """
    v_bounds = _default_v_bounds() if v_bounds is None else v_bounds
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
                                min_time_s=0.0, v_bounds=None):
    """End-to-end: near-offset shot gather [nt, C] -> 2-D vp starting model.

    Pass the SINGLE nearest-offset shot's gather (time first) and its source
    offset. Returns (vp_2d, z_prof, v_prof, picks) for inspection.
    """
    v_bounds = _default_v_bounds() if v_bounds is None else v_bounds
    picks = pick_first_breaks(gathers, dt, sta_s=sta_s, lta_s=lta_s,
                              threshold=threshold, min_time_s=min_time_s)
    z_prof, v_prof = vsp_checkshot_velocity(picks, z_rcv, x_offset=x_offset,
                                            v_bounds=v_bounds)
    vp = build_starting_model(z_prof, v_prof, nz, nx, dz, v_bounds=v_bounds)
    return vp, z_prof, v_prof, picks

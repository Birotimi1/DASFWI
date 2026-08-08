"""Near-surface handling for field DAS-FWI: air layer, bounds, anisotropic smoothing.

Three faults found on 2026-08-03 by comparing our field driver against Park et
al.'s FORGE methodology. All three corrupt the SAME shallow zone -- the upper
~1 km that the DAS-VSP exists to constrain -- so they are fixed together.

1. NO AIR LAYER. Park: "The inclined nature of the surface topography requires
   careful handling... we set the highest point of the model as depth 0 and
   incorporate an AIR LAYER in the inversion process." We matched the datum
   convention but modelled the space between the datum plane and the real ground
   surface as ROCK. Every source below the highest then radiates through
   fictitious rock.

2. THE Vp BOUND MADE AN AIR LAYER IMPOSSIBLE. `VP_BOUND=(1500,6000)` clamps air
   (340 m/s) to 1500. Park's INV2 bounds are 1.0-5.9 km/s -- ours also forbade
   the slow alluvium (zone I) outright, so that error had to be absorbed
   elsewhere in the model.

3. ISOTROPIC GRADIENT SMOOTHING. Park smooth 200 m x 100 m (INV1) and
   100 m x 25 m (INV2) -- 2:1 and 4:1 HORIZONTAL:VERTICAL, i.e. along layers,
   not across. ADFWI's `smooth2d(grad, span=...)` takes ONE span, so we smoothed
   vertically as hard as horizontally and threw away the VERTICAL RESOLUTION
   that is the whole point of a VSP.

HOW THE AIR LAYER IS ENFORCED (no new machinery -- ADFWI already has both parts,
they were simply never wired up):
  * `water_layer_mask` -> in AcousticModel.clip_params the masked cells keep
    their UNCLAMPED value, so air at 340 m/s survives a (1000, 6000) bound.
  * `grad_mute` + `marine_or_land="marine"` -> grad_taper zeroes the gradient in
    the top `grad_mute` rows, so the air is never updated.
Both are required: the mask alone would let the optimizer walk the air away, and
the mute alone would let clip_params clamp it to the lower bound.
"""
import numpy as np

#: speed of sound in air (m/s). Park model an air layer; proxy_model uses 340.
V_AIR = 340.0

#: Park's INV2 bounds, in m/s. The LOWER bound matters: unconsolidated alluvium
#: reaches ~1 km/s, and our previous 1500 forbade it.
VP_BOUND_FIELD = (1000.0, 6000.0)


def air_cells(z_air, dz):
    """Number of grid rows occupied by the air layer (>= 0)."""
    if z_air is None or z_air <= 0:
        return 0
    return int(np.ceil(float(z_air) / float(dz)))


def topography_relief(src_z):
    """Relief spanned by the sources, in metres.

    MEASURE THIS BEFORE BUILDING AN AIR LAYER. If the relief is small compared
    with a wavelength (v_min/f_max; ~75 m at 20 Hz and 1500 m/s) the flat-datum
    approximation is defensible and the air layer buys nothing but cost.
    `src_z` is depth below datum, so its max IS the relief.
    """
    z = np.asarray(src_z, float)
    return float(np.nanmax(z) - np.nanmin(z))


def surface_profile(src_x, src_z, nx, dx, x0=0.0):
    """Ground depth below datum for EVERY grid column, from the source elevations.

    MEASURED AT FORGE (2026-08-03, 318 shots): the topography is an almost
    perfect RAMP -- correlation(x, z) = +0.994, dropping 161.6 m over ~2960 m
    (~3 degrees), with at most 6.2 m between neighbouring shots. So linear
    interpolation between shots, with edge hold beyond them, is faithful; there
    are no cliffs to alias.

    A UNIFORM-THICKNESS air slab is WRONG at this site: at one end of the line
    the ground IS the datum (zero air) and at the other it is 158 m below it.
    That is exactly what Park mean by "the inclined nature of the surface
    topography", and why they add an air layer rather than just a datum shift.
    """
    sx = np.asarray(src_x, float).ravel()
    sz = np.asarray(src_z, float).ravel()
    o = np.argsort(sx)
    sx, sz = sx[o], sz[o]
    xs = x0 + np.arange(int(nx)) * float(dx)
    return np.interp(xs, sx, sz)            # np.interp holds the edge values


def air_mask_topo(nz, nx, ground_depth, dz):
    """Boolean [nz, nx] air mask whose lower edge FOLLOWS the topography.

    `ground_depth` is metres below datum per column (see `surface_profile`).
    Cell (i, j) is air when its depth is above that column's ground.
    """
    depth = (np.arange(int(nz))[:, None] + 0.5) * float(dz)
    return depth < np.asarray(ground_depth, float)[None, :]


def with_air_layer_topo(vp, ground_depth, dz, v_air=V_AIR):
    """Copy of `vp` with air ABOVE the topographic surface, column by column."""
    out = np.array(vp, dtype=float, copy=True)
    m = air_mask_topo(out.shape[0], out.shape[1], ground_depth, dz)
    out[m] = float(v_air)
    return out


def air_mask(nz, nx, n_air):
    """Boolean [nz, nx] marking the air rows -- for `water_layer_mask`.

    True where the value must be EXEMPT from clip_params, i.e. the air.
    """
    m = np.zeros((int(nz), int(nx)), dtype=bool)
    if n_air > 0:
        m[:int(n_air), :] = True
    return m


def with_air_layer(vp, n_air, v_air=V_AIR):
    """Return a copy of `vp` whose top `n_air` rows are air."""
    out = np.array(vp, dtype=float, copy=True)
    if n_air > 0:
        out[:int(n_air), :] = float(v_air)
    return out


def anisotropic_span(v_min, f_max, dx, dz, fraction=0.25, aspect=4.0,
                     min_span=1):
    """(span_x, span_z) in CELLS for gradient smoothing, Park-style.

    Returns a HORIZONTAL and a VERTICAL span. `aspect` is the horizontal:vertical
    ratio of the smoothing LENGTHS -- Park use 2:1 for the surface-seismic
    tomography and 4:1 for the DAS-VSP, and 4:1 is the right default here
    because a VSP's resolution is vertical: smoothing z as hard as x discards
    exactly what the fibre measures.

    The vertical length is lambda*fraction (the classic "do not pretend to
    resolve finer than this"), and the horizontal length is `aspect` times that.
    """
    if not (v_min > 0 and f_max > 0 and dx > 0 and dz > 0):
        raise ValueError(f"need positive v_min/f_max/dx/dz, got "
                         f"{v_min}/{f_max}/{dx}/{dz}")
    lam_frac = float(fraction) * (float(v_min) / float(f_max))    # metres
    span_z = max(int(min_span), int(round(lam_frac / float(dz))))
    span_x = max(int(min_span), int(round(lam_frac * float(aspect) / float(dx))))
    return span_x, span_z


def smooth2d_anisotropic(z, span_x, span_z):
    """Separable Gaussian smoothing with different H and V spans.

    ADFWI's `smooth2d` is isotropic (one `span`), and its kernel is built with
    `np.linspace(-2*span, 2*span, 2*span+1)`, so `span` must be an INT -- the
    same requirement that broke wavelength_span. Implemented separably here
    rather than by patching the vendored propagator.
    """
    a = np.array(z, dtype=float, copy=True)
    for axis, span in ((0, int(span_z)), (1, int(span_x))):
        if span <= 0:
            continue
        x = np.arange(-2 * span, 2 * span + 1, dtype=float)
        k = np.exp(-0.5 * (x / float(span)) ** 2)
        k /= k.sum()
        a = np.apply_along_axis(
            lambda v: np.convolve(np.pad(v, span * 2, mode="edge"), k,
                                  mode="same")[span * 2:-span * 2],
            axis, a)
    return a


def cfl_dt(vmax, dx, safety=0.45, order=4):
    """Largest STABLE time step for the FD scheme, in seconds.

    MEASURED, not taken on faith. At the FORGE setup (dx=10 m, vmax=5896 m/s)
    Park's dt = 1 ms sits at 0.97x the nominal 4th-order limit, and the run
    LOOKS FINE then blows up slowly:
        0.6 s -> max|u| 9.9      1.2 s -> 196      2.0 s -> NaN
    So a short smoke passes and the real 2 s record dies -- the worst possible
    failure shape. `safety=0.45` is deliberately below the nominal 0.606 because
    the air/rock contrast (17:1) and the PML both erode the margin.

    We cannot simply copy Park's dt: their scheme's stability limit is not ours.
    """
    if not (vmax > 0 and dx > 0):
        raise ValueError(f"need positive vmax/dx, got {vmax}/{dx}")
    return float(safety) * float(dx) / float(vmax)


def stable_time_axis(vmax, dx, record_s, dt_wanted=None, safety=0.45):
    """(dt, nt) that keep `record_s` seconds AND stay stable.

    Shortening dt without lengthening nt would silently TRUNCATE the record,
    which is its own quiet corruption -- the far offsets would simply lose their
    arrivals. So nt grows to preserve the record length.
    """
    dt_max = cfl_dt(vmax, dx, safety)
    dt = min(dt_wanted, dt_max) if dt_wanted else dt_max
    return float(dt), int(np.ceil(float(record_s) / dt))


def describe(nz, nx, dz, z_air, v_min, f_max, dx, src_z=None, aspect=4.0):
    """One-line summary for the job log, so the near-surface setup is VISIBLE.

    Silent near-surface handling is how these three faults survived: the model
    looked fine and the error went into the shallow velocities.
    """
    n_air = air_cells(z_air, dz)
    sx, sz = anisotropic_span(v_min, f_max, dx, dz, aspect=aspect)
    parts = [f"air {n_air} rows ({n_air * dz:.0f} m @ {V_AIR:.0f} m/s)"
             if n_air else "NO air layer",
             f"vp_bound {VP_BOUND_FIELD[0]:.0f}-{VP_BOUND_FIELD[1]:.0f} m/s",
             f"grad smooth {sx}x{sz} cells = {sx * dx:.0f}x{sz * dz:.0f} m (H:V {aspect:g}:1)"]
    if src_z is not None:
        relief = topography_relief(src_z)
        lam = v_min / f_max
        parts.append(f"relief {relief:.0f} m vs lambda {lam:.0f} m")
        if relief > 0.25 * lam:
            # the flat-datum error is a SPURIOUS free-surface ghost: a source
            # sitting h below a flat datum reflects off it 2h/v late
            ghost_ms = 2.0 * relief / v_min * 1000.0
            parts.append(f"*** AIR LAYER REQUIRED: a flat datum fabricates a "
                         f"free-surface ghost {ghost_ms:.0f} ms late "
                         f"= {ghost_ms / (1000.0 / (2 * f_max)):.1f} half-cycles")
        else:
            parts.append("relief < lambda/4: flat datum ok, air layer not needed")
    return "near-surface: " + "; ".join(parts)


def imprint_mask(nz, nx, dz, dx, src_iz, src_ix, rcv_iz, rcv_ix,
                 radius_m=100.0, taper_m=None):
    """Zero the gradient around sources and receivers -- Noe et al. 2025.

    >>> THE PIECE OF NOE'S CONDITIONING WE NEVER IMPLEMENTED. <<<
    From their FWI workflow, alongside the three we did build (amplitude-
    dependent channel weighting, arrival windowing, lambda/4 gradient
    smoothing):

        "We remove source and receiver imprints by setting the kernels to zero
         within a radius of 100 m around their respective locations."

    The adjoint kernel is singular AT a source or receiver: the wavefield there
    is dominated by the near-field term, which carries no information about the
    medium but has enormous amplitude. Left in, it stamps a bright spot at every
    source and receiver and streaks radiate outward along the raypaths. That is
    exactly what our FORGE sections show -- a fan centred on the wellhead, which
    is the RECEIVER imprint, plus surface artefacts at the shot line.

    Tapered rather than cut, so the mask does not imprint its own hard edge in
    place of the one it removes. `radius_m` defaults to Noe's 100 m; pass a
    wavelength-scaled value at other frequencies.
    """
    taper_m = radius_m if taper_m is None else float(taper_m)
    m = np.ones((int(nz), int(nx)), float)
    zz = np.arange(nz)[:, None] * float(dz)
    xx = np.arange(nx)[None, :] * float(dx)
    pts = list(zip(np.atleast_1d(src_iz), np.atleast_1d(src_ix))) + \
          list(zip(np.atleast_1d(rcv_iz), np.atleast_1d(rcv_ix)))
    for iz, ix in pts:
        r = np.hypot(zz - float(iz) * dz, xx - float(ix) * dx)
        # 0 inside radius, ramping to 1 at radius+taper
        w = np.clip((r - radius_m) / max(taper_m, 1e-9), 0.0, 1.0)
        m = np.minimum(m, w)
    return m

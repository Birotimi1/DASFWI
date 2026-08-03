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
        parts.append(f"relief {relief:.0f} m vs lambda {lam:.0f} m"
                     + ("  *** relief > lambda/4: air layer MATTERS"
                        if relief > 0.25 * lam else "  (relief small: flat datum ok)"))
    return "near-surface: " + "; ".join(parts)

"""Model-recovery metrics following Liu's ADFWI paper:

  MAPE - Mean Absolute Percentage Error (Hyndman & Koehler, 2006), eq. (9):
      MAPE = 100/(Nx*Nz) * sum_{x,z} |m(x,z) - m_hat(x,z)| / |m(x,z)|     [%]
      lower is better; 0 = perfect.

  SSIM - Structural SIMilarity index (Wang et al., 2004), eq. (10):
      windowed (mu, sigma, cross-covariance, C1/C2) similarity of the true and
      inverted models; higher is better, 1 = identical. This is exactly
      skimage.metrics.structural_similarity with gaussian_weights (the Wang 2004
      formulation), data_range taken from the TRUE model's dynamic range.

m is the true model, m_hat the inverted model (2D velocity arrays).
"""
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:                                    # pragma: no cover
    _ssim = None


def mape(true, inv):
    """Mean Absolute Percentage Error (%). Velocity models never hit 0, but the
    denominator is guarded anyway."""
    true = np.asarray(true, dtype=float)
    inv = np.asarray(inv, dtype=float)
    denom = np.abs(true)
    denom = np.where(denom > 0, denom, np.nan)
    return float(100.0 * np.nanmean(np.abs(true - inv) / denom))


def ssim(true, inv):
    """Structural similarity index in [-1, 1] (1 = identical). Gaussian window
    (Wang 2004), data_range from the true model. Returns NaN if scikit-image is
    unavailable."""
    true = np.asarray(true, dtype=float)
    inv = np.asarray(inv, dtype=float)
    if _ssim is None:
        return float("nan")
    dr = float(true.max() - true.min()) or 1.0
    # win_size must be odd and <= smallest dim; 11 is skimage's gaussian default
    win = min(11, min(true.shape) | 1 if min(true.shape) % 2 == 0
              else min(true.shape))
    win = win if win % 2 == 1 else win - 1
    return float(_ssim(true, inv, data_range=dr, gaussian_weights=True,
                       win_size=max(3, win)))


def model_scores(true, inv, deep=None):
    """Return {mape, ssim, mape_deep, ssim_deep} for a true/inverted pair.
    `deep` is an optional row-slice for the deep-region metrics."""
    out = {"mape": mape(true, inv), "ssim": ssim(true, inv)}
    if deep is not None:
        out["mape_deep"] = mape(true[deep], inv[deep])
        out["ssim_deep"] = ssim(true[deep], inv[deep])
    return out


def boundary_depth(vp, dz, v_contour=4500.0):
    """Depth [m] of the first crossing of `v_contour`, per column. NaN if none."""
    v = np.asarray(vp, float)
    out = np.full(v.shape[1], np.nan)
    for j in range(v.shape[1]):
        i = np.flatnonzero(v[:, j] >= v_contour)
        if i.size:
            out[j] = i[0] * float(dz)
    return out


def dip_recovery(vp_true, vp_init, vp_final, dz, dx, v_contour=4500.0,
                 edge_frac=0.15):
    """How much of a TRUE lateral dip did the inversion actually recover?

    >>> THE QUESTION THIS PROJECT COULD NOT ANSWER. <<<
    Park's FORGE section has a basement rising ~650 m across it, and ours are
    flat -- but Park's dip is an INPUT to their FWI (it comes from 3-D surface
    seismic via INV1), not an output of it. So "we do not match Park" never
    established whether a single-well DAS-VSP can recover a dip AT ALL.

    With a dip in the synthetic and a flat starting model, it can be measured:

        recovered_frac = (dip_final - dip_init) / (dip_true - dip_init)

    1.0 = fully recovered, 0.0 = the inversion added no lateral structure,
    negative = it moved the wrong way. Columns within `edge_frac` of either edge
    are dropped: they are outside the shot aperture and scoring them would
    measure the taper.
    """
    nx = np.asarray(vp_true).shape[1]
    k = max(1, int(edge_frac * nx))
    sl = slice(k, nx - k)
    x = np.arange(nx)[sl] * float(dx) / 1000.0

    def slope(v):
        b = boundary_depth(v, dz, v_contour)[sl]
        m = np.isfinite(b)
        if m.sum() < 4:
            return float("nan")
        return float(np.polyfit(x[m], b[m], 1)[0])        # m per km

    d_t, d_i, d_f = slope(vp_true), slope(vp_init), slope(vp_final)
    denom = d_t - d_i
    frac = (d_f - d_i) / denom if np.isfinite(denom) and abs(denom) > 1e-9 \
        else float("nan")
    return dict(dip_true=d_t, dip_init=d_i, dip_final=d_f,
                recovered_frac=frac, v_contour=v_contour)

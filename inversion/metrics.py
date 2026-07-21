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

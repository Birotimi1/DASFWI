"""Illumination (diagonal-pseudo-Hessian) gradient preconditioner.

FWI gradients decay with depth (geometric spreading + weak deep illumination,
worse with limited-aperture DAS), so shallow updates dominate and deep
structure is under-recovered. The standard fix divides the gradient by an
estimate of the source illumination -- the diagonal of the (Gauss-Newton)
Hessian -- which is the time-integrated forward-wavefield energy per cell:

    diag(H)_ij  ~  illum_ij  =  sum_t | forward_wavefield_ij(t) |^2

The ADFWI propagators already accumulate this (record_waveform[
"forward_wavefield_*"]). This module turns it into a per-cell weight
W = 1 / precond^power (precond = smoothed illum, normalized, floored) so that
`grad *= W` lifts weakly-illuminated deep cells. Default power=2 matches ADFWI's
acoustic GradProcessor (grad / illumination^2 -- aggressive deep compensation);
power=1 is the gentler textbook diagonal-Hessian form (grad / diag(H)).
"""
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:                                # pragma: no cover
    gaussian_filter = None


def illumination_weight(illum, power=2.0, epsilon=1e-3, sigma=6.0, max_boost=1e3):
    """Return a per-cell preconditioner weight from a forward-illumination map.

    Args:
        illum: 2D forward-wavefield energy (torch tensor or ndarray), (nz, nx).
        power: exponent on the (inverse) illumination (2 = ADFWI acoustic
            GradProcessor default; 1 = gentler diagonal Hessian).
        epsilon: relative floor on the normalized illumination, so the raw
            boost is 1/epsilon**power (epsilon=1e-3, power=2 -> up to 1e6).
        sigma: Gaussian smoothing (nodes) applied to the illumination before
            forming the weight; stabilizes the preconditioner. 0 disables.
        max_boost: hard cap on the weight. The raw power-2 boost can reach 1e6,
            which pushed fragile misfits (envelope) into a NaN over 300 iters;
            capping at ~1e3 keeps a strong deep boost but prevents blow-up.

    Returns:
        same type as `illum` (torch tensor on the same device/dtype, or ndarray)
        with W in [1, max_boost]; multiply the gradient by it.
    """
    is_torch = hasattr(illum, "detach")
    if is_torch:
        import torch
        a = illum.detach().to("cpu", torch.float64).numpy()
    else:
        a = np.asarray(illum, dtype=float)
    a = np.nan_to_num(np.abs(a), nan=0.0, posinf=0.0, neginf=0.0)   # guard input

    if sigma and sigma > 0 and gaussian_filter is not None:
        a = gaussian_filter(a, sigma=sigma, mode="reflect")

    mx = float(a.max())
    if mx <= 0:                                  # no illumination -> no-op
        w = np.ones_like(a)
    else:
        precond = np.clip(a / mx, epsilon, 1.0)
        w = np.power(1.0 / precond, power)       # in [1, 1/epsilon**power]
        w = np.minimum(w, max_boost)             # bound the deep amplification
    w = np.nan_to_num(w, nan=1.0, posinf=max_boost, neginf=1.0)

    if is_torch:
        import torch
        return torch.as_tensor(w, dtype=illum.dtype, device=illum.device)
    return w

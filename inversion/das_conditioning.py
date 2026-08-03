"""DAS-specific data conditioning and gradient regularisation (Noe et al. 2025).

Noe, Tuinstra, Klaasen, Krischer & Fichtner (2025, GJI 244:1-17) close with a
warning this module exists to answer:

    "any inversion on fibre-optic data will need careful data processing and
     regularization that reflects its specific environment, including sources,
     noise, scattering effects and coupling of the fibre to the ground."

Their FWI applies three conditioning steps we did not have. All three are
OPTIONS that compose with any misfit and any optimizer, so they can be A/B'd
against our existing results rather than silently changing them.

  1. wavelength_span  - smooth the gradient at ~lambda/4, i.e. FREQUENCY-AWARE
     rather than a fixed Tikhonov weight. Coarse smoothing at low frequency,
     fine at high, matching the resolution the data can actually support.
  2. arrival_window   - keep only `pre` s before and `post` s after each trace's
     peak. Noe use 2 s / 4 s. Besides removing noise and coda, this drops the
     LATE arrivals that have accumulated the most phase error, so it is itself
     cycle-skip mitigation.
  3. channel_weights  - weight channels by amplitude so poorly illuminated ones
     do not dominate the gradient. On DAS this is not only about fibre coupling:
     BROADSIDE INSENSITIVITY means a channel whose fibre axis is perpendicular
     to the incoming wave records almost nothing by geometry alone, and that is
     present in synthetic data too.

CRITICAL DESIGN POINT for 2 and 3: both the window and the weights are derived
from the OBSERVED data and then held FIXED. Deriving them from the synthetic
would move the window and re-weight the traces at every iteration, so the misfit
would change shape as the model changed -- the inversion would be chasing a
moving objective rather than minimising a fixed one.
"""
import numpy as np
import torch

from ADFWI.fwi.misfit import Misfit


# --------------------------------------------------------------------------- #
# 1. wavelength-scaled gradient smoothing
# --------------------------------------------------------------------------- #
def wavelength_span(v_min, f_max, dx, fraction=0.25, min_span=1):
    """Gradient-smoothing span in CELLS for a lambda*`fraction` length scale.

    lambda = v_min / f_max is the shortest wavelength the data contains, so
    lambda/4 is the classic "do not pretend to resolve finer than this" scale.
    Returns a span for GradProcessor(grad_smooth=...), which passes it to
    smooth2d.

    Acoustic Marmousi (v_min 1500 m/s, dx 40 m): 6.25 Hz -> 1.5 -> 2 cells,
    3 Hz -> 3.1 -> 3 cells. Low bands get smoothed harder, as they should.

    RETURNS AN INT, and that is a hard requirement, not a nicety: smooth2d
    builds its kernel with `np.linspace(-2*span, 2*span, 2*span + 1)`, and
    linspace's `num` must be an integer. Returning the raw float raised
    "TypeError: 'float' object cannot be interpreted as an integer" inside the
    gradient processor and killed every conditioned cell on Bridges-2 at the
    first gradient. The float was unit-tested against the FORMULA and never
    once passed to its actual consumer -- see the smooth2d test.
    """
    if not (v_min > 0 and f_max > 0 and dx > 0):
        raise ValueError(f"need positive v_min/f_max/dx, got {v_min}/{f_max}/{dx}")
    return max(int(min_span), int(round(float(fraction) * (v_min / f_max) / dx)))


# --------------------------------------------------------------------------- #
# 2 + 3. trace conditioning, derived from the OBSERVED data
# --------------------------------------------------------------------------- #
def arrival_window(obs, dt, pre=2.0, post=4.0, time_axis=1, taper_s=0.25):
    """Cosine-tapered window around each trace's peak, from the OBSERVED data.

    Returns a mask shaped like `obs` (1 inside the window, 0 outside, cosine
    ramps of `taper_s` at the edges so the cut does not inject a step).
    """
    o = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
    nt = o.shape[time_axis]
    peak = o.abs().argmax(dim=time_axis, keepdim=True)          # per-trace onset
    idx = torch.arange(nt, device=o.device).reshape(
        [-1 if i == time_axis else 1 for i in range(o.ndim)])
    d = (idx - peak).to(torch.float64) * float(dt)              # seconds from peak
    ntap = max(float(taper_s), 1e-9)
    # 1 inside [-pre, post], cosine ramp over `taper_s` outside it
    left = torch.clamp((d + pre) / ntap + 1.0, 0.0, 1.0)
    right = torch.clamp((post - d) / ntap + 1.0, 0.0, 1.0)
    w = torch.minimum(left, right).clamp(0.0, 1.0)
    return (0.5 - 0.5 * torch.cos(np.pi * w)).to(o.dtype)       # smoothstep


def channel_weights(obs, time_axis=1, power=1.0, floor=0.05, normalize=True):
    """Per-channel weights from OBSERVED amplitude (broadside/coupling aware).

    weight ~ (channel RMS)**power, floored so a dead channel contributes ~0 but
    never NaN, and normalised to mean 1 so the loss scale is unchanged.

    >>> MEASURED WARNING -- DO NOT USE WITH A NON-QUADRATIC MISFIT AS-IS. <<<
    These weights are applied to the DATA (see ConditionedMisfit.forward), but
    Noe's intent is to weight each channel's CONTRIBUTION TO THE MISFIT. Those
    coincide only for a quadratic misfit. Pushing scaled data through a
    NONLINEAR misfit gives an uncontrolled effective exponent: measured on a
    weak channel with w=0.066, its share of the misfit is scaled by 0.066 under
    L2 but by 0.0003 under envelope^1.5 -- a 220x stronger suppression than
    intended.

    That is not academic. On the Bridges-2 A/B (2026-08-03) `switch+c` at the
    SKIP starter collapsed from 0.742 to 0.26-0.30, while `l2+c` was unharmed
    at 0.614: the switch's rescue IS its envelope stage, and weighting silently
    gutted it. Fixing this properly means weighting per-channel misfit
    contributions rather than the data -- which the black-box Misfit interface
    cannot express without one forward call per channel.
    """
    o = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
    rms = o.to(torch.float64).pow(2).mean(dim=time_axis, keepdim=True).sqrt()
    w = (rms / rms.max().clamp_min(1e-30)).pow(float(power))
    w = w.clamp_min(float(floor))
    if normalize:
        w = w / w.mean().clamp_min(1e-30)
    return w.to(o.dtype)


class ConditionedMisfit(Misfit):
    """Applies Noe-style conditioning, then delegates to any inner misfit.

    Composes with everything: BlendedMisfit, StagedMisfit, or a plain misfit.
    The window and weights are derived from the OBSERVED gather on every call
    (see _conditioning for why they are not cached), so the objective is fixed
    with respect to the MODEL while still tracking whichever shots a batch
    contains.
    """

    def __init__(self, inner, dt, window=None, weight=False,
                 window_pre=2.0, window_post=4.0, weight_power=1.0,
                 time_axis=1):
        super().__init__()
        self.inner, self.dt = inner, float(dt)
        self.window = bool(window)
        self.weight = bool(weight)
        self.window_pre, self.window_post = float(window_pre), float(window_post)
        self.weight_power = float(weight_power)
        self.time_axis = int(time_axis)

    def __getattr__(self, name):
        """Delegate anything we do not define to the WRAPPED misfit.

        Without this the wrapper is opaque, and wrapping SILENTLY BREAKS the
        controller: run_switch.py drives the misfit through `set_lambda`,
        `set_stage`, `active_name` and `lam`, none of which exist here, so every
        conditioned cell died with AttributeError at the first controller update
        -- after printing its setup line, which is exactly where the Bridges-2
        logs stopped. The class docstring claimed it "composes with everything";
        this is what makes that true rather than aspirational.

        Delegation rather than an explicit forwarding list because that API has
        grown repeatedly (set_lambda -> set_stage -> active_name -> lam), and an
        explicit list silently rots the next time it grows.

        Python calls __getattr__ only when normal lookup has already failed, so
        `forward` and the conditioning helpers still resolve here first. Reading
        `inner` out of __dict__ avoids recursing when it is not yet assigned.
        """
        if name.startswith("__") or "inner" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["inner"], name)

    @staticmethod
    def _obs_of(a, b):
        """The observed gather is the one WITHOUT the autograd graph."""
        if torch.is_tensor(a) and a.requires_grad:
            return b
        if torch.is_tensor(b) and b.requires_grad:
            return a
        return b

    def _conditioning(self, obs):
        # Derived from obs EVERY call, deliberately not cached across calls.
        # Caching looked like a cheap win but is wrong under shot batching: with
        # batch_size < n_shots each call sees a DIFFERENT subset of shots, whose
        # arrivals peak at different times, so a mask cached from the first batch
        # would window every later batch at the wrong times. Recomputing is an
        # argmax plus arithmetic -- negligible beside a wave propagation -- and
        # the property that matters (the window never moves as the MODEL changes)
        # comes from deriving it from obs, not from caching it.
        m = None
        if self.window:
            m = arrival_window(obs, self.dt, self.window_pre, self.window_post,
                               self.time_axis)
        if self.weight:
            w = channel_weights(obs, self.time_axis, self.weight_power)
            m = w if m is None else m * w
        return m

    def forward(self, a, b):
        m = self._conditioning(self._obs_of(a, b))
        if m is None:
            return self.inner.forward(a, b)
        m = m.to(a.device) if torch.is_tensor(a) else m
        # SAME conditioning on both sides -- anything else biases the misfit
        return self.inner.forward(a * m, b * m)

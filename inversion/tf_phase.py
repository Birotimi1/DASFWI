"""Fichtner time-frequency PHASE misfit (Fichtner et al. 2008, GJI 175:665-685).

The literature's standard single-misfit answer to cycle skipping, and what BOTH
published DAS FWI studies use -- so it is the direct COMPETITOR to our adaptive
switch, not merely another amplitude fix.

METHOD. Take the Gabor (windowed Fourier) transform of each trace,

    u~(t, w) = INT u(tau) h*(tau - t) e^{-i w tau} dtau ,   h = Gaussian window

write it in polar form u~ = |u~| e^{i phi}, and penalise the PHASE only:

    E_p^2 = INT INT W(t,w)^2 [phi_syn(t,w) - phi_obs(t,w)]^2 dt dw

Amplitude is discarded entirely, which is why it suits field DAS: absolute
amplitudes depend on fibre coupling, and our FORGE QC (inversion/das_qc.py)
showed coupling there acts as a SCALAR -- so a phase-only measurement is exactly
the right observable at that site rather than a compromise.

>>> THREE THINGS THAT MAKE OR BREAK THIS, ALL LEARNED THE HARD WAY <<<

1. NEVER difference two wrapped phases. `angle(syn) - angle(obs)` jumps by 2*pi
   whenever either crosses the branch cut, so two nearly-identical traces can
   report a phase error of ~2*pi. The difference is taken from the CROSS-
   SPECTRUM instead, arg(u~_syn * conj(u~_obs)), which lands in (-pi, pi] by
   construction and has no spurious jumps.

2. PHASE IS MEANINGLESS WHERE THERE IS NO ENERGY, and worse than meaningless
   for the adjoint: d(arg z)/dz ~ 1/|z|, so empty (t,w) cells contribute
   unbounded gradient. Cells below `amp_floor` x max|u~_obs| are hard-masked
   AND their atan2 arguments are replaced by safe constants, so no NaN can be
   produced and then propagated by autograd. (A related unsigned-phase mistake
   already produced a false shape-distortion verdict on real FORGE data -- see
   inversion/das_qc.py. Phase statistics punish carelessness quietly.)

3. IT DOES NOT ABOLISH CYCLE SKIPPING, and any claim that it does is wrong.
   For a time shift dt the phase difference is dphi = -w * dt (SIGN: with the
   e^{-i w tau} transform convention a DELAYED synthetic gets a NEGATIVE phase;
   use `time_shift` rather than reading dphi/w directly). It WRAPS when
   |w dt| > pi, i.e. |dt| > T/2 -- the identical cycle-skip condition. What
   TF-phase buys is that the LOW-frequency cells of the plane wrap later, so
   they still supply an unwrapped, quasi-convex measurement when the high ones
   do not. That is the same mechanism as multiscale, obtained without running a
   cascade. `wrap_limit_s` reports it explicitly per frequency.

   CONSEQUENCE FOR OUR MARMOUSI TEST, stated in advance rather than after the
   fact: the integrated Ricker at F0=5 spans 3.0-6.25 Hz = 1.06 OCTAVES, so
   there is very little low-frequency plane to lean on, exactly the deficiency
   that made a multiscale cascade useless on this dataset. If TF-phase
   underperforms on Marmousi that is the FIRST hypothesis to test, not evidence
   the method is bad -- FORGE has 4.4 octaves.

Stateless (no iteration counter), so it composes inside BlendedMisfit /
StagedMisfit -- see inversion/adaptive_misfit._reject_stateful.
"""
import numpy as np
import torch

from ADFWI.fwi.misfit import Misfit


def gaussian_window(n, sigma_frac=0.3, device=None, dtype=torch.float32):
    """Gaussian window of length n, std = sigma_frac * n.

    Fichtner uses a Gaussian because it is the unique window minimising the
    time-bandwidth product, i.e. it gives the sharpest joint (t, w) localisation
    available -- the whole method rests on reading a phase at a place in time
    AND frequency, so window choice is not cosmetic.
    """
    k = torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0
    return torch.exp(-0.5 * (k / (sigma_frac * n)) ** 2)


def _as_traces(x):
    """[S, nt, C] or [nt, C] or [nt] -> (2-D [N, nt] view, restore-shape)."""
    if x.dim() == 3:                      # [S, nt, C] -> [S*C, nt]
        S, nt, C = x.shape
        return x.permute(0, 2, 1).reshape(S * C, nt), (S, C, nt)
    if x.dim() == 2:                      # [nt, C] -> [C, nt]
        nt, C = x.shape
        return x.transpose(0, 1), (1, C, nt)
    return x.reshape(1, -1), (1, 1, x.shape[0])


def gabor(x, n_fft, hop, window):
    """Differentiable Gabor transform of [N, nt] -> complex [N, F, T]."""
    return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, center=True, pad_mode="constant",
                      normalized=False, onesided=True, return_complex=True)


def wrap_limit_s(freqs):
    """Largest time shift each frequency can express before its phase WRAPS.

    |dphi| = |w dt| <= pi  ->  |dt| <= 1 / (2 f).  Identical to the cycle-skip
    threshold T/2, which is the point: TF-phase inherits the skip limit
    frequency by frequency, and only the low rows stay unwrapped.
    """
    f = np.asarray(freqs, float)
    with np.errstate(divide="ignore"):
        return np.where(f > 0, 1.0 / (2.0 * np.maximum(f, 1e-30)), np.inf)


class Misfit_TFPhase(Misfit):
    """Fichtner TF-phase misfit. Stateless; safe inside BlendedMisfit.

    dt          propagator time step [s]
    win_s       Gabor window length [s]. Must span a few periods of the LOWEST
                frequency of interest or that row is unresolved; but a window
                approaching the record length destroys time localisation. On a
                narrowband low-frequency synthetic this trade-off is tight.
    f_min/f_max restrict the plane to the band that carries signal. Rows outside
                are dropped entirely -- they are noise, and noise phase is
                uniformly distributed, so including them adds variance and no
                information.
    amp_floor   fraction of max|u~_obs| below which a cell is masked out.
    """

    def __init__(self, dt, win_s=1.0, hop_frac=0.25, sigma_frac=0.3,
                 f_min=None, f_max=None, amp_floor=0.05, weight_power=1.0,
                 eps=1e-12):
        super().__init__()
        self.dt = float(dt)
        self.win_s = float(win_s)
        self.hop_frac = float(hop_frac)
        self.sigma_frac = float(sigma_frac)
        self.f_min, self.f_max = f_min, f_max
        self.amp_floor = float(amp_floor)
        self.weight_power = float(weight_power)
        self.eps = float(eps)
        self._cache = {}

    # ------------------------------------------------------------------ #
    def _plan(self, nt, device, dtype):
        """n_fft/hop/window/freqs for this trace length (cached per shape)."""
        key = (nt, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        n_fft = 1 << max(4, int(np.ceil(np.log2(max(8, self.win_s / self.dt)))))
        n_fft = min(n_fft, 1 << int(np.floor(np.log2(max(8, nt)))))
        hop = max(1, int(n_fft * self.hop_frac))
        win = gaussian_window(n_fft, self.sigma_frac, device=device, dtype=dtype)
        freqs = np.fft.rfftfreq(n_fft, self.dt)
        keep = np.ones(freqs.shape, bool)
        keep[0] = False                                  # DC carries no phase
        if self.f_min is not None:
            keep &= freqs >= self.f_min
        if self.f_max is not None:
            keep &= freqs <= self.f_max
        if keep.sum() < 2:                               # band too narrow: keep all but DC
            keep = np.ones(freqs.shape, bool); keep[0] = False
        idx = torch.as_tensor(np.flatnonzero(keep), device=device)
        plan = (n_fft, hop, win, freqs[keep], idx)
        self._cache[key] = plan
        return plan

    def _tf(self, x2d, plan):
        n_fft, hop, win, _, idx = plan
        return gabor(x2d, n_fft, hop, win).index_select(1, idx)

    # ------------------------------------------------------------------ #
    def phase_difference(self, syn, obs):
        """(dphi, weight, freqs) on the masked TF plane. dphi in (-pi, pi].

        Exposed separately from `forward` because it is the quantity to inspect
        when a result looks odd: dphi/w should be a flat estimate of the time
        shift wherever the weight is non-zero.
        """
        a, _ = _as_traces(syn)
        b, _ = _as_traces(obs.to(syn.dtype) if torch.is_tensor(obs)
                          else torch.as_tensor(obs, dtype=syn.dtype))
        b = b.to(a.device)
        plan = self._plan(a.shape[-1], a.device, a.dtype)
        A, B = self._tf(a, plan), self._tf(b, plan)

        # --- 1. phase difference from the CROSS-SPECTRUM, never by differencing
        #        two wrapped angles (which jumps 2*pi at the branch cut).
        cross = A * torch.conj(B)
        re, im = cross.real, cross.imag

        # --- 2. mask empty cells: phase is undefined there and its derivative
        #        ~1/|z| is unbounded, so they would dominate the adjoint.
        env = B.abs()
        thr = self.amp_floor * env.amax(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        keep = (env > thr) & (cross.abs() > self.eps)
        # substitute SAFE constants inside the masked cells so atan2 cannot make
        # a NaN that autograd would then propagate into every parameter.
        re = torch.where(keep, re, torch.ones_like(re))
        im = torch.where(keep, im, torch.zeros_like(im))
        dphi = torch.atan2(im, re)                       # = 0 where masked

        w = (env / env.amax(dim=(1, 2), keepdim=True).clamp_min(self.eps)
             ).pow(self.weight_power) * keep.to(env.dtype)
        return dphi, w, plan[3]

    def time_shift(self, syn, obs):
        """Weighted estimate of the syn-minus-obs time shift [s], per gather.

        THE SIGN LIVES HERE so no caller has to rederive it. With the
        e^{-i w tau} convention a synthetic DELAYED by dt carries phase
        -w*dt, so dt = -dphi/w. Reading dphi/w directly returns the shift with
        the wrong sign -- which is exactly what the first version of the test
        caught (recovered -0.00399 s for a true +0.00400 s).

        Only meaningful while |dt| < 1/(2 f) for the rows carrying the weight;
        beyond that the phase has wrapped and this reports the folded value.
        Useful as a standalone traveltime diagnostic, not just for debugging.
        """
        dphi, w, freqs = self.phase_difference(syn, obs)
        om = torch.as_tensor(2 * np.pi * freqs, dtype=dphi.dtype,
                             device=dphi.device).reshape(1, -1, 1)
        num = (w ** 2 * dphi * om).sum()
        den = (w ** 2 * om ** 2).sum().clamp_min(self.eps)
        return -(num / den)                       # minus: see the note above

    def forward(self, syn, obs):
        dphi, w, _ = self.phase_difference(syn, obs)
        num = (w * dphi).pow(2).sum()
        den = w.pow(2).sum().clamp_min(self.eps)
        return num / den                                  # mean squared phase error

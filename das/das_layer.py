"""Differentiable DAS observation operator (T3).

Implements E3, the exact endpoint-difference form of the gauge-averaged axial
strain rate for a straight fiber, as a ``torch.nn.Module`` consuming ADFWI's
velocity records:

    E3: eps_bar_dot(s_c, t) = [ v_e(s_c + l/2, t) - v_e(s_c - l/2, t) ] / l,
        v_e = e_x * v_x + e_z * v_z.

Inputs are the propagator's receiver gathers ``u`` (horizontal velocity) and
``w`` (vertical velocity), each ``[n_shot, nt, n_rcv]`` with receivers laid out
in ``FiberGeometry.rcv_pos`` order. Output is the channel gather
``[n_shot, nt, C]`` of strain rate [1/s] (or strain via E4 causal cumsum).

The units check (spec E0): [m/s]/[m] = [1/s] — the operator output is strain
rate natively; no time derivative or integral anywhere in the chain.

The adjoint is never hand-coded: the layer is a pure differentiable tensor
expression, so ``loss.backward()`` generates it (a +-(1/l)*S(t)*e force couple
at the gauge ends in acoustic). Verified by the dot-product test in T4.

Everything is float64 by default (the registered ``e_x``/``e_z`` buffers), but
``forward`` follows the dtype and device of its inputs, so float32 works too.
"""

from __future__ import annotations

import torch

from .geometry import FiberGeometry


class DASObservationLayer(torch.nn.Module):
    """E3 endpoint-difference DAS observation operator (differentiable).

    Args:
        geometry: a constructed :class:`~das.geometry.FiberGeometry`. Provides
            the endpoint index maps (``idx_minus``/``idx_plus`` into the
            deduplicated receiver list), the channel unit tangents, and ``l``.
        output: ``"strain_rate"`` (native, default) or ``"strain"`` (E4 causal
            cumulative sum of strain rate times ``dt``).
        dt: time-sample interval in seconds. Required only when
            ``output == "strain"``; ignored otherwise.

    Buffers (registered so they follow ``module.to(device)``):
        idx_plus, idx_minus: LongTensor ``[C]`` — receiver-column indices of
            each channel's ``+l/2`` / ``-l/2`` gauge endpoint.
        e_x, e_z: float64 ``[C]`` — channel unit tangent components (all
            ``(0, 1)`` for the vertical constructor). Cast to the input dtype
            inside ``forward`` so outputs follow input precision.
    """

    _OUTPUTS = ("strain_rate", "strain")

    def __init__(self, geometry: FiberGeometry, output: str = "strain_rate",
                 dt: float = None):
        super().__init__()
        if output not in self._OUTPUTS:
            raise ValueError(
                f"output must be one of {self._OUTPUTS}, got {output!r}"
            )
        if output == "strain" and dt is None:
            raise ValueError(
                'output="strain" requires dt (time-sample interval in s) '
                "for the E4 causal cumsum; pass dt=... to __init__"
            )

        self.output = output
        self.l = float(geometry.l)
        self.dt = None if dt is None else float(dt)
        self.n_rcv = geometry.n_rcv

        self.register_buffer(
            "idx_plus", geometry.idx_plus.clone().to(torch.long))
        self.register_buffer(
            "idx_minus", geometry.idx_minus.clone().to(torch.long))
        self.register_buffer(
            "e_x", torch.as_tensor(geometry.e_x, dtype=torch.float64).clone())
        self.register_buffer(
            "e_z", torch.as_tensor(geometry.e_z, dtype=torch.float64).clone())

    def forward(self, rcv_u: torch.Tensor, rcv_w: torch.Tensor) -> torch.Tensor:
        """Apply E3 (and E4 if ``output == "strain"``).

        Args:
            rcv_u: horizontal velocity records ``[S, T, R]`` with
                ``R == len(geometry.rcv_pos)``.
            rcv_w: vertical velocity records, same shape.

        Returns:
            ``[S, T, C]`` tensor of gauge-averaged axial strain rate (or
            strain), in the dtype and on the device of the inputs.
        """
        # Cast tangent buffers to the input dtype/device so the output follows
        # the inputs (buffers are stored float64; inputs may be float32).
        e_x = self.e_x.to(dtype=rcv_u.dtype, device=rcv_u.device)
        e_z = self.e_z.to(dtype=rcv_w.dtype, device=rcv_w.device)

        v_e_plus = e_x * rcv_u[..., self.idx_plus] + e_z * rcv_w[..., self.idx_plus]
        v_e_minus = e_x * rcv_u[..., self.idx_minus] + e_z * rcv_w[..., self.idx_minus]
        out = (v_e_plus - v_e_minus) / self.l  # [S, T, C]  (E3)

        if self.output == "strain":
            # E4: eps_bar(t_n) = dt * sum_{m<=n} eps_bar_dot(t_m) (causal cumsum
            # over the time axis).
            out = torch.cumsum(out, dim=1) * self.dt

        return out

    def channel_mask(self, rcv_mask: torch.Tensor) -> torch.Tensor:
        """Map a receiver-level mask to a channel-level mask.

        A channel is alive iff BOTH of its gauge endpoints are alive.

        Args:
            rcv_mask: boolean ``[S, R]``.

        Returns:
            boolean ``[S, C]``.
        """
        return rcv_mask[:, self.idx_plus] & rcv_mask[:, self.idx_minus]

    def forward_pointwise(self, wavefield_vx, wavefield_vz, N_g: int = 5):
        """Pointwise gauge-integrated operator (E5) — Phase F, NOT implemented.

        E5 (pointwise mode — curved fiber / elastic):

            eps_bar_dot(s_c, t) = sum_g w_g * [ e_x^2 * dvx/dx
                                              + e_x*e_z * (dvx/dz + dvz/dx)
                                              + e_z^2 * dvz/dz ]_{s_{c,g}}

        with gauge points s_{c,g} = s_c + (g/(N_g - 1) - 1/2) * l (inclusive
        linspace; N_g = 1 -> single center point; see ``das.geometry.
        gauge_points``), spatial derivatives by FD stencils on the velocity
        wavefield, bilinear interpolation to the gauge points, and
        trapezoid-rule weights w_g normalized to unit sum.

        WARNING (curvature): never implement the curved case by differencing
        v_e along the fiber. The identity d(v.e)/ds = e^T eps_dot e + v.(de/ds)
        shows the along-fiber derivative of v_e contains a curvature term
        v.(de/ds) in addition to the axial strain rate e^T eps_dot e; that
        curvature term is separate physical sensitivity and must not be
        silently included in the observation operator.

        Args:
            wavefield_vx: horizontal velocity wavefield snapshots (Phase F
                signature to be finalized; expected [S, T, nz, nx]).
            wavefield_vz: vertical velocity wavefield snapshots, same shape.
            N_g: number of gauge integration points.

        Raises:
            NotImplementedError: always, until Phase F.
        """
        raise NotImplementedError(
            "Pointwise/elastic gauge integration (E5) is Phase F. "
            "See this method's docstring for the formula and the curvature "
            "warning; use the E3 endpoint-difference forward() for straight "
            "fibers."
        )

    def extra_repr(self) -> str:
        return (f"C={self.idx_plus.numel()}, n_rcv={self.n_rcv}, "
                f"l={self.l}, output={self.output!r}, dt={self.dt}")

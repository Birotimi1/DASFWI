"""Differentiable E5 pointwise DAS observation operator (deviated/curved fiber).

Consumes the propagator's velocity records ``u`` (=v_x) and ``w`` (=v_z) at the
grid nodes emitted by :class:`PointwiseFiberGeometry`. Each spatial derivative
at a gauge point is a precomputed sparse linear functional (bilinear interp +
central FD folded together): ``d(field)/dx = sum_k coeff_ddx * field[node]``,
likewise ``d/dz``. From the four derivatives it forms the axial strain rate and
trapezoid-sums over the gauge:

    dvx/dx = <coeff_ddx, u>   dvz/dx = <coeff_ddx, w>
    dvx/dz = <coeff_ddz, u>   dvz/dz = <coeff_ddz, w>
    axial  = e_x^2 dvx/dx + e_x e_z (dvx/dz + dvz/dx) + e_z^2 dvz/dz
    eps_bar_dot = sum_g w_g * axial_g                                   (E5)

The operator NEVER differences velocity along the fiber, so the curvature
sensitivity is excluded by construction (see PointwiseFiberGeometry). The whole
thing is out-of-place tensor algebra, so ``loss.backward()`` builds the adjoint.

Cross-validated (2-factor) against Noe's pyber operator
(``pyber.utils.project_strain_tensor_on_unit_vector``, which computes the axial
strain as ``unit_vector @ (unit_vector @ strain_tensor)`` = ``e^T eps e``, then
gauge-averages via ``pyber.gauge_length.GaugeLength``): our axial contraction
``e_x^2 eps_xx + 2 e_x e_z eps_xz + e_z^2 eps_zz`` and trapezoid gauge weights
reproduce it to machine precision on arbitrary fibre orientations
(``tests/test_pointwise.py::test_matches_noe_pyber_contraction``). Both project
the strain tensor pointwise onto the local tangent and gauge-average, so both
exclude the curvature term identically. Noe works with strain from a strain-
capable solver; we form strain RATE from velocity-record derivatives (the DAS
observable is strain rate), the natural ADFWI-side equivalent.
"""

from __future__ import annotations

import torch

from .geometry_pointwise import PointwiseFiberGeometry


class DASPointwiseLayer(torch.nn.Module):
    """E5 pointwise gauge-integrated axial strain-rate operator.

    Args:
        geometry: a constructed :class:`PointwiseFiberGeometry`.
        output: ``"strain_rate"`` (default) or ``"strain"`` (E4 causal cumsum
            of strain rate times ``dt``; requires ``dt``).
        dt: time step [s], required only for ``output == "strain"``.
    """

    _OUTPUTS = ("strain_rate", "strain")

    def __init__(self, geometry: PointwiseFiberGeometry,
                 output: str = "strain_rate", dt: float = None):
        super().__init__()
        if output not in self._OUTPUTS:
            raise ValueError(f"output must be one of {self._OUTPUTS}")
        if output == "strain" and dt is None:
            raise ValueError('output="strain" requires dt')
        self.output = output
        self.dt = None if dt is None else float(dt)
        self.n_rcv = geometry.n_rcv

        self.register_buffer("node_idx", geometry.node_idx.clone())
        self.register_buffer(
            "coeff_ddx", torch.as_tensor(geometry.coeff_ddx, dtype=torch.float64))
        self.register_buffer(
            "coeff_ddz", torch.as_tensor(geometry.coeff_ddz, dtype=torch.float64))
        self.register_buffer(
            "gauge_ex", torch.as_tensor(geometry.gauge_ex, dtype=torch.float64))
        self.register_buffer(
            "gauge_ez", torch.as_tensor(geometry.gauge_ez, dtype=torch.float64))
        self.register_buffer(
            "gauge_w", torch.as_tensor(geometry.gauge_w, dtype=torch.float64))

    def forward(self, rcv_u: torch.Tensor, rcv_w: torch.Tensor) -> torch.Tensor:
        """rcv_u, rcv_w: ``[S, T, R]`` -> ``[S, T, C]`` strain rate (or strain).

        ``node_idx`` is ``[C, N_g, K]``; gathering maps ``[S, T, R]`` to
        ``[S, T, C, N_g, K]`` and the coefficient contraction over K gives each
        derivative at every gauge point.
        """
        cast = dict(dtype=rcv_u.dtype, device=rcv_u.device)
        ex = self.gauge_ex.to(**cast)
        ez = self.gauge_ez.to(**cast)
        gw = self.gauge_w.to(**cast)
        cddx = self.coeff_ddx.to(**cast)
        cddz = self.coeff_ddz.to(**cast)

        u = rcv_u[..., self.node_idx]            # [S, T, C, N_g, K]
        w = rcv_w[..., self.node_idx]
        dvx_dx = (u * cddx).sum(-1)              # [S, T, C, N_g]
        dvz_dx = (w * cddx).sum(-1)
        dvx_dz = (u * cddz).sum(-1)
        dvz_dz = (w * cddz).sum(-1)

        axial = (ex * ex * dvx_dx
                 + ex * ez * (dvx_dz + dvz_dx)
                 + ez * ez * dvz_dz)             # [S, T, C, N_g]
        out = (axial * gw).sum(dim=-1)           # trapezoid over gauge -> [S,T,C]

        if self.output == "strain":
            out = torch.cumsum(out, dim=1) * self.dt
        return out

    def extra_repr(self) -> str:
        return (f"C={self.node_idx.shape[0]}, N_g={self.node_idx.shape[1]}, "
                f"n_rcv={self.n_rcv}, output={self.output!r}")

"""Pointwise gauge geometry for deviated / curved fibers (E5, Phase F).

E3 (``geometry.py`` + ``das_layer.py``) is the EXACT operator for a straight
fiber: the gauge integral of the axial strain rate telescopes to an endpoint
difference of the axial velocity. That telescoping requires a CONSTANT tangent
``e``. For a curved fiber ``e`` varies along the arc, and

    d(v.e)/ds = e^T eps_dot e + v.(de/ds)

so differencing ``v_e`` along the fiber would fold the curvature term
``v.(de/ds)`` — a separate physical sensitivity — into the measurement. E5
instead evaluates the axial strain rate POINTWISE and quadratures it over the
gauge, never differencing along the fiber:

    eps_bar_dot(s_c, t) = sum_g w_g * [ e_x^2 dvx/dx
                                      + e_x e_z (dvx/dz + dvz/dx)
                                      + e_z^2 dvz/dz ]_{s_{c,g}}

with gauge points at arc length ``s_{c,g} = s_c + (g/(N_g-1) - 1/2) l``,
trapezoid weights ``w_g`` (``geometry.gauge_points``), and the tangent
``e = (e_x, e_z)`` evaluated at each gauge point.

This module builds the geometry the pointwise layer consumes: channel centers
and tangents along an arbitrary polyline trajectory, the gauge points, and —
because the operator needs spatial velocity derivatives — a linear functional
over grid-node velocity RECORDS at each gauge point (records-based, so ADFWI's
propagator stays untouched). Each spatial derivative at an off-node gauge point
is a BILINEAR interpolation of the four surrounding nodes' central-difference
derivatives; because both bilinear interpolation and central differencing are
linear, the whole per-derivative operator collapses to one sparse set of
``(node, coefficient)`` pairs. This keeps ``N_g`` meaningful even when the gauge
sub-interval is finer than the grid (naive nearest-node snapping would collapse
several gauge points onto one node); it is exact for velocity fields linear in
space.

A straight (vertical OR tilted) fiber is the two-vertex special case; the E5
result then converges to the exact E3 value as ``N_g`` grows and the grid
refines. E5's real purpose is the curved case E3 cannot represent.
"""

from __future__ import annotations

import numpy as np
import torch

from .geometry import gauge_points


def _arc_table(vertices):
    """Cumulative arc length at each polyline vertex."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 2:
        raise ValueError("vertices must be [[x, z], ...] with >= 2 points")
    seg = np.diff(v, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    if np.any(seglen <= 0):
        raise ValueError("polyline has a zero-length segment")
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return v, s, seglen


def _position_tangent(v, s_table, seglen, s):
    """(x, z, e_x, e_z) at arc length ``s`` along the polyline.

    Tangent is the unit direction of the segment containing ``s``. Beyond the
    fiber ends (``s < 0`` or ``s > total``) the position is EXTRAPOLATED along
    the end segment's tangent, NOT clamped: a channel's gauge window near the
    fiber top/bottom extends l/2 past the last channel centre, and those gauge
    points are real spatial locations (the wavefield exists there). Clamping
    would fold them onto the endpoint and bias the gauge quadrature (it shifts
    the top/bottom channels' centroid by ~l/4). The user must leave >= l/2 grid
    margin at the fibre ends (enforced by the stencil edge check).
    """
    total = s_table[-1]
    if s <= 0.0:
        e = (v[1] - v[0]) / seglen[0]
        p = v[0] + e * s                       # s <= 0 -> extrapolate backward
    elif s >= total:
        e = (v[-1] - v[-2]) / seglen[-1]
        p = v[-1] + e * (s - total)            # extrapolate forward
    else:
        k = int(np.searchsorted(s_table, s, side="right") - 1)
        k = max(0, min(k, len(seglen) - 1))
        e = (v[k + 1] - v[k]) / seglen[k]
        p = v[k] + e * (s - s_table[k])
    return p[0], p[1], e[0], e[1]


class PointwiseFiberGeometry:
    """Fiber geometry for the E5 pointwise operator (deviated / curved).

    Args:
        vertices: polyline ``[[x, z], ...]`` in metres (>= 2 points). A
            vertical fiber is ``[[x, z_top], [x, z_bot]]``; a tilted straight
            fiber offsets the toe; a curved fiber adds intermediate vertices.
        n_channels: number of channel centres, spaced ``dch`` along arc length
            from the top vertex.
        dch: channel spacing along the fibre [m].
        l: gauge length [m].
        dx, dz: inversion grid spacing [m].
        N_g: gauge integration points (odd >= 3 recommended; 1 -> single
            centre point, no gauge averaging).

    Attributes (after construction):
        channel_x, channel_z [C]: channel-centre coordinates.
        e_x, e_z [C]: channel-centre unit tangent.
        rcv_pos: list of unique ``(z_index, x_index)`` grid nodes (same
            convention as ``FiberGeometry``) for the ADFWI receivers.
        node_idx: LongTensor ``[C, N_g, K]`` — indices into ``rcv_pos`` of the
            nodes each gauge point's derivative functionals touch (padded).
        coeff_ddx, coeff_ddz: float64 ``[C, N_g, K]`` — linear coefficients so
            that ``d/dx(field) = sum_k coeff_ddx * field[node_idx]`` (and
            likewise for ``d/dz``), combining bilinear interp + central FD.
        gauge_ex, gauge_ez: float64 ``[C, N_g]`` tangent at each gauge point.
        gauge_w: float64 ``[C, N_g]`` trapezoid weights (rows sum to 1).
    """

    def __init__(self, vertices, n_channels, dch, l, dx=5.0, dz=5.0, N_g=5):
        self.l = float(l)
        self.dx = float(dx)
        self.dz = float(dz)
        self.N_g = int(N_g)
        self.n_channels = int(n_channels)
        self.dch = float(dch)
        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1")

        v, s_table, seglen = _arc_table(vertices)
        offs, w = gauge_points(self.N_g)        # offsets in [-1/2, 1/2] of l

        chan = [_position_tangent(v, s_table, seglen, c * self.dch)
                for c in range(self.n_channels)]
        self.channel_x = np.array([c[0] for c in chan], dtype=np.float64)
        self.channel_z = np.array([c[1] for c in chan], dtype=np.float64)
        self.e_x = np.array([c[2] for c in chan], dtype=np.float64)
        self.e_z = np.array([c[3] for c in chan], dtype=np.float64)

        rcv_pos = []
        index_of = {}

        def _register(iz, ix):
            key = (int(iz), int(ix))
            pos = index_of.get(key)
            if pos is None:
                pos = len(rcv_pos)
                index_of[key] = pos
                rcv_pos.append(key)
            return pos

        C, Ng = self.n_channels, self.N_g
        gauge_ex = np.empty((C, Ng), np.float64)
        gauge_ez = np.empty((C, Ng), np.float64)
        # per (c, g): dict node_key -> [coeff_ddx, coeff_ddz]
        per_gauge = [[None] * Ng for _ in range(C)]

        for c in range(C):
            s_c = c * self.dch
            for g in range(Ng):
                gx, gz, ex, ez = _position_tangent(
                    v, s_table, seglen, s_c + offs[g] * self.l)
                gauge_ex[c, g], gauge_ez[c, g] = ex, ez
                per_gauge[c][g] = self._stencil(gx, gz)

        # flatten to padded [C, Ng, K] arrays
        K = max(len(per_gauge[c][g]) for c in range(C) for g in range(Ng))
        node_idx = np.zeros((C, Ng, K), np.int64)
        coeff_ddx = np.zeros((C, Ng, K), np.float64)
        coeff_ddz = np.zeros((C, Ng, K), np.float64)
        for c in range(C):
            for g in range(Ng):
                for k, (key, cx, cz) in enumerate(per_gauge[c][g]):
                    node_idx[c, g, k] = _register(*key)
                    coeff_ddx[c, g, k] = cx
                    coeff_ddz[c, g, k] = cz

        self.rcv_pos = rcv_pos
        self.node_idx = torch.as_tensor(node_idx, dtype=torch.long)
        self.coeff_ddx = coeff_ddx
        self.coeff_ddz = coeff_ddz
        self.gauge_ex = gauge_ex
        self.gauge_ez = gauge_ez
        self.gauge_w = np.broadcast_to(w, (C, Ng)).copy()

    def _stencil(self, gx, gz):
        """Bilinear-interp + central-FD functional at continuous (gx, gz).

        Returns a list of ``((iz, ix), coeff_ddx, coeff_ddz)`` so that
        ``d(field)/dx ~ sum coeff_ddx * field[node]`` and likewise ddz.
        """
        fx_full = gx / self.dx
        fz_full = gz / self.dz
        ix0, iz0 = int(np.floor(fx_full)), int(np.floor(fz_full))
        fx, fz = fx_full - ix0, fz_full - iz0
        if ix0 < 1 or iz0 < 1:
            raise ValueError(
                "gauge stencil reaches the grid edge (ix/iz < 1); move the "
                "fibre into the interior or enlarge the grid")
        corners = [(iz0, ix0, (1 - fx) * (1 - fz)),
                   (iz0, ix0 + 1, fx * (1 - fz)),
                   (iz0 + 1, ix0, (1 - fx) * fz),
                   (iz0 + 1, ix0 + 1, fx * fz)]
        acc = {}   # (iz, ix) -> [ddx, ddz]
        for (iz, ix, wc) in corners:
            if wc == 0.0:
                continue
            # central difference at this corner, weighted by the bilinear wc
            for (nz, nx, coef, axis) in (
                    (iz, ix + 1, +wc / (2 * self.dx), 0),
                    (iz, ix - 1, -wc / (2 * self.dx), 0),
                    (iz + 1, ix, +wc / (2 * self.dz), 1),
                    (iz - 1, ix, -wc / (2 * self.dz), 1)):
                slot = acc.setdefault((nz, nx), [0.0, 0.0])
                slot[axis] += coef
        return [(key, cc[0], cc[1]) for key, cc in acc.items()]

    @property
    def n_rcv(self):
        return len(self.rcv_pos)

    @property
    def C(self):
        return self.n_channels

    @classmethod
    def vertical(cls, x_well, z_top, z_bot, n_channels, dch, l,
                 dx=5.0, dz=5.0, N_g=5):
        """Convenience: a vertical fiber (the E3-exact special case)."""
        return cls([[x_well, z_top], [x_well, z_bot]], n_channels, dch, l,
                   dx=dx, dz=dz, N_g=N_g)

    @classmethod
    def tilted(cls, x_top, z_top, dip_deg, length, n_channels, dch, l,
               dx=5.0, dz=5.0, N_g=5):
        """Convenience: a straight fiber dipping ``dip_deg`` from vertical
        (0 = vertical). Toe offset in +x."""
        th = np.radians(dip_deg)
        x_bot = x_top + length * np.sin(th)
        z_bot = z_top + length * np.cos(th)
        return cls([[x_top, z_top], [x_bot, z_bot]], n_channels, dch, l,
                   dx=dx, dz=dz, N_g=N_g)

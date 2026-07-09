"""DAS fiber geometry (T2).

Turns a fiber trajectory into the quantities the differentiable observation
operator (T3, ``das/das_layer.py``) needs:

* channel centers and unit tangents,
* the *deduplicated* list of gauge-endpoint receiver positions (grid indices)
  handed to the ADFWI propagator, and
* per-channel index maps (``idx_minus``, ``idx_plus``) into that receiver list
  so channel ``c``'s two gauge endpoints are ``rcv_pos[idx_minus[c]]`` and
  ``rcv_pos[idx_plus[c]]``.

Only the vertical FORGE fiber is implemented here (constructor
``FiberGeometry``). The general polyline / curved-fiber case is Phase F; the
gauge-point convention it will use is provided now as :func:`gauge_points`.

Operator math (see build spec section 1, E1--E3):

    E1 (vertical fiber): channel centers x_c = (x_well, z_c),
        z_c = z_top + c * dch, tangent e = (0, 1),
        endpoints x_c+- = (x_well, z_c +- l/2).
    E3 (straight fiber): the gauge-averaged axial strain rate telescopes to an
        endpoint difference,
            eps_bar_dot(s_c, t) = [ v_e(s_c + l/2, t) - v_e(s_c - l/2, t) ] / l,
        with v_e = e_x * v_x + e_z * v_z. For a vertical fiber v_e = v_z (the
        propagator's ``w`` records).

Everything is float64. Only the two integer index maps are torch (LongTensor);
they exist so the das_layer can gather velocity records by receiver column.
"""

from __future__ import annotations

import numpy as np
import torch


def gauge_points(N_g):
    """General gauge-point convention (E2), verified against pyber.

    Returns ``N_g`` sample offsets along the fiber, expressed as *fractions of
    the gauge length* ``l`` in ``[-1/2, +1/2]``, together with trapezoid-rule
    quadrature weights normalized to unit sum. Multiply the offsets by ``l`` to
    get physical distances from the channel center (matching pyber's
    ``distances_in_m``).

    Convention (spec section 1):
        s_{c,g} = s_c + (g/(N_g - 1) - 1/2) * l,   g = 0 ... N_g - 1
    i.e. an inclusive ``linspace(-1/2, +1/2, N_g)``. ``N_g = 1`` collapses to a
    single center point with unit weight.

    Weights: the trapezoid rule (pyber ``GaugeLength.get_integration_points_and
    _weights``): each interior point gets a half-contribution from each adjacent
    interval, endpoints get one half-interval; here normalized so the weights
    sum to 1. For ``N_g = 2`` this yields ``[1/2, 1/2]``.

    Args:
        N_g: number of gauge integration points (>= 1).

    Returns:
        (offsets, weights): two float64 ``ndarray`` of shape ``[N_g]``.
        ``offsets`` in ``[-1/2, +1/2]`` (fractions of ``l``), ``weights`` sum to 1.

    Note:
        This is the pointwise/curved-fiber convention used in Phase F (E5). The
        vertical acoustic operator implemented by :class:`FiberGeometry` uses the
        exact endpoint-difference form (E3), which is the ``N_g = 2`` special
        case with signed weights ``[-1/l, +1/l]`` (pyber ``simple_difference``).
    """
    N_g = int(N_g)
    if N_g < 1:
        raise ValueError("N_g must be >= 1")

    if N_g == 1:
        # pyber GaugeLength.single_point(): distances=[0.0], weights=[1.0]
        return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)

    # Inclusive linspace over [-1/2, +1/2] (fractions of l).
    offsets = np.linspace(-0.5, 0.5, N_g, dtype=np.float64)

    # Trapezoid rule, written per integration point (cf. pyber
    # get_integration_points_and_weights). Uniform base weights of 1.
    base = np.ones(N_g, dtype=np.float64)
    dx = np.diff(offsets)
    weights = np.zeros(N_g, dtype=np.float64)
    weights[:-1] += base[:-1] / 2.0 * dx
    weights[1:] += base[1:] / 2.0 * dx

    # Normalize to unit sum (pyber .normalize()).
    weights /= weights.sum()
    return offsets, weights


class FiberGeometry:
    """Vertical DAS fiber geometry for ADFWI.

    Builds channel centers, unit tangents, the deduplicated gauge-endpoint
    receiver list, and the channel->endpoint index maps for a straight vertical
    fiber at horizontal position ``x_well``.

    Two placement modes:

    * ``snap_to_nodes=True`` (field mode, the default): one channel is selected
      per grid depth node ``z = k * dz`` that lies within the physical fiber
      span ``[z_top, z_top + (n_channels - 1) * dch]``. The selected channel is
      the *physical* channel (at ``z_top + c * dch``) nearest that node, so the
      channel_z values are true channel depths, snapped to within ``dch / 2`` of
      a node. This reproduces the field channel-subset rule (spec E0): use the
      channel nearest each 5 m node, position error <= dch/2 ~ 0.51 m.

    * ``snap_to_nodes=False`` (synthetic mode): channels are placed *exactly* on
      consecutive grid nodes with spacing ``dz`` (channel_z = z_top + c * dz).
      Used for inverse-crime synthetics where the fiber lives on the grid.

    In both modes each channel's gauge endpoints (center +- l/2 along the fiber)
    are mapped to grid nodes. With ``dz = l/2`` (the production setting) endpoints
    fall exactly on nodes and consecutive channels share endpoints, so the
    receiver list is deduplicated: for the aligned vertical case
    ``n_rcv == C + 2`` (all nodes from ``k_min - 1`` to ``k_max + 1``), with
    ``idx_minus[c] == c`` and ``idx_plus[c] == c + 2``.

    Attributes:
        x_well, z_top, n_channels, dch, l, dx, dz, snap_to_nodes: constructor args.
        channel_z (ndarray [C], float64): depth of each selected channel center.
        e_x, e_z (ndarray [C], float64): unit tangent components; (0, 1) for all.
        rcv_pos (list[tuple[int, int]]): unique ``(z_index, x_index)`` grid-index
            tuples, in deterministic first-seen order, for the ADFWI receivers.
        idx_minus, idx_plus (torch.LongTensor [C]): indices into ``rcv_pos`` for
            each channel's ``-l/2`` and ``+l/2`` gauge endpoints respectively.
    """

    def __init__(self, x_well, z_top, n_channels, dch=1.02, l=10.0,
                 dx=5.0, dz=5.0, snap_to_nodes=True):
        self.x_well = float(x_well)
        self.z_top = float(z_top)
        self.n_channels = int(n_channels)
        self.dch = float(dch)
        self.l = float(l)
        self.dx = float(dx)
        self.dz = float(dz)
        self.snap_to_nodes = bool(snap_to_nodes)

        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1")

        # Horizontal grid index of the well (shared by every receiver).
        x_index = int(round(self.x_well / self.dx))

        if self.snap_to_nodes:
            # Field mode: one channel per grid depth node within the fiber span.
            # Physical channel depths available on the fiber:
            phys_z = self.z_top + np.arange(self.n_channels) * self.dch
            z_bottom = phys_z[-1]

            # Grid nodes within the physical span [z_top, z_bottom].
            k_lo = int(np.ceil(self.z_top / self.dz))
            k_hi = int(np.floor(z_bottom / self.dz))
            if k_hi < k_lo:
                raise ValueError(
                    "fiber span too short to contain a grid node; "
                    "increase n_channels or lower z_top onto the grid"
                )
            node_ks = np.arange(k_lo, k_hi + 1)
            node_z = node_ks * self.dz

            # For each node pick the nearest physical channel depth.
            # (broadcast |phys_z - node_z|, argmin over channels)
            nearest = np.abs(phys_z[None, :] - node_z[:, None]).argmin(axis=1)
            channel_z = phys_z[nearest]
        else:
            # Synthetic mode: channels exactly on consecutive grid nodes.
            channel_z = self.z_top + np.arange(self.n_channels) * self.dz

        self.channel_z = np.asarray(channel_z, dtype=np.float64)
        C = self.channel_z.shape[0]

        # Vertical fiber -> unit tangent e = (0, 1) for every channel.
        self.e_x = np.zeros(C, dtype=np.float64)
        self.e_z = np.ones(C, dtype=np.float64)

        # Gauge endpoints along the fiber: center +- l/2 in depth.
        z_minus = self.channel_z - self.l / 2.0
        z_plus = self.channel_z + self.l / 2.0

        # Map endpoint depths to grid z-indices (nearest node).
        k_minus = np.round(z_minus / self.dz).astype(int)
        k_plus = np.round(z_plus / self.dz).astype(int)

        # Deduplicate endpoint grid indices generically, by exact (z, x) index,
        # preserving deterministic first-seen order. This yields C + 2 receivers
        # for the aligned vertical case but stays correct off-alignment.
        rcv_pos = []
        index_of = {}  # (z_index, x_index) -> position in rcv_pos

        def _register(k_z):
            key = (int(k_z), x_index)
            pos = index_of.get(key)
            if pos is None:
                pos = len(rcv_pos)
                index_of[key] = pos
                rcv_pos.append(key)
            return pos

        idx_minus = np.empty(C, dtype=np.int64)
        idx_plus = np.empty(C, dtype=np.int64)
        # Register in a deterministic order: all minus endpoints ascending, then
        # plus endpoints. For the aligned case this produces the natural node
        # ordering k_min-1 .. k_max+1 with idx_minus[c]=c, idx_plus[c]=c+2.
        for c in range(C):
            idx_minus[c] = _register(k_minus[c])
        for c in range(C):
            idx_plus[c] = _register(k_plus[c])

        self.rcv_pos = rcv_pos
        self.idx_minus = torch.as_tensor(idx_minus, dtype=torch.long)
        self.idx_plus = torch.as_tensor(idx_plus, dtype=torch.long)

    @property
    def n_rcv(self):
        """Number of unique gauge-endpoint receivers."""
        return len(self.rcv_pos)

    @property
    def C(self):
        """Number of channels."""
        return self.channel_z.shape[0]

    def endpoint_z(self, c, sign):
        """Physical depth of channel ``c``'s gauge endpoint on the grid.

        Args:
            c: channel index.
            sign: ``-1`` for the ``-l/2`` endpoint, ``+1`` for ``+l/2``.

        Returns:
            float: ``z_index * dz`` of the mapped grid node.
        """
        idx = self.idx_minus if sign < 0 else self.idx_plus
        z_index, _ = self.rcv_pos[int(idx[c])]
        return z_index * self.dz

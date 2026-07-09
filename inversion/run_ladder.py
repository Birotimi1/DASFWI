"""Degradation-ladder campaign script - SKELETON (T8; HPC stage only).

The ladder degrades the inverse-crime setup toward field realism one rung at
a time (noise, wavelet error, geometry error, band limits, ...) and scores
each rung's inversion with the frozen E18 criterion:

    PASS iff (J_final/J_0 <= tau_J) AND (dz_b <= tau_b) AND (RMS_Vp <= tau_v)

Thresholds tau_J, tau_b, tau_v are FROZEN before any rung runs
(forge.io_preprocess.ladder_pass).
"""


def run_ladder(config):
    """Planned steps (HPC stage):

    1. Freeze (tau_J, tau_b, tau_v) in the config; write them to the results
       header BEFORE the first rung.
    2. Build the rung list: each rung = base inverse-crime config
       (inversion.run_inverse_crime) + one additional degradation.
    3. For each rung: run_inverse_crime(rung_config) -> losses, vp_final.
    4. Score: J_final/J_0 from iter_loss; dz_b from
       forge.io_preprocess.boundary_depth_error on the fiber-well profile;
       RMS_Vp from forge.io_preprocess.rms_vp over the sonic interval.
    5. Record ladder_pass(...) per rung; the last passing rung defines the
       validated realism level.
    """
    raise NotImplementedError(
        "run_ladder is an HPC-stage script; local phase is code+tests only "
        "(spec section 3). See the docstring for the planned wiring.")


if __name__ == "__main__":
    run_ladder(None)

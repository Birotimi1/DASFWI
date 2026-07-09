"""Field-data DAS FWI run script - SKELETON (T8; executed at HPC stage only).

Wires: preprocessed field strain rate (forge.io_preprocess.apply_chain) ->
SeismicData({"strain_rate": ...}) -> AcousticFWI(das_layer=..., obs_key=
"strain_rate") with multiscale bands whose low cut respects the E16
usable-f_low diagnostic.

Local rule (spec section 3): do NOT invert or download field data locally.
This file only fixes the interface so the HPC campaign is a config change.
"""


def run_field(config):
    """Planned steps (HPC stage):

    1. Load raw shot gathers for wells 78A-32 (1010 ch) / 78B-32 (1206 ch),
       field observable = strain rate [1/s]; channel spacing 1.02 m, gauge
       length 10 m (spec E0). No time derivative/integral anywhere.
    2. FiberGeometry(snap_to_nodes=True): one channel per 5 m node
       (position error <= dch/2 ~ 0.51 m).
    3. Preprocess per band: forge.io_preprocess.apply_chain (E14 order).
    4. E16 diagnostic: snr_spectrum + usable_f_low per band; only invert
       bands with f >= f_low.
    5. SeismicData.record_data({"strain_rate": gathers}); receiver masks for
       dead channels -> AcousticFWI channel masks via das_layer.channel_mask.
    6. AcousticFWI(..., das_layer=layer, obs_key="strain_rate"),
       AdamW(weight_decay=0.0), ScheduledMisfit or GC; multiscale
       fwi.forward(cutoff_freq=f, start_iter=...) bookkeeping as in
       run_inverse_crime.
    7. E17 validation: rms_vp vs the 58-32 sonic; boundary_depth_error vs
       823 m + air offset in 78B-32.
    """
    raise NotImplementedError(
        "run_field is an HPC-stage script; local phase is code+tests only "
        "(spec section 3). See the docstring for the planned wiring.")


if __name__ == "__main__":
    run_field(None)

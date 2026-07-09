"""Deterministic tiny STOCK pressure-FWI run (T5 no-regression harness).

This module is imported by tests/test_upstream_patch.py and was also executed
once BEFORE patching ADFWI/fwi/acoustic_fwi.py to freeze the reference losses
(see REFERENCE_LOSSES in the test). Everything here uses only stock ADFWI
behavior - no das_layer, obs_key left at its default - so its results must be
bit-identical before and after the T5 patch.

Fully deterministic: no randomness anywhere (fixed geometry, fixed bump
perturbation, AdamW with default deterministic updates, float64).
"""

import numpy as np
import torch

from ADFWI.model import AcousticModel
from ADFWI.survey import Source, Receiver, Survey, SeismicData
from ADFWI.propagator import AcousticPropagator, GradProcessor
from ADFWI.utils.wavelets import wavelet
from ADFWI.fwi.misfit import Misfit_waveform_L2
from ADFWI.fwi import AcousticFWI

NX = NZ = 70
DX = DZ = 5.0
NT = 800
DT = 4e-4
F0 = 10.0
VP0 = 2000.0


def _make_model(vp, vp_grad=False):
    rho = 0.31 * 1000.0 * np.asarray(vp) ** 0.25
    return AcousticModel(0, 0, NX, NZ, DX, DZ,
                         vp=np.asarray(vp, dtype=np.float64), rho=rho,
                         vp_grad=vp_grad, rho_grad=False,
                         auto_update_rho=False, free_surface=False,
                         abc_type="PML", nabc=20,
                         device="cpu", dtype=torch.float64)


def _make_survey():
    src = Source(NT, DT, F0)
    wl = wavelet(NT, DT, F0)[1]
    src.add_source(15, 2, wl)
    src.add_source(50, 2, wl)
    rcv = Receiver(NT, DT)
    rcv.add_receivers(np.arange(5, 65), np.full(60, 2), "pr")
    return Survey(src, rcv)


def run_stock_fwi(iterations=2):
    """Run the tiny stock pressure FWI; return the list of per-iter losses."""
    survey = _make_survey()

    # observed data from a bumped true model (inverse crime, pressure records)
    zz, xx = np.meshgrid(np.arange(NZ), np.arange(NX), indexing="ij")
    vp_true = VP0 + 80.0 * np.exp(-(((zz - 35.0) ** 2 + (xx - 35.0) ** 2)
                                    / (2 * 6.0 ** 2)))
    prop_true = AcousticPropagator(_make_model(vp_true), survey,
                                   device="cpu", dtype=torch.float64)
    with torch.no_grad():
        rec = prop_true.forward(checkpoint_segments=4)
    obs_data = SeismicData(survey)
    obs_data.record_data({"p": rec["p"]})

    # inversion from the homogeneous model
    model = _make_model(np.full((NZ, NX), VP0), vp_grad=True)
    prop = AcousticPropagator(model, survey, device="cpu", dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,
                                                gamma=1.0)
    fwi = AcousticFWI(prop, model, optimizer, scheduler,
                      loss_fn=Misfit_waveform_L2(dt=DT),
                      obs_data=obs_data,
                      gradient_processor=GradProcessor(),
                      waveform_normalize=True,
                      cache_result=True,
                      save_fig_epoch=-1)
    fwi.forward(iteration=iterations, checkpoint_segments=4)
    return fwi.iter_loss


if __name__ == "__main__":
    losses = run_stock_fwi()
    for i, l in enumerate(losses):
        print(f"iter {i}: {l!r}")

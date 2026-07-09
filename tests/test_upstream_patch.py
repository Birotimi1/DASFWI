"""T5 tests: the (only) upstream patch to ADFWI/fwi/acoustic_fwi.py.

1. No-regression: the stock pressure-FWI path must be bit-identical to the
   unpatched class. REFERENCE_LOSSES below were produced by running
   tests/stock_fwi_runner.py BEFORE the patch was applied (same env, same
   machine, float64, fully deterministic - no randomness in the chain).
2. DAS path smoke test: AcousticFWI(..., das_layer=..., obs_key="strain_rate")
   on the T4 tiny setup with inverse-crime observed data; 2 AdamW iterations
   must decrease the loss and produce finite, nonzero vp gradients.
"""

import numpy as np
import pytest
import torch

from ADFWI.model import AcousticModel
from ADFWI.survey import Source, Receiver, Survey, SeismicData
from ADFWI.propagator import AcousticPropagator, GradProcessor
from ADFWI.fwi import AcousticFWI

from das.das_layer import DASObservationLayer
from tests.stock_fwi_runner import run_stock_fwi
from tests.test_adjoint import (GCMisfit64, make_geometry, make_model,
                                make_survey, NZ, NX, VP0)

# Frozen pre-patch reference (see module docstring). Loss VALUES are what the
# unpatched code produced; equality must be exact (bit-identical execution).
REFERENCE_LOSSES = [0.13282959772031983, 0.31729675449796246]


def test_no_regression_stock_path():
    losses = run_stock_fwi(iterations=2)
    assert len(losses) == len(REFERENCE_LOSSES)
    for i, (got, ref) in enumerate(zip(losses, REFERENCE_LOSSES)):
        assert got == ref, (
            f"stock-path loss changed at iter {i}: {got!r} != {ref!r} "
            "(the T5 patch must be bit-identical when das_layer is not used)")


def test_das_path_smoke():
    geometry = make_geometry()
    layer = DASObservationLayer(geometry, output="strain_rate")

    # inverse-crime observed strain rate from a bumped true model
    zz, xx = np.meshgrid(np.arange(NZ), np.arange(NX), indexing="ij")
    vp_true = (np.full((NZ, NX), VP0)
               + 100.0 * np.exp(-(((zz - 30.0) ** 2 + (xx - 40.0) ** 2)
                                  / (2 * 8.0 ** 2))))
    survey = make_survey(geometry)
    prop_true = AcousticPropagator(make_model(vp_true), survey,
                                   device="cpu", dtype=torch.float64)
    with torch.no_grad():
        rec = prop_true.forward(checkpoint_segments=10)
        obs_sr = layer(rec["u"], rec["w"])
    obs_data = SeismicData(survey)
    obs_data.record_data({"strain_rate": obs_sr})

    # inversion from the homogeneous model through the DAS path
    model = make_model(np.full((NZ, NX), VP0), vp_grad=True)
    prop = AcousticPropagator(model, survey, device="cpu", dtype=torch.float64)
    # lr modest: the inverse-crime GC loss starts near its floor (correlation
    # ~0.997), so a large step overshoots and the 2-iteration decrease check
    # fails for optimization (not gradient) reasons; E8 validates the gradient.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,
                                                gamma=1.0)
    fwi = AcousticFWI(prop, model, optimizer, scheduler,
                      loss_fn=GCMisfit64(dt=1),
                      obs_data=obs_data,
                      gradient_processor=GradProcessor(),
                      waveform_normalize=True,
                      cache_result=True,
                      save_fig_epoch=-1,
                      das_layer=layer,
                      obs_key="strain_rate")
    fwi.forward(iteration=2, checkpoint_segments=10)

    assert len(fwi.iter_loss) == 2
    assert fwi.iter_loss[1] < fwi.iter_loss[0], (
        f"DAS-path loss did not decrease: {fwi.iter_loss}")
    grad = np.asarray(fwi.iter_vp_grad[-1])
    assert np.isfinite(grad).all(), "vp gradient not finite"
    assert np.abs(grad).max() > 0, "vp gradient identically zero"


def test_das_path_rejects_offset_mute():
    geometry = make_geometry()
    layer = DASObservationLayer(geometry, output="strain_rate")
    survey = make_survey(geometry)
    model = make_model(np.full((NZ, NX), VP0), vp_grad=True)
    prop = AcousticPropagator(model, survey, device="cpu", dtype=torch.float64)
    obs_data = SeismicData(survey)
    obs_data.record_data({"strain_rate": np.zeros((1, prop.nt, geometry.C))})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)
    with pytest.raises(ValueError):
        AcousticFWI(prop, model, optimizer, scheduler,
                    loss_fn=GCMisfit64(dt=1), obs_data=obs_data,
                    gradient_processor=GradProcessor(),
                    waveform_mute_offset=100.0,
                    das_layer=layer, obs_key="strain_rate")

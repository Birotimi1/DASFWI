"""END-TO-END readiness: elastic generation -> ACOUSTIC inversion with the switch.

Every other test here checks a component. This one checks that the whole FORGE
chain actually RUNS, because that is the thing that has repeatedly not been
true: components passed their own tests while the assembled pipeline died on
the first iteration. It caught `scheduler=None` crashing AcousticFWI.forward,
which no unit test could have -- the Route B starter would have failed the
moment it reached the cluster.

Deliberately tiny (60x80, 3 shots, 6 iterations) so it runs in the suite. It
asserts the pipeline WORKS, not that the inversion is GOOD -- inversion quality
is the experiment, not a unit test.
"""
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from ADFWI.fwi import AcousticFWI
from ADFWI.model import AcousticModel
from ADFWI.propagator import AcousticPropagator, GradProcessor

from das.geometry import merge_fibers
from forge.proxy_model import (forge_proxy_vp, forge_fibers, vibroseis_line,
                               DX, DZ, V_AIR)
from forge.realistic_synthetic import elastic_observed
from inversion import config, near_surface as ns
from inversion.adaptive_misfit import BlendedMisfit, SkipSwitch
from inversion.skip_diagnostic import skip_fraction

NZ, NX, NT, DT, F0, ZAIR = 60, 80, 600, 0.001, 25.0, 50.0


@pytest.fixture(scope="module")
def chain():
    """Generate ELASTICALLY (+noise), invert ACOUSTICALLY -- the field's real
    physics mismatch, which an inverse-crime synthetic cannot reproduce."""
    vp_true = forge_proxy_vp(NZ, NX, dz=DZ, z_air=ZAIR)
    geom = merge_fibers(list(forge_fibers(NZ, 150.0, 250.0, 80.0, 20, DZ)))
    src = vibroseis_line(NT, DT, F0, [10, 40, 70], int(ZAIR / DZ))
    obs, survey, layer = elastic_observed(vp_true, geom, src, DX, DZ, nabc=10,
                                          snr_db=20.0, device="cpu",
                                          dtype=torch.float32)
    n_air = ns.air_cells(ZAIR, DZ)
    vp0 = ns.with_air_layer(gaussian_filter(vp_true, sigma=6.0), n_air)
    model = AcousticModel(0, 0, NX, NZ, DX, DZ, vp0,
                          np.power(vp0, 0.25) * 310.0,
                          vp_bound=list(ns.VP_BOUND_FIELD), vp_grad=True,
                          water_layer_mask=ns.air_mask(NZ, NX, n_air),
                          free_surface=True, abc_type="PML", nabc=10,
                          device="cpu", dtype=torch.float32)
    prop = AcousticPropagator(model, survey, device="cpu", dtype=torch.float32)
    loss = BlendedMisfit(config.build_misfit("gc", dt=DT, iterations=6),
                         config.build_misfit("envelope", dt=DT, iterations=6),
                         lam=1.0, normalize=True)
    opt = torch.optim.Adam(model.parameters(), lr=20.0)
    fwi = AcousticFWI(propagator=prop, model=model, optimizer=opt,
                      # scheduler.step() is called UNCONDITIONALLY: None crashes
                      scheduler=torch.optim.lr_scheduler.StepLR(opt, 10 ** 9, 1.0),
                      loss_fn=loss, obs_data=obs,
                      gradient_processor=GradProcessor(grad_mute=n_air,
                                                       marine_or_land="marine"),
                      waveform_normalize=True, cache_result=True,
                      save_fig_epoch=-1, das_layer=layer,
                      obs_key="strain_rate")
    obs_arr = np.asarray(obs.data["strain_rate"])
    ctrl, skips, lams = SkipSwitch(), [], []
    for it in range(3):
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=1)
            syn = layer(rec["u"], rec["w"]).cpu()
        sk = float(skip_fraction(syn, obs_arr, DT, F0)["skip_fraction"])
        lam = ctrl.update(sk)
        loss.set_lambda(lam)
        skips.append(sk); lams.append(float(lam))
        fwi.forward(iteration=2, start_iter=it * 2, batch_size=None,
                    checkpoint_segments=1)
    return dict(vp_true=vp_true, vp0=vp0, vp=model.vp.detach().cpu().numpy(),
                loss=np.asarray(fwi.iter_loss), obs=obs_arr, n_air=n_air,
                skips=skips, lams=lams)


def test_elastic_data_is_usable(chain):
    d = chain["obs"]
    assert d.ndim == 3 and d.shape[0] == 3
    assert np.isfinite(d).all() and np.abs(d).max() > 0


def test_the_pipeline_runs_and_the_loss_decreases(chain):
    L = chain["loss"]
    assert np.isfinite(L).all(), "NaN in the loss"
    assert L[-1] < L[0], f"loss did not decrease: {L[0]:.3e} -> {L[-1]:.3e}"


def test_the_air_layer_survives_inversion(chain):
    """water_layer_mask exempts air from clip_params, grad_mute stops it being
    updated. BOTH are needed; either alone destroys the air layer."""
    assert chain["vp"][0, 0] == pytest.approx(V_AIR, abs=1e-3)
    assert np.allclose(chain["vp"][:chain["n_air"]], V_AIR, atol=1e-3)


def test_the_model_stays_physical(chain):
    vp = chain["vp"]
    assert np.isfinite(vp).all()
    below = vp[chain["n_air"]:]
    assert below.min() >= ns.VP_BOUND_FIELD[0] - 1e-6
    assert below.max() <= ns.VP_BOUND_FIELD[1] + 1e-6


def test_the_switch_measures_and_hands_over_on_FIELD_LIKE_data(chain):
    """The controller needs only syn vs obs -- no true model -- so it works on
    field data. Here it runs on ELASTIC, NOISY data inverted acoustically, and
    must still track the skip fraction down and hand over."""
    skips, lams = chain["skips"], chain["lams"]
    assert all(np.isfinite(skips)), skips
    assert skips[-1] < skips[0], f"skip did not fall: {skips}"
    assert set(lams) <= {0.0, 1.0}, f"lambda must stay binary: {lams}"
    assert lams[0] == 1.0, "must START in the robust stage at a skipped start"

"""NON-CRIME FORGE synthetic: elastic generation, acoustic inversion.

>>> WHY THIS EXISTS <<<
`proxy_model.generate_observed` says so itself: "inverse crime by construction
(same propagator family, same operator, no noise)". Every conditioning tool we
built was then tested against data containing none of the errors it exists to
remove, and we drew the obvious wrong conclusions:

  * ARRIVAL WINDOWING (`w`) "hurt" (-0.028) -- on data with no noise, no coda
    and no surface waves to window OUT. Park report STRONG SURFACE WAVES in the
    near-offset FORGE gathers, which an ACOUSTIC inversion cannot model at all.
  * CHANNEL WEIGHTING (`c`) "hurt" -- on data with no poorly-coupled channels.
  * lambda/4 SMOOTHING (`g`) "hurt" -- on data whose fine structure was
    perfectly resolvable because it came from the same propagator.
  * MULTISCALE "hurt" -- partly a rigged budget, partly 1.06 octaves.

A synthetic that shares the inversion's physics cannot test physics-mismatch
mitigation. So this module introduces the four errors the field actually has:

  1. ELASTIC forward modelling, ACOUSTIC inversion. Rayleigh and converted waves
     exist in the data and are UNMODELLABLE by the inverter -- exactly Park's
     situation. This is the big one, and the only way to make surface-wave
     muting testable.
  2. A DIFFERENT SOURCE WAVELET from the one the inversion assumes (Park assume
     a 10 Hz Ricker; the true source is unknown).
  3. NOISE, at a controllable SNR.
  4. TOPOGRAPHY / air layer, already in `proxy_model`.

USE: generate with `elastic_observed(...)`, invert with the ACOUSTIC driver.
Any result that survives this is far more likely to survive the field.
"""
import numpy as np
import torch

from ADFWI.model import IsotropicElasticModel
from ADFWI.propagator import ElasticPropagator
from ADFWI.survey import SeismicData

from das.das_layer import DASObservationLayer
from forge.proxy_model import build_survey, V_AIR


#: Vs from Vp. sqrt(3) is the Poisson-solid value; the AIR layer must keep
#: Vs = 0 or the elastic solver will propagate a shear wave through air.
SQRT3 = float(np.sqrt(3.0))


def vs_from_vp(vp, v_air_max=V_AIR + 1.0):
    """Poisson-solid Vs, with Vs = 0 in the air slab.

    Identified by VELOCITY, never by a fixed row count, so it stays correct
    under topography where the air/ground interface is not flat.
    """
    vs = np.asarray(vp, float) / SQRT3
    vs[np.asarray(vp, float) <= v_air_max] = 0.0
    return vs


def add_noise(data, snr_db, seed=0):
    """Additive Gaussian noise at a given SNR, per gather.

    Field DAS is not noiseless, and `channel_weights`/`arrival_window` exist to
    handle exactly that. Scaled per gather so a quiet shot is not swamped.
    """
    x = np.asarray(data, float)
    rng = np.random.default_rng(seed)
    p_sig = np.mean(x ** 2, axis=tuple(range(1, x.ndim)), keepdims=True)
    p_noise = p_sig / (10.0 ** (float(snr_db) / 10.0))
    return x + rng.standard_normal(x.shape) * np.sqrt(p_noise)


def elastic_observed(vp, geometry, source, dx, dz, nabc=20, rho=None, vs=None,
                     snr_db=None, noise_seed=0, checkpoint_segments=10,
                     device="cpu", dtype=torch.float32, free_surface=True):
    """Observed DAS strain rate from an ELASTIC forward model.

    The inversion is ACOUSTIC, so the returned data contain physics the
    inverter cannot reproduce -- surface waves, converted waves, and the elastic
    amplitude behaviour of a free surface. That mismatch is the point: it is
    what the field has, and what a same-propagator synthetic hides.

    Returns (obs_data, survey, layer) exactly like proxy_model.generate_observed,
    so it is a drop-in replacement in any script that used the crime version.
    """
    vp = np.asarray(vp, float)
    vs = vs_from_vp(vp) if vs is None else np.asarray(vs, float)
    if rho is None:
        # Gardner, with air handled by velocity as in proxy_model.gardner_rho
        rho = 310.0 * np.power(np.maximum(vp, 1.0), 0.25)
        rho[vp <= V_AIR + 1.0] = 1.225
    nz, nx = vp.shape

    model = IsotropicElasticModel(0, 0, nx, nz, dx, dz, vp, vs, rho,
                                  free_surface=free_surface, abc_type="PML",
                                  nabc=nabc, device=device, dtype=dtype)
    survey = build_survey(source, geometry)
    layer = DASObservationLayer(geometry, output="strain_rate")
    prop = ElasticPropagator(model, survey, device=device, dtype=dtype)
    with torch.no_grad():
        rec = prop.forward(checkpoint_segments=checkpoint_segments)
        # the DAS layer takes the particle-velocity fields; the elastic
        # propagator names them the same way the acoustic one does
        obs = layer(rec["vx"] if "vx" in rec else rec["u"],
                    rec["vz"] if "vz" in rec else rec["w"])
    obs_np = obs.detach().cpu().numpy()
    if snr_db is not None:
        obs_np = add_noise(obs_np, snr_db, noise_seed)
    obs_data = SeismicData(survey)
    obs_data.record_data({"strain_rate": torch.as_tensor(obs_np, dtype=obs.dtype)})
    return obs_data, survey, layer


def mismatched_wavelet(nt, dt, f0_true, f0_assumed, t0=None):
    """(true, assumed) Ricker wavelets -- the wavelet-sensitivity experiment.

    Park assume a 10 Hz Ricker; the real source is unknown. Generating with one
    and inverting with another measures what that assumption costs, which nobody
    has reported for this site. L2 fits amplitude AND phase, so it must absorb
    the difference into the VELOCITY MODEL; `convsi` is source-independent and
    should not. That contrast is the experiment.
    """
    from ADFWI.utils.wavelets import wavelet
    t0 = (1.2 / float(f0_true)) if t0 is None else t0
    # wavelet() returns (time_axis, amplitude) -- taking the whole tuple gives a
    # (2, nt) array whose first row is TIME, which silently produces a nonsense
    # source. Caught by the width test; take [1].
    true = np.asarray(wavelet(nt, dt, f0_true, t0=t0, type="Ricker"))[1]
    assumed = np.asarray(wavelet(nt, dt, f0_assumed, t0=t0, type="Ricker"))[1]
    return true, assumed

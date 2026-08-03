"""Non-crime FORGE synthetic: elastic generation for an acoustic inversion.

proxy_model.generate_observed is an inverse crime by its own docstring -- same
propagator, same operator, no noise. Every conditioning tool was then tested
against data containing none of the errors it removes, and we concluded they
were useless. These tests pin the properties that make the synthetic able to
FALSIFY that conclusion.
"""
import numpy as np
import pytest

from forge.realistic_synthetic import (vs_from_vp, add_noise, SQRT3,
                                       mismatched_wavelet)
from forge.proxy_model import V_AIR


def test_air_gets_zero_shear_velocity():
    """An elastic solver would otherwise propagate a SHEAR wave through air.
    Identified by VELOCITY, not a row count, so it stays right under topography
    where the air/ground interface is not flat."""
    vp = np.array([[V_AIR, V_AIR], [1500.0, 2500.0], [5500.0, 5900.0]])
    vs = vs_from_vp(vp)
    assert (vs[0] == 0.0).all()
    assert vs[1, 0] == pytest.approx(1500.0 / SQRT3)
    assert vs[2, 1] == pytest.approx(5900.0 / SQRT3)
    assert (vs[1:] > 0).all()


def test_air_detection_is_by_velocity_not_by_row():
    """Topography means the air/ground boundary is NOT a flat row."""
    vp = np.full((4, 3), 2000.0)
    vp[0, :] = V_AIR
    vp[1, 0] = V_AIR                      # a lower-lying column: air reaches deeper
    vs = vs_from_vp(vp)
    assert vs[1, 0] == 0.0 and vs[1, 1] > 0.0


def test_noise_hits_the_requested_snr():
    """Conditioning (`w`, `c`) exists for noise. A noiseless synthetic cannot
    test it, which is why the A/B said those tools were useless."""
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((3, 400, 12))
    for snr in (20.0, 10.0, 3.0):
        noisy = add_noise(clean, snr, seed=1)
        got = 10 * np.log10(np.mean(clean ** 2) /
                            np.mean((noisy - clean) ** 2))
        assert got == pytest.approx(snr, abs=0.6), f"asked {snr} dB, got {got:.2f}"
    assert not np.allclose(add_noise(clean, 10, seed=1),
                           add_noise(clean, 10, seed=2))     # seed is honoured


def test_noise_is_scaled_per_gather():
    """A quiet shot must not be swamped by noise sized from a loud one."""
    x = np.stack([np.ones((200, 5)), 1e-4 * np.ones((200, 5))])
    n = add_noise(x, 20.0, seed=0) - x
    loud, quiet = np.std(n[0]), np.std(n[1])
    assert quiet < loud / 100, f"noise not per-gather: {loud:.3e} vs {quiet:.3e}"


def test_mismatched_wavelets_really_differ():
    """The wavelet-sensitivity experiment (#50): Park ASSUME a 10 Hz Ricker and
    the true source is unknown. Generating with one and inverting with another
    measures what that assumption costs -- unreported for this site."""
    true, assumed = mismatched_wavelet(1000, 0.001, f0_true=14.0, f0_assumed=10.0)
    assert true.shape == assumed.shape == (1000,)
    assert not np.allclose(true, assumed)
    # the assumed wavelet must be the LOWER-frequency one -> broader in time
    w_true = np.sum(np.abs(true) > 0.05 * np.abs(true).max())
    w_asm = np.sum(np.abs(assumed) > 0.05 * np.abs(assumed).max())
    assert w_asm > w_true
    same, same2 = mismatched_wavelet(1000, 0.001, 10.0, 10.0)
    assert np.allclose(same, same2)        # the control: no mismatch


def test_elastic_generation_is_importable_and_documented():
    """Guard the module contract: it must be a DROP-IN for the crime version,
    returning (obs_data, survey, layer)."""
    import inspect
    from forge.realistic_synthetic import elastic_observed
    sig = inspect.signature(elastic_observed)
    for p in ("vp", "geometry", "source", "dx", "dz", "snr_db"):
        assert p in sig.parameters
    assert "ACOUSTIC" in inspect.getdoc(elastic_observed)

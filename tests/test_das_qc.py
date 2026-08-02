"""DAS waveform-shape QC (inversion/das_qc.py) on CONTROLLED distortions.

The QC has to survive being pointed at an unfamiliar site, so it is tested
against cases whose answer we already know by construction:

    clean                -> amplitude-only  (no false alarm)
    scalar per channel   -> amplitude-only  (harmless, correctly dismissed)
    fixed MAGNITUDE      -> SHAPE DISTORTION, repeatable  (needs deconvolution
      filter                -- must not be missed)
    fixed ALL-PASS       -> SHAPE DISTORTION via the PHASE test, which is the
      filter (mild)         only one that can see it: |R(f)| == 1, so the
                            magnitude test reports a perfectly flat "scalar"
    magnitude varying    -> shape mismatch, NOT repeatable (noise, not the fibre)
      shot to shot
    PHASE varying        -> likewise noise. This is the FALSE POSITIVE that an
      shot to shot          unsigned phase statistic produced on real FORGE data
    incoherent channels  -> INCONCLUSIVE, never a false "shape distortion"

The last one is the deployment guard: the whole method assumes neighbouring
channels see the same wavefield, and at a site with coarse spacing or poor SNR
that assumption fails and looks exactly like distortion.

Every threshold these tests pin is CALIBRATED against these measurements rather
than chosen -- see the notes in `verdict` and `qc_das`.
"""
import numpy as np
import pytest

from inversion.das_qc import (qc_das, auto_band, channel_report, verdict,
                              spacing_is_adequate, recommended_settings,
                              format_report)

DT, NT, NCH, NSH, F0 = 0.001, 800, 60, 6, 20.0


def _clean(n_shots=NSH, seed=0):
    """[S, nt, C] Ricker arrivals with a gentle moveout -- the coherent case a
    real fibre produces, since channels are metres apart and lambda is ~100 m."""
    rng = np.random.default_rng(seed)
    t = np.arange(NT) * DT
    g = np.zeros((n_shots, NT, NCH))
    for s in range(n_shots):
        for c in range(NCH):
            tt = t - (0.15 + 0.02 * s + 0.0008 * c)      # 0.8 ms/channel moveout
            g[s, :, c] = (1 - 2 * (np.pi * F0 * tt) ** 2) * \
                np.exp(-(np.pi * F0 * tt) ** 2)
    return g + 0.01 * rng.standard_normal(g.shape)


def _apply_allpass(g, beta, f_ref=20.0):
    """Multiply channel c by exp(i*beta[c]*(f/f_ref)**2) -- an ALL-PASS filter.
    |H(f)| == 1 exactly, so the magnitude test (2) is BLIND to it, yet the
    waveform is genuinely deformed. Only the phase test (4) can see this."""
    S, nt, C = g.shape
    f = np.fft.rfftfreq(nt, DT)
    ph = np.outer((f / f_ref) ** 2, beta)                        # [nf, C]
    G = np.fft.rfft(g, axis=1) * np.exp(1j * ph)[None]
    return np.fft.irfft(G, n=nt, axis=1)


def _apply_tilt(g, alpha, f_ref=20.0):
    """Multiply channel c of shot s by (f/f_ref)**alpha[s, c] -- a per-channel
    FILTER, i.e. exactly the shape distortion poor coupling would cause. The
    log-log spectral-ratio slope the QC measures IS alpha, so the ground truth
    is known analytically."""
    S, nt, C = g.shape
    f = np.fft.rfftfreq(nt, DT)
    w = np.clip(f / f_ref, 1e-3, None)
    G = np.fft.rfft(g, axis=1)                                   # [S, nf, C]
    G = G * (w[None, :, None] ** alpha[:, None, :])
    return np.fft.irfft(G, n=nt, axis=1)


# --------------------------------------------------------------------------- #
# the four cases
# --------------------------------------------------------------------------- #
def test_clean_data_raises_no_alarm():
    res = qc_das(_clean(), DT, f_lo=5.0, f_hi=60.0)
    assert res["amplitude_only"] and not res["shape_distortion"]
    assert res["median_neighbour_corr"] > 0.95
    assert res["frac_spectral_tilt"] == 0.0
    assert res["frac_phase_nonlinear"] == 0.0
    assert res["premise"]["holds"]


def test_pure_amplitude_distortion_is_dismissed():
    """A per-channel SCALAR is the harmless case: per-trace normalisation
    already removes it, so flagging it would send us deconvolving for nothing."""
    rng = np.random.default_rng(1)
    g = _clean() * np.exp(rng.normal(0, 1.5, (1, 1, NCH)))    # x0.03 .. x30
    res = qc_das(g, DT, f_lo=5.0, f_hi=60.0)
    assert res["amplitude_only"], res["verdict"]
    assert res["frac_spectral_tilt"] == 0.0        # a scalar has NO tilt
    assert not recommended_settings(res)["deconvolve_channels"]


def test_repeatable_per_channel_filter_is_caught():
    """The case that MUST NOT be missed: the same filter on the same channels in
    every shot -- a property of the fibre, not the shot. This is the one no
    misfit choice can fix."""
    rng = np.random.default_rng(2)
    alpha = np.zeros((NSH, NCH))
    bad = rng.choice(NCH, size=NCH // 3, replace=False)
    alpha[:, bad] = rng.choice([-1.5, 1.5], size=len(bad))     # fixed across shots
    res = qc_das(_apply_tilt(_clean(), alpha), DT, f_lo=5.0, f_hi=60.0)
    assert res["shape_distortion"], res["verdict"]
    assert res["frac_spectral_tilt"] > 0.15
    assert res["frac_tilt_repeatable"] > 0.5
    assert recommended_settings(res)["deconvolve_channels"]


def test_all_pass_dispersion_is_caught_by_the_phase_test_alone():
    """THE BLIND SPOT the magnitude test cannot cover. An all-pass filter has
    |R(f)| == 1, so test 2 reports a perfectly flat 'scalar' ratio -- yet the
    waveform IS deformed, and dispersion is the one distortion a phase misfit
    cannot absorb either. Without test 4 we would have declared FORGE clean on
    evidence that could not have detected this.

    Mild and isolated on purpose -- that is the case only test 4 can reach.
    Severe dispersion destroys neighbour correlation and is caught by test 1
    instead; see test_severe_dispersion_is_not_called_clean."""
    beta = np.zeros(NCH)
    beta[np.arange(0, NCH, 6)] = 3.0                   # same filter every shot
    res = qc_das(_apply_allpass(_clean(), beta), DT, f_lo=5.0, f_hi=60.0)
    assert res["frac_spectral_tilt"] == 0.0            # magnitude test is BLIND
    assert res["frac_phase_nonlinear"] > 0.10          # phase test sees it
    assert res["dispersive"] and res["shape_distortion"], res["verdict"]
    assert "ALL-PASS" in res["verdict"]
    assert recommended_settings(res)["deconvolve_channels"]


def test_severe_dispersion_is_not_called_clean():
    """Coverage check for the known limit. Dispersion strong enough to destroy
    neighbour correlation falls outside test 4's coherence gate -- so the thing
    that must be guaranteed is that it is not silently passed as clean."""
    beta = np.zeros(NCH)
    beta[np.arange(0, NCH, 6)] = 8.0
    res = qc_das(_apply_allpass(_clean(), beta), DT, f_lo=5.0, f_hi=60.0)
    assert not res["amplitude_only"], res["verdict"]
    assert res["frac_shape_mismatch"] > 0.05           # test 1 carries it


def test_phase_noise_is_not_mistaken_for_dispersion():
    """REGRESSION -- this one produced a FALSE POSITIVE on real FORGE data.

    An RMS (unsigned) phase departure SATURATES, so noise-driven phase scatter
    reads as 'consistently ~0.5 rad' in every shot and passes a std-below-
    magnitude repeatability test. Two fixes are pinned here: the statistic is
    SIGNED, so random curvature cancels in the median; and repeatability is
    judged against the median's STANDARD ERROR, so a handful of shots cannot
    manufacture consistency by chance."""
    rng = np.random.default_rng(7)
    g = np.stack([_apply_allpass(_clean()[s:s + 1], rng.normal(0, 3.0, NCH))[0]
                  for s in range(NSH)])                # NEW filter every shot
    res = qc_das(g, DT, f_lo=5.0, f_hi=60.0)
    assert not res["dispersive"], res["verdict"]
    assert not res["shape_distortion"], res["verdict"]


def test_shot_varying_distortion_is_called_noise_not_coupling():
    """Same symptom, different cause: a distortion that changes shot to shot
    cannot be the cable. Test 3 (repeatability) is what separates them, and the
    prescription differs -- weighting/windowing, not deconvolution."""
    rng = np.random.default_rng(3)
    alpha = rng.normal(0, 2.0, (NSH, NCH))                     # new every shot
    res = qc_das(_apply_tilt(_clean(), alpha), DT, f_lo=5.0, f_hi=60.0)
    assert not res["amplitude_only"]
    assert not res["shape_distortion"]                         # NOT the fibre
    assert "NOT repeatable" in res["verdict"]


# --------------------------------------------------------------------------- #
# the deployment guard
# --------------------------------------------------------------------------- #
def test_incoherent_channels_are_inconclusive_not_a_false_alarm():
    """THE guard. If neighbours are incoherent -- coarse spacing, scattering,
    bad SNR -- the premise fails and the data look distorted. Reporting SHAPE
    DISTORTION there would send a new site chasing a defect that is not real, so
    the answer must be INCONCLUSIVE with the reason attached."""
    rng = np.random.default_rng(4)
    res = qc_das(rng.standard_normal((NSH, NT, NCH)), DT, f_lo=5.0, f_hi=60.0)
    assert res["inconclusive"]
    assert not res["shape_distortion"]              # never a false positive
    assert not res["premise"]["holds"]
    assert "premise" in res["verdict"] and "not evidence" in res["verdict"].lower()
    assert "conservative" in recommended_settings(res)["note"]


def test_premise_guard_can_be_switched_off():
    rng = np.random.default_rng(4)
    res = qc_das(rng.standard_normal((NSH, NT, NCH)), DT, f_lo=5.0, f_hi=60.0,
                 check_premise=False)
    assert not res["premise"]["checked"] and not res["inconclusive"]
    # And with the guard OFF the verdict must STILL not cry distortion. For
    # incoherent traces the cross-spectrum phase is random, so its non-linearity
    # saturates in every shot -- large AND stable, which mimics a repeatable
    # filter. `coherent_min` is what stops that, and it must hold on its own.
    assert not res["shape_distortion"], res["verdict"]


def test_premise_threshold_separates_incoherence_from_real_distortion():
    """The guard's threshold is a CALIBRATION, not a guess: it must sit below
    the correlation that severe distortion leaves behind and above the
    correlation of genuinely incoherent channels. Too high and true positives
    are downgraded to 'inconclusive'; too low and the guard never fires."""
    rng = np.random.default_rng(6)
    beta = np.zeros(NCH)
    beta[rng.choice(NCH, size=NCH // 3, replace=False)] = 8.0     # severe
    distorted = qc_das(_apply_allpass(_clean(), beta), DT, 5.0, 60.0)
    incoherent = qc_das(rng.standard_normal((NSH, NT, NCH)), DT, 5.0, 60.0)
    thr = distorted["premise"]["threshold"]
    assert incoherent["premise"]["low_band_corr"] < thr \
        < distorted["premise"]["low_band_corr"]


def test_geometric_premise_check():
    """Advisory metadata check: the neighbour span must sit well inside one
    wavelength. FORGE (1.02 m, 1500 m/s, 40 Hz) passes comfortably."""
    ok, ratio = spacing_is_adequate(1.02, 1500.0, 40.0, n_neigh=2)
    assert ok and ratio < 0.1
    coarse, ratio2 = spacing_is_adequate(25.0, 1500.0, 40.0, n_neigh=2)
    assert not coarse and ratio2 > 1.0


# --------------------------------------------------------------------------- #
# site-agnostic plumbing
# --------------------------------------------------------------------------- #
def test_auto_band_finds_the_source_band():
    """A new site has a different source and sample rate, so the band must be
    measured, not inherited from FORGE."""
    lo, hi = auto_band(_clean(), DT)
    assert lo < F0 < hi                       # brackets the 20 Hz Ricker peak
    assert hi < 0.5 / DT                      # below Nyquist


def test_auto_band_is_used_when_no_band_is_given():
    res = qc_das(_clean(), DT)
    assert res["band"][0] < F0 < res["band"][1]
    assert res["amplitude_only"]


def test_rejects_bad_input_shapes():
    with pytest.raises(ValueError):
        qc_das(np.zeros((NT, NCH)), DT)                    # missing shot axis
    with pytest.raises(ValueError):
        channel_report(np.zeros((2, NT, 3)), DT, 5.0, 60.0, n_neigh=2)  # too few


def test_report_is_printable_and_json_safe():
    import json
    res = qc_das(_clean(), DT, f_lo=5.0, f_hi=60.0)
    assert "VERDICT" in format_report(res)
    json.dumps(res)                                        # no numpy scalars


def test_verdict_handles_all_dead_channels():
    rep = channel_report(np.zeros((2, NT, NCH)), DT, 5.0, 60.0)
    v = verdict(rep)
    assert v["n_channels_scored"] == 0 and v["inconclusive"]

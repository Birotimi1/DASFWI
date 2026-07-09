# DASFWI

Differentiable DAS (distributed acoustic sensing) observation operator and full-waveform
inversion workflow for the 2022 FORGE stimulation VSP experiment, built on top of
[ADFWI](https://github.com/liufeng2317/ADFWI). The package implements the gauge-averaged
axial strain-rate operator as an exact endpoint difference of the axial particle velocity
(`ε̄̇(s_c,t) = [v_e(s_c+l/2,t) − v_e(s_c−l/2,t)]/l`), inserted as a differentiable
`torch.nn.Module` between ADFWI's acoustic propagator and its misfit machinery, so the
adjoint source is generated automatically by autograd. Developed per the build spec
`DASFWI_CodeDev_Handoff_v6b.md` (local CPU development phase; HPC campaigns follow).

## Build status

| Task | File(s) | Status |
|------|---------|--------|
| T1 | package skeleton, `env.yml` | ✅ done |
| T2 | `das/geometry.py` + tests | ✅ done (16 tests) |
| T3 | `das/das_layer.py` + tests | ✅ done (12 tests) |
| T4 | `tests/test_adjoint.py` (E6–E9 gates) | ✅ done (E8 rel. err ≤ 4e−5) |
| T5 | ADFWI `fwi/acoustic_fwi.py` patch + tests | ✅ done (bit-identical no-regression; patch archived in `patches/`) |
| T6 | `inversion/misfit_schedule.py` + tests | pending |
| T7 | `forge/proxy_model.py`, `inversion/run_inverse_crime.py` | pending |
| T8 | `forge/io_preprocess.py`, run-script skeletons + tests | pending |

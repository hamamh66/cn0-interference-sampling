# Sampling One Interference Pattern Twice

[![tests](https://github.com/hamamh66/cn0-interference-sampling/actions/workflows/test.yml/badge.svg)](https://github.com/hamamh66/cn0-interference-sampling/actions)
[![Archive](https://img.shields.io/badge/archive-FRDR%20(DOI%20pending)-lightgrey.svg)](https://www.frdr-dfdr.ca/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion code and executed notebook for the manuscript

> H. Hamam, *Sampling One Interference Pattern Twice: Detectability of GNSS Multipath Classes from Spatial and
> Inter-Frequency C/N0 Differences under Receiver Quantization and Noise*, 2026 (under review).

The package simulates the carrier-to-noise-density ratio (C/N0) reported by a GNSS receiver for four reception
classes — line-of-sight, ground multipath, building multipath and non-line-of-sight — using an image-source plane-wave
model with Fresnel reflection coefficients, and evaluates how much of the inter-antenna or inter-frequency C/N0
difference survives receiver estimation noise and quantization. It reproduces every figure and table of the paper.

## What is in the repository

| Path | Content |
|---|---|
| `cn0sampling/simulation.py` | Propagation, antenna-channel and receiver models; analytic phase step and weak-signal RMS; quantized binary Bayes error; self-checks |
| `cn0sampling/run_experiments.py` | Experiments E1–E5, paired seeds, LaTeX table generation (`cn0-run-experiments`) |
| `cn0sampling/make_diagrams.py` | Framework and operation schematics (`cn0-make-diagrams`) |
| `notebooks/cn0_difference_detectability.ipynb` | Executed notebook: self-checks, full run, figures, tables and interpretation |
| `results/results.json` | Archived results with per-seed metrics, validation checks, software versions and source hashes |
| `results/figures/`, `results/tables/` | Archived figures (PDF/PNG/SVG) and LaTeX tables used in the manuscript |
| `tests/` | Physical self-checks and a reproduction test against the archived results |

## Reproduce

```bash
git clone https://github.com/hamamh66/cn0-interference-sampling.git
cd cn0-interference-sampling
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"          # or: pip install -r requirements.txt for the exact archived versions
pytest -q                          # self-checks + reproduction of results/results.json (~30 s)
cn0-run-experiments --out results  # full run, ~15 s on a laptop CPU
cn0-make-diagrams
```

Or open the notebook: locally with `jupyter lab notebooks/cn0_difference_detectability.ipynb`, or in Google Colab
(the first cell clones this repository into the Colab local disk; nothing is written to Drive).

Results are deterministic for fixed seeds; the archived file was generated with the versions in `requirements.txt`.
Other versions may change results at the last reported decimal.

## Model summary

* Signals: GPS L1 (1575.42 MHz) and L5 (1176.45 MHz); direct ray, single ground reflection, single vertical-wall
  reflection; Fresnel coefficients for circular polarization with a scalar cross-polar leakage term.
* Receiver: `Y = round((mu + s + eta)/q)·q`, Gaussian `eta` with per-channel standard deviation `sigma` and optional
  inter-channel correlation `rho`; quantization step `q`.
* Experiments: E1 phase step versus elevation; E2 median |ΔC/N0| versus baseline length; E3 four-class decision-tree
  classification under six (`sigma`, `q`) settings, five seeds, paired across nine feature configurations;
  E4 phase-marginalized quantized binary Bayes error; E5 frozen-model stress tests (bias, calibration, correlated
  noise, scene shift).

All evidence is simulated. The model omits diffraction, diffuse scattering, finite reflector extent, tracking loss and
correlator response; no measured receiver data are used.

## Archive
Each tagged release is deposited in the [Federated Research Data Repository (FRDR)](https://www.frdr-dfdr.ca/), the Digital Research Alliance of Canada's repository, which assigns a persistent DOI. The DOI for v1.0.0 will be listed here and in `CITATION.cff` once the deposit is published.

## Citing

Please cite the article once published and the archived release (FRDR) (see `CITATION.cff`; GitHub shows a "Cite this
repository" button). Until the article is published:

```bibtex
@misc{Hamam2026GitHub,
  author    = {H. Hamam},
  title     = {cn0sampling: {GNSS} multipath-class detectability from spatial and inter-frequency {C/N0} differences under receiver quantization and noise},
  editor    = {Repository},
  publisher = {GitHub},
  year      = {2026},
  url       = {https://github.com/hamamh66/cn0-interference-sampling},
  note      = {Open-source repository providing the fully reproducible implementation, executed notebook and archived results for the article.}
}
```

## License

MIT (see `LICENSE`).

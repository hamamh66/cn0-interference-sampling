# cn0-interference-sampling

[![tests](https://github.com/hamamh66/cn0-interference-sampling/actions/workflows/test.yml/badge.svg)](https://github.com/hamamh66/cn0-interference-sampling/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)

Simulates the carrier-to-noise-density ratio (C/N0) reported by a GNSS receiver for four reception classes —
line-of-sight, ground multipath, building multipath and non-line-of-sight — using an image-source plane-wave model
with Fresnel reflection coefficients, and evaluates how much of the inter-antenna or inter-frequency C/N0 difference
survives receiver estimation noise and quantization.

Typical questions it answers:

* Which antenna-pair geometries have an exact C/N0-difference null for a given reflector (e.g. a horizontal pair over
  flat ground)?
* At what vertical spacing does the ground-reflection response first peak, as a function of elevation?
* How much four-class separability remains once the receiver rounds C/N0 to 0.25 dB and adds 0.5 dB of noise?
* For a specified two-ray alternative, what is the minimum error of *any* detector operating on the quantized
  difference (numerical Bayes error)?
* How do differential channel biases, correlated noise and a change of scene affect a classifier trained on the
  nominal model?

## Installation

```bash
git clone https://github.com/hamamh66/cn0-interference-sampling.git
cd cn0-interference-sampling
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

Python ≥ 3.10; dependencies are NumPy, SciPy, scikit-learn and Matplotlib. `requirements.txt` pins the exact
versions used to produce the archived results.

## Quick start

```bash
pytest -q                          # physical self-checks + reproduction of results/results.json (~30 s)
cn0-run-experiments --out results  # experiments E1–E5, ~15 s on a laptop CPU
cn0-make-diagrams                  # framework and operation schematics
```

Or open `notebooks/cn0_difference_detectability.ipynb` (Jupyter, or Google Colab — the first cell clones this
repository into the Colab local disk).

Python API:

```python
from cn0sampling import simulation as sim
s  = sim.make_scene(4000, seed=0)                      # balanced LOS / MPG / MPB / NLOS scenes
d  = sim.signal(s, baseline=(0, 0, 0.10)) - sim.signal(s)   # inter-antenna C/N0 difference, vertical 10 cm
y  = sim.measure(sim.signal(s)[:, None], seed=1, sigma=0.5, q=0.25)   # receiver noise + quantization
pe = sim.bayes_error(alpha=0.05, dphi=3.14159, sigma=0.5, q=0.25)      # quantized binary Bayes error
```

## Repository layout

| Path | Content |
|---|---|
| `cn0sampling/simulation.py` | Propagation, antenna-channel and receiver models; analytic phase step and weak-signal RMS; quantized binary Bayes error; `self_checks()` |
| `cn0sampling/run_experiments.py` | Experiments E1–E5, paired seeds, LaTeX table generation (`cn0-run-experiments`) |
| `cn0sampling/make_diagrams.py` | Schematics (`cn0-make-diagrams`) |
| `notebooks/cn0_difference_detectability.ipynb` | Executed notebook: self-checks, full run, figures, tables and interpretation |
| `results/results.json` | Archived results with per-seed metrics, validation checks, software versions and source hashes |
| `results/figures/`, `results/tables/` | Archived figures (PDF/PNG/SVG) and LaTeX tables |
| `tests/` | Physical self-checks and a reproduction test against the archived results (run in CI) |

## Model

* **Signals:** GPS L1 (1575.42 MHz) and L5 (1176.45 MHz); direct ray, single specular ground reflection, single
  vertical-wall reflection; image-source plane waves with positive path excess; Fresnel coefficients for circular
  polarization with a scalar cross-polar leakage term; RHCP hemispherical antenna pattern with below-horizon
  suppression.
* **Classes (balanced):** LOS (direct), MPG (direct + ground), MPB (direct + ground + wall), NLOS (wall only).
* **Receiver:** `Y = round((mu + s + eta)/q)·q` — Gaussian estimation noise `eta` (per-channel σ, optional
  inter-channel correlation ρ) followed by uniform quantization with step `q`.
* **Experiments:** E1 relative phase step versus elevation; E2 median |ΔC/N0| versus baseline length; E3 four-class
  decision-tree classification under six (σ, q) settings, five seeds, paired across nine feature configurations
  (single channel, horizontal / vertical / oblique pairs, H+V and H+H triplets, L1–L5); E4 phase-marginalized
  quantized binary Bayes error; E5 frozen-model stress tests (differential bias, open-sky calibration, correlated
  noise, scene shift).

Results are deterministic for fixed seeds. Other library versions may change values at the last reported decimal.

## Scope and limitations

Everything here is simulated. The model is planar, specular and narrowband: it omits diffraction, diffuse scattering,
finite reflector extent, multiple bounces, antenna coupling, vector polarization patterns, tracking loss and correlator
response. Bayes errors apply to the specified binary observable, not to the four-class classifier. No measured
receiver data are included, and no positioning solution is computed.

## Archive

Tagged releases are deposited in the [Federated Research Data Repository (FRDR)](https://www.frdr-dfdr.ca/), which
assigns a persistent DOI; the DOI for each release is recorded in `CITATION.cff` and in the release notes.

## Citing

See `CITATION.cff` (GitHub's "Cite this repository" button).

## License

MIT — see `LICENSE`.

# uav-rbpc-inspection

Certified scheduling of a UAV inspection fleet operating from a **continuously
sailing mother vessel** in an offshore wind farm.

This repository implements the model and solver of the paper
*"Certified Scheduling of UAV Fleet Inspection from a Sailing Mother Vessel:
Launch Timing, Routing, and Robust Recovery"*:

- **Space–time-coupled sortie columns** $\omega=(\tau,\pi,h)$: launch time,
  visiting sequence, and recovery horizon decided jointly.
- **Distributionally robust chance constraints** over a decision-dependent,
  unimodal-moment ambiguity set of the vessel-position forecast error, with a
  two-dimensional geometric refinement of the convex return distance and a
  hard wave/wind landing gate.
- **R-BPC**, a resource-aware branch-price-and-cut framework that prices
  complete columns through the unchanged physical–meteorological oracle and
  returns **machine-verifiable lexicographic optimality certificates**
  (coverage first, energy second).

## Repository layout

| Path | Content |
|---|---|
| `step1`–`step8` | Data pipeline: AIS fetch, wind-farm matching, turbine/wind/wave retrieval, track export, forecast-error (`xi`) moments, recovery scenarios |
| `step9`–`step10` | Physical model: power, wind triangle, energy/time budgets, DRCC margins, ambiguity sets |
| `step11`–`step12` | R-BPC: exact pricing through the physical oracle, master problem, certified prefix bounds, branching, certificates |
| `step13`–`step14` | Model/algorithm experiment runners (E1/E2, A1/A2) |
| `step15` | Out-of-sample safety replay |
| `step16`–`step19` | Figures, paper tables, diagnostics |
| `step20` | Release preflight (metadata-only checks) |
| `selftest.py` | Full internal test suite |
| `docs/` | Detailed documentation (in Chinese): model, algorithm, parameters, experiments, proofs, data |
| `results/` | Selected certified artifacts backing the paper's tables (see below) |

## Install

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip on Linux
```

Requires Python 3.9+ with numpy, pandas, scipy, matplotlib, geopandas,
xarray, cdsapi (see `requirements.txt`).

## Data

Raw data are **not included** in this repository. Running the pipeline from
scratch requires fetching public sources with your own API credentials:

1. Register at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
   and put your personal key in `~/.cdsapirc` (see the header of
   `step4_fetch_wind_era5.py`). **No API keys are stored in this repository.**
2. Run `step1`–`step5` to fetch AIS tracks, the wind-farm layout (OSM), ERA5
   winds, and CMEMS wave data for the study area.
3. Run `step6`–`step8` to export tracks, estimate forecast-error moments, and
   generate recovery scenarios.

The exact instance files and all formal command lines used in the paper are
documented in `docs/doc_experiments.md` (Chinese).

## Reproducing the paper

The main certified instance (Rodsand II, n = 10 turbines, tier-M UAV, K = 2,
B = 7, VP unimodal criterion) is one command line, recorded verbatim in
`docs/doc_experiments.md`. Typical runtime on one desktop core: ~36 min.

## Verifying the certificates

Every certified artifact carries a `proof_contract_sha256` column that binds
it to the proof-critical source bytes. Recompute the digest of the released
code and compare:

```python
import step12_branch_price as S
print(S.FORMAL_PROOF_CONTRACT_SHA256)
# baf6e6d0e6d4fa9e513d21cd32ad2c4e5f349ea564c079d95ff4c56d0e9fc766
```

This value must equal the `proof_contract_sha256` recorded in each artifact
under `results/`.

## Selected artifacts under `results/`

| Artifact | Backs |
|---|---|
| `experiments/E1_n10*`, `algorithm_experiments/A2_speed` | Acceleration ladder (paper Table 2, n = 10 rows) |
| `experiments/E1_n12`, `E1_n15_v31`, `E1_n18` | Scale envelope 10→18 turbines (Table 2) |
| `experiments/E1_k1_v31` | K = 1 boundary study, incl. profiling artifacts (Section on the certificate envelope) |
| `experiments/E1_n10_reference` | Table 2 reference row (5.54 h baseline; from git tag `baseline-reference`) |
| `diagnostics/` | K = 1 cProfile evidence backing the string-rendering attribution (19.4M calls) |
| `tools/figures/` | Generator for all paper figures (reads `results/`, writes `results/figures/`) |
| `experiments/E1_k3` | Fleet-size sensitivity (Tables 2–3, K = 3) |
| `experiments/E1_frontier_n8` | Certified fleet–battery frontier, full route universe |
| `algorithm_experiments/A1_accuracy` | Safety-criterion comparison (Table 4) |
| `experiments/E1_channels_n10.json` | Out-of-sample replay channels (7,141 draws) |

## Provenance notes

- The five-version runtime ladder of Table 2 (reference / compiled / pruned /
  cached / hoisted) compares successive solver generations; each generation's
  artifact carries the proof digest of the code snapshot it ran on (the
  reference generation corresponds to git tag `baseline-reference` in the
  authors' working repository). Only the `hoisted` generation is the current
  code.
- Table 4 (safety-criterion comparison) is backed by
  `experiments/E2_criteria_recourse_compat/`; `E1_criteria_comparison/` is an
  earlier diagnostic variant and `algorithm_experiments/A1_accuracy/` is the
  solution-quality ladder benchmark.
- Naming note: the criterion key `cantelli` in the code and the paper's
  "Cantelli (one-sided Chebyshev)" label both refer to the bound
  sqrt((1-eps)/eps). The paper's function-name reference is historical;
  the name is kept unchanged because the proof-code digest binds the
  artifact certificates to the exact source bytes.

## Tests

```bash
.venv/Scripts/python selftest.py
```

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite the accompanying paper (citation entry
will be added upon publication).

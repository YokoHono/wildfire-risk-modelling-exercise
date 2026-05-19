# Wildfire Risk Modelling Exercise — OroraTech Submission

Our submission for the
[UL Wildfire Risk Modeling Exercise](https://web.cvent.com/event/9351958c-e489-433d-b3e9-a29c4bdfa039/summary?environment=P2).
Two scenarios, one risk score per building:

- *Town of Forests* — Lake Tahoe, CA — mountainous WUI / Intermix
- *Town of Prairies* — Amarillo, TX — flat grassland WUI

Each property gets a single risk score in [0, 1] built from the five components
the manual defines (hazard, exposure, vulnerability, severity of consequence,
capacity to withstand impact):

```
risk_gross = (Hazard × Vulnerability × Exposure × Severity)^(1/4)
risk_net   = risk_gross × (1 − 0.5 × Capacity)
risk_score = min-max rescaled to [0, 1] within the scenario
```

See [METHODOLOGY.html](METHODOLOGY.html) for the full evaluator report —
methodology, output GeoJSON schema, ArcGIS submission details, validation,
limitations, references. Open it in a browser and use *Print → Save as PDF*
if you need a PDF version.

## Quick start

```bash
bash install_packages.sh
source .venv/bin/activate
jupyter trust Forest_submission.ipynb Prairie_submission.ipynb
jupyter notebook
```

Open either submission notebook and run the cells top to bottom. With cached
artefacts each notebook runs in 1–2 minutes; from a cold start `python run.py`
adds ~15 min for the Monte Carlo sim and 5–10 min for the one-time DINS / MTBS /
FPA-FOD / WUI downloads.

## System requirements

Python 3.11 (3.9 also works), ~4 GB RAM, no GPU. About 2 GB of disk for the
cached reference data.

## Workflow

| Step | What | How |
|---|---|---|
| 1 | Download and cache reference data, run the Monte Carlo sim, write the 3-component intermediate | `python run.py --scenario both` |
| 2 | Pull per-property ignition timestamps out of the OroraTech sim HTML | `python extract_property_ignitions.py` |
| 3 | Apply the 5-component framework and produce the submission | run `Forest_submission.ipynb` and `Prairie_submission.ipynb` |
| 4 | Regenerate notebooks from the template (optional) | `python make_notebooks.py` |

## Repository layout

```
.
├── README.md
├── METHODOLOGY.html                full evaluator report (open in browser → Print to PDF)
├── requirements.txt / install_packages.sh
│
├── Forest_submission.ipynb
├── Prairie_submission.ipynb
│
├── run.py                          3-component pipeline + Monte Carlo sim
├── make_notebooks.py               regenerates the submission notebooks
├── extract_property_ignitions.py   parses OroraTech sim → per-property GeoJSON
├── structure_overlay.py            Purnomo (2024/2025) WUI structure-ignition overlay for ForeFire
│
├── src/                            model code (config, loaders, components, viz)
├── outputs/                        scored GeoJSONs + folium HTML maps
├── data_cache/                     cached reference data (MTBS, FPA-FOD, WUI, fire-spread sims)
└── Forest/, Prairie/               synthetic scenario data (git-ignored, auto-downloaded from scil-data.sdsc.edu)
```

## Inputs

The synthetic exercise data is used as provided. The model also pulls in four
real-world datasets, all cached under `data_cache/`:

| Dataset | Source | What it gives us |
|---|---|---|
| CAL FIRE DINS damage inspections | gis.data.cnra.ca.gov | training labels for vulnerability |
| MTBS burn perimeters | edcintl.cr.usgs.gov | training labels for the climate hazard classifier |
| FPA-FOD wildfire ignitions | Kaggle `rtatman/188-million-us-wildfires` | ignition-density features for climate hazard |
| USFS WUI 2020 (Radeloff / SILVIS Lab) | geoserver.silvis.forest.wisc.edu | exposure modifier for WUI-overlapping parcels |

Both scenarios also use our coupled fire-spread simulation (Rothermel 1972
surface spread + Purnomo et al. 2024/2025 WUI structure ignition, implemented
in [structure_overlay.py](structure_overlay.py)) for the `hazard_sim`
component — see the References below.

Section **§2b** of each submission notebook lists every input with its source,
version, record count, CRS, and how individual attributes are used.

## Outputs

Everything submission-relevant lives in [outputs/](outputs/).

The headline deliverable per scenario is `outputs/<scenario>_risk_scored_5component.geojson` —
every building polygon plus 17 columns (all five components, the capacity
sub-scores, the gross / net / final risk, and the decile bucket). Schema in §6c
of either notebook.

For evaluators there are three interactive folium maps per scenario:

- `outputs/<scenario>_risk_map.html` — single-layer map of the final risk
- `outputs/<scenario>_risk_components.html` — seven togglable layers (top 10
  highlighted plus each component and the final risk), per-layer adaptive
  colourbar
- `outputs/<scenario>_risk_components_raw.html` — same six layers with a fixed
  `vmin = 0` scale for honest comparison

We also upload the same data as a hosted feature service on ArcGIS Online, shared
into the exercise team group, per §6.1.2.b of the manual.

## Caveats

- `risk_score` is rescaled per scenario, so Forest and Prairie scores aren't
  directly comparable.
- The vulnerability regressor reports MAE ≈ 0.28 and R² ≈ 0.47 on a held-out DINS
  split. Good for relative ranking; not calibrated for absolute damage probabilities.
- The hazard for both scenarios is `max(climate hazard, OroraTech sim)`.
- No formal validation against historical losses. Appropriate for relative
  ranking and mitigation prioritisation, not for insurance pricing.

§6c of each submission notebook covers downstream-use guidance in more detail.

## References

The OroraTech Fire Spread Simulation couples two published wildfire spread
models:

- Rothermel, R.C. (1972). *A mathematical model for predicting fire spread in
  wildland fuels.* USDA Forest Service Research Paper INT-115, Ogden, UT.
- Purnomo, D.M.J., Qin, Y., Theodori, M., Zamanialaei, M., Lautenberger, C.,
  Trouvé, A., & Gollner, M. (2024). *Reconstructing modes of destruction in
  wildland–urban interface fires using a semi-physical level-set model.*
  Proceedings of the Combustion Institute **40**, 105755.
  <https://doi.org/10.1016/j.proci.2024.105755>

Reference datasets:

- CAL FIRE Damage Inspection (DINS) — <https://gis.data.cnra.ca.gov>
- MTBS — <https://www.mtbs.gov>
- Short, K.C. — Spatial wildfire occurrence data for the United States,
  1992–2015 (FPA-FOD; Kaggle mirror `rtatman/188-million-us-wildfires`)
- Radeloff, V.C. et al. — USFS Wildland-Urban Interface 2020, SILVIS Lab —
  <https://silvis.forest.wisc.edu/data/wui-change/>

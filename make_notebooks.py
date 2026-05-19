"""Generate the two submission notebooks (Forest, Prairie) as .ipynb JSON.

Each notebook is a concise, presentation-oriented walkthrough that:
  - loads the scenario inputs (buildings, ignition, weather)
  - produces / loads the per-property risk score
  - shows summary statistics, top-risk buildings, and an interactive map
"""

from __future__ import annotations

import json
import os
import textwrap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [(l + "\n") for l in lines],
    }


def code(source: str) -> dict:
    src = textwrap.dedent(source).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in src.splitlines()],
    }


def build_notebook(
    scenario: str,
    *,
    location: str,
    ignition_date: str,
    n_buildings: int,
    wildland_sim_geojson: str | None = None,
    property_sim_geojson: str | None = None,
    sim_animation_html: str | None = None,
):
    other = "Prairie" if scenario == "Forest" else "Forest"
    cells = [
        md(
            f"# Wildfire Risk — {scenario} scenario",
            "",
            f"**Location:** {location}  ",
            f"**Ignition:** {ignition_date}  ",
            f"**Buildings:** ~{n_buildings:,}",
            "",
            "> _Risk is the potential consequence which could occur to a "
            "system (i.e. landscape, tree, animal species, infrastructure "
            "site, etc.), society or a community, often determined "
            "probabilistically as a function of **hazard, exposure, "
            "vulnerability, severity of consequence, and capacity to "
            "withstand impact**._",
            "",
            "Each property is rated **[0, 1]** on five components:",
            "",
            "| # | Component | Meaning | Sign |",
            "|---|---|---|---|",
            "| 1 | **Hazard** | P(fire reaches the property), `max(climate hazard, scenario sim hazard)` | amplifier |",
            "| 2 | **Vulnerability** | P(damage \\| fire impingement) from construction attrs | amplifier |",
            "| 3 | **Exposure** | Asset value & footprint at stake (property value, sqft × usage, adjacency) | amplifier |",
            "| 4 | **Severity of consequence** | Loss-of-life / disruption magnitude (usage-weighted occupancy load) | amplifier |",
            "| 5 | **Capacity to withstand impact** | Defensible space + modern construction + ember resistance + access | **dampener** |",
            "",
            "**Aggregation:**",
            "```",
            "risk_gross = (Hazard × Vulnerability × Exposure × Severity)^(1/4)",
            "risk_net   = risk_gross × (1 − 0.5 × Capacity)",
            "risk_score = min-max rescale to [0, 1]",
            "```",
        ),
        md("## 1 · Setup"),
        md(
            "First, install dependencies into the active kernel. Idempotent — "
            "pip skips anything already satisfied. `git-lfs` is included so "
            "the LFS-tracked GeoJSON / zip / shapefile inputs can be pulled "
            "on a fresh clone where the system `git-lfs` wasn't installed "
            "before `git clone`.",
        ),
        code(
            """
            %pip install -q -r requirements.txt
            """
        ),
        md(
            "Pull the LFS-tracked binaries. Without this step, files like "
            "`outputs/<scenario>_risk_scored.geojson` are still ~150-byte "
            "pointer texts and downstream cells fail with "
            "`DataSourceError: not recognized as a supported file format`.",
        ),
        code(
            """
            import os, subprocess
            subprocess.run(['git', 'lfs', 'install', '--local'], check=False)
            subprocess.run(['git', 'lfs', 'pull'], check=False)

            # Sanity check: any LFS-tracked file we'll read below should be
            # a real binary now, not a pointer.
            def _assert_not_lfs_pointer(path):
                if not os.path.exists(path):
                    return
                with open(path, 'rb') as fh:
                    head = fh.read(64)
                if head.startswith(b'version https://git-lfs'):
                    raise RuntimeError(
                        f"{path} is still a git-lfs pointer. "
                        "`git lfs pull` failed — run it manually from the "
                        "repo root, then re-execute this cell."
                    )
            """
        ),
        code(
            """
            import sys, os
            sys.path.insert(0, os.path.abspath('.'))

            import geopandas as gpd
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import IFrame, display

            from src.config import SCENARIOS
            from src.loaders import load_buildings, load_ignition, load_weather
            from src.risk_framework import (
                severity_score, capacity_score, aggregate_5component,
            )

            SCENARIO = SCENARIOS[%r]
            """ % scenario
        ),
        md(
            "## 1b · Parameters and assumptions",
            "",
            "All tunable knobs in one place, with the rationale next to each:",
            "",
            "| Parameter | Value | Rationale |",
            "|---|---|---|",
            "| `HORIZON_HOURS` | 12 h | Length of the scenario fire simulation provided by the data |",
            "| Hazard combination rule | `max(hazard_climate, hazard_sim)` | Preserves the strongest single signal; the 50/50 blend would halve the sim score because the MTBS-trained climate prior is near-zero on most of the Forest extent |",
            "| Vulnerability defensible-space modifier | up to **−40 %** at fully-compliant Zone 0/1/2 | Mirror of the proportional risk reduction observed in CAL FIRE DINS records for compliant parcels |",
            "| Severity usage weights | Multi-residence 1.4, Mixed 1.2, Single-residence 1.0, Other 0.8, Commercial 0.6, Industrial 0.5 | Occupancy load × usage; residential weighted higher for life-safety |",
            "| Capacity sub-weights | defensible space 0.35, modern construction 0.25, ember resistance 0.20, access 0.15, WUI maintenance 0.05 | Building safety literature emphasises defensible space and modern code |",
            "| Capacity dampener | risk_net = risk_gross × (1 − **0.5** × capacity) | A fully-resilient property's risk is halved, not zeroed |",
            "| Aggregation form | `(H · V · E · S)^(1/4)` | Geometric mean: each component must be non-trivial for the score to be high |",
            "| Map vmax (per layer) | `round(column.max() + 0.005, 2)` | Stretches the colour ramp across the actual data range so the bright end isn't wasted |",
            "| Map vmin auto-gap | upper-cluster edge − 0.02 if the largest gap in the data ≥ 0.15 | Bimodal layers (Forest hazard) get the full palette on the upper cluster |",
            "",
            "**Coordinate reference systems (CRS):**  inputs and outputs in **EPSG:4326** "
            "(WGS84). Intermediate centroid distances computed in **EPSG:3857** for "
            "metric units. Outputs preserve the input CRS.",
        ),
        md("## 2 · Inputs"),
        md(
            "We are given a synthetic dataset for this scenario containing:",
            "* per-property building polygons with construction attributes and a property value",
            "* an ignition point and timestamp",
            "* a 30-day synoptic weather record near the ignition",
            "* fuel, terrain, and moisture rasters for the simulation extent",
        ),
        code(
            """
            buildings = load_buildings(SCENARIO)
            lon, lat, ig_time = load_ignition(SCENARIO)
            weather = load_weather(SCENARIO)

            print(f"buildings:  {len(buildings):,}")
            print(f"ignition:   ({lat:.4f}, {lon:.4f})  @ {ig_time}")
            print(f"weather:    {len(weather):,} observations, "
                  f"mean wind {weather['wind_ms'].mean():.1f} m/s")
            buildings[['bldgid', 'usage', 'property_value',
                       'building_sqft', 'stories_above_ground']].head()
            """
        ),
        md(
            "## 2b · Data sources and provenance",
            "",
            "All inputs to the model — both the synthetic core data provided by the exercise "
            "and the supplemental real-world reference datasets — are listed below "
            "with their source, version, record counts, CRS, and how each is used.",
            "",
            "### Core synthetic exercise data *(provided by the exercise; used unchanged)*",
            "",
            "| Dataset | File | CRS | Used by |",
            "|---|---|---|---|",
            f"| {scenario} buildings | `{scenario}/{scenario}/{scenario}_generated_buildings_fireprops_Parcel.geojson` | EPSG:4326 | per-property attributes feed **vulnerability** (roof / siding / eaves / vents / windows / deck / fence / year_built / property_value), **exposure** (property_value, building_sqft, stories, neighborhood_count, structural_sep_dist), **severity** (usage, sqft × stories), **capacity** (zone_0/1/2 compliance, year_built, fire-resistant materials) |",
            f"| Ignition point | `{scenario}/{scenario}/{scenario}_ignition.geojson` | EPSG:4326 | t₀ for the per-property sim arrival times |",
            f"| Synoptic weather | `{scenario}/{scenario}/{scenario}_Synoptic_Weather_Data.csv` | n/a | mean wind speed & direction; used by the MC sim (Prairie) and by the proximity step's wind-alignment when enabled |",
            f"| LANDFIRE rasters | `{scenario}/{scenario}/{scenario}_SB40/depth/moist*/rhof*/SAV/elevation.tif` | EPSG:4326 | climate-hazard classifier features (Scott & Burgan fuel model, fuel loading, moisture, SAV, depth, elevation, derived slope) |",
            f"| Treelist | `{scenario}/{scenario}/{scenario}_Treelist.geojson` | EPSG:4326 | per-tree CBD, CBH, MOIST → canopy-fuel features in climate hazard; canopy ROS boost in MC sim |",
            f"| Voxels | `{scenario}/{scenario}/voxels/trees*.dat` | local metric | 3-D fuel arrays reduced to 2-D summaries (rhof, moist, depth); used for canopy ROS boost and as climate-hazard features |",
            f"| Roads | `{scenario}/{scenario}/{scenario}_roads.geojson` | EPSG:4326 | rasterised fuel-break mask for MC sim; `dist_to_road_m` for the **access** sub-score of capacity |",
            "",
            "### Supplemental real-world reference data",
            "",
            "| Dataset | Source | Version / date | Records | Used by |",
            "|---|---|---|---|---|",
            "| **CAL FIRE DINS** post-fire damage inspections | `https://gis.data.cnra.ca.gov` item `994d3dc4569640caadbbc3198d5a3da1` | accessed 2026-05; 132 522 records, filtered to ≈ 95 k fire-only structural records | 94 942 used | Train the **vulnerability** regressor (HistGradientBoostingRegressor on 22 binary/numeric construction features → damage in [0, 1]) |",
            "| **MTBS** burn-severity perimeters | `https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/...` | 1984 – present; 30 730 fires | Forest: Angora 2007 overlaps extent; Prairie: Stone Ridge 2011 overlaps | Train the **climate hazard** classifier (HistGradientBoostingClassifier on landscape rasters) |",
            "| **FPA-FOD** wildfire ignitions | Kaggle mirror `rtatman/188-million-us-wildfires` (auth via `~/.kaggle/access_token`) | 1992 – 2015; 1.88 M ignitions | Forest: 24 280; Prairie: 4 250 within 100 km | `ignition_density_5km` and `ignition_density_25km` features for climate hazard |",
            "| **USFS WUI 2020** (Radeloff / SILVIS Lab) | `https://geoserver.silvis.forest.wisc.edu/geodata/wui_change_2020_v4/...` | 2020 census-block WUI classification | CA: 565 809; TX: 751 557 | **Exposure × 1.15** modifier for properties intersecting an Intermix or Interface block |",
            f"| **OroraTech Fire Spread Simulation** — ForeFire (Rothermel 1972 surface fire spread) coupled with the Purnomo et al. (2024 / 2025) semi-physical level-set model for WUI structure ignition, via [structure_overlay.py](structure_overlay.py) (see references below) | `data_cache/fire_spread_simulations/{scenario.lower()}_scenario/forecast_{scenario.lower()}_wui.geojson` and `forecast_{scenario.lower()}_properties.geojson` | 73 cumulative wildland-perimeter slices over the 12 h scenario, plus per-property ignition timestamps for the structures the fire reaches | provided per-property ignition data | `hazard_sim` for each ignited building, from arrival time |",
            "",
            "**References for the OroraTech Fire Spread Simulation:**",
            "",
            "* Rothermel, R.C. (1972). *A mathematical model for predicting fire spread in wildland fuels.* USDA Forest Service Research Paper INT-115, Ogden, UT.",
            "* Purnomo, D.M.J., Qin, Y., Theodori, M., Zamanialaei, M., Lautenberger, C., Trouvé, A., & Gollner, M. (2024). *Reconstructing modes of destruction in wildland–urban interface fires using a semi-physical level-set model.* Proceedings of the Combustion Institute **40**, 105755. <https://doi.org/10.1016/j.proci.2024.105755>",
            "",
            "**Attribute-level usage examples** *(per §6.1.3.c of the manual)*:",
            "",
            "* `roof_wood = 1` and `roof_non_combust = 0` increase vulnerability by ~0.3 (DINS-trained).",
            "* `zone_0_compliant + zone_1_defensible + zone_2_defensible = 3` reduces vulnerability by **40 %** and lifts the **defensible space** sub-score to 1.0.",
            "* `ventscreens_1_8 = 1` lifts the **ember resistance** sub-score by 0.30.",
            "* `usage == \"Multi Residence\"` applies a **severity** weight of 1.4 and an **exposure** weight of 1.2.",
            "* `dist_to_road_m = 30` gives the **access** capacity sub-score ≈ 0.61 via `exp(−30/60)`.",
        ),
        md(
            "## 2c · Synthetic data NOT used (and why)",
            "",
            "Most of the synthetic dataset feeds the model — see the table in §2b. "
            "A handful of items are intentionally **not** consumed, for the reasons below.",
            "",
            "| Item | Why we don't use it |",
            "|---|---|",
            "| **`<city>_Treelist.txt`** | Exact text-format duplicate of `<city>_Treelist.geojson`. We read the GeoJSON, which is faster to parse with geopandas and carries the same fields. |",
            "| **`<city>_ReadDatFiles.py`** | Reference reader from LANL for the voxel `.dat` files. We re-implement equivalent logic in `src/canopy.py:voxel_2d_summary` so it integrates cleanly with our caching pipeline (the 2-D Z-summary is saved to `data_cache/*_voxel_summary.npz`). |",
            "| **3-D canopy reconstruction via LANL Trees** *(suggested in §5.3 of the manual)* | A full 3-D canopy field is only required for CFD-style coupled simulations (HIGRAD/FIRETEC etc.), which we don't run. Our hazard pipeline captures vertical fuel via two coarser-but-cheaper summaries: the per-tree treelist (CBD / CBH / MOIST) and the voxel `treesrhof` collapsed along Z. |",
            "| **Time-resolved synoptic weather** *(per-timestep temp / RH / wind)* | We use the scenario-mean wind speed and direction. The OroraTech sim already consumed the full time series upstream (its 10-min wildland-perimeter slices encode time-varying wind effects); for our static climate-hazard classifier and severity / capacity / exposure components, per-timestep weather doesn't change the score. |",
            "| **Building attributes flagged as nullable** in the data README — `wildland_area`, `wui_zone`, `parcel_median_size`, `acres_treated_wui` | We use them where they carry values (substituting 0.5 / column-median when missing); for `wui_zone` specifically we cross-check against the USFS WUI 2020 product, which has fewer nulls. |",
            "| **Coarse building-class flags** — `strip_mall_1970`, `big_box_2010`, `med_home_1800`, `small_home_1100`, `condo_townhome_20`, `small_apartment` | Redundant with the finer-grained construction attributes (`roof_*`, `siding_*`, `eaves_*`, `year_built_*`, etc.) that the DINS-trained vulnerability model uses directly. |",
            "| **Garage flags** — `attached_garage`, `detached_garage`, `garage type` | DINS shows no strong standalone signal for these and they're partly captured already by `building_sqft`. |",
            "| **Numeric building geometry** — `building width / height / length [m]`, `nominal lot length / width [m]`, `roof height [m]`, `eave height [m]` | Redundant with `building_sqft` × `stories_above_ground` (which we do use); the manual notes that some lot fields can be null. |",
            "| **Free-text `info` field** | Descriptive label ('Forest school/hospital/industrial unit' etc.), not a numeric model input. |",
            "| **Vegetation type strings** — `wildland_veg_type`, `wui_veg_type`, `urban_veg_type`, `veg_density`, `topography` | Per-building categorical attributes that aren't part of the DINS-trained vulnerability schema. The equivalent landscape signal comes from the LANDFIRE-derived rasters (SB40, fuel loading, depth, SAV) sampled at each building's centroid. |",
            "| **`avg_slope`** *(per-building)* | We compute slope from the `<city>_elevation.tif` raster at each centroid instead; the raster-derived value is what the climate hazard classifier was trained on. |",
            "",
            "If any of these would materially change the score for a particular downstream use, "
            "they're cheap to wire in — most are already loaded as columns on the buildings "
            "GeoDataFrame, just not currently fed to the model.",
        ),
        md("## 3 · Compute the risk score"),
        md(
            "Set `RECOMPUTE = True` to run the full pipeline end-to-end",
            "(~5 minutes for 100 Monte-Carlo realisations on this scenario).",
            "Otherwise we load the cached output from a previous run.",
            "",
            "The cached file blends climate and scenario hazard 50/50. We "
            "replace that with `hazard = max(hazard_climate, hazard_sim)` so "
            "the scenario sim isn't diluted by the climate prior for ignited "
            "buildings, while non-ignited buildings still inherit any "
            "non-zero baseline from the climate model.",
        ),
        code(
            """
            RECOMPUTE = False
            REALIZATIONS = 100

            os.makedirs('outputs', exist_ok=True)
            out_path = f"outputs/{SCENARIO.name}_risk_scored.geojson"

            if RECOMPUTE or not os.path.exists(out_path):
                from src.risk import run_all
                results = run_all(
                    scenarios=[SCENARIO.name],
                    realizations=REALIZATIONS,
                    downsample=15,
                )
                scored = results[SCENARIO.name]
            else:
                _assert_not_lfs_pointer(out_path)
                scored = gpd.read_file(out_path)
                print(f"loaded cached scores from {out_path}")

            # Replace the 50/50 hazard blend from run.py with the max() rule.
            scored['hazard'] = np.maximum(
                scored['hazard_climate'], scored['hazard_sim']
            ).clip(0, 1).round(4)

            print(f"scored {len(scored):,} buildings")
            """
        ),
        *([
            md(
                "## 3b · Override hazard with the per-property fire-spread simulation",
                "",
                "The OroraTech Fire Spread Simulation is a coupled model that pairs "
                "Rothermel (1972)'s mathematical model for surface fire spread in "
                "wildland fuels with the Purnomo et al. (2024 / 2025) semi-physical "
                "level-set model for WUI structure ignition (see §2b for full "
                "references). The structure-overlay implementation lives at "
                "[structure_overlay.py](structure_overlay.py). It provides a "
                "per-structure ignition timestamp (extracted from the HTML viewer "
                f"into `forecast_{scenario.lower()}_properties.geojson`). For each "
                "ignited property we have an *arrival hour from ignition*, "
                "converted to a hazard score:",
                "",
                "$$\\text{hazard}_\\text{sim} = \\max\\!\\Big(0,\\; "
                "1 - \\frac{\\text{arrival hours}}{12}\\Big)$$",
                "",
                "Buildings that never ignite get `hazard_sim = 0`. We then "
                "combine with the climate prior as "
                "`hazard = max(hazard_climate, hazard_sim)` so the arrival "
                "time drives hazard for ignited properties without being "
                "halved by the near-zero climate term. The final 5-component "
                "risk score is recomputed in section **3e**.",
            ),
            code(textwrap.dedent("""
                PROPERTY_SIM = __PROPERTY_SIM_PATH__

                # Parse the per-property ignition timestamps out of the
                # OroraTech HTML viewer if the cached GeoJSON isn't already
                # on disk (or is still a git-lfs pointer from a stale clone).
                def _needs_regenerate(p):
                    if not os.path.exists(p):
                        return True
                    with open(p, 'rb') as fh:
                        return fh.read(64).startswith(b'version https://git-lfs')

                if _needs_regenerate(PROPERTY_SIM):
                    from extract_property_ignitions import SPECS, run_one
                    run_one(SPECS[SCENARIO.name])

                prop_sim = gpd.read_file(PROPERTY_SIM)
                print(f"per-property ignitions: {len(prop_sim):,}"
                      f"  ({100*len(prop_sim)/len(scored):.1f}% of buildings)")
                print(f"arrival range: {prop_sim['arrival_hours_from_ignition'].min():.2f}h"
                      f"  to {prop_sim['arrival_hours_from_ignition'].max():.2f}h")

                # `parcel_row` in prop_sim is the row index in the Parcel
                # buildings file, which is the same ordering as `scored`.
                arrival_by_row = pd.Series(
                    prop_sim['arrival_hours_from_ignition'].to_numpy(dtype=float),
                    index=prop_sim['parcel_row'].astype(int).to_numpy(),
                )

                arrival = np.full(len(scored), np.inf)
                for idx, hrs in arrival_by_row.items():
                    if 0 <= idx < len(scored):
                        arrival[idx] = hrs

                reached = np.isfinite(arrival)
                hazard_sim_new = np.where(
                    reached, np.clip(1 - arrival / 12.0, 0, 1), 0
                ).astype(float)
                print(f"buildings ignited within 12 h: {reached.sum():,}"
                      f" / {len(scored):,} ({100*reached.mean():.1f}%)")
                print(f"new hazard_sim  mean={hazard_sim_new.mean():.3f}"
                      f"  std={hazard_sim_new.std():.3f}")

                # Override hazard_sim and re-derive hazard = max(climate, sim).
                scored['ignition_hour_from_sim'] = np.where(reached, arrival, np.nan)
                scored['hazard_sim'] = hazard_sim_new.round(4)
                scored['hazard'] = np.maximum(
                    scored['hazard_climate'], scored['hazard_sim']
                ).clip(0, 1).round(4)

                fig, ax = plt.subplots(figsize=(8, 2.8))
                ax.hist(arrival[reached], bins=30, color='#cf4446',
                        alpha=0.85, edgecolor='white')
                ax.set_xlabel('Hours from ignition until the building ignites')
                ax.set_ylabel('buildings')
                ax.set_title(f'{SCENARIO.name} sim — per-property ignition time distribution')
                plt.show()
                """).replace("__PROPERTY_SIM_PATH__", repr(property_sim_geojson))),
        ] if property_sim_geojson else []),
        md(
            "## 3c · Severity of consequence",
            "",
            "Loss-of-life and service-disruption potential — distinct from "
            "**Exposure** (the *physical* asset). Built from occupancy load "
            "(`building_sqft × stories_above_ground`) weighted by usage type, "
            "then rank-normalised:",
            "",
            "| Usage | Weight | Rationale |",
            "|---|---|---|",
            "| Multi-residence | 1.4 | many overnight occupants |",
            "| Mixed comm./res. | 1.2 | residential + business hours |",
            "| Single residence | 1.0 | baseline |",
            "| Other | 0.8 | unknown |",
            "| Commercial | 0.6 | mainly business hours |",
            "| Industrial | 0.5 | minimal occupancy |",
        ),
        code(
            """
            severity = severity_score(scored)
            scored['severity'] = np.round(severity, 4)
            print(f'severity  mean={severity.mean():.3f}  std={severity.std():.3f}')
            """
        ),
        md(
            "## 3d · Capacity to withstand impact *(dampener)*",
            "",
            "Resilience features that **reduce** expected damage when the "
            "building is impacted. Weighted sum of five sub-scores:",
            "",
            "| Sub-score | Source fields | Weight |",
            "|---|---|---|",
            "| Defensible space | mean(zone_0_compliant, zone_1_defensible, zone_2_defensible) | 0.35 |",
            "| Modern construction | year_built_2000_2025 + non-combust siding / roof + fire-resist siding | 0.25 |",
            "| Ember resistance | fine vent screens, tempered windows, enclosed eaves | 0.20 |",
            "| Access | `exp(−dist_to_road_m / 60)` — closer = faster response/evac | 0.15 |",
            "| WUI maintenance | rank of `acres_treated_wui` (nullable → neutral 0.5) | 0.05 |",
        ),
        code(
            """
            cap = capacity_score(scored)
            capacity = cap['capacity']
            scored['capacity'] = np.round(capacity, 4)
            for k in ('cap_defensible_space', 'cap_construction_quality',
                      'cap_ember_resistance', 'cap_access', 'cap_wui_maintenance'):
                scored[k] = np.round(cap[k], 4)

            print(f"capacity  mean={capacity.mean():.3f}  std={capacity.std():.3f}")
            for k in ['cap_defensible_space', 'cap_construction_quality',
                      'cap_ember_resistance', 'cap_access', 'cap_wui_maintenance']:
                v = scored[k]
                print(f"  {k:30s}  mean={v.mean():.3f}  std={v.std():.3f}")
            """
        ),
        md(
            "## 3e · Aggregation",
            "",
            "```",
            "risk_gross = (hazard × vulnerability × exposure × severity)^(1/4)",
            "risk_net   = risk_gross × (1 − 0.5 × capacity)",
            "risk_score = min-max rescale to [0, 1]",
            "```",
        ),
        code(
            """
            agg = aggregate_5component(
                scored['hazard'].to_numpy(),
                scored['vulnerability'].to_numpy(),
                scored['exposure'].to_numpy(),
                severity, capacity,
            )
            scored['risk_gross'] = np.round(agg['risk_gross'], 4)
            scored['risk_net']   = np.round(agg['risk_net'],   4)
            scored['risk_score'] = np.round(agg['risk_score'], 4)
            scored['risk_decile'] = agg['risk_decile'].astype(int)

            print(f"risk_score  mean={scored['risk_score'].mean():.3f}"
                  f"  std={scored['risk_score'].std():.3f}"
                  f"  min={scored['risk_score'].min():.3f}"
                  f"  max={scored['risk_score'].max():.3f}")
            """
        ),
        md("## 4 · Score summary"),
        code(
            """
            cols = ['hazard', 'vulnerability', 'exposure',
                    'severity', 'capacity', 'risk_score']
            scored[cols].describe().round(3)
            """
        ),
        md("### Distribution of each component"),
        code(
            """
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            panels = [
                ('hazard',        '#d83a3a'),
                ('vulnerability', '#f1932d'),
                ('exposure',      '#2a8db8'),
                ('severity',      '#7a4fb8'),
                ('capacity',      '#3aa860'),
            ]
            for ax, (col, color) in zip(axes, panels):
                vals = scored[col].dropna()
                ax.hist(vals, bins=30, color=color, alpha=0.85,
                        edgecolor='white', linewidth=0.5)
                ax.set_title(col); ax.set_xlim(0, 1)
                ax.text(0.95, 0.92,
                        f'mu={vals.mean():.2f}\\nsigma={vals.std():.2f}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.25',
                                  facecolor='white', edgecolor='none', alpha=0.85))
            fig.suptitle(f'{SCENARIO.name}: five-component score distributions',
                         y=1.05, fontweight='semibold')
            fig.tight_layout()
            plt.show()
            """
        ),
        md("## 5 · Highest-risk properties"),
        md(
            "The top-ranked buildings combine **all three** components — they are",
            "structurally vulnerable, located in or adjacent to high-hazard fuels,",
            "and high-value / dense-adjacency.",
        ),
        code(
            """
            top10 = (
                scored
                .nlargest(10, 'risk_score')
                .loc[:, ['bldgid', 'usage', 'property_value',
                         'hazard', 'vulnerability', 'exposure',
                         'severity', 'capacity',
                         'risk_score', 'risk_decile']]
                .reset_index(drop=True)
                .round(3)
            )
            top10
            """
        ),
        md(
            "Component contribution for the same 10 buildings:",
        ),
        code(
            """
            top = scored.nlargest(10, 'risk_score').reset_index(drop=True)
            N = len(top)
            # Muted component palette so the final risk_score (dark, slightly
            # wider) reads as the headline bar in each group.
            COMPONENTS = [
                ('hazard',        '#c98382'),   # dusty rose
                ('vulnerability', '#d4ae7c'),   # muted amber
                ('exposure',      '#8aabc0'),   # slate blue
                ('severity',      '#a698c2'),   # lavender
                ('capacity',      '#88b094'),   # sage
            ]
            n_bars = len(COMPONENTS) + 1               # +1 for risk_score
            bar_w = 0.12                               # component bar width
            risk_w = 0.18                              # risk_score bar — wider
            spacing = 1.7                              # x-step between groups
            x = np.arange(N) * spacing

            fig, ax = plt.subplots(figsize=(13.5, 4.2))
            # Light alternating background bands so adjacent groups are easy to tell apart
            for i in range(N):
                if i % 2 == 0:
                    ax.axvspan(x[i] - spacing/2, x[i] + spacing/2,
                                color='#f5f5f3', zorder=0)

            # Centre the group around x. Components first, then a slightly
            # wider risk_score bar at the right edge of each group.
            comp_offsets = [(j - (n_bars - 1) / 2) * bar_w for j in range(len(COMPONENTS))]
            risk_offset = (len(COMPONENTS) - (n_bars - 1) / 2) * bar_w + (risk_w - bar_w) / 2
            for (col, color), off in zip(COMPONENTS, comp_offsets):
                ax.bar(x + off, top[col], bar_w, color=color, alpha=0.85,
                       label=col, edgecolor='white', linewidth=0.5, zorder=2)
            ax.bar(x + risk_offset, top['risk_score'], risk_w,
                   color='#1a1a1a', label='risk_score',
                   edgecolor='#000', linewidth=0.6, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels([f'#{i+1}' for i in range(N)], fontsize=10)
            ax.set_xlim(-spacing/2, x[-1] + spacing/2)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('score')
            ax.set_title(f'{SCENARIO.name}: top-10 risk buildings — component breakdown')
            ax.legend(ncol=n_bars, fontsize=9, frameon=False, loc='upper right')
            ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, zorder=1)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()
            """
        ),
        *([
            md(
                "## 5b · Wildland fire-spread animation",
                "",
                "The pre-rendered interactive map below (built outside this "
                "notebook) animates the wildland-fire perimeter at "
                "10-minute steps over the 12 h horizon, with the wind "
                "speed/direction, fuel moisture and rate-of-spread shown "
                "per step. Use the time slider at the bottom.",
            ),
            code(
                """
                from IPython.display import IFrame
                IFrame(%r, width='100%%', height=620)
                """ % sim_animation_html
            ),
        ] if sim_animation_html else []),
        md(
            "## 6 · Risk map",
            "",
            "Interactive map with **togglable layers** — switch between the "
            "final risk score and each individual component (hazard, "
            "vulnerability, exposure, severity, capacity) using the layer "
            "control in the top-right corner. Inferno-palette continuous "
            "gradient with a per-layer colourbar at the bottom.",
            "",
            "We render the map **twice**:",
            "",
            "* **6a · Raw scale.** Every layer's colour gradient runs from "
            "`vmin = 0` to that layer's actual max. Faithful to the data "
            "but layers with a bimodal distribution (e.g. Forest hazard, "
            "where 75 % of buildings sit near 0 and 25 % sit in [0.6, 0.9]) "
            "look mostly purple-or-orange with little within-cluster "
            "differentiation.",
            "* **6b · Optimised for visualisation.** We auto-detect a 'gap' "
            "in each layer's distribution and use it as `vmin`. For "
            "bimodal layers this puts the upper cluster across the full "
            "inferno palette (purple → magenta → orange → yellow). "
            "Continuous layers are unaffected (`vmin = 0` is kept).",
            "",
            "Building polygons are simplified to ~1 m tolerance before "
            "embedding so the inline maps stay under the notebook "
            "output-size limit. Each rendering is also saved to disk: "
            "`<scenario>_risk_components_raw.html` and "
            "`<scenario>_risk_components.html` respectively.",
        ),
        md("### 6a · Raw scale (`vmin = 0` for every layer)"),
        code(
            """
            import folium
            import branca.colormap as cm

            # matplotlib's 'inferno' colormap, sampled at 10 stops
            SCORE_PALETTE = ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60',
                             '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4']

            # Slim the geometry for inline rendering (1 m simplification is
            # imperceptible at city zoom) and keep only the columns we display.
            keep = ['geometry', 'bldgid', 'usage', 'property_value',
                    'hazard', 'vulnerability', 'exposure',
                    'severity', 'capacity', 'risk_score']
            slim = scored[keep].copy()
            slim['geometry'] = (slim.set_geometry('geometry').to_crs(3857)
                                  .geometry.simplify(1.0, preserve_topology=True)
                                  .to_crs(4326))
            for c in ['hazard', 'vulnerability', 'exposure',
                      'severity', 'capacity', 'risk_score']:
                slim[c] = slim[c].round(3)
            slim['property_value'] = slim['property_value'].round(0)

            tooltip_fields = ['bldgid', 'usage', 'risk_score',
                              'hazard', 'vulnerability', 'exposure',
                              'severity', 'capacity', 'property_value']
            LAYER_SPEC = [
                ('risk_score',    'Final risk score', True),
                ('hazard',        'Hazard',           False),
                ('vulnerability', 'Vulnerability',    False),
                ('exposure',      'Exposure',         False),
                ('severity',      'Severity',         False),
                ('capacity',      'Capacity',         False),
            ]

            def _layer_vmax(series, floor=0.1):
                s = series.dropna()
                if len(s) == 0:
                    return 1.0
                return max(floor, round(float(s.max()) + 0.005, 2))

            def _layer_vmin(series, min_gap=0.15, buffer=0.02):
                # Auto-detect a 'gap' in the distribution. Returns 0 if
                # the data is continuous (no bimodal gap).
                s = series.dropna().to_numpy()
                s = s[s > 1e-3]
                if len(s) < 50:
                    return 0.0
                s = np.sort(s)
                diffs = np.diff(s)
                if len(diffs) == 0:
                    return 0.0
                max_idx = int(np.argmax(diffs))
                if float(diffs[max_idx]) < min_gap:
                    return 0.0
                return max(0.0, round(float(s[max_idx + 1]) - buffer, 2))

            def build_components_map(use_gap_vmin: bool, out_suffix: str = ''):
                cent = slim.geometry.unary_union.centroid
                fmap = folium.Map(location=[cent.y, cent.x], zoom_start=14,
                                  tiles='cartodbpositron')
                for col, label, show in LAYER_SPEC:
                    vmin = _layer_vmin(slim[col]) if use_gap_vmin else 0.0
                    vmax = _layer_vmax(slim[col])
                    lc = cm.LinearColormap(SCORE_PALETTE, vmin=vmin, vmax=vmax)
                    lc.caption = f'{label} ({vmin:.2f} – {vmax:.2f})'

                    def _styler(feat, _lc=lc, _col=col):
                        v = float(feat['properties'].get(_col, 0) or 0)
                        return {'fillColor': _lc(v), 'color': '#333',
                                'weight': 0.3, 'fillOpacity': 0.85}

                    fg = folium.FeatureGroup(name=label, show=show)
                    folium.GeoJson(
                        slim,
                        style_function=_styler,
                        tooltip=folium.GeoJsonTooltip(
                            fields=tooltip_fields, localize=True),
                    ).add_to(fg)
                    fg.add_to(fmap)
                    lc.add_to(fmap)
                # Top-10 highest-risk buildings — bright outline + numbered marker
                top10 = slim.nlargest(10, 'risk_score').reset_index(drop=True)
                fg_top = folium.FeatureGroup(name='Top 10 highest risk', show=True)
                folium.GeoJson(
                    top10,
                    style_function=lambda _f: {
                        'fillColor': '#00ffff',
                        'color': '#00ffff',
                        'weight': 3,
                        'fillOpacity': 0.0,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, localize=True),
                ).add_to(fg_top)
                for i, row in top10.iterrows():
                    c = row.geometry.centroid
                    folium.Marker(
                        location=[c.y, c.x],
                        icon=folium.DivIcon(
                            icon_size=(28, 28),
                            icon_anchor=(14, 14),
                            html=(f'<div style="background:#00ffff;color:#000;'
                                  f'border:2px solid #003344;border-radius:50%;'
                                  f'width:24px;height:24px;line-height:20px;'
                                  f'text-align:center;font-weight:700;'
                                  f'font-family:sans-serif;font-size:12px;'
                                  f'box-shadow:0 1px 3px rgba(0,0,0,.4);">'
                                  f'{i+1}</div>'),
                        ),
                    ).add_to(fg_top)
                fg_top.add_to(fmap)

                folium.LayerControl(collapsed=False).add_to(fmap)
                os.makedirs('outputs', exist_ok=True)
                fname = f'outputs/{SCENARIO.name}_risk_components{out_suffix}.html'
                fmap.save(fname)
                return fmap

            # Render 6a — raw scale
            fmap_raw = build_components_map(use_gap_vmin=False, out_suffix='_raw')
            fmap_raw   # ← renders inline
            """
        ),
        md("### 6b · Optimised for visualisation (auto-detected `vmin` per layer)"),
        code(
            """
            fmap_opt = build_components_map(use_gap_vmin=True, out_suffix='')
            fmap_opt   # ← renders inline
            """
        ),
        md(
            "## 6c · Output schema and intended downstream use",
            "",
            "### Output GeoJSON column schema",
            "",
            "Each feature in `outputs/" + scenario + "_risk_scored_5component.geojson` "
            "carries the original building polygon (EPSG:4326) plus the columns below. "
            "All scores are dimensionless in **[0, 1]** unless noted.",
            "",
            "| Column | Type | Range | Interpretation |",
            "|---|---|---|---|",
            "| `bldgid` | int | — | building identifier from the synthetic dataset (not unique) |",
            "| `usage` | str | — | Residential / Commercial / Industrial / Other |",
            "| `property_value` | float (USD) | ≥ 0 | replacement-value proxy from the synthetic data |",
            "| `building_sqft` | float | ≥ 0 | footprint area in ft² |",
            "| `hazard_climate` | float | [0, 1] | P(burn) from the MTBS-trained landscape classifier |",
            "| `hazard_sim` | float | [0, 1] | scenario-specific hazard from the per-property arrival time `max(0, 1 − arrival/12)` |",
            "| `hazard` | float | [0, 1] | `max(hazard_climate, hazard_sim)` — final hazard input |",
            "| `ignition_hour_from_sim` | float (h) or NaN | [0, 12] | when the building first ignites in the OroraTech sim; NaN if not ignited |",
            "| `vulnerability` | float | [0, 1] | DINS-trained P(damage \\| fire), with up to −40 % defensible-space modifier |",
            "| `exposure` | float | [0, 1] | rank-normalized blend of value (0.5) + footprint × usage (0.3) + adjacency (0.2), optionally × 1.15 in WUI |",
            "| `exposure_value`, `exposure_footprint`, `exposure_adjacency` | float | [0, 1] | individual exposure sub-scores |",
            "| `severity` | float | [0, 1] | usage-weighted occupancy-load rank (loss-of-life / disruption magnitude) |",
            "| `capacity` | float | [0, 1] | weighted blend of the five resilience sub-scores below |",
            "| `cap_defensible_space` | float | [0, 1] | mean of `zone_0_compliant`, `zone_1_defensible`, `zone_2_defensible` |",
            "| `cap_construction_quality` | float | [0, 1] | post-2000 build + non-combust siding/roof + fire-resistant siding |",
            "| `cap_ember_resistance` | float | [0, 1] | fine vent screens + tempered windows + enclosed eaves |",
            "| `cap_access` | float | [0, 1] | `exp(−dist_to_road_m / 60)` |",
            "| `cap_wui_maintenance` | float | [0, 1] | rank of `acres_treated_wui` (0.5 if null) |",
            "| `risk_gross` | float | [0, 1] | `(H · V · E · S)^(1/4)` |",
            "| `risk_net` | float | [0, 1] | `risk_gross · (1 − 0.5 · capacity)` |",
            "| **`risk_score`** | **float** | **[0, 1]** | **final per-property wildfire risk score**, min–max rescaled to [0, 1] within this scenario |",
            "| `risk_decile` | int | 1 – 10 | decile bucket of `risk_score` (1 = lowest, 10 = highest) |",
            "",
            "### How to interpret `risk_score`",
            "",
            "A value near **1.0** means this property is in the worst combination of all five "
            "amplifiers (fire reaches it, it's vulnerable to damage if reached, it's "
            "high-value / densely surrounded, and losing it has major occupancy/service "
            "consequences) with little resilience to absorb that loss. A value near **0** "
            "means at least one component is essentially zero — either fire never reaches "
            "the property, or its construction is fire-resistant, or its replacement value "
            "and occupancy are negligible, or its resilience is high enough to dampen "
            "an otherwise high score by half.",
            "",
            "The `risk_score` is **scenario-relative**: a 0.7 in Forest is comparable to "
            "another 0.7 in Forest, but not directly comparable across to Prairie because "
            "the rescaling is per-scenario.",
            "",
            "### Intended downstream use",
            "",
            "*(Per §6.1.3.e of the manual — which of the listed uses we'd expect this model to serve.)*",
            "",
            "* **Community Wildfire Protection Planning** — rank buildings to prioritise outreach and education.",
            "* **Mitigation prioritisation** — identify the highest-`capacity_dampener` opportunity targets (low capacity + high gross risk); use the `cap_*` sub-scores to point at the *specific* mitigation that would most reduce risk for a given parcel (defensible-space inspection, structural hardening, ember-resistant vents, road-access improvements, fuel-break maintenance).",
            "* **Pre-incident response planning** — high-`risk_score` clusters identify likely first-priority defensive zones if this scenario occurred.",
            "* **Resource allocation** — choose where to position water tenders, brush units, evacuation marshals.",
            "* **Insurance / loss-modelling triage** — single rank score for portfolio sorting (not for premium-setting; the model isn't calibrated against historical losses).",
            "",
            "**Not appropriate for:**  pricing decisions, life-safety alerts to individual residents, or any cross-scenario comparison (the rescale is per-scenario).",
            "",
            "### Hardware / runtime",
            "",
            "* CPU: any modern laptop class (no GPU required).",
            "* Memory: ~3 GB peak when loading all DINS + MTBS + scored GeoJSON in pandas/geopandas.",
            "* Notebook end-to-end (with cached scored file and cached real-world reference data): **≈ 2 min** for vulnerability training + override + 5-component aggregation + map rendering on Forest, ≈ 1 min on Prairie.",
            "* Cold start (no caches): add ~15 min for the upstream `python run.py` Monte-Carlo simulation, plus ~5–10 min for the one-off DINS / MTBS / FPA-FOD / WUI downloads.",
            "",
            "### Validation",
            "",
            "Per §6.1.3.f: the model is **not formally validated against historical losses**. "
            "The vulnerability component reports MAE ≈ 0.28 / R² ≈ 0.47 on a held-out 20 % "
            "split of DINS records — usable for relative ranking, not for absolute "
            "damage-probability claims. The hazard climate classifier reports per-scenario "
            "AUC ≈ 0.95 (Forest) and 0.99 (Prairie) on held-out grid points, but this is "
            "in-sample to a single historical fire per site. The OroraTech sim is the "
            "authoritative scenario hazard for Forest; we treat its ignition timestamps "
            "as ground truth for `hazard_sim`.",
        ),
        md("## 7 · Output"),
        md(
            f"The per-property risk score is written to "
            f"`outputs/{scenario}_risk_scored_5component.geojson` — each feature "
            "carries all five component scores plus the final blended risk.",
        ),
        code(
            """
            os.makedirs('outputs', exist_ok=True)
            out_path = f"outputs/{SCENARIO.name}_risk_scored_5component.geojson"
            # risk_raw is the obsolete 3-component intermediate (H·V·E)^(1/3);
            # superseded by risk_gross. index_right is a stale sjoin artefact.
            drop_cols = [c for c in ('risk_raw', 'index_right') if c in scored.columns]
            scored.drop(columns=drop_cols).to_file(out_path, driver='GeoJSON')
            print(f"output: {out_path}")
            print(f"size:   {os.path.getsize(out_path)/1024:.0f} KB")
            print("\\nfields per feature:")
            for c in ['bldgid', 'hazard_climate', 'hazard_sim', 'hazard',
                      'vulnerability', 'exposure', 'severity', 'capacity',
                      'cap_defensible_space', 'cap_construction_quality',
                      'cap_ember_resistance', 'cap_access', 'cap_wui_maintenance',
                      'risk_gross', 'risk_net', 'risk_score', 'risk_decile']:
                print(f"  - {c}")
            """
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main():
    forest_sim_dir = "data_cache/fire_spread_simulations/forest_scenario"
    prairie_sim_dir = "data_cache/fire_spread_simulations/prairie_scenario"
    notebooks = {
        "Forest_submission.ipynb": build_notebook(
            "Forest", location="Lake Tahoe, CA",
            ignition_date="2024-10-27 10:00", n_buildings=4417,
            wildland_sim_geojson=f"{forest_sim_dir}/forecast_forest_wui.geojson",
            property_sim_geojson=f"{forest_sim_dir}/forecast_forest_properties.geojson",
            sim_animation_html=f"{forest_sim_dir}/forecast_forest_wui_map_coupled_20m_5m.html",
        ),
        "Prairie_submission.ipynb": build_notebook(
            "Prairie", location="Amarillo, TX",
            ignition_date="2026-03-09 10:00", n_buildings=2085,
            wildland_sim_geojson=f"{prairie_sim_dir}/forecast_prairie_wui_minSpeed0015.geojson",
            property_sim_geojson=f"{prairie_sim_dir}/forecast_prairie_properties.geojson",
            sim_animation_html=f"{prairie_sim_dir}/forecast_prairie_wui_map_minSpeed0015.html",
        ),
    }
    for fname, nb in notebooks.items():
        path = os.path.join(BASE_DIR, fname)
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()

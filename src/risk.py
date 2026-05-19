"""Top-level pipeline: combine Hazard × Vulnerability × Exposure → risk score."""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd

from . import data_sources, exposure, hazard_climate, hazard_simulation, vulnerability
from .config import (
    BASE_DIR, HAZARD_CLIMATE_WEIGHT, HAZARD_SIM_WEIGHT,
    SCENARIOS, Scenario,
)
from .data_sources import SCENARIO_STATE
from .loaders import load_buildings


def _combine_hazard(climate: np.ndarray, sim: np.ndarray) -> np.ndarray:
    return np.clip(
        HAZARD_CLIMATE_WEIGHT * climate + HAZARD_SIM_WEIGHT * sim, 0, 1,
    )


def _geo_mean(h: np.ndarray, v: np.ndarray, e: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return np.cbrt(np.clip(h, eps, 1) * np.clip(v, eps, 1) * np.clip(e, eps, 1))


def _rescale_unit(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def run_scenario(
    scenario: Scenario,
    vuln_model,
    mtbs,
    realizations: int = 100,
    downsample: int = 15,
) -> gpd.GeoDataFrame:
    print(f"\n{'=' * 60}\nSCENARIO: {scenario.name}\n{'=' * 60}")
    buildings = load_buildings(scenario)
    print(f"[load] {len(buildings):,} buildings")

    # ---- HAZARD: climate (MTBS) ----
    buildings = hazard_climate.attach_landscape_features(buildings, scenario)
    climate_model, feat_cols = hazard_climate.train_climate_hazard(scenario, mtbs)
    h_climate = hazard_climate.score_buildings(buildings, climate_model, feat_cols)
    print(f"[hazard-climate] mean={h_climate.mean():.3f}  std={h_climate.std():.3f}")

    # ---- HAZARD: simulation (Monte Carlo) ----
    pburn, grid = hazard_simulation.run_simulation(
        scenario, realizations=realizations, downsample=downsample,
    )
    h_sim = hazard_simulation.score_buildings(buildings, pburn, grid)
    print(f"[hazard-sim] mean={h_sim.mean():.3f}  std={h_sim.std():.3f}")

    h = _combine_hazard(h_climate, h_sim)

    # ---- VULNERABILITY ----
    v = vulnerability.score_buildings(vuln_model, buildings)
    print(f"[vuln] mean={v.mean():.3f}  std={v.std():.3f}")

    # ---- EXPOSURE ----
    wui = data_sources.get_wui(SCENARIO_STATE.get(scenario.name, ""))
    modifier = exposure.wui_modifier_from_geodataframe(buildings, wui)
    exp_parts = exposure.compute_exposure(buildings, wui_modifier=modifier)
    e = exp_parts["exposure"]
    print(f"[expo] mean={e.mean():.3f}  std={e.std():.3f}  "
          f"(WUI modifier: {'on' if modifier is not None else 'off'})")

    # ---- RISK = (H × V × E)^(1/3), rescaled ----
    risk_raw = _geo_mean(h, v, e)
    risk_score = _rescale_unit(risk_raw)
    risk_decile = pd.qcut(risk_score, 10, labels=False, duplicates="drop") + 1

    buildings["hazard_climate"] = np.round(h_climate, 4)
    buildings["hazard_sim"] = np.round(h_sim, 4)
    buildings["hazard"] = np.round(h, 4)
    buildings["vulnerability"] = np.round(v, 4)
    buildings["exposure_value"] = np.round(exp_parts["value_rank"], 4)
    buildings["exposure_footprint"] = np.round(exp_parts["footprint_rank"], 4)
    buildings["exposure_adjacency"] = np.round(exp_parts["adjacency"], 4)
    buildings["exposure"] = np.round(e, 4)
    buildings["risk_raw"] = np.round(risk_raw, 4)
    buildings["risk_score"] = np.round(risk_score, 4)
    buildings["risk_decile"] = risk_decile.astype(int)

    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scenario.name}_risk_scored.geojson")
    # Drop join artefacts that occasionally leak through the pipeline.
    buildings.drop(columns=[c for c in ("index_right",) if c in buildings.columns],
                   inplace=True)
    buildings.to_file(out_path, driver="GeoJSON")
    print(f"[save] {out_path}")

    print(f"\n--- {scenario.name} score summary ---")
    for col in ["hazard_climate", "hazard_sim", "hazard", "vulnerability",
                "exposure", "risk_score"]:
        vals = buildings[col]
        print(f"  {col:20s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")
    return buildings


def run_all(scenarios: list[str], realizations: int, downsample: int):
    print("=" * 60)
    print("STEP 0: real-world data acquisition")
    print("=" * 60)
    dins = data_sources.get_dins()
    if dins is None:
        raise RuntimeError("DINS unavailable — cannot train vulnerability model.")
    mtbs = data_sources.get_mtbs()
    if mtbs is None:
        print("[MTBS] unavailable — climate hazard will fall back to heuristic")
        mtbs = gpd.GeoDataFrame()

    print("=" * 60)
    print("STEP 1: train vulnerability model on DINS")
    print("=" * 60)
    vuln_model = vulnerability.train_vulnerability_model(dins)

    results = {}
    for name in scenarios:
        scenario = SCENARIOS[name]
        results[name] = run_scenario(
            scenario, vuln_model, mtbs,
            realizations=realizations, downsample=downsample,
        )
    return results

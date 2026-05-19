"""Exposure component: per-building value/consequence weight ∈ [0, 1].

Captures *what* is at stake if a building burns. Combines:
  * property value  — log-scaled, rank-normalised
  * physical footprint × usage type — proxy for the size of the loss
  * adjacency — denser clusters / smaller structural separation amplify
    exposure because losing one structure threatens neighbours
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from .config import (
    EXPOSURE_W_ADJACENCY, EXPOSURE_W_FOOTPRINT, EXPOSURE_W_VALUE,
    USAGE_WEIGHT,
)


def _rank_pct(arr: np.ndarray) -> np.ndarray:
    s = pd.Series(arr)
    return s.rank(pct=True, method="average").to_numpy()


def _usage_weight(s: pd.Series) -> np.ndarray:
    return s.fillna("other").astype(str).str.lower().map(USAGE_WEIGHT).fillna(USAGE_WEIGHT["other"]).to_numpy()


def compute_exposure(
    buildings: gpd.GeoDataFrame,
    wui_modifier: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return a dict with sub-component scores and final exposure ∈ [0, 1]."""
    n = len(buildings)

    pv = pd.to_numeric(buildings.get("property_value"), errors="coerce").fillna(0).to_numpy()
    pv_log = np.log10(np.clip(pv, 1, None))
    value_rank = _rank_pct(pv_log)

    sqft = pd.to_numeric(buildings.get("building_sqft"), errors="coerce").fillna(
        buildings.get("building width [m]", pd.Series(0, index=buildings.index)).fillna(0)
        * buildings.get("building length [m]", pd.Series(0, index=buildings.index)).fillna(0)
        * 10.764  # m² → ft²
    ).to_numpy()
    stories = pd.to_numeric(buildings.get("stories_above_ground"), errors="coerce").fillna(1).to_numpy()
    footprint_volume = sqft * np.clip(stories, 1, None)

    usage = buildings.get("usage", pd.Series("other", index=buildings.index))
    uw = _usage_weight(usage)
    footprint_rank = _rank_pct(footprint_volume) * uw
    # Re-normalise to [0, 1] (usage weight can push slightly above 1)
    footprint_rank = footprint_rank / max(footprint_rank.max(), 1e-9)

    nc = pd.to_numeric(buildings.get("neighborhood_count"), errors="coerce")
    sep = pd.to_numeric(buildings.get("structural_sep_dist"), errors="coerce")
    if nc.notna().sum() > 0:
        nc_rank = _rank_pct(nc.fillna(nc.median()).to_numpy())
    else:
        nc_rank = np.full(n, 0.5)
    if sep.notna().sum() > 0:
        # Smaller separation → higher exposure → rank inversely
        sep_rank = 1.0 - _rank_pct(sep.fillna(sep.median()).to_numpy())
    else:
        sep_rank = np.full(n, 0.5)
    adjacency = 0.5 * nc_rank + 0.5 * sep_rank

    exposure = (
        EXPOSURE_W_VALUE * value_rank
        + EXPOSURE_W_FOOTPRINT * footprint_rank
        + EXPOSURE_W_ADJACENCY * adjacency
    )

    if wui_modifier is not None:
        exposure = np.clip(exposure * wui_modifier, 0, 1)

    return {
        "value_rank": value_rank,
        "footprint_rank": footprint_rank,
        "adjacency": adjacency,
        "exposure": np.clip(exposure, 0, 1),
    }


def wui_modifier_from_geodataframe(
    buildings: gpd.GeoDataFrame,
    wui: gpd.GeoDataFrame | None,
) -> np.ndarray | None:
    """Per-building modifier ∈ [1.0, 1.15] based on WUI class:
       intermix or interface → 1.15, otherwise 1.0. Returns None if WUI absent."""
    if wui is None or len(wui) == 0:
        return None
    try:
        if wui.crs != buildings.crs:
            wui = wui.to_crs(buildings.crs)
    except Exception:
        return None
    # USFS WUI shapefiles encode "WUIFLAG2020" as one of:
    #   0 = no WUI, 1 = intermix, 2 = interface
    wui_col = next((c for c in wui.columns if "WUIFLAG" in c.upper()), None)
    if wui_col is None:
        return None
    wui_active = wui[wui[wui_col].astype(str).isin(["1", "2"])]
    if len(wui_active) == 0:
        return np.full(len(buildings), 1.0)
    joined = gpd.sjoin(
        buildings[["geometry"]].copy(), wui_active[["geometry"]].copy(),
        how="left", predicate="intersects",
    )
    in_wui = joined.index_right.notna().to_numpy()
    # If a building intersects multiple WUI polygons, sjoin duplicates rows; collapse
    if len(in_wui) != len(buildings):
        per_building = joined.groupby(joined.index).size() > 0
        in_wui = per_building.reindex(buildings.index, fill_value=False).to_numpy()
    return np.where(in_wui, 1.15, 1.0)

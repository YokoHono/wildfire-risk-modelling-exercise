"""5-component risk framework.

Risk is the potential consequence to a system, society or community, often
determined probabilistically as a function of:

    1. Hazard                          — probability/intensity of the event
    2. Exposure                        — what's exposed to the event
    3. Vulnerability                   — fragility given the event hits
    4. Severity of consequence         — magnitude of impact if loss occurs
    5. Capacity to withstand impact    — resilience / mitigation present

Each component is rated on a [0, 1] scale per property. Components 1–4
amplify risk; component 5 *dampens* it. Aggregation:

    risk_gross = (H · V · E · S) ** (1/4)        # geometric mean of amplifiers
    risk_net   = risk_gross · (1 - β · C)        # resilience modifier (β = 0.5)
    risk_score = min-max rescale of risk_net to [0, 1] within the scenario
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

# ----------------------------------------------------------------------------
# 1. HAZARD — from a time-stamped fire-perimeter sequence (OroraTech sim)
# ----------------------------------------------------------------------------

def hazard_from_fire_sequence(
    buildings: gpd.GeoDataFrame,
    sim_polygons: gpd.GeoDataFrame,
    ignition_time: pd.Timestamp,
    horizon_hours: float = 12.0,
) -> np.ndarray:
    """For each building, look up the earliest time-stamped polygon whose
    geometry contains the building's centroid. Convert arrival time to
    hazard ∈ [0, 1]:

        hazard = 1 - (arrival_hours / horizon_hours)      if reached
        hazard = 0                                        otherwise

    A property reached at ignition has hazard = 1, never-reached = 0.
    """
    sims = sim_polygons.copy()
    sims["date"] = pd.to_datetime(sims["date"], utc=True)
    sims = sims.sort_values("date").reset_index(drop=True)
    sims["t_hours"] = (sims["date"] - ignition_time).dt.total_seconds() / 3600.0

    # Use the union of polygon at each timestamp (handles MultiPolygon naturally)
    # Centroid of each building (reproject to UTM for accurate centroid, back to WGS84)
    centroids = buildings.to_crs(epsg=3857).geometry.centroid
    centroids = gpd.GeoSeries(centroids, crs=3857).to_crs(buildings.crs)

    # Spatial index over the centroid points
    bldg = gpd.GeoDataFrame(geometry=centroids, crs=buildings.crs).reset_index(drop=True)
    n = len(bldg)
    arrival = np.full(n, np.inf)

    # Walk sim polygons in chronological order; only check buildings not yet
    # marked as reached (cumulative spread → once reached, stays reached).
    for _, row in sims.iterrows():
        if not np.isfinite(arrival).all():  # there are still unreached buildings
            unreached_idx = np.where(np.isinf(arrival))[0]
            if len(unreached_idx) == 0:
                break
            sub = bldg.iloc[unreached_idx]
            within = sub.within(row.geometry)
            now_reached = unreached_idx[within.to_numpy()]
            arrival[now_reached] = row["t_hours"]
        else:
            break

    reached = np.isfinite(arrival)
    hazard = np.zeros(n, dtype=np.float32)
    hazard[reached] = np.clip(
        1.0 - arrival[reached] / horizon_hours, 0.0, 1.0,
    ).astype(np.float32)
    return hazard, arrival


# ----------------------------------------------------------------------------
# 4. SEVERITY OF CONSEQUENCE
# ----------------------------------------------------------------------------

# Severity captures the *non-property* component of "how bad is this loss":
#   - Loss-of-life potential: residential occupancy > commercial > industrial
#   - Occupants per building proxy: sqft × stories with a residential weighting
#   - Service disruption: industrial / commercial weighted lower since
#     occupants are typically present only during business hours
#
# Property value is *not* included here — that already lives in Exposure.

_SEVERITY_USAGE_WEIGHT = {
    "single residence": 1.0,
    "multi residence": 1.4,
    "multiple residence": 1.4,
    "multifamily": 1.4,
    "mixed commercial/residential": 1.2,
    "residential": 1.0,
    "commercial": 0.6,
    "industrial": 0.5,
    "other": 0.8,
}


def severity_score(buildings: gpd.GeoDataFrame) -> np.ndarray:
    """Severity ∈ [0, 1] reflecting loss-of-life and disruption potential."""
    usage = buildings.get("usage", pd.Series("other", index=buildings.index))
    uw = (
        usage.fillna("other").astype(str).str.lower()
        .map(_SEVERITY_USAGE_WEIGHT).fillna(_SEVERITY_USAGE_WEIGHT["other"])
        .to_numpy()
    )
    sqft = pd.to_numeric(buildings.get("building_sqft"), errors="coerce").fillna(0).to_numpy()
    stories = pd.to_numeric(buildings.get("stories_above_ground"), errors="coerce").fillna(1).to_numpy()
    occ_units = sqft * np.clip(stories, 1, None) * uw

    # Rank to [0, 1] so the score is scenario-relative and robust to outliers
    s = pd.Series(occ_units).rank(pct=True, method="average").to_numpy()
    return s.astype(np.float32)


# ----------------------------------------------------------------------------
# 5. CAPACITY TO WITHSTAND IMPACT
# ----------------------------------------------------------------------------

# Capacity is the inverse of vulnerability at the *system* level: the
# resilience, preparedness, and access features that mitigate damage when a
# property is impacted by fire. Components:
#   - defensible space (zone_0_compliant + zone_1_defensible + zone_2_defensible)
#   - modern construction (post-2000 building code, fire-resistant features)
#   - ember-resistant detailing (fine vent screens, tempered windows)
#   - firefighter / evacuation access (distance to nearest road, inverted)
#   - wildland-area maintenance (acres_treated_wui — best-effort; nullable in data)
#
# Each sub-component → [0, 1]. Final capacity = weighted mean.

_CAPACITY_WEIGHTS = {
    "defensible_space": 0.35,
    "construction_quality": 0.25,
    "ember_resistance": 0.20,
    "access": 0.15,
    "wui_maintenance": 0.05,
}


def _safe_pct_rank(x: np.ndarray, *, invert: bool = False) -> np.ndarray:
    """Rank-normalize to [0, 1]; invert=True puts the smallest values at 1."""
    r = pd.Series(x).rank(pct=True, method="average").to_numpy(dtype=np.float32)
    return 1.0 - r if invert else r


def capacity_score(buildings: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    """Capacity ∈ [0, 1] plus its sub-components."""
    n = len(buildings)

    def _col(name, default=0):
        s = pd.to_numeric(buildings.get(name), errors="coerce").fillna(default)
        return s.to_numpy(dtype=np.float32)

    # 1. Defensible space — mean of three zone-compliance booleans
    ds = np.nanmean(
        np.vstack([_col("zone_0_compliant"), _col("zone_1_defensible"), _col("zone_2_defensible")]),
        axis=0,
    ).astype(np.float32)

    # 2. Modern construction — indicators of post-2000 build + fire-resistant materials
    cq = (
        0.4 * _col("year_built_2000_2025")
        + 0.3 * _col("siding_fire_resist")
        + 0.2 * _col("siding_non_combust")
        + 0.1 * _col("roof_non_combust")
    )
    cq = np.clip(cq, 0, 1).astype(np.float32)

    # 3. Ember resistance — fine vent screens, tempered windows, enclosed eaves
    er = (
        0.30 * _col("ventscreens_1_8")             # 1/8" is the finest in our schema
        + 0.20 * _col("ventscreens_1_4")
        + 0.20 * _col("window_multi_tempered")
        + 0.15 * _col("window_single_tempered")
        + 0.15 * _col("eaves_enclosed")
    )
    er = np.clip(er, 0, 1).astype(np.float32)

    # 4. Access — closer to a road = faster response and easier evacuation.
    # Uses the dist_to_road_m feature if present (added by hazard_climate flow).
    if "dist_to_road_m" in buildings.columns:
        dr = _col("dist_to_road_m", default=300.0)
        # 0 m → 1.0, 100 m+ → 0; smooth exponential decay
        access = np.clip(np.exp(-dr / 60.0), 0, 1).astype(np.float32)
    else:
        access = np.full(n, 0.5, dtype=np.float32)

    # 5. WUI maintenance — `acres_treated_wui` is nullable per data README; if
    # available, rank it. Otherwise treat as neutral (0.5).
    treated = pd.to_numeric(buildings.get("acres_treated_wui"), errors="coerce")
    if treated.notna().sum() > 0:
        wui_m = _safe_pct_rank(treated.fillna(treated.median()).to_numpy())
    else:
        wui_m = np.full(n, 0.5, dtype=np.float32)

    w = _CAPACITY_WEIGHTS
    capacity = (
        w["defensible_space"] * ds
        + w["construction_quality"] * cq
        + w["ember_resistance"] * er
        + w["access"] * access
        + w["wui_maintenance"] * wui_m
    ).astype(np.float32)
    capacity = np.clip(capacity, 0, 1)
    return {
        "capacity": capacity,
        "cap_defensible_space": ds,
        "cap_construction_quality": cq,
        "cap_ember_resistance": er,
        "cap_access": access,
        "cap_wui_maintenance": wui_m,
    }


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate_5component(
    hazard: np.ndarray,
    vulnerability: np.ndarray,
    exposure: np.ndarray,
    severity: np.ndarray,
    capacity: np.ndarray,
    *,
    capacity_max_reduction: float = 0.5,
) -> dict[str, np.ndarray]:
    """Combine the five components into a single risk score ∈ [0, 1].

    Geometric mean of the four amplifiers, then dampened by capacity:

        risk_gross = (H · V · E · S) ** (1/4)
        risk_net   = risk_gross · (1 - capacity_max_reduction · C)
        risk_score = min-max rescale of risk_net to [0, 1]
    """
    eps = 1e-6
    H = np.clip(hazard, eps, 1.0)
    V = np.clip(vulnerability, eps, 1.0)
    E = np.clip(exposure, eps, 1.0)
    S = np.clip(severity, eps, 1.0)
    C = np.clip(capacity, 0.0, 1.0)

    risk_gross = (H * V * E * S) ** 0.25
    risk_net = risk_gross * (1.0 - capacity_max_reduction * C)
    lo, hi = float(np.min(risk_net)), float(np.max(risk_net))
    risk_score = (risk_net - lo) / max(hi - lo, 1e-9)
    risk_decile = pd.qcut(risk_score, 10, labels=False, duplicates="drop") + 1
    return {
        "risk_gross": risk_gross.astype(np.float32),
        "risk_net": risk_net.astype(np.float32),
        "risk_score": risk_score.astype(np.float32),
        "risk_decile": risk_decile.astype(int),
    }

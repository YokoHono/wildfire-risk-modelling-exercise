"""Climatological hazard: P(burn) from landscape features, trained on MTBS.

For each scenario we find MTBS fire perimeters that overlap the simulation
extent, label a grid of sample points as burned/unburned, and train a
classifier on the simulation's own LANDFIRE-derived rasters. The trained
model is applied to each building centroid.
"""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import RASTER_FEATURES, Scenario
from .loaders import (
    building_centroid_lonlat,
    clean_raster,
    compute_slope_deg,
    sample_array_at_points,
    sample_raster_at_points,
)


LANDSCAPE_FEATURE_COLS = [f"raster_{f}" for f in RASTER_FEATURES] + [
    "raster_slope", "ignition_density_5km", "ignition_density_25km",
    "treelist_cbd_30m", "treelist_cbh_30m", "treelist_count_30m",
    "canopy_rhof_total", "canopy_moist", "canopy_fuel_depth",
    "dist_to_road_m",
]

# Lon/lat → metres scale factor at the scenario's latitude (used only to
# convert km radii into degrees for the KDTree distance query).
_DEG_PER_METRE = 1.0 / 111320.0


def _scenario_bounds(scenario: Scenario):
    with rasterio.open(os.path.join(scenario.dir, f"{scenario.prefix}_elevation.tif")) as src:
        return src.bounds, src.transform, src.res, (src.height, src.width)


def _ignition_density(
    sample_xs: np.ndarray, sample_ys: np.ndarray,
    fire_lons: np.ndarray, fire_lats: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """Count historical ignitions within `radius_km` of each sample point.

    Projects all coordinates to a local equidistant XY (metres) centred on
    the mean sample location to make a simple KDTree distance query valid.
    """
    if len(fire_lons) == 0 or len(sample_xs) == 0:
        return np.zeros(len(sample_xs))
    lat0 = float(np.mean(sample_ys))
    mx = 111320.0
    my = 111320.0 * np.cos(np.radians(lat0))
    fire_xy = np.column_stack([fire_lons * my, fire_lats * mx])
    sample_xy = np.column_stack([sample_xs * my, sample_ys * mx])
    tree = cKDTree(fire_xy)
    return np.array(tree.query_ball_point(sample_xy, r=radius_km * 1000.0, return_length=True),
                     dtype=float)


def _load_ignition_points(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Return (lons, lats) of FPA-FOD ignitions within ~1° of the scenario."""
    from . import data_sources
    path = data_sources.get_fpa_fod()
    if not path:
        return np.array([]), np.array([])
    bounds, *_ = _scenario_bounds(scenario)
    margin = 1.0  # degrees, ≈ 100 km
    df = data_sources.load_fpa_fod_points(
        path,
        (bounds.left - margin, bounds.bottom - margin,
         bounds.right + margin, bounds.top + margin),
    )
    return df["LONGITUDE"].to_numpy(), df["LATITUDE"].to_numpy()


def _build_training_grid(scenario: Scenario, mtbs: gpd.GeoDataFrame) -> pd.DataFrame | None:
    bounds, transform, res, _ = _scenario_bounds(scenario)
    sim_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    overlapping = mtbs[mtbs.intersects(sim_box)]
    if len(overlapping) == 0:
        print(f"[hazard-climate {scenario.name}] no MTBS fires overlap simulation extent")
        return None

    print(f"[hazard-climate {scenario.name}] {len(overlapping)} overlapping MTBS fire(s):")
    for _, fire in overlapping.iterrows():
        print(f"    - {fire.get('Incid_Name')}  {fire.get('Ig_Date')}  {fire.get('BurnBndAc')} ac")

    fire_union = overlapping.geometry.unary_union

    # sample grid at 3× raster spacing
    xs = np.arange(bounds.left + res[0], bounds.right, res[0] * 3)
    ys = np.arange(bounds.bottom + res[1], bounds.top, res[1] * 3)
    xx, yy = np.meshgrid(xs, ys)
    px = xx.ravel()
    py = yy.ravel()
    labels = np.fromiter(
        (1 if fire_union.contains(Point(x, y)) else 0 for x, y in zip(px, py)),
        dtype=int, count=px.size,
    )

    feats = pd.DataFrame({"x": px, "y": py, "burned": labels})
    for fname in RASTER_FEATURES:
        path = os.path.join(scenario.dir, f"{scenario.prefix}_{fname}.tif")
        if os.path.exists(path):
            feats[f"raster_{fname}"] = sample_raster_at_points(path, px, py)

    # slope from elevation
    elev_path = os.path.join(scenario.dir, f"{scenario.prefix}_elevation.tif")
    with rasterio.open(elev_path) as src:
        elev = clean_raster(src.read(1))
        slope = compute_slope_deg(elev, src.transform)
        feats["raster_slope"] = sample_array_at_points(slope, src.transform, px, py)

    # FPA-FOD historical ignition density (best-effort)
    fire_lons, fire_lats = _load_ignition_points(scenario)
    feats["ignition_density_5km"] = _ignition_density(px, py, fire_lons, fire_lats, 5.0)
    feats["ignition_density_25km"] = _ignition_density(px, py, fire_lons, fire_lats, 25.0)
    if len(fire_lons) > 0:
        print(f"[hazard-climate {scenario.name}] "
              f"FPA-FOD ignitions in area: {len(fire_lons):,}  "
              f"(mean within 5km of grid = {feats['ignition_density_5km'].mean():.1f})")

    # Treelist canopy summary (per-point neighbourhood)
    from . import canopy as _canopy
    from . import roads as _roads
    tl = _canopy.sample_treelist_at_points(scenario, px, py, radius_m=30.0)
    feats["treelist_cbd_30m"] = tl["sum_cbd"]
    feats["treelist_cbh_30m"] = tl["mean_cbh"]
    feats["treelist_count_30m"] = tl["tree_count"]

    # Voxel-derived vertical-integral canopy fuel (best-effort)
    vx = _canopy.sample_voxel_at_points(scenario, px, py)
    if vx is not None:
        feats["canopy_rhof_total"] = vx["canopy_rhof_total"]
        feats["canopy_moist"] = vx["canopy_moist"]
        feats["canopy_fuel_depth"] = vx["canopy_fuel_depth"]

    feats["dist_to_road_m"] = _roads.distance_to_road_m(scenario, px, py)

    feats = feats.dropna(subset=[f"raster_{RASTER_FEATURES[0]}"])
    print(f"[hazard-climate {scenario.name}] training grid: "
          f"{len(feats):,} samples  ({int(feats['burned'].sum()):,} burned, "
          f"{int((1 - feats['burned']).sum()):,} unburned)")
    return feats


def train_climate_hazard(scenario: Scenario, mtbs: gpd.GeoDataFrame):
    training = _build_training_grid(scenario, mtbs)
    if training is None or training["burned"].sum() < 20 or (1 - training["burned"]).sum() < 20:
        print(f"[hazard-climate {scenario.name}] insufficient training data — falling back")
        return None, []

    cols = [c for c in LANDSCAPE_FEATURE_COLS if c in training.columns]
    X = training[cols]
    y = training["burned"]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = HistGradientBoostingClassifier(
        max_depth=5, max_iter=200, learning_rate=0.05, random_state=42,
    )
    model.fit(Xt, yt)
    auc = roc_auc_score(yv, model.predict_proba(Xv)[:, 1])
    print(f"[hazard-climate {scenario.name}] AUC-ROC = {auc:.3f}")
    model.fit(X, y)
    return model, cols


def _heuristic_hazard(buildings: gpd.GeoDataFrame) -> np.ndarray:
    """Fallback used when no overlapping MTBS fire — rank-normalize fuel × slope."""
    parts = []
    for c in ["raster_rhof1", "raster_rhof10", "raster_rhof100", "raster_depth"]:
        if c in buildings.columns:
            v = buildings[c].fillna(0)
            if v.max() > 0:
                parts.append(v.rank(pct=True).to_numpy())
    if "raster_slope" in buildings.columns:
        v = buildings["raster_slope"].fillna(0)
        if v.max() > 0:
            parts.append(v.rank(pct=True).to_numpy())
    for c in ["raster_moist1", "raster_moist10", "raster_moist100"]:
        if c in buildings.columns:
            v = buildings[c].fillna(v.mean() if (v := buildings[c]).notna().any() else 0)
            parts.append(1 - v.rank(pct=True).to_numpy())
    if not parts:
        return np.full(len(buildings), 0.5)
    return np.clip(np.mean(np.column_stack(parts), axis=1), 0, 1)


def attach_landscape_features(buildings: gpd.GeoDataFrame, scenario: Scenario) -> gpd.GeoDataFrame:
    """Add raster_* and ignition_density_* columns sampled at each building centroid."""
    xs, ys = building_centroid_lonlat(buildings)
    for fname in RASTER_FEATURES:
        path = os.path.join(scenario.dir, f"{scenario.prefix}_{fname}.tif")
        if os.path.exists(path):
            buildings[f"raster_{fname}"] = sample_raster_at_points(path, xs, ys)
    elev_path = os.path.join(scenario.dir, f"{scenario.prefix}_elevation.tif")
    if os.path.exists(elev_path):
        with rasterio.open(elev_path) as src:
            elev = clean_raster(src.read(1))
            slope = compute_slope_deg(elev, src.transform)
            buildings["raster_slope"] = sample_array_at_points(slope, src.transform, xs, ys)

    fire_lons, fire_lats = _load_ignition_points(scenario)
    buildings["ignition_density_5km"] = _ignition_density(xs, ys, fire_lons, fire_lats, 5.0)
    buildings["ignition_density_25km"] = _ignition_density(xs, ys, fire_lons, fire_lats, 25.0)

    from . import canopy as _canopy
    from . import roads as _roads
    tl = _canopy.sample_treelist_at_points(scenario, xs, ys, radius_m=30.0)
    buildings["treelist_cbd_30m"] = tl["sum_cbd"]
    buildings["treelist_cbh_30m"] = tl["mean_cbh"]
    buildings["treelist_count_30m"] = tl["tree_count"]
    vx = _canopy.sample_voxel_at_points(scenario, xs, ys)
    if vx is not None:
        buildings["canopy_rhof_total"] = vx["canopy_rhof_total"]
        buildings["canopy_moist"] = vx["canopy_moist"]
        buildings["canopy_fuel_depth"] = vx["canopy_fuel_depth"]
    buildings["dist_to_road_m"] = _roads.distance_to_road_m(scenario, xs, ys)
    return buildings


def score_buildings(buildings: gpd.GeoDataFrame, model, feat_cols: list[str]) -> np.ndarray:
    """Climate hazard ∈ [0, 1] per building."""
    if model is None or not feat_cols:
        return _heuristic_hazard(buildings)
    X = buildings[feat_cols]
    return model.predict_proba(X)[:, 1]

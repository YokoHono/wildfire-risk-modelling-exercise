"""Load synthetic-scenario inputs: buildings, ignition, weather, rasters."""

from __future__ import annotations

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from .config import ELEV_NODATA_ALT, RASTER_NODATA, Scenario
from .data_sources import get_scenario_data


def _ensure_scenario_data(scenario: Scenario) -> None:
    if not os.path.exists(scenario.buildings):
        get_scenario_data(scenario.name)


def load_buildings(scenario: Scenario) -> gpd.GeoDataFrame:
    _ensure_scenario_data(scenario)
    gdf = gpd.read_file(scenario.buildings)
    gdf.columns = [c.lower() for c in gdf.columns]
    return gdf


def load_ignition(scenario: Scenario) -> tuple[float, float, pd.Timestamp]:
    _ensure_scenario_data(scenario)
    with open(scenario.ignition) as f:
        d = json.load(f)
    feat = d["features"][0]
    lon, lat = feat["geometry"]["coordinates"]
    start = pd.to_datetime(feat["properties"]["start"], utc=True)
    return lon, lat, start


def load_weather(scenario: Scenario) -> pd.DataFrame:
    """Parse the synoptic CSV (7 header lines, then columns + units row)."""
    _ensure_scenario_data(scenario)
    df = pd.read_csv(scenario.weather, skiprows=7)
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = [
        "station", "datetime", "temp_f", "rh",
        "wind_mph", "wind_dir", "wind_gust_mph",
    ]
    for c in ["temp_f", "rh", "wind_mph", "wind_dir", "wind_gust_mph"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    df["wind_ms"] = df["wind_mph"] * 0.44704
    return df


def clean_raster(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float)
    out[out == RASTER_NODATA] = np.nan
    out[out == ELEV_NODATA_ALT] = np.nan
    return out


def load_raster(scenario: Scenario, name: str) -> tuple[np.ndarray, rasterio.Affine, tuple]:
    _ensure_scenario_data(scenario)
    path = os.path.join(scenario.dir, f"{scenario.prefix}_{name}.tif")
    with rasterio.open(path) as src:
        arr = clean_raster(src.read(1))
        return arr, src.transform, (src.height, src.width)


def compute_slope_deg(elev: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """Slope in degrees from an elevation array. Pixel size approximated from
    geographic transform by converting degrees → metres at the equator
    (sufficient for relative slope at our latitudes)."""
    py = abs(transform.e) * 111320
    px = abs(transform.a) * 111320
    dy, dx = np.gradient(elev, py, px)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))


def sample_raster_at_points(
    path: str, xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    """Sample a single-band raster at lon/lat points."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        H, W = arr.shape
        out = np.full(len(xs), np.nan)
        for i, (x, y) in enumerate(zip(xs, ys)):
            r, c = rowcol(src.transform, x, y)
            if 0 <= r < H and 0 <= c < W:
                v = arr[r, c]
                if v != RASTER_NODATA and v != ELEV_NODATA_ALT:
                    out[i] = v
        return out


def sample_array_at_points(
    arr: np.ndarray, transform: rasterio.Affine, xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    H, W = arr.shape
    out = np.full(len(xs), np.nan)
    for i, (x, y) in enumerate(zip(xs, ys)):
        r, c = rowcol(transform, x, y)
        if 0 <= r < H and 0 <= c < W:
            out[i] = arr[r, c]
    return out


def building_centroid_lonlat(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Centroid (lon, lat). Computes in a local projected CRS to silence the
    geographic-CRS warning, then converts back."""
    if gdf.crs is not None and gdf.crs.is_geographic:
        proj = gdf.to_crs(epsg=3857)
        cent_proj = proj.geometry.centroid
        cent = gpd.GeoSeries(cent_proj, crs=3857).to_crs(gdf.crs)
    else:
        cent = gdf.geometry.centroid
    return cent.x.to_numpy(), cent.y.to_numpy()

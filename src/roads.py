"""Road network features.

Two uses:
  1. Fuel-break mask for the spread simulation — cells overlapping a road
     get their ROS scaled down by ``ROAD_ROS_FACTOR``.
  2. Per-building distance-to-nearest-road (metres) for exposure /
     evac-access reasoning (not currently used to modify exposure, but
     produced for downstream analysis).
"""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString

from .config import Scenario


ROAD_ROS_FACTOR = 0.10  # ROS scaling inside road cells (fuel break)


def load_roads(scenario: Scenario) -> gpd.GeoDataFrame:
    path = os.path.join(scenario.dir, f"{scenario.prefix}_roads.geojson")
    if not os.path.exists(path):
        return gpd.GeoDataFrame(geometry=[])
    return gpd.read_file(path)


def road_mask(
    scenario: Scenario, transform: Affine, shape: tuple[int, int],
) -> np.ndarray:
    """Return a boolean mask of cells intersected by any road segment."""
    roads = load_roads(scenario)
    if len(roads) == 0:
        return np.zeros(shape, dtype=bool)
    geoms = [(geom, 1) for geom in roads.geometry if geom is not None]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    mask = rasterize(
        geoms,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    ).astype(bool)
    return mask


def _explode_lines(roads: gpd.GeoDataFrame) -> list[np.ndarray]:
    """Return list of (N, 2) lon/lat arrays for every road segment vertex."""
    out: list[np.ndarray] = []
    for geom in roads.geometry:
        if geom is None:
            continue
        if isinstance(geom, LineString):
            out.append(np.asarray(geom.coords))
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                out.append(np.asarray(part.coords))
    return out


def distance_to_road_m(
    scenario: Scenario, xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    """Distance (m) from each (lon, lat) point to the nearest road vertex.

    Approximate: uses the closest vertex via KDTree rather than the closest
    point on a line segment, which over-estimates by at most ~½ vertex
    spacing (typically <20 m for OSM data).
    """
    roads = load_roads(scenario)
    if len(roads) == 0 or len(xs) == 0:
        return np.full(len(xs), np.inf, dtype=np.float32)
    coords = np.concatenate(_explode_lines(roads), axis=0)
    if coords.size == 0:
        return np.full(len(xs), np.inf, dtype=np.float32)
    lat0 = float(np.mean(ys))
    mx = 111320.0 * np.cos(np.radians(lat0))
    my = 111320.0
    road_xy = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])
    pt_xy = np.column_stack([xs * mx, ys * my])
    tree = cKDTree(road_xy)
    dists, _ = tree.query(pt_xy, k=1)
    return dists.astype(np.float32)

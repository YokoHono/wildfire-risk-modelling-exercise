"""Canopy fuel features derived from the synthetic Treelist and voxel data.

The treelist GeoJSON holds per-tree attributes in WGS84:
  HT       canopy height [m]
  CBH      canopy base height [m]
  DIA      maximum canopy diameter [m]
  CBD      canopy bulk density [kg/m³]
  MOIST    foliar moisture [fraction]
  SS       fine fuel size scale

The voxels (treesrhof.dat, treesmoist.dat, treesfueldepth.dat, treesss.dat) are
3-D binary arrays in a local metric frame (~2 m cells, ~24–43 vertical
layers). They're large (760 MB Prairie, 4.2 GB Forest) so we memory-map
them, reduce along Z, and cache the 2-D summaries to ``data_cache/`` for
subsequent runs.
"""

from __future__ import annotations

import os
import re

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol

from .config import DATA_CACHE, Scenario


# ---------------------------------------------------------------------------
# Treelist → per-cell rasters on a requested grid (geographic, WGS84)
# ---------------------------------------------------------------------------

def rasterize_treelist(
    scenario: Scenario,
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Bin trees into the supplied (transform, shape) raster grid.

    Returns four arrays of the same shape:
      cbd_sum       sum of canopy bulk density per cell (kg/m³ · tree_count)
      cbh_mean      weighted mean canopy base height per cell (m)
      moist_mean    mean foliar moisture per cell
      tree_count    integer count of trees per cell
    Cells with no trees get 0.
    """
    path = os.path.join(scenario.dir, f"{scenario.prefix}_Treelist.geojson")
    if not os.path.exists(path):
        H, W = shape
        return {
            "cbd_sum": np.zeros((H, W), dtype=np.float32),
            "cbh_mean": np.zeros((H, W), dtype=np.float32),
            "moist_mean": np.zeros((H, W), dtype=np.float32),
            "tree_count": np.zeros((H, W), dtype=np.int32),
        }

    trees = gpd.read_file(path)
    xs = trees["X"].to_numpy() if "X" in trees.columns else trees.geometry.x.to_numpy()
    ys = trees["Y"].to_numpy() if "Y" in trees.columns else trees.geometry.y.to_numpy()
    cbd = trees["CBD"].fillna(0).to_numpy(dtype=np.float32)
    cbh = trees["CBH"].fillna(0).to_numpy(dtype=np.float32)
    moist = trees["MOIST"].fillna(1.0).to_numpy(dtype=np.float32)

    H, W = shape
    cbd_sum = np.zeros((H, W), dtype=np.float32)
    cbh_sum = np.zeros((H, W), dtype=np.float32)
    moist_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)

    # Vectorised row/col lookup via the inverse affine
    inv = ~transform
    cols, rows = inv * (xs, ys)
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows = rows[valid]
    cols = cols[valid]
    cbd = cbd[valid]
    cbh = cbh[valid]
    moist = moist[valid]

    np.add.at(cbd_sum, (rows, cols), cbd)
    np.add.at(cbh_sum, (rows, cols), cbh)
    np.add.at(moist_sum, (rows, cols), moist)
    np.add.at(count, (rows, cols), 1)

    safe = np.where(count > 0, count, 1).astype(np.float32)
    return {
        "cbd_sum": cbd_sum,
        "cbh_mean": np.where(count > 0, cbh_sum / safe, 0).astype(np.float32),
        "moist_mean": np.where(count > 0, moist_sum / safe, 0).astype(np.float32),
        "tree_count": count,
    }


def sample_treelist_at_points(
    scenario: Scenario,
    xs: np.ndarray,
    ys: np.ndarray,
    radius_m: float = 30.0,
) -> dict[str, np.ndarray]:
    """For each (lon, lat), summarise nearby trees (within `radius_m`).

    Returns three arrays (sum_cbd, mean_cbh, tree_count). At our latitudes a
    30 m radius is small enough to be approximated as Euclidean in
    locally-scaled lon/lat units.
    """
    path = os.path.join(scenario.dir, f"{scenario.prefix}_Treelist.geojson")
    if not os.path.exists(path) or len(xs) == 0:
        z = np.zeros(len(xs), dtype=np.float32)
        return {"sum_cbd": z.copy(), "mean_cbh": z.copy(), "tree_count": z.copy()}

    from scipy.spatial import cKDTree

    trees = gpd.read_file(path)
    tx = trees["X"].to_numpy() if "X" in trees.columns else trees.geometry.x.to_numpy()
    ty = trees["Y"].to_numpy() if "Y" in trees.columns else trees.geometry.y.to_numpy()
    cbd = trees["CBD"].fillna(0).to_numpy()
    cbh = trees["CBH"].fillna(0).to_numpy()

    lat0 = float(np.mean(ys)) if len(ys) else 0.0
    mx = 111320.0 * np.cos(np.radians(lat0))
    my = 111320.0
    tree_xy = np.column_stack([tx * mx, ty * my])
    pt_xy = np.column_stack([xs * mx, ys * my])

    tree_kd = cKDTree(tree_xy)
    sum_cbd = np.zeros(len(xs), dtype=np.float32)
    sum_cbh = np.zeros(len(xs), dtype=np.float32)
    count = np.zeros(len(xs), dtype=np.float32)
    for i, p in enumerate(pt_xy):
        idx = tree_kd.query_ball_point(p, r=radius_m)
        if idx:
            sum_cbd[i] = cbd[idx].sum()
            sum_cbh[i] = cbh[idx].sum()
            count[i] = len(idx)
    safe = np.where(count > 0, count, 1)
    return {
        "sum_cbd": sum_cbd,
        "mean_cbh": (sum_cbh / safe).astype(np.float32),
        "tree_count": count,
    }


# ---------------------------------------------------------------------------
# Voxels → 2-D vertical-integral summaries (cached)
# ---------------------------------------------------------------------------

def _voxel_dims(scenario: Scenario) -> tuple[int, int, int, float] | None:
    """Parse nx, ny, nz, dx from the scenario's ReadDatFiles.py."""
    path = os.path.join(scenario.dir, "voxels", f"{scenario.prefix}_ReadDatFiles.py")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    def _grab(key: str, cast):
        m = re.search(rf"^\s*{key}\s*=\s*([\d.]+)", text, flags=re.M)
        return cast(m.group(1)) if m else None

    nx = _grab("nx", int)
    ny = _grab("ny", int)
    nz = _grab("nz", int)
    dx = _grab("dx", float)
    if None in (nx, ny, nz, dx):
        return None
    return nx, ny, nz, dx


def _voxel_path(scenario: Scenario, fname: str) -> str | None:
    """Return the .dat path; prefer the unzipped subdirectory if present."""
    base = os.path.join(scenario.dir, "voxels")
    sub = os.path.join(base, f"{scenario.prefix}_voxels")
    for d in (sub, base):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def _read_voxel_layer(fh, nx: int, ny: int, nz: int, zi: int) -> np.ndarray:
    """Read a single z-slice from a Fortran-ordered (nx, ny, nz) 4-byte float
    array with a leading 4-byte record marker."""
    offset = 4 + zi * nx * ny * 4
    fh.seek(offset)
    arr = np.frombuffer(fh.read(nx * ny * 4), dtype=np.float32)
    return arr.reshape((ny, nx))  # Fortran order: x varies fastest


def voxel_2d_summary(scenario: Scenario) -> dict[str, np.ndarray] | None:
    """Reduce the 3-D voxel arrays to 2-D summaries (Fortran nx, ny grid).

    Caches the result so subsequent runs are instant.
    """
    dims = _voxel_dims(scenario)
    if dims is None:
        return None
    nx, ny, nz, dx = dims

    cache = os.path.join(DATA_CACHE, f"{scenario.prefix}_voxel_summary.npz")
    if os.path.exists(cache):
        with np.load(cache) as d:
            return {k: d[k] for k in d.files} | {"nx": nx, "ny": ny, "dx": dx}

    rhof_path = _voxel_path(scenario, "treesrhof.dat")
    moist_path = _voxel_path(scenario, "treesmoist.dat")
    depth_path = _voxel_path(scenario, "treesfueldepth.dat")
    if rhof_path is None:
        print(f"[canopy {scenario.name}] no voxel files found")
        return None

    print(f"[canopy {scenario.name}] reducing voxels "
          f"({nx}×{ny}×{nz} cells, this may take a few minutes)…")

    rhof_z = np.zeros((ny, nx), dtype=np.float32)
    moist_z = np.zeros((ny, nx), dtype=np.float32)
    depth_z = np.zeros((ny, nx), dtype=np.float32)
    moist_count = np.zeros((ny, nx), dtype=np.float32)

    with open(rhof_path, "rb") as fh:
        for zi in range(nz):
            rhof_z += _read_voxel_layer(fh, nx, ny, nz, zi)
    if moist_path:
        with open(moist_path, "rb") as fh:
            for zi in range(nz):
                layer = _read_voxel_layer(fh, nx, ny, nz, zi)
                mask = layer > 0
                moist_z += np.where(mask, layer, 0)
                moist_count += mask.astype(np.float32)
        safe = np.where(moist_count > 0, moist_count, 1)
        moist_z = moist_z / safe
    if depth_path:
        with open(depth_path, "rb") as fh:
            for zi in range(nz):
                depth_z = np.maximum(depth_z, _read_voxel_layer(fh, nx, ny, nz, zi))

    np.savez_compressed(cache,
                         rhof_z=rhof_z, moist_z=moist_z, depth_z=depth_z)
    print(f"[canopy {scenario.name}] voxel summary cached → {cache}")
    return {"rhof_z": rhof_z, "moist_z": moist_z, "depth_z": depth_z,
            "nx": nx, "ny": ny, "dx": dx}


def voxel_geographic_bounds(scenario: Scenario, summary: dict) -> tuple[float, float, float, float]:
    """Heuristic: assume the voxel domain is centred on the scenario's
    SB40 raster extent. Returns (lon_min, lat_min, lon_max, lat_max)."""
    sb40_path = os.path.join(scenario.dir, f"{scenario.prefix}_SB40.tif")
    with rasterio.open(sb40_path) as src:
        b = src.bounds
    lon_c = 0.5 * (b.left + b.right)
    lat_c = 0.5 * (b.bottom + b.top)
    half_x_m = summary["nx"] * summary["dx"] / 2.0
    half_y_m = summary["ny"] * summary["dx"] / 2.0
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.radians(lat_c)))
    return (
        lon_c - half_x_m * deg_per_m_lon,
        lat_c - half_y_m * deg_per_m_lat,
        lon_c + half_x_m * deg_per_m_lon,
        lat_c + half_y_m * deg_per_m_lat,
    )


def sample_voxel_at_points(
    scenario: Scenario, xs: np.ndarray, ys: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Return per-point canopy rhof/moist/depth from the voxel summary.

    Returns ``None`` if voxel data is not available; arrays of zeros for
    points outside the voxel domain.
    """
    summary = voxel_2d_summary(scenario)
    if summary is None:
        return None
    lon_min, lat_min, lon_max, lat_max = voxel_geographic_bounds(scenario, summary)
    nx, ny = summary["nx"], summary["ny"]
    rhof_z = summary["rhof_z"]
    moist_z = summary["moist_z"]
    depth_z = summary["depth_z"]

    fx = (xs - lon_min) / (lon_max - lon_min)
    fy = (ys - lat_min) / (lat_max - lat_min)
    col = np.clip(np.floor(fx * nx).astype(int), 0, nx - 1)
    row = np.clip(np.floor(fy * ny).astype(int), 0, ny - 1)
    in_domain = (fx >= 0) & (fx <= 1) & (fy >= 0) & (fy <= 1)

    # Trees are point-located in the voxel grid (most 2 m cells are empty),
    # so point sampling is too noisy. Mean over a ~10 m radius window via a
    # summed-area table (constant cost per query).
    win = max(1, int(round(10.0 / summary["dx"])))

    def _avg(arr: np.ndarray) -> np.ndarray:
        sat = arr.cumsum(0).cumsum(1)
        sat = np.pad(sat, ((1, 0), (1, 0)), mode="constant")
        r0 = np.clip(row - win, 0, ny)
        r1 = np.clip(row + win + 1, 0, ny)
        c0 = np.clip(col - win, 0, nx)
        c1 = np.clip(col + win + 1, 0, nx)
        area = (r1 - r0) * (c1 - c0)
        total = sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]
        out = np.where(area > 0, total / np.maximum(area, 1), 0).astype(np.float32)
        return np.where(in_domain, out, 0)

    return {
        "canopy_rhof_total": _avg(rhof_z),
        "canopy_moist": _avg(moist_z),
        "canopy_fuel_depth": _avg(depth_z),
    }

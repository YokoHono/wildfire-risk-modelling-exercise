"""Scenario hazard: Monte Carlo wildfire spread on a downsampled raster grid.

Implements a priority-queue front tracker (Dijkstra-like) where edge cost is
travel time in minutes between adjacent cells. Per-cell rate of spread is a
simplified Rothermel-style product of:

    ROS = base_ROS(SB40) × wind_factor × slope_factor × moisture_factor

Wind factor uses a cosine alignment between travel direction and observed wind.
Slope factor uses the Rothermel φ_s form. Moisture damps ROS toward zero as
fine-fuel moisture climbs.

Per realization we perturb wind speed (Gaussian) and direction (uniform within
±SIM_WIND_DIR_JITTER_DEG) and apply a stochastic downwind spot ignition. We
record the arrival-time raster for each realization, then summarize:
    P_sim(cell) = fraction of realizations where arrival_time ≤ horizon.
Buildings inherit P_sim from their centroid cell.
"""

from __future__ import annotations

import heapq
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject

from .config import (
    SIM_HORIZON_MIN,
    SIM_REALIZATIONS_DEFAULT,
    SIM_SPOT_COUNT,
    SIM_SPOT_DIST_M,
    SIM_WIND_DIR_JITTER_DEG,
    Scenario,
)
from .loaders import clean_raster, load_ignition, load_weather

# SB40 base rate-of-spread (m/min). Sources: Scott & Burgan (2005) table 5,
# coarsely averaged to the 9 fuel-model families.
SB40_BASE_ROS_MPM: dict[int, float] = {}
for c in range(91, 100):
    SB40_BASE_ROS_MPM[c] = 1.0      # NB non-burnable / GR short
for c in range(101, 110):
    SB40_BASE_ROS_MPM[c] = 4.0      # GR tall grass
for c in range(121, 130):
    SB40_BASE_ROS_MPM[c] = 2.0      # GS grass-shrub
for c in range(141, 150):
    SB40_BASE_ROS_MPM[c] = 1.8      # SH shrub
for c in range(161, 166):
    SB40_BASE_ROS_MPM[c] = 1.0      # TU timber understory
for c in range(181, 190):
    SB40_BASE_ROS_MPM[c] = 0.5      # TL timber litter
for c in range(201, 205):
    SB40_BASE_ROS_MPM[c] = 0.3      # SB slash-blowdown
SB40_BASE_ROS_MPM[0] = 0.0


def _base_ros(sb40: np.ndarray) -> np.ndarray:
    """Map SB40 fuel-model codes to base ROS in m/min."""
    out = np.zeros_like(sb40, dtype=float)
    for code, ros in SB40_BASE_ROS_MPM.items():
        out[sb40 == code] = ros
    return out


def _downsample(arr: np.ndarray, factor: int, agg: str = "mean") -> np.ndarray:
    """Block-aggregate a 2D array by an integer factor."""
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    a = arr[:h, :w].reshape(h // factor, factor, w // factor, factor)
    if agg == "mean":
        return np.nanmean(a, axis=(1, 3))
    if agg == "max":
        return np.nanmax(a, axis=(1, 3))
    if agg == "mode":
        # Cheap: use the first non-nan in each block
        flat = a.reshape(a.shape[0], a.shape[2], -1)
        return flat[:, :, 0]
    raise ValueError(agg)


def _load_scenario_grid(scenario: Scenario, downsample: int):
    """Load and downsample the rasters needed for spread modelling.

    The elevation raster is at a much coarser native resolution than the
    LANDFIRE-derived rasters (e.g. 180×230 vs 3506×4548 for Forest), so we
    reproject it onto the fuel-raster grid before downsampling.
    """
    def _open(name):
        path = os.path.join(scenario.dir, f"{scenario.prefix}_{name}.tif")
        return rasterio.open(path)

    with _open("SB40") as sb_src, _open("moist1") as m_src, _open("elevation") as e_src:
        sb40 = clean_raster(sb_src.read(1))
        moist1 = clean_raster(m_src.read(1))
        transform = sb_src.transform
        crs = sb_src.crs

        # Reproject elevation onto SB40 grid (bilinear)
        elev = np.full(sb40.shape, np.nan, dtype=np.float32)
        reproject(
            source=e_src.read(1).astype(np.float32),
            destination=elev,
            src_transform=e_src.transform, src_crs=e_src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.bilinear,
            src_nodata=e_src.nodata if e_src.nodata is not None else -32768,
            dst_nodata=np.nan,
        )
        elev = np.where(elev <= -1000, np.nan, elev)

    # Trim to identical shape (1-pixel mismatches possible)
    H = min(sb40.shape[0], elev.shape[0], moist1.shape[0])
    W = min(sb40.shape[1], elev.shape[1], moist1.shape[1])
    sb40 = sb40[:H, :W]
    elev = elev[:H, :W]
    moist1 = moist1[:H, :W]

    # SB40 codes are valid only in [0, 200]; clamp out-of-range values to 0
    sb40_clean = np.where((sb40 >= 0) & (sb40 <= 200), sb40, 0)
    sb40_clean = np.nan_to_num(sb40_clean, nan=0.0).astype(np.int32)
    sb40_ds = _downsample(sb40_clean, downsample, agg="mode")
    elev_ds = _downsample(np.nan_to_num(elev, nan=0.0), downsample, agg="mean")
    moist_ds = _downsample(np.nan_to_num(moist1, nan=0.1), downsample, agg="mean")

    # Adjust the affine transform for the downsampled grid
    new_a = transform.a * downsample
    new_e = transform.e * downsample
    new_transform = rasterio.Affine(
        new_a, transform.b, transform.c, transform.d, new_e, transform.f
    )

    # Physical pixel size in metres (approximate: 1° lat ≈ 111320 m)
    cell_m = abs(new_a) * 111320

    # Slope from downsampled elevation
    dy, dx = np.gradient(np.nan_to_num(elev_ds), cell_m, cell_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    # Aspect (radians, clockwise from north): direction of steepest descent
    aspect = np.arctan2(-dx, -dy)  # 0 = north

    # Road fuel-break mask on the downsampled grid
    from . import roads as _roads
    road = _roads.road_mask(scenario, new_transform, sb40_ds.shape)

    # Canopy bulk density on the downsampled grid (treelist + voxel)
    from . import canopy as _canopy
    tl_grid = _canopy.rasterize_treelist(scenario, new_transform, sb40_ds.shape)
    cbd_field = tl_grid["cbd_sum"].astype(np.float32)
    # Voxel rhof (vertically integrated) — resample onto the grid by point sampling
    vx_summary = _canopy.voxel_2d_summary(scenario)
    if vx_summary is not None:
        # Sample voxel grid at each cell centre
        H, W = sb40_ds.shape
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        lon = new_transform.c + (cc + 0.5) * new_transform.a
        lat = new_transform.f + (rr + 0.5) * new_transform.e
        vx_field = _canopy.sample_voxel_at_points(
            scenario, lon.ravel(), lat.ravel(),
        )
        canopy_rhof = vx_field["canopy_rhof_total"].reshape(sb40_ds.shape)
    else:
        canopy_rhof = np.zeros(sb40_ds.shape, dtype=np.float32)

    return {
        "sb40": sb40_ds,
        "elev": elev_ds,
        "slope_deg": slope_deg,
        "aspect": aspect,
        "moist": moist_ds,
        "transform": new_transform,
        "cell_m": cell_m,
        "shape": sb40_ds.shape,
        "road": road,
        "cbd": cbd_field,
        "canopy_rhof": canopy_rhof,
    }


def _ros_grid(grid: dict, wind_speed_ms: float, wind_dir_deg: float):
    """Compute directional ROS multipliers between every cell and its 8 neighbours.

    Returns: dict mapping (dr, dc) → 2D ROS field (m/min) for travel from any
    cell along that direction. We bake the wind alignment and slope effect into
    a per-direction multiplier on top of the base ROS.
    """
    base = _base_ros(grid["sb40"])
    moist = grid["moist"]
    moist_factor = np.clip(1.0 - 2.0 * np.nan_to_num(moist, nan=0.1), 0.05, 1.0)
    slope_deg = np.nan_to_num(grid["slope_deg"], nan=0.0)
    aspect = np.nan_to_num(grid["aspect"], nan=0.0)

    # Rothermel-ish slope factor (simplified): φ_s = 5.275 × tan²(slope)
    phi_s = 5.275 * np.tan(np.radians(np.clip(slope_deg, 0, 60))) ** 2
    slope_factor = 1.0 + np.clip(phi_s, 0, 5.0)

    # Convert wind direction (degrees, met-convention "from") to radians "to"
    wind_to = np.radians((wind_dir_deg + 180) % 360)
    # Wind strength factor. Observed wind is at 6-10 m; we apply a 0.5
    # reduction to approximate mid-flame wind, then a BehavePlus-style
    # power law (kept gentler than published coefficients to avoid
    # runaway spread on continuous grass fuels).
    midflame = wind_speed_ms * 0.5
    wind_mag = 1.0 + 0.35 * (midflame ** 1.15)

    fields = {}
    # Canopy ROS boost: a cell with substantial canopy fuel (treelist CBD or
    # voxel rhof) carries more total combustible mass per area, supporting
    # higher surface ROS and crown-fire potential. Normalise both fields to
    # [0, 1] across the scenario, take the max, and apply a multiplicative
    # boost up to (1 + CANOPY_BOOST_MAX).
    CANOPY_BOOST_MAX = 1.5
    def _norm(arr):
        m = float(arr.max())
        return arr / m if m > 0 else np.zeros_like(arr)
    canopy_norm = np.maximum(_norm(grid.get("cbd", np.zeros_like(base))),
                              _norm(grid.get("canopy_rhof", np.zeros_like(base))))
    canopy_factor = 1.0 + CANOPY_BOOST_MAX * canopy_norm

    # Road fuel-break factor (multiplicative; 1.0 outside roads, small in road cells)
    from .roads import ROAD_ROS_FACTOR
    road = grid.get("road")
    road_factor = np.where(road, ROAD_ROS_FACTOR, 1.0) if road is not None else 1.0

    # 8-neighbour offsets and their direction-of-travel azimuth (radians from north, clockwise)
    neigh = {
        (-1, 0): 0.0,            # north
        (-1, 1): np.pi / 4,
        (0, 1): np.pi / 2,       # east
        (1, 1): 3 * np.pi / 4,
        (1, 0): np.pi,           # south
        (1, -1): 5 * np.pi / 4,
        (0, -1): 3 * np.pi / 2,  # west
        (-1, -1): 7 * np.pi / 4,
    }
    for off, theta in neigh.items():
        # Wind alignment: cosine between travel and wind-to direction
        wind_align = np.cos(theta - wind_to)            # scalar
        wf = 1.0 + (wind_mag - 1.0) * max(wind_align, 0)
        # Slope alignment: travel cells uphill if travel direction aligns with -aspect
        slope_align = np.cos(theta - aspect)
        # Combine: only apply slope boost if going uphill
        sf = 1.0 + (slope_factor - 1.0) * np.clip(slope_align, 0, 1)
        fields[off] = base * moist_factor * wf * sf * canopy_factor * road_factor
    return fields


def _peak_wind(weather: pd.DataFrame, ignition_time: pd.Timestamp) -> tuple[float, float]:
    """Mean wind speed/direction in the 6 h after ignition, or fallback mean."""
    window = weather[(weather["datetime"] >= ignition_time) &
                      (weather["datetime"] <= ignition_time + pd.Timedelta(hours=6))]
    if len(window) < 3:
        window = weather
    ws = float(window["wind_ms"].mean())
    wd = float(window["wind_dir"].mean())
    return ws, wd


def _ignition_cell(grid: dict, lon: float, lat: float) -> tuple[int, int]:
    r, c = rowcol(grid["transform"], lon, lat)
    r = int(np.clip(r, 0, grid["shape"][0] - 1))
    c = int(np.clip(c, 0, grid["shape"][1] - 1))
    return r, c


def _drain(pq, arrival, ros, neighbour_dist, horizon_min, H, W):
    while pq:
        t, r, c = heapq.heappop(pq)
        if t > arrival[r, c]:
            continue
        if t > horizon_min:
            continue
        for (dr, dc), dist in neighbour_dist.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            speed = ros[(dr, dc)][r, c]
            if speed <= 0.01:
                continue
            t_new = t + dist / speed
            if t_new < arrival[nr, nc] and t_new <= horizon_min:
                arrival[nr, nc] = t_new
                heapq.heappush(pq, (t_new, nr, nc))


def _simulate_one(grid, ignition_rc, wind_speed, wind_dir, horizon_min,
                  spot_dist_m, spot_count, rng) -> np.ndarray:
    """Return arrival-time raster (minutes; np.inf where unreached)."""
    H, W = grid["shape"]
    arrival = np.full((H, W), np.inf)
    ros = _ros_grid(grid, wind_speed, wind_dir)
    cell_m = grid["cell_m"]
    diag = cell_m * np.sqrt(2)

    r0, c0 = ignition_rc
    arrival[r0, c0] = 0.0
    pq = [(0.0, r0, c0)]
    neighbour_dist = {
        (-1, 0): cell_m, (1, 0): cell_m, (0, -1): cell_m, (0, 1): cell_m,
        (-1, -1): diag, (-1, 1): diag, (1, -1): diag, (1, 1): diag,
    }

    # Initial spread without spotting
    _drain(pq, arrival, ros, neighbour_dist, horizon_min, H, W)

    # Three rounds of stochastic spotting interleaved with re-spread
    wind_to = np.radians((wind_dir + 180) % 360)
    for _round in range(3):
        burned = np.argwhere(np.isfinite(arrival))
        if len(burned) == 0:
            break
        K = min(spot_count, max(1, len(burned) // 20))
        idx = rng.choice(len(burned), size=K, replace=False)
        for k in idx:
            br, bc = burned[k]
            d = rng.uniform(spot_dist_m * 0.25, spot_dist_m) / cell_m
            jitter = rng.uniform(-0.35, 0.35)  # ±20° spread of spots
            theta = wind_to + jitter
            dr = -int(round(np.cos(theta) * d))
            dc = int(round(np.sin(theta) * d))
            sr = int(br + dr)
            sc = int(bc + dc)
            if 0 <= sr < H and 0 <= sc < W:
                t_seed = arrival[br, bc] + rng.uniform(3, 15)
                if t_seed < arrival[sr, sc] and t_seed <= horizon_min:
                    arrival[sr, sc] = t_seed
                    heapq.heappush(pq, (t_seed, sr, sc))
        _drain(pq, arrival, ros, neighbour_dist, horizon_min, H, W)

    return arrival


def run_simulation(
    scenario: Scenario,
    realizations: int = SIM_REALIZATIONS_DEFAULT,
    downsample: int = 15,
    horizon_min: int = SIM_HORIZON_MIN,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Run Monte Carlo simulation, return P(burn) raster and grid metadata."""
    print(f"[hazard-sim {scenario.name}] preparing grid (downsample={downsample})")
    grid = _load_scenario_grid(scenario, downsample)
    print(f"[hazard-sim {scenario.name}] grid {grid['shape']} cell≈{grid['cell_m']:.1f} m")

    lon, lat, ig_time = load_ignition(scenario)
    weather = load_weather(scenario)
    ws_mean, wd_mean = _peak_wind(weather, ig_time)
    ws_sigma = max(1.0, float(weather["wind_ms"].std()))
    print(f"[hazard-sim {scenario.name}] mean wind {ws_mean:.1f} m/s @ {wd_mean:.0f}° "
          f"(σ_speed={ws_sigma:.1f})")

    ig_rc = _ignition_cell(grid, lon, lat)
    print(f"[hazard-sim {scenario.name}] ignition cell {ig_rc}")

    rng = np.random.default_rng(seed)
    burn_count = np.zeros(grid["shape"], dtype=np.float32)

    for i in range(realizations):
        ws = max(0.5, ws_mean + rng.normal(0, ws_sigma * 0.5))
        wd = wd_mean + rng.uniform(-SIM_WIND_DIR_JITTER_DEG, SIM_WIND_DIR_JITTER_DEG)
        arr = _simulate_one(
            grid, ig_rc, ws, wd, horizon_min, SIM_SPOT_DIST_M, SIM_SPOT_COUNT, rng,
        )
        burn_count += np.isfinite(arr).astype(np.float32)
        if (i + 1) % max(1, realizations // 10) == 0:
            mean_burned_pct = 100 * burn_count.mean() / (i + 1)
            print(f"  realization {i + 1}/{realizations}  mean burned = {mean_burned_pct:.1f}%")

    pburn = burn_count / realizations
    return pburn, grid


def score_buildings(
    buildings: gpd.GeoDataFrame,
    pburn: np.ndarray,
    grid: dict,
    neighbourhood_radius: int = 4,
) -> np.ndarray:
    """P(building exposed to fire) = max P(burn) in a small neighbourhood of
    cells around the building centroid. Buildings often sit on cleared lots
    (SB40=0) that can't burn directly, so the fire reaches the *fuel cells
    adjacent* to the building. The neighbourhood max captures this exposure
    via radiant/ember impingement at short range."""
    from .loaders import building_centroid_lonlat
    xs, ys = building_centroid_lonlat(buildings)
    H, W = grid["shape"]
    out = np.zeros(len(xs))
    R = neighbourhood_radius
    for i, (x, y) in enumerate(zip(xs, ys)):
        r, c = rowcol(grid["transform"], x, y)
        if not (0 <= r < H and 0 <= c < W):
            continue
        r0, r1 = max(0, r - R), min(H, r + R + 1)
        c0, c1 = max(0, c - R), min(W, c + R + 1)
        out[i] = float(pburn[r0:r1, c0:c1].max())
    return np.clip(out, 0, 1)

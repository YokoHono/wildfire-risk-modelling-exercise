"""Download and cache real-world reference datasets.

All downloads are idempotent: if the cached file exists, no network call is
made. Failures are non-fatal — callers receive `None` and should fall back
gracefully.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from .config import DATA_CACHE, BASE_DIR

# ----------------------------------------------------------------------------
# Synthetic scenario data — mirror at SDSC. Keeps a fresh clone self-bootstrapping
# without committing the (gitignored) Forest/ and Prairie/ directories.
# ----------------------------------------------------------------------------

SCENARIO_MIRROR_BASE = "https://scil-data.sdsc.edu/data/ul-risk-modeling-2026"

# Per-scenario list of files to fetch. Paths are relative to BASE_DIR;
# URLs are relative to SCENARIO_MIRROR_BASE/<scenario>/.
_SCENARIO_FILES = [
    "{p}_SB40.tif",
    "{p}_depth.tif",
    "{p}_elevation.tif",
    "{p}_moist1.tif",
    "{p}_moist10.tif",
    "{p}_moist100.tif",
    "{p}_rhof1.tif",
    "{p}_rhof10.tif",
    "{p}_rhof100.tif",
    "{p}_SAV.tif",
    "{p}_Treelist.geojson",
    "{p}_Treelist.txt",
    "{p}_generated_buildings_fireprops.geojson",
    "{p}_generated_buildings_fireprops_addAttr.geojson",
    "{p}_generated_buildings_fireprops_Parcel.geojson",
    "{p}_roads.geojson",
    "{p}_Synoptic_Weather_Data.csv",
    "{p}_ignition.geojson",
]
_SCENARIO_VOXEL_FILES = [
    "voxels/{p}_ReadDatFiles.py",
    "voxels/{p}_voxels.zip",
]


def _download(url: str, dest: str, label: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            os.replace(tmp, dest)
        return True
    except Exception as e:
        print(f"[{label}] download failed: {e}")
        return False


def get_scenario_data(scenario: str) -> bool:
    """Download the synthetic scenario data into <BASE_DIR>/<scenario>/<scenario>/.

    Idempotent: files already on disk are skipped. Voxel zip is extracted once.
    Returns True if every required file is present at the end, False otherwise.
    """
    p = scenario  # prefix == scenario name in the dataset
    scen_dir = os.path.join(BASE_DIR, scenario, scenario)
    voxels_dir = os.path.join(scen_dir, "voxels")
    os.makedirs(voxels_dir, exist_ok=True)

    missing_before = False
    for rel in _SCENARIO_FILES + _SCENARIO_VOXEL_FILES:
        rel_named = rel.format(p=p)
        dest = os.path.join(scen_dir, rel_named)
        if os.path.exists(dest):
            continue
        missing_before = True
        url = f"{SCENARIO_MIRROR_BASE}/{scenario}/{rel_named}"
        print(f"[{scenario}] fetching {rel_named}")
        if not _download(url, dest, scenario):
            return False

    voxel_zip = os.path.join(voxels_dir, f"{p}_voxels.zip")
    voxel_marker = os.path.join(voxels_dir, "treesfueldepth.dat")
    if os.path.exists(voxel_zip) and not os.path.exists(voxel_marker):
        try:
            with zipfile.ZipFile(voxel_zip) as z:
                z.extractall(voxels_dir)
        except Exception as e:
            print(f"[{scenario}] voxel extract failed: {e}")
            return False

    if missing_before:
        print(f"[{scenario}] scenario data ready in {scen_dir}")
    return True


# ----------------------------------------------------------------------------
# DINS — CAL FIRE damage inspections (vulnerability training labels)
# ----------------------------------------------------------------------------

DINS_URL = (
    "https://gis.data.cnra.ca.gov/api/download/v1/items/"
    "994d3dc4569640caadbbc3198d5a3da1/csv?layers=0"
)
DINS_CACHE = os.path.join(BASE_DIR, "dins_data.csv")


def get_dins() -> pd.DataFrame | None:
    if not os.path.exists(DINS_CACHE):
        print(f"[DINS] downloading from {DINS_URL}")
        try:
            r = requests.get(DINS_URL, stream=True, timeout=300)
            r.raise_for_status()
            with open(DINS_CACHE, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        except Exception as e:
            print(f"[DINS] download failed: {e}")
            return None
    try:
        df = pd.read_csv(DINS_CACHE, low_memory=False)
        print(f"[DINS] {len(df):,} records")
        return df
    except Exception as e:
        print(f"[DINS] read failed: {e}")
        return None


# ----------------------------------------------------------------------------
# MTBS — Monitoring Trends in Burn Severity perimeters (climate hazard labels)
# ----------------------------------------------------------------------------

MTBS_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/"
    "data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
)
MTBS_DIR = os.path.join(DATA_CACHE, "mtbs_perimeters")


def get_mtbs() -> gpd.GeoDataFrame | None:
    shp = _find_file(MTBS_DIR, ".shp", contains="perim")
    if shp is None:
        os.makedirs(MTBS_DIR, exist_ok=True)
        zip_path = os.path.join(MTBS_DIR, "mtbs_perimeters.zip")
        if not os.path.exists(zip_path):
            print(f"[MTBS] downloading (~350 MB)…")
            try:
                r = requests.get(MTBS_URL, stream=True, timeout=900)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            except Exception as e:
                print(f"[MTBS] download failed: {e}")
                return None
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(MTBS_DIR)
            shp = _find_file(MTBS_DIR, ".shp", contains="perim")
        except Exception as e:
            print(f"[MTBS] extract failed: {e}")
            return None
    if shp is None:
        return None
    try:
        gdf = gpd.read_file(shp)
        print(f"[MTBS] {len(gdf):,} fire perimeters")
        return gdf
    except Exception as e:
        print(f"[MTBS] read failed: {e}")
        return None


# ----------------------------------------------------------------------------
# FPA-FOD — Fire occurrence database. The canonical USFS RDS download is
# gated behind a JS catalog UI (no stable direct URL), so we use the Kaggle
# mirror (rtatman/188-million-us-wildfires, CC0) authenticated with a KGAT
# bearer token from ~/.kaggle/access_token. The fall-back chain after that
# tries the historical RDS endpoints in case they come back online.
# ----------------------------------------------------------------------------

FPA_FOD_KAGGLE_DATASET = "rtatman/188-million-us-wildfires"
KAGGLE_ACCESS_TOKEN_PATH = os.path.expanduser("~/.kaggle/access_token")

FPA_FOD_URLS_FALLBACK = [
    "https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.7/RDS-2013-0009.7_SQLITE.zip",
    "https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.6/RDS-2013-0009.6_GPKG.zip",
]
FPA_FOD_DIR = os.path.join(DATA_CACHE, "fpa_fod")


def _kaggle_token() -> str | None:
    if not os.path.exists(KAGGLE_ACCESS_TOKEN_PATH):
        return None
    with open(KAGGLE_ACCESS_TOKEN_PATH) as f:
        tok = f.read().strip()
    return tok or None


def _extract_fpa_fod_zip(zip_path: str) -> str | None:
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(FPA_FOD_DIR)
    except Exception as e:
        print(f"[FPA-FOD] extract failed: {e}")
        return None
    return _find_file(FPA_FOD_DIR, ".sqlite") or _find_file(FPA_FOD_DIR, ".gpkg")


def _download_fpa_fod_from_kaggle() -> str | None:
    tok = _kaggle_token()
    if not tok:
        return None
    os.makedirs(FPA_FOD_DIR, exist_ok=True)
    zip_path = os.path.join(FPA_FOD_DIR, "rtatman_188M_wildfires.zip")
    if not os.path.exists(zip_path):
        url = f"https://www.kaggle.com/api/v1/datasets/download/{FPA_FOD_KAGGLE_DATASET}"
        print(f"[FPA-FOD] downloading from Kaggle ({FPA_FOD_KAGGLE_DATASET})…")
        try:
            r = requests.get(
                url, stream=True, timeout=900,
                headers={"Authorization": f"Bearer {tok}"},
            )
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        except Exception as e:
            print(f"[FPA-FOD] Kaggle download failed: {e}")
            return None
    return _extract_fpa_fod_zip(zip_path)


def get_fpa_fod() -> str | None:
    """Return path to a usable FPA-FOD SQLITE/GPKG, or None if unavailable."""
    cached = _find_file(FPA_FOD_DIR, ".sqlite") or _find_file(FPA_FOD_DIR, ".gpkg")
    if cached:
        print(f"[FPA-FOD] cached: {cached}")
        return cached

    # Cached zip from a previous run? Extract without re-downloading.
    cached_zip = _find_file(FPA_FOD_DIR, ".zip")
    if cached_zip:
        extracted = _extract_fpa_fod_zip(cached_zip)
        if extracted:
            return extracted

    via_kaggle = _download_fpa_fod_from_kaggle()
    if via_kaggle:
        return via_kaggle

    os.makedirs(FPA_FOD_DIR, exist_ok=True)
    for url in FPA_FOD_URLS_FALLBACK:
        zip_path = os.path.join(FPA_FOD_DIR, os.path.basename(url))
        try:
            r = requests.get(url, stream=True, timeout=600)
            if r.status_code != 200:
                continue
            print(f"[FPA-FOD] downloading {url}")
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(FPA_FOD_DIR)
            cached = _find_file(FPA_FOD_DIR, ".gpkg") or _find_file(FPA_FOD_DIR, ".sqlite")
            if cached:
                return cached
        except Exception:
            continue

    print("[FPA-FOD] unavailable — proceeding without ignition-density feature")
    return None


def load_fpa_fod_points(sqlite_path: str, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Load ignition points (LATITUDE, LONGITUDE, FIRE_YEAR, FIRE_SIZE) within
    a (lon_min, lat_min, lon_max, lat_max) bounding box."""
    import sqlite3
    lon_min, lat_min, lon_max, lat_max = bbox
    with sqlite3.connect(sqlite_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT LATITUDE, LONGITUDE, FIRE_YEAR, FIRE_SIZE, STAT_CAUSE_DESCR
            FROM Fires
            WHERE LATITUDE BETWEEN ? AND ?
              AND LONGITUDE BETWEEN ? AND ?
            """,
            conn,
            params=(lat_min, lat_max, lon_min, lon_max),
        )
    return df


# ----------------------------------------------------------------------------
# USFS WUI — Wildland-Urban Interface classification (exposure modifier)
# ----------------------------------------------------------------------------

WUI_URL_TEMPLATE = (
    "https://geoserver.silvis.forest.wisc.edu/geodata/wui_change_2020_v4/"
    "zip/shp/{state}_wui_block_1990_2020_change_v4_shp.zip"
)
WUI_DIR = os.path.join(DATA_CACHE, "wui")

# Scenario → state postal code (controlling which WUI download to fetch)
SCENARIO_STATE = {"Forest": "CA", "Prairie": "TX"}


def get_wui(state: str) -> gpd.GeoDataFrame | None:
    """Return the state WUI shapefile as a GeoDataFrame, or None on failure."""
    state_dir = os.path.join(WUI_DIR, state)
    shp = _find_file(state_dir, ".shp")
    if shp is None:
        os.makedirs(state_dir, exist_ok=True)
        zip_path = os.path.join(state_dir, f"{state}_wui.zip")
        if not os.path.exists(zip_path):
            url = WUI_URL_TEMPLATE.format(state=state)
            try:
                print(f"[WUI {state}] downloading…")
                r = requests.get(url, stream=True, timeout=600)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            except Exception as e:
                print(f"[WUI {state}] download failed: {e}")
                return None
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(state_dir)
            shp = _find_file(state_dir, ".shp")
        except Exception as e:
            print(f"[WUI {state}] extract failed: {e}")
            return None
    if shp is None:
        return None
    try:
        gdf = gpd.read_file(shp)
        print(f"[WUI {state}] {len(gdf):,} block-level WUI features")
        return gdf
    except Exception as e:
        print(f"[WUI {state}] read failed: {e}")
        return None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _find_file(root: str, suffix: str, contains: str | None = None) -> str | None:
    if not os.path.exists(root):
        return None
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith(suffix) and (contains is None or contains.lower() in f.lower()):
                return os.path.join(r, f)
    return None

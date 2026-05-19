"""Static configuration: paths, scenario metadata, weights."""

from __future__ import annotations

import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CACHE = os.path.join(BASE_DIR, "data_cache")
os.makedirs(DATA_CACHE, exist_ok=True)


@dataclass(frozen=True)
class Scenario:
    name: str
    dir: str
    prefix: str
    buildings: str
    ignition: str
    weather: str
    center_lonlat: tuple[float, float]
    utm_epsg: int


SCENARIOS: dict[str, Scenario] = {
    "Forest": Scenario(
        name="Forest",
        dir=os.path.join(BASE_DIR, "Forest", "Forest"),
        prefix="Forest",
        buildings=os.path.join(
            BASE_DIR, "Forest", "Forest",
            "Forest_generated_buildings_fireprops_Parcel.geojson",
        ),
        ignition=os.path.join(BASE_DIR, "Forest", "Forest", "Forest_ignition.geojson"),
        weather=os.path.join(
            BASE_DIR, "Forest", "Forest", "Forest_Synoptic_Weather_Data.csv"
        ),
        center_lonlat=(-120.0306, 38.9014),
        utm_epsg=32610,
    ),
    "Prairie": Scenario(
        name="Prairie",
        dir=os.path.join(BASE_DIR, "Prairie", "Prairie"),
        prefix="Prairie",
        buildings=os.path.join(
            BASE_DIR, "Prairie", "Prairie",
            "Prairie_generated_buildings_fireprops_Parcel.geojson",
        ),
        ignition=os.path.join(BASE_DIR, "Prairie", "Prairie", "Prairie_ignition.geojson"),
        weather=os.path.join(
            BASE_DIR, "Prairie", "Prairie", "Prairie_Synoptic_Weather_Data.csv"
        ),
        center_lonlat=(-101.912, 35.236),
        utm_epsg=32614,
    ),
}

RASTER_NODATA = 9999
ELEV_NODATA_ALT = -32768

# Landscape raster features used for climatological hazard
RASTER_FEATURES = [
    "SB40", "depth", "moist1", "moist10", "moist100",
    "rhof1", "rhof10", "rhof100", "SAV", "elevation",
]

# Blend weights
HAZARD_CLIMATE_WEIGHT = 0.5
HAZARD_SIM_WEIGHT = 0.5

# Defensible-space modifier: vulnerability = raw * (1 - DEFENSIBLE_REDUCTION * compliance)
DEFENSIBLE_REDUCTION = 0.4

# Exposure sub-weights
EXPOSURE_W_VALUE = 0.5
EXPOSURE_W_FOOTPRINT = 0.3
EXPOSURE_W_ADJACENCY = 0.2

USAGE_WEIGHT = {
    "single residence": 1.0,
    "multi residence": 1.2,
    "multifamily": 1.2,
    "multiple residence": 1.2,
    "mixed commercial/residential": 1.0,
    "residential": 1.0,
    "commercial": 0.9,
    "industrial": 0.7,
    "other": 0.8,
}

# Simulation defaults
SIM_REALIZATIONS_DEFAULT = 100
SIM_WIND_DIR_JITTER_DEG = 20.0
SIM_SPOT_DIST_M = 400.0
SIM_SPOT_COUNT = 60
SIM_HORIZON_MIN = 720  # 12 hours

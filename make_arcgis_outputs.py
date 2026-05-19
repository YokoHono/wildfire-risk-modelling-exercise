"""Slim the 5-component risk GeoJSONs down to the columns that matter for
the ArcGIS Online feature service submission (§2b of the exercise manual).

The full *_risk_scored_5component.geojson files carry every input attribute
the pipeline consumed (108 columns for Forest, 99 for Prairie). That's useful
for auditing a score inside the notebook, but evaluators viewing the layer in
ArcGIS only need the score components plus a handful of inputs they might
want to drill into. This script keeps the score columns and the most
informative inputs, drops the rest, and writes
``outputs/<scenario>_risk_arcgis.geojson``.

Usage:
    python make_arcgis_outputs.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"

# Columns to keep, in the order they'll appear on the feature popup.
KEEP_COLUMNS = [
    # Identity & context
    "bldgid", "usage", "property_value", "building_sqft", "stories_above_ground",
    # Final risk
    "risk_score", "risk_decile", "risk_gross", "risk_net",
    # Hazard
    "hazard", "hazard_climate", "hazard_sim", "ignition_hour_from_sim",
    # Vulnerability
    "vulnerability",
    # Exposure
    "exposure", "exposure_value", "exposure_footprint", "exposure_adjacency",
    # Severity
    "severity",
    # Capacity
    "capacity",
    "cap_defensible_space", "cap_construction_quality",
    "cap_ember_resistance", "cap_access", "cap_wui_maintenance",
    # Audit-useful inputs — construction
    "roof_wood", "roof_non_combust",
    "siding_wood", "siding_non_combust", "siding_fire_resist",
    "eaves_open", "eaves_enclosed",
    "window_multi_tempered", "window_single_tempered",
    "window_multi_untempered", "window_single_untempered",
    "ventscreens_1_4", "ventscreens_1_2", "ventscreens_1_8",
    "deck_above_grade_wood", "fence_combustible",
    # Audit-useful inputs — defensible space
    "zone_0_compliant", "zone_1_defensible", "zone_2_defensible",
    # Audit-useful inputs — year of construction
    "year_built_2000_2025", "year_built_1970_1999", "year_built_1940_1969",
    # Audit-useful inputs — adjacency / parcel
    "neighborhood_count", "structural_sep_dist",
    "dist_to_road_m", "acres_treated_wui",
]


def slim(scenario: str) -> None:
    src = OUT_DIR / f"{scenario}_risk_scored_5component.geojson"
    dst = OUT_DIR / f"{scenario}_risk_arcgis.geojson"
    with open(src) as f:
        gj = json.load(f)

    # Build the column set actually present in the source.
    src_cols = list(gj["features"][0]["properties"].keys())
    keep = [c for c in KEEP_COLUMNS if c in src_cols]
    dropped = [c for c in src_cols if c not in keep]

    for feat in gj["features"]:
        feat["properties"] = {c: feat["properties"].get(c) for c in keep}

    with open(dst, "w") as f:
        json.dump(gj, f)

    src_size = os.path.getsize(src)
    dst_size = os.path.getsize(dst)
    print(
        f"{scenario}: {len(gj['features']):,} features  "
        f"{len(src_cols)} → {len(keep)} cols  "
        f"({src_size/1024/1024:.1f} MB → {dst_size/1024/1024:.1f} MB)"
    )
    print(f"  wrote {dst.relative_to(BASE_DIR)}")
    print(f"  dropped {len(dropped)} columns")


def main() -> None:
    for scenario in ("Forest", "Prairie"):
        slim(scenario)


if __name__ == "__main__":
    main()

"""Extract per-property ignition timestamps from the OroraTech HTML viewer and
write a GeoJSON of building footprints with the `ignition` attribute.

The HTML viewer embeds a JavaScript constant ``S`` — an array of one object
per structure with keys ``{lat, lon, sid, mat, size, ftp, t_s, t_h, color}``.
Properties that ignited carry a non-null ``t_s`` (seconds from the simulation
start); the rest have ``t_s = null``.

For each scenario the script:

1. Parses the ``S`` array out of the HTML.
2. Joins each ignited structure (``sid`` = row index) to the corresponding
   building polygon in the ``<scenario>_generated_buildings_fireprops`` file.
3. Spatially attaches the richer attributes from the Parcel-joined building
   file (``bldgid``, ``usage``, ``property_value``, ``building_sqft``) and
   prefers the more detailed parcel polygon as the output geometry, falling
   back to the simpler fireprops rectangle when no parcel match exists.
4. Writes ``data_cache/fire_spread_simulations/<scenario>_scenario/
   forecast_<scenario>_properties.geojson``.

Usage:
    python extract_property_ignitions.py                # both scenarios
    python extract_property_ignitions.py --scenario Forest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SCENARIOS
from src.loaders import load_ignition


@dataclass(frozen=True)
class SimSpec:
    scenario: str
    html_path: str
    out_path: str


SPECS: dict[str, SimSpec] = {
    "Forest": SimSpec(
        scenario="Forest",
        html_path=(
            "data_cache/fire_spread_simulations/forest_scenario/"
            "forecast_forest_wui_map_coupled_20m_5m.html"
        ),
        out_path=(
            "data_cache/fire_spread_simulations/forest_scenario/"
            "forecast_forest_properties.geojson"
        ),
    ),
    "Prairie": SimSpec(
        scenario="Prairie",
        html_path=(
            "data_cache/fire_spread_simulations/prairie_scenario/"
            "forecast_prairie_wui_map_minSpeed0015.html"
        ),
        out_path=(
            "data_cache/fire_spread_simulations/prairie_scenario/"
            "forecast_prairie_properties.geojson"
        ),
    ),
}


def _extract_S_array(html_text: str, html_path: str) -> list[dict]:
    """Pull the ``S`` array out of the HTML by bracket-matching."""
    key = "const S="
    start = html_text.find(key)
    if start < 0:
        raise RuntimeError(f"Could not find {key!r} in {html_path}")
    i = start + len(key)
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < len(html_text):
        c = html_text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    return json.loads(html_text[i : j + 1])


def run_one(spec: SimSpec) -> None:
    scenario = SCENARIOS[spec.scenario]
    buildings_path = os.path.join(
        scenario.dir, f"{scenario.prefix}_generated_buildings_fireprops.geojson"
    )
    parcel_path = os.path.join(
        scenario.dir,
        f"{scenario.prefix}_generated_buildings_fireprops_Parcel.geojson",
    )

    print(f"\n=== {spec.scenario} ===")
    print(f"  HTML: {spec.html_path}")
    with open(spec.html_path) as f:
        html = f.read()
    S = _extract_S_array(html, spec.html_path)
    print(f"  parsed S array: {len(S):,} structures")
    ignited = [s for s in S if s.get("t_s") is not None]
    print(f"  ignited: {len(ignited):,}  ({100 * len(ignited) / len(S):.1f}%)")

    # Buildings file — sid in S indexes rows of this GeoJSON in their native order.
    buildings = gpd.read_file(buildings_path)
    buildings.columns = [c.lower() for c in buildings.columns]
    if len(buildings) != len(S):
        print(f"  WARNING: building count {len(buildings)} != S length {len(S)}")

    # Also pull richer per-building attributes from the Parcel-joined file.
    parcel = gpd.read_file(parcel_path)
    parcel.columns = [c.lower() for c in parcel.columns]
    keep = [
        c for c in [
            "bldgid", "usage", "property_value",
            "building_sqft", "stories_above_ground",
        ]
        if c in parcel.columns
    ]

    _, _, ig_time = load_ignition(scenario)
    ig_time = pd.to_datetime(ig_time).tz_convert("UTC")

    # Map fireprops row (sid) → parcel row via centroid-in-parcel sjoin.
    # This gives us the *detailed* polygon used by <Scenario>_risk_map.html for
    # each ignited building. Buildings whose centroid falls in no parcel keep
    # their simpler fireprops rectangle.
    bld_centroids = (
        buildings.assign(_sid=range(len(buildings)))
        .reset_index(drop=True)
        .to_crs(3857)
        .set_geometry(buildings.to_crs(3857).geometry.centroid)
        .to_crs(buildings.crs)[["_sid", "geometry"]]
    )
    parcel_with_idx = (
        parcel.reset_index(drop=True)
        .assign(_parcel_row=lambda d: range(len(d)))
        [["_parcel_row"] + keep + ["geometry"]]
    )
    sid_to_parcel = gpd.sjoin(
        bld_centroids, parcel_with_idx,
        how="left", predicate="within",
    ).drop_duplicates(subset=["_sid"]).set_index("_sid")

    features: list[dict] = []
    n_parcel_geom = 0
    n_fallback_geom = 0
    for s in ignited:
        sid = int(s["sid"])
        if not (0 <= sid < len(buildings)):
            continue
        t_s = float(s["t_s"])
        ignition_ts = (ig_time + pd.Timedelta(seconds=t_s)).isoformat()

        parcel_row_val = (
            sid_to_parcel.loc[sid, "_parcel_row"]
            if sid in sid_to_parcel.index else None
        )
        if pd.notna(parcel_row_val):
            prow = int(parcel_row_val)
            geom = parcel.iloc[prow].geometry
            n_parcel_geom += 1
        else:
            prow = None
            geom = buildings.iloc[sid].geometry
            n_fallback_geom += 1
        geom_json = json.loads(
            gpd.GeoSeries([geom]).to_json()
        )["features"][0]["geometry"]

        extra: dict = {}
        if prow is not None:
            prow_data = parcel.iloc[prow]
            for c in keep:
                v = prow_data.get(c)
                if pd.notna(v):
                    extra[c] = (
                        int(v) if c == "bldgid" else (
                            float(v) if isinstance(v, (int, float)) else str(v)
                        )
                    )

        row = buildings.iloc[sid]
        props = {
            "sid":         sid,
            "parcel_row":  prow,
            "bldgid":      extra.get(
                "bldgid",
                int(row.get("bldgid")) if pd.notna(row.get("bldgid")) else None,
            ),
            "usage":       extra.get(
                "usage",
                str(row.get("usage")) if pd.notna(row.get("usage")) else None,
            ),
            "property_value":  extra.get("property_value"),
            "building_sqft":   extra.get("building_sqft"),
            "ignition":        ignition_ts,
            "arrival_hours_from_ignition":   round(t_s / 3600.0, 3),
            "arrival_seconds_from_ignition": int(t_s),
            "material":        s.get("mat"),
            "color":           s.get("color"),
        }
        props = {k: v for k, v in props.items() if v is not None}
        features.append({
            "type":       "Feature",
            "properties": props,
            "geometry":   geom_json,
        })

    out = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(spec.out_path), exist_ok=True)
    with open(spec.out_path, "w") as f:
        json.dump(out, f)
    print(f"  wrote {spec.out_path}  ({os.path.getsize(spec.out_path)/1024:.0f} KB)")
    print(f"    features: {len(features):,}")
    print(f"    detailed parcel polygons: {n_parcel_geom:,}")
    print(f"    rectangle fallback:        {n_fallback_geom:,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario", choices=list(SPECS) + ["both"], default="both",
    )
    args = ap.parse_args()
    targets = list(SPECS) if args.scenario == "both" else [args.scenario]
    for name in targets:
        spec = SPECS[name]
        if not os.path.exists(spec.html_path):
            print(f"[skip] {name}: HTML not found at {spec.html_path}")
            continue
        run_one(spec)


if __name__ == "__main__":
    main()

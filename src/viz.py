"""Folium maps: per-building risk choropleth + components."""

from __future__ import annotations

import os

import branca.colormap as cm
import folium
import geopandas as gpd

from .config import BASE_DIR


_DECILE_COLORS = [
    # matplotlib 'inferno' colormap sampled at 10 stops
    "#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60",
    "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4",
]


def _scenario_bounds(gdf: gpd.GeoDataFrame):
    bx = gdf.total_bounds  # minx, miny, maxx, maxy
    return [(bx[1], bx[0]), (bx[3], bx[2])]


def _decile_style_fn(score_col: str, vmin: float = 0.0, vmax: float = 1.0):
    cmap = cm.LinearColormap(_DECILE_COLORS, vmin=vmin, vmax=vmax, caption=score_col)

    def style(feat):
        v = feat["properties"].get(score_col, 0)
        return {
            "fillColor": cmap(float(v) if v is not None else 0.0),
            "color": "#444",
            "weight": 0.3,
            "fillOpacity": 0.78,
        }
    return style, cmap


def _tooltip_fields(gdf: gpd.GeoDataFrame, extra: list[str]):
    base = ["bldgid" if "bldgid" in gdf.columns else "index", "usage"]
    return [c for c in base + extra if c in gdf.columns]


def _layer_vmax(series, floor: float = 0.1) -> float:
    """Round the column's max up to 2 decimals (with a small floor) so the
    colourbar uses the full palette without spending the top of the gradient
    on values the data never reaches."""
    s = series.dropna()
    if len(s) == 0:
        return 1.0
    raw = float(s.max())
    return max(floor, round(raw + 0.005, 2))


def _layer_vmin(series, min_gap: float = 0.15, buffer: float = 0.02) -> float:
    """Auto-detect a 'gap' in the distribution and use it as vmin.

    For bimodal distributions (e.g., Forest hazard, where 75 % of buildings
    sit near 0 and 25 % sit in [0.6, 0.9] with an empty middle), this places
    the upper cluster at the dark end of the palette and stretches it across
    the full inferno range. Returns 0 for continuous distributions.

    `min_gap` is the smallest gap (in absolute units) considered meaningful.
    `buffer` is subtracted from the upper-cluster start so the lower edge of
    the cluster lands just above the darkest colour, not on it.
    """
    import numpy as np
    s = series.dropna().to_numpy()
    s = s[s > 1e-3]                     # ignore numerical noise / zeros
    if len(s) < 50:
        return 0.0
    s = np.sort(s)
    diffs = np.diff(s)
    if len(diffs) == 0:
        return 0.0
    max_idx = int(np.argmax(diffs))
    if float(diffs[max_idx]) < min_gap:
        return 0.0
    upper_start = float(s[max_idx + 1])
    return max(0.0, round(upper_start - buffer, 2))


def make_risk_map(gdf: gpd.GeoDataFrame, scenario_name: str) -> str:
    centroid = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles="cartodbpositron")
    vmin = _layer_vmin(gdf["risk_score"])
    vmax = _layer_vmax(gdf["risk_score"])
    style, cmap = _decile_style_fn("risk_score", vmin=vmin, vmax=vmax)
    fields = _tooltip_fields(gdf, [
        "risk_score", "risk_decile", "hazard", "vulnerability", "exposure",
        "property_value",
    ])
    folium.GeoJson(
        gdf, name="Risk score",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=fields, localize=True),
    ).add_to(m)
    cmap.caption = f"{scenario_name} wildfire risk score ({vmin:.2f} – {vmax:.2f})"
    cmap.add_to(m)
    m.fit_bounds(_scenario_bounds(gdf))
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{scenario_name}_risk_map.html")
    m.save(out)
    print(f"[viz] {out}")
    return out


def make_components_map(gdf: gpd.GeoDataFrame, scenario_name: str) -> str:
    centroid = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles="cartodbpositron")

    # Add a layer per component that the GeoDataFrame carries. Severity and
    # capacity are optional (only present in 5-component scoring); the final
    # risk_score is always last and shown by default.
    layer_specs = [
        ("hazard",        "Hazard"),
        ("vulnerability", "Vulnerability"),
        ("exposure",      "Exposure"),
    ]
    if "severity" in gdf.columns:
        layer_specs.append(("severity", "Severity"))
    if "capacity" in gdf.columns:
        layer_specs.append(("capacity", "Capacity"))
    layer_specs.append(("risk_score", "Final risk"))

    for col, label in layer_specs:
        if col not in gdf.columns:
            continue
        vmin = _layer_vmin(gdf[col])
        vmax = _layer_vmax(gdf[col])
        style, cmap = _decile_style_fn(col, vmin=vmin, vmax=vmax)
        cmap.caption = f"{label} ({vmin:.2f} – {vmax:.2f})"
        fg = folium.FeatureGroup(name=label, show=(col == "risk_score"))
        folium.GeoJson(
            gdf, name=label,
            style_function=style,
            tooltip=folium.GeoJsonTooltip(
                fields=_tooltip_fields(gdf, [col, "risk_score"]),
                localize=True,
            ),
        ).add_to(fg)
        fg.add_to(m)
        # Per-layer colourbar — each component's gradient runs across its
        # actual [vmin, vmax] range, so the full palette is used per layer.
        # For bimodal distributions, vmin is auto-set to the gap so the
        # upper cluster spans the entire palette.
        cmap.add_to(m)

    # Top-10 highest-risk buildings — bright outline + numbered marker
    if "risk_score" in gdf.columns:
        top10 = gdf.nlargest(10, "risk_score").reset_index(drop=True)
        fg_top = folium.FeatureGroup(name="Top 10 highest risk", show=True)
        folium.GeoJson(
            top10,
            style_function=lambda _f: {
                "fillColor": "#00ffff",
                "color": "#00ffff",
                "weight": 3,
                "fillOpacity": 0.0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=_tooltip_fields(gdf, ["risk_score", "risk_decile", "hazard",
                                              "vulnerability", "exposure",
                                              "severity", "capacity",
                                              "property_value"]),
                localize=True,
            ),
        ).add_to(fg_top)
        for i, row in top10.iterrows():
            c = row.geometry.centroid
            folium.Marker(
                location=[c.y, c.x],
                icon=folium.DivIcon(
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                    html=(f'<div style="background:#00ffff;color:#000;'
                          f'border:2px solid #003344;border-radius:50%;'
                          f'width:24px;height:24px;line-height:20px;'
                          f'text-align:center;font-weight:700;'
                          f'font-family:sans-serif;font-size:12px;'
                          f'box-shadow:0 1px 3px rgba(0,0,0,.4);">'
                          f'{i + 1}</div>'),
                ),
            ).add_to(fg_top)
        fg_top.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(_scenario_bounds(gdf))
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{scenario_name}_risk_components.html")
    m.save(out)
    print(f"[viz] {out}")
    return out


def make_all_maps(gdf: gpd.GeoDataFrame, scenario_name: str) -> tuple[str, str]:
    return make_risk_map(gdf, scenario_name), make_components_map(gdf, scenario_name)

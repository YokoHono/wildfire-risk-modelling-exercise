"""WUI structure overlay for ForeFire.

Implements the urban fire spread model of Purnomo et al. 2024 (IJWF
WF24102) and 2025 (Nature Communications, doi:10.1038/s41467-025-63386-2)
on top of a ForeFire wildland simulation:

    * Direct flame contact (DFC) heat from neighbouring burning structures
      (Eqns 2-4 of Purnomo 2024).
    * Radiation heat from neighbouring burning structures, point-source
      with 100 m cutoff (Eqn 5).
    * Ember (firebrand) deposition from any burning cell, lognormal
      landing PDF of Sardoy et al. 2008 (Eqns 6-9).
    * Cumulative flux-time-product ignition criterion (Eqn 13, Lee 2009).
    * Wildland -> urban heat coupling via HRR = I_f * dx (Eqn 15).

The overlay does *not* run inside ForeFire's C++ event loop. Instead it is
ticked once per ForeFire step from ``ForeFireWrapper.predict``: it reads
the current fireline intensity / arrival-time / wind state, accumulates
heat on every "structure cell" that is not yet ignited, and once a cell's
cumulative absorbed flux crosses its FTP threshold the overlay emits a
``startFire[loc=...; t=...]`` command so ForeFire continues the spread
through that cell with a structure-specific rate of spread.

Coupling is one-directional in the current implementation: structure
ignitions become wildland ignitions in ForeFire (so neighbouring grass /
ornamental vegetation inside or adjacent to the building footprint is
carried by the existing Rothermel front). Bidirectional radiative
feedback into wildland cells is captured implicitly because each ignited
structure cell radiates onto its neighbours in subsequent ticks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import structlog
from shapely.geometry import Point

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameters & per-structure state
# ---------------------------------------------------------------------------


@dataclass
class StructureProperties:
    """Per-structure physical attributes that drive the urban model.

    All fields default to the configured defaults; any of them can be
    overridden from the source GeoDataFrame columns (typically populated
    from a `fireprops` attribute table).
    """

    ftp_kJ_per_m2: float
    combustibility: float           # 0-1, scales FTP inversely
    combustible_fraction: float     # alpha_c, 0-1 (Eqn 12)
    radiation_absorptivity: float   # 0-1, multiplied with combustible_fraction
    peak_hrr_kW_per_m2: float
    peak_hrr_duration_s: float
    growth_s: float
    decay_s: float
    size_m: float


@dataclass
class StructureState:
    """Mutable per-structure run-time state."""

    structure_id: int
    centroid_xy: Tuple[float, float]
    cell_xy: Tuple[float, float]    # snapped to ForeFire raster center
    properties: StructureProperties
    ignited: bool = False
    ignition_time_s: Optional[float] = None
    cumulative_flux_kJ_per_m2: float = 0.0
    near_vegetation: bool = False   # set by mark_vegetation_proximity()
    nearest_veg_xy: Optional[Tuple[float, float]] = None  # emission point for startFire[]
    # Trail of (time_s, q_dot_total_kW_per_m2) for diagnostics.
    flux_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class WildlandSource:
    """One sampled point along the wildland fire front perimeter.

    Each instance represents ``sample_spacing_m`` metres of front arc-length.
    Total HRR contribution = fireline_intensity_kW_per_m × sample_spacing_m,
    so summing across all samples correctly integrates intensity along the
    perimeter without double-counting.
    """

    x: float                            # UTM easting
    y: float                            # UTM northing
    fireline_intensity_kW_per_m: float  # Byram I_f [kW/m]
    sample_spacing_m: float             # arc-length between adjacent samples

    @property
    def hrr_kW(self) -> float:
        return self.fireline_intensity_kW_per_m * self.sample_spacing_m


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


class StructureOverlay:
    """Urban-fire overlay for ForeFireWrapper.

    Lifecycle:

        overlay = StructureOverlay.from_geojson(buildings_path, config)
        overlay.attach(forefire_wrapper, cell_size_m=resolution)
        # ... in predict() between steps:
        overlay.tick(simulated_time_s, dt_s, wind_speed_mps, wind_dir_deg)
    """

    def __init__(
        self,
        structures: List[StructureState],
        config: Dict[str, Any],
        epsg: int,
    ):
        self.structures = structures
        self.config = config
        self.epsg = epsg
        self.cell_size_m: float = 10.0  # set in attach()
        self._ff_wrapper = None         # set in attach()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        config: Dict[str, Any],
        target_epsg: int,
        attribute_map: Optional[Dict[str, str]] = None,
    ) -> "StructureOverlay":
        """Load a building GeoJSON and return an overlay instance.

        The GeoJSON must contain Polygon / MultiPolygon features. Optional
        per-feature attributes (controlled by ``attribute_map``) override
        the defaults from the config:

            attribute_map = {
                "ftp_kJ_per_m2":          "FTP",
                "combustible_fraction":   "alpha_c",
                "radiation_absorptivity": "alpha_r",
                "size_m":                 "footprint_m",
                "material_class":         "roof_material",
            }
        """
        gdf = gpd.read_file(path)
        return cls.from_geodataframe(
            gdf, config=config, target_epsg=target_epsg, attribute_map=attribute_map,
            source=str(path),
        )

    @classmethod
    def from_geodataframe(
        cls,
        gdf: "gpd.GeoDataFrame",
        config: Dict[str, Any],
        target_epsg: int,
        attribute_map: Optional[Dict[str, str]] = None,
        source: str = "<in-memory>",
    ) -> "StructureOverlay":
        """Build an overlay from an in-memory GeoDataFrame.

        Convenience for callers that want to pre-derive columns
        (e.g. compute ``material_class`` from boolean roof / siding fields)
        before constructing the overlay.
        """
        gdf = gdf.to_crs(epsg=target_epsg)

        attribute_map = attribute_map or {}
        defaults = config["structure"]
        material_classes = config.get("_material_classes", {})

        states: List[StructureState] = []
        for sid, (_, row) in enumerate(gdf.iterrows()):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid
            size_attr = attribute_map.get("size_m")
            if size_attr and size_attr in row and row[size_attr] is not None:
                size_m = float(row[size_attr])
            else:
                # Use sqrt(area) as a robust proxy for "characteristic size".
                size_m = float(math.sqrt(max(geom.area, 1.0)))

            material = None
            mat_attr = attribute_map.get("material_class")
            if mat_attr and mat_attr in row and row[mat_attr] is not None:
                material = material_classes.get(str(row[mat_attr]).lower())

            def pick(key: str, default_key: str) -> float:
                attr = attribute_map.get(key)
                if attr and attr in row and row[attr] is not None:
                    return float(row[attr])
                if material and key in material:
                    return float(material[key])
                return float(defaults[default_key])

            combustibility = pick(
                "combustibility", "combustibility_default"
            )
            ftp = (
                float(defaults["ftp_baseline_kJ_per_m2"]) / max(combustibility, 1e-3)
            )
            attr_ftp = attribute_map.get("ftp_kJ_per_m2")
            if attr_ftp and attr_ftp in row and row[attr_ftp] is not None:
                ftp = float(row[attr_ftp])

            props = StructureProperties(
                ftp_kJ_per_m2=ftp,
                combustibility=combustibility,
                combustible_fraction=pick(
                    "combustible_fraction", "combustible_fraction_default"
                ),
                radiation_absorptivity=pick(
                    "radiation_absorptivity", "radiation_absorptivity_default"
                ),
                peak_hrr_kW_per_m2=float(
                    defaults["peak_hrr_kW_per_m2_default"]
                ),
                peak_hrr_duration_s=float(
                    defaults["peak_hrr_duration_seconds_default"]
                ),
                growth_s=float(defaults["growth_seconds"]),
                decay_s=float(defaults["decay_seconds"]),
                size_m=size_m,
            )
            states.append(
                StructureState(
                    structure_id=int(sid),
                    centroid_xy=(centroid.x, centroid.y),
                    cell_xy=(centroid.x, centroid.y),  # snapped in attach()
                    properties=props,
                )
            )

        logger.info(
            "Loaded WUI structure overlay.",
            n_structures=len(states),
            target_epsg=target_epsg,
            source=source,
        )
        return cls(structures=states, config=config, epsg=target_epsg)

    @staticmethod
    def load_config(path: str | Path) -> Dict[str, Any]:
        with open(path, "r") as fh:
            return json.load(fh)

    # -- attachment ----------------------------------------------------------

    def attach(self, ff_wrapper, cell_size_m: float) -> None:
        """Bind the overlay to a ForeFireWrapper and record the wildland grid size.

        We deliberately do NOT snap structure centroids to that grid: the
        wildland raster (e.g. 30 m LANDFIRE) is typically much coarser than
        building spacing (5-15 m), so snapping collapses many structures
        onto the same cell and saturates the DFC engulfment fraction.
        Distances are therefore computed from real centroids; ``cell_size_m``
        is only used when we eventually emit ignitions back to ForeFire.
        """
        self._ff_wrapper = ff_wrapper
        self.cell_size_m = float(cell_size_m)
        for s in self.structures:
            s.cell_xy = s.centroid_xy

    def mark_vegetation_proximity(
        self,
        fuel_map_path: "str | Path",
        proximity_m: float = 50.0,
    ) -> None:
        """Flag structures within ``proximity_m`` of a burnable fuel pixel.

        Used by :meth:`_emit_start_fire` in ``"near_vegetation"`` coupling
        mode: only structures adjacent to vegetated land cover emit
        ``startFire[]`` back into ForeFire, avoiding degenerate micro-fronts
        in dense NODATA urban cores while still letting fire re-enter grass /
        park corridors at the urban edge.
        """
        import rasterio
        from pyproj import Transformer
        from rasterio.transform import rowcol
        from rasterio.transform import xy as raster_xy

        with rasterio.open(fuel_map_path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            transform = src.transform
            raster_epsg = src.crs.to_epsg()
            pixel_deg = abs(transform.a)
            # Convert pixel size to metres (handles geographic CRS).
            pixel_m = pixel_deg * 111_320.0 if raster_epsg == 4326 else pixel_deg

        valid = arr > 0
        if nodata is not None:
            valid &= arr != float(nodata)

        radius_px = max(1, int(math.ceil(proximity_m / pixel_m)))

        # Forward: overlay UTM → raster CRS.  Back: raster CRS → overlay UTM.
        to_raster: Optional[Any] = None
        to_overlay: Optional[Any] = None
        if raster_epsg is not None and raster_epsg != self.epsg:
            to_raster = Transformer.from_crs(
                f"EPSG:{self.epsg}", f"EPSG:{raster_epsg}", always_xy=True
            )
            to_overlay = Transformer.from_crs(
                f"EPSG:{raster_epsg}", f"EPSG:{self.epsg}", always_xy=True
            )

        n_marked = 0
        for s in self.structures:
            cx, cy = s.centroid_xy
            rcx, rcy = to_raster.transform(cx, cy) if to_raster else (cx, cy)
            try:
                r, c = rowcol(transform, rcx, rcy)
            except Exception:
                continue
            r0 = max(0, r - radius_px)
            r1 = min(valid.shape[0], r + radius_px + 1)
            c0 = max(0, c - radius_px)
            c1 = min(valid.shape[1], c + radius_px + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            window = valid[r0:r1, c0:c1]
            if not window.any():
                continue

            s.near_vegetation = True
            n_marked += 1

            # Find the nearest vegetated pixel and store its projected coords.
            # startFire[] must land on a valid fuel pixel, not on the NODATA
            # structure centroid, or ForeFire will read fuel index 32767 and crash.
            vrows, vcols = np.where(window)
            vrows = vrows + r0
            vcols = vcols + c0
            dists = (vrows - r) ** 2 + (vcols - c) ** 2
            best = int(np.argmin(dists))
            px, py = raster_xy(transform, int(vrows[best]), int(vcols[best]))
            if to_overlay is not None:
                px, py = to_overlay.transform(px, py)
            s.nearest_veg_xy = (float(px), float(py))

        logger.info(
            "Marked structures near vegetation.",
            n_near=n_marked,
            total=len(self.structures),
            proximity_m=proximity_m,
        )

    # -- per-step update -----------------------------------------------------

    def tick(
        self,
        sim_time_s: float,
        dt_s: float,
        wind_speed_mps: float,
        wind_dir_deg: float,
        wildland_sources: Optional[List["WildlandSource"]] = None,
    ) -> List[StructureState]:
        """Advance the overlay by one ForeFire step.

        Returns the list of structures newly ignited during this tick.
        For each newly ignited structure the overlay also emits a
        ``startFire[loc=(x,y,0); t=...]`` command on the bound ForeFire
        instance so that subsequent ForeFire steps carry the front
        through that cell.

        ``wildland_sources`` is a list of sampled points along the current
        wildland fire front perimeter (see ``WildlandSource``). When provided,
        each unignited structure receives additional heat from the wildland
        fire via radiation, direct flame contact, and embers — independent of
        any already-ignited structures.
        """
        if self._ff_wrapper is None:
            raise RuntimeError(
                "StructureOverlay.tick called before attach()."
            )

        # 1. Collect every burning emitter (already-ignited structures).
        burning = [s for s in self.structures if s.ignited]
        unignited = [s for s in self.structures if not s.ignited]
        if not unignited:
            return []

        wind_vec = self._wind_vector(wind_speed_mps, wind_dir_deg)

        # 2. Aggregate heat on each unignited structure.
        new_ignitions: List[StructureState] = []
        for target in unignited:
            q_dfc = self._dfc_heat_kW_per_m2(target, burning, sim_time_s, wind_vec)
            q_rad = self._radiation_heat_kW_per_m2(target, burning, sim_time_s)
            q_ember = self._ember_heat_kW_per_m2(target, burning, wind_speed_mps)
            # Wildland fire front → structure coupling (Purnomo 2024, Eqn 15).
            if wildland_sources:
                q_rad += self._wildland_rad_kW_per_m2(target, wildland_sources)
                q_dfc += self._wildland_dfc_kW_per_m2(
                    target, wildland_sources, wind_vec
                )
                q_ember += self._wildland_ember_kW_per_m2(
                    target, wildland_sources, wind_speed_mps
                )
            # Eqn 11 Purnomo: q_t = alpha_c * q_c + alpha_r * q_r''
            alpha_c = target.properties.combustible_fraction
            alpha_r = alpha_c * target.properties.radiation_absorptivity
            q_total = alpha_c * (q_dfc + q_ember) + alpha_r * q_rad

            target.cumulative_flux_kJ_per_m2 += q_total * dt_s  # kW/m^2 * s = kJ/m^2
            target.flux_history.append((sim_time_s, q_total))

            # Eqn 13 Purnomo: ignite when sum(q_t * dt) >= FTP.
            if target.cumulative_flux_kJ_per_m2 >= target.properties.ftp_kJ_per_m2:
                target.ignited = True
                target.ignition_time_s = sim_time_s
                new_ignitions.append(target)

        # 3. Emit startFire[] for newly ignited structures.
        # startFire[loc=...] sizes the initial triangle at 2×perimeterResolution,
        # so we temporarily reduce resolution to building scale before the batch
        # and restore production values immediately after.  No step[] is called
        # in between, so the wildland front is unaffected.
        if new_ignitions:
            coupling = self.config.get("coupling", {})
            mode = coupling.get("emit_start_fires", False)
            if mode not in (False, "false"):
                ign_pres = float(
                    coupling.get("start_fire_perimeter_resolution_m", 5.0)
                )
                prod_pres = getattr(
                    self._ff_wrapper, "_perimeter_resolution_prod", 30
                )
                prod_inc = getattr(
                    self._ff_wrapper, "_spatial_increment_prod", 1.0
                )
                self._ff_wrapper._execute(
                    f"trigger[resolution;"
                    f"perimeterResolution={ign_pres:.1f};"
                    f"spatialIncrement={ign_pres / 3.0:.2f}]"
                )
                for s in new_ignitions:
                    self._emit_start_fire(s, sim_time_s)
                self._ff_wrapper._execute(
                    f"trigger[resolution;"
                    f"perimeterResolution={prod_pres};"
                    f"spatialIncrement={prod_inc}]"
                )
            else:
                for s in new_ignitions:
                    self._emit_start_fire(s, sim_time_s)

        if new_ignitions:
            logger.info(
                "WUI overlay ignited new structures.",
                count=len(new_ignitions),
                sim_time_s=sim_time_s,
                ids=[s.structure_id for s in new_ignitions],
            )
        return new_ignitions

    # -- physics helpers -----------------------------------------------------

    def _hrr_profile_kW(self, source: StructureState, sim_time_s: float) -> float:
        """Transient HRR per Purnomo 2024 supplementary Fig. S1."""
        if source.ignition_time_s is None:
            return 0.0
        t = sim_time_s - source.ignition_time_s
        if t < 0:
            return 0.0
        peak_kW_per_m2 = source.properties.peak_hrr_kW_per_m2
        peak = peak_kW_per_m2 * (source.properties.size_m ** 2)  # kW
        if t < source.properties.growth_s:
            return peak * (t / source.properties.growth_s)
        if t < source.properties.growth_s + source.properties.peak_hrr_duration_s:
            return peak
        decay_start = source.properties.growth_s + source.properties.peak_hrr_duration_s
        if t < decay_start + source.properties.decay_s:
            frac = 1.0 - (t - decay_start) / source.properties.decay_s
            return max(0.0, peak * frac)
        return 0.0

    def _dfc_heat_kW_per_m2(
        self,
        target: StructureState,
        burning: Sequence[StructureState],
        sim_time_s: float,
        wind_vec: Tuple[float, float],
    ) -> float:
        """Direct flame contact contribution (Eqns 2-4)."""
        if not burning:
            return 0.0
        wx, wy = wind_vec
        wind_mag = math.hypot(wx, wy)
        const_a = self.config["dfc"]["downwind_const_a"]
        const_b = self.config["dfc"]["upwind_const_b"]
        intercept = self.config["dfc"]["intercept_m"]

        total_kW_per_m2 = 0.0
        for src in burning:
            dx = target.cell_xy[0] - src.cell_xy[0]
            dy = target.cell_xy[1] - src.cell_xy[1]
            dist = math.hypot(dx, dy)
            if dist <= 0.0:
                continue
            # Project distance onto wind direction to decide downwind/upwind.
            if wind_mag > 1e-6:
                proj = (dx * wx + dy * wy) / wind_mag
            else:
                proj = 0.0
            # Eqns 3-4: a (downwind) and b (upwind) flame reach measured
            # from the source centroid.
            d_half = src.properties.size_m / 2.0
            if proj >= 0:
                reach = const_a * wind_mag + intercept + d_half
            else:
                reach = const_b * wind_mag + intercept + d_half
            target_radius = target.properties.size_m / 2.0
            # Distance from the target's nearest face to the flame envelope.
            face_to_flame = dist - target_radius - reach
            if face_to_flame >= 0.0:
                continue
            hrr_kW = self._hrr_profile_kW(src, sim_time_s)
            if hrr_kW <= 0:
                continue
            # Eqn 2 reinterpreted on a per-structure (not per-grid-cell) basis:
            # the flame surface flux experienced by a fully engulfed neighbour
            # is the HRR per unit floor area of the source; the engulfed
            # fraction scales linearly with the overlap depth into the target.
            engulfed_frac = max(
                0.0,
                min(1.0, -face_to_flame / max(target.properties.size_m, 1e-3)),
            )
            flame_flux_kW_per_m2 = hrr_kW / max(
                src.properties.size_m ** 2, 1e-3
            )
            total_kW_per_m2 += flame_flux_kW_per_m2 * engulfed_frac
        return total_kW_per_m2

    def _radiation_heat_kW_per_m2(
        self,
        target: StructureState,
        burning: Sequence[StructureState],
        sim_time_s: float,
    ) -> float:
        """Point-source radiation (Eqn 5)."""
        if not burning:
            return 0.0
        frac = self.config["radiation"]["fraction_of_hrr"]
        rmax = self.config["radiation"]["max_distance_m"]
        total = 0.0
        for src in burning:
            dx = target.cell_xy[0] - src.cell_xy[0]
            dy = target.cell_xy[1] - src.cell_xy[1]
            R = math.hypot(dx, dy)
            if R <= 0.0 or R >= rmax:
                continue
            hrr = self._hrr_profile_kW(src, sim_time_s)
            if hrr <= 0:
                continue
            total += frac * hrr / (4.0 * math.pi * R * R)
        return total

    def _ember_heat_kW_per_m2(
        self,
        target: StructureState,
        burning: Sequence[StructureState],
        wind_speed_mps: float,
    ) -> float:
        """Probabilistic ember contribution (structure-to-structure).

        In stochastic mode (``embers.stochastic: true``) the number of embers
        landing on the target is sampled from a Poisson distribution whose
        mean is the Sardoy expected-value. This introduces variance that allows
        rare high-ember events to ignite structures that would never cross the
        FTP threshold under the deterministic expected-value model.
        """
        cfg = self.config["embers"]
        if not cfg.get("enabled", True):
            return 0.0
        if not burning:
            return 0.0

        stochastic = cfg.get("stochastic", False)
        v = max(wind_speed_mps, 1e-3)
        g = self.config["physical_constants"]["g_m_per_s2"]
        Lc = cfg["lc_characteristic_plume_length_m"]
        Fr = (v ** 2) / (g * Lc)
        n_embers = float(cfg["embers_per_step_per_burning_cell"])
        p_ig = float(cfg["ignition_probability_per_landed_ember"])
        p_trunc = float(cfg["min_landing_probability_truncation"])

        contrib = 0.0
        for src in burning:
            hrr_kW = src.properties.peak_hrr_kW_per_m2 * (src.properties.size_m ** 2)
            If = max(hrr_kW / max(src.properties.size_m, 1.0), 1e-3)
            if Fr <= 1.0:
                mu = 1.47 * If ** 0.54 * v ** -0.55 + 1.14
                sigma = 0.86 * If ** -0.21 * v ** 0.44 + 0.19
            else:
                mu = 1.32 * If ** 0.26 * v ** 0.11 - 0.02
                sigma = 4.95 * If ** -0.01 * v ** -0.02 - 3.48
            sigma = max(sigma, 1e-3)
            dx = target.cell_xy[0] - src.cell_xy[0]
            dy = target.cell_xy[1] - src.cell_xy[1]
            x = math.hypot(dx, dy)
            if x <= 0.0:
                continue
            p = (
                1.0
                / (math.sqrt(2.0 * math.pi) * sigma * x)
                * math.exp(-((math.log(x) - mu) ** 2) / (2.0 * sigma * sigma))
            )
            if p < p_trunc:
                continue
            if stochastic:
                actual = int(np.random.poisson(p * n_embers))
                if actual == 0:
                    continue
                contrib += actual * p_ig * If
            else:
                contrib += p * n_embers * p_ig * If
        return contrib

    # -- wildland fire → structure coupling -------------------------------------

    @property
    def structure_bbox(self) -> Tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) bounding box of all structure centroids."""
        if not self.structures:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [s.cell_xy[0] for s in self.structures]
        ys = [s.cell_xy[1] for s in self.structures]
        return min(xs), min(ys), max(xs), max(ys)

    def _wildland_rad_kW_per_m2(
        self,
        target: StructureState,
        sources: List["WildlandSource"],
    ) -> float:
        """Point-source radiation from sampled wildland front (Purnomo 2024, Eqn 15)."""
        frac = self.config["radiation"]["fraction_of_hrr"]
        rmax = self.config["radiation"]["max_distance_m"]
        total = 0.0
        for src in sources:
            dx = target.cell_xy[0] - src.x
            dy = target.cell_xy[1] - src.y
            R = math.hypot(dx, dy)
            if R <= 0.0 or R >= rmax:
                continue
            total += frac * src.hrr_kW / (4.0 * math.pi * R * R)
        return total

    def _wildland_dfc_kW_per_m2(
        self,
        target: StructureState,
        sources: List["WildlandSource"],
        wind_vec: Tuple[float, float],
    ) -> float:
        """Direct flame contact from wildland fire front onto interface structures."""
        wx, wy = wind_vec
        wind_mag = math.hypot(wx, wy)
        const_a = self.config["dfc"]["downwind_const_a"]
        const_b = self.config["dfc"]["upwind_const_b"]
        intercept = self.config["dfc"]["intercept_m"]
        total = 0.0
        for src in sources:
            dx = target.cell_xy[0] - src.x
            dy = target.cell_xy[1] - src.y
            dist = math.hypot(dx, dy)
            if dist <= 0.0:
                continue
            proj = (dx * wx + dy * wy) / wind_mag if wind_mag > 1e-6 else 0.0
            # Wildland flame reach from the front perimeter (no source "size").
            reach = (const_a if proj >= 0 else const_b) * wind_mag + intercept
            face_to_flame = dist - target.properties.size_m / 2.0 - reach
            if face_to_flame >= 0.0:
                continue
            # Flux: I_f [kW/m] normalised over the sample's arc-length gives
            # an intensity per unit area of the impinging flame surface.
            flame_flux = src.fireline_intensity_kW_per_m / max(
                src.sample_spacing_m, 1.0
            )
            engulfed_frac = max(
                0.0,
                min(1.0, -face_to_flame / max(target.properties.size_m, 1e-3)),
            )
            total += flame_flux * engulfed_frac
        return total

    def _wildland_ember_kW_per_m2(
        self,
        target: StructureState,
        sources: List["WildlandSource"],
        wind_speed_mps: float,
    ) -> float:
        """Ember flux from wildland fire front — Sardoy et al. 2008 lognormal PDF."""
        cfg = self.config["embers"]
        if not cfg.get("enabled", True):
            return 0.0
        stochastic = cfg.get("stochastic", False)
        v = max(wind_speed_mps, 1e-3)
        g = self.config["physical_constants"]["g_m_per_s2"]
        Lc = cfg["lc_characteristic_plume_length_m"]
        Fr = (v ** 2) / (g * Lc)
        n_embers = float(cfg["embers_per_step_per_burning_cell"])
        p_ig = float(cfg["ignition_probability_per_landed_ember"])
        p_trunc = float(cfg["min_landing_probability_truncation"])
        contrib = 0.0
        for src in sources:
            If = max(src.fireline_intensity_kW_per_m, 1e-3)
            if Fr <= 1.0:
                mu = 1.47 * If ** 0.54 * v ** -0.55 + 1.14
                sigma = 0.86 * If ** -0.21 * v ** 0.44 + 0.19
            else:
                mu = 1.32 * If ** 0.26 * v ** 0.11 - 0.02
                sigma = 4.95 * If ** -0.01 * v ** -0.02 - 3.48
            sigma = max(sigma, 1e-3)
            dx = target.cell_xy[0] - src.x
            dy = target.cell_xy[1] - src.y
            x = math.hypot(dx, dy)
            if x <= 0.0:
                continue
            try:
                p = (
                    1.0
                    / (math.sqrt(2.0 * math.pi) * sigma * x)
                    * math.exp(
                        -((math.log(x) - mu) ** 2) / (2.0 * sigma * sigma)
                    )
                )
            except (ValueError, OverflowError):
                continue
            if p < p_trunc:
                continue
            if stochastic:
                actual = int(np.random.poisson(p * n_embers))
                if actual == 0:
                    continue
                contrib += actual * p_ig * If
            else:
                contrib += p * n_embers * p_ig * If
        return contrib

    @staticmethod
    def _wind_vector(speed_mps: float, dir_deg: float) -> Tuple[float, float]:
        """Convert meteorological wind direction (FROM) to a vector pointing TO."""
        rad = math.radians((dir_deg + 180.0) % 360.0)
        return speed_mps * math.sin(rad), speed_mps * math.cos(rad)

    def _emit_start_fire(self, s: StructureState, sim_time_s: float) -> None:
        mode = self.config.get("coupling", {}).get("emit_start_fires", False)
        if mode is False or mode == "false":
            return
        if mode == "near_vegetation" and not s.near_vegetation:
            return
        # mode is True  OR  mode == "near_vegetation" and structure qualifies.
        # Emit at the nearest vegetated pixel, not the structure centroid:
        # the centroid sits on NODATA (SB40 = 32767) and ForeFire would crash
        # trying to look up fuel properties at that index.
        x, y = s.nearest_veg_xy if s.nearest_veg_xy is not None else s.cell_xy
        self._ff_wrapper._execute(
            f"startFire[loc=({x},{y},0.);t={sim_time_s}]"
        )

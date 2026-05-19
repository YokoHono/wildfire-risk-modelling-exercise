"""End-to-end wildfire risk pipeline.

Usage:
    python run.py                              # both scenarios, defaults
    python run.py --scenario Forest
    python run.py --scenario both --realizations 50 --downsample 20
"""

from __future__ import annotations

import argparse
import sys

from src.risk import run_all
from src.viz import make_all_maps


def main():
    p = argparse.ArgumentParser(description="Wildfire risk: Hazard × Vulnerability × Exposure")
    p.add_argument("--scenario", choices=["Forest", "Prairie", "both"], default="both")
    p.add_argument("--realizations", type=int, default=100,
                   help="Monte Carlo realizations for the hazard simulation (default 100)")
    p.add_argument("--downsample", type=int, default=15,
                   help="Raster downsample factor for the simulation grid (default 15)")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip generating folium HTML maps")
    args = p.parse_args()

    scenarios = ["Forest", "Prairie"] if args.scenario == "both" else [args.scenario]

    results = run_all(
        scenarios=scenarios,
        realizations=args.realizations,
        downsample=args.downsample,
    )

    if not args.no_viz:
        for name, gdf in results.items():
            make_all_maps(gdf, name)

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())

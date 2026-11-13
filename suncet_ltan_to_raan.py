"""
Compute RAAN from LTAN for Sun-synchronous orbits using poliastro.

Load orbit targets from orbit_config.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import astropy.units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time
from astropy.utils import iers

with Path(__file__).with_name("orbit_config.json").open(encoding="utf-8") as stream:
    _orbit_config = json.load(stream)
DEFAULT_LAUNCH_NAME = _orbit_config["default_launch"]
LAUNCHES = _orbit_config["launches"]
DEFAULT_LAUNCH = LAUNCHES[DEFAULT_LAUNCH_NAME]


def ltan_string_to_quantity(ltan: str) -> u.Quantity:
    """Parse 'HH:MM:SS' into an astropy Quantity in hourangle (poliastro expects this)."""
    parts = ltan.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"LTAN must be 'HH:MM:SS', got {ltan!r}")
    h, m, s = (int(parts[0]), int(parts[1]), float(parts[2]))
    return Angle(f"{h}h{m}m{s}s").to(u.hourangle)


# Keep the planning model and command-line defaults tied to the launch settings.
DEFAULT_MLTAN_HOURS = float(
    ltan_string_to_quantity(DEFAULT_LAUNCH["ltan"]).to_value(u.hourangle)
)


def raan_for_launch(launch_name: str) -> Longitude:
    """RAAN in GCRS for a named launch from LAUNCHES, wrapped to [0°, 360°)."""
    from poliastro.earth.util import raan_from_ltan

    if launch_name not in LAUNCHES:
        raise KeyError(f"Unknown launch {launch_name!r}; known: {list(LAUNCHES)}")
    meta = LAUNCHES[launch_name]
    epoch = Time(meta["epoch"], format="isot", scale="utc")
    ltan = ltan_string_to_quantity(meta["ltan"])
    raan = raan_from_ltan(epoch, ltan)
    return Longitude(raan).wrap_at(360 * u.deg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", choices=LAUNCHES, help="Select a single launch.")
    args = parser.parse_args()

    with (iers.conf.set_temp("auto_download", False),
          iers.conf.set_temp("iers_degraded_accuracy", "warn")):
        for name in (args.launch,) if args.launch else LAUNCHES:
            epoch = Time(LAUNCHES[name]["epoch"], format="isot", scale="utc")
            raan_deg = raan_for_launch(name)
            print(f"{name}:")
            print(f"  epoch: {epoch.isot} ({epoch.scale})")
            print(f"  LTAN:  {LAUNCHES[name]['ltan']}")
            print(f"  RAAN:  {raan_deg.to(u.deg):.4f}")


if __name__ == "__main__":
    main()

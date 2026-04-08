"""
Compute RAAN from LTAN for Sun-synchronous orbits using poliastro.

Launch metadata is keyed by a short launch name; each entry has an ISO8601
epoch and an LTAN time string (HH:MM:SS).
"""

from __future__ import annotations

import astropy.units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time
from poliastro.earth.util import raan_from_ltan

# Launch name -> { "epoch": ISO8601 string, "ltan": "HH:MM:SS" }
LAUNCHES = {
    "twilight": {
        "epoch": "2026-01-11T13:44:50Z",
        "ltan": "18:00:00",
    },
    "transporter_17": {
        "epoch": "2026-06-15T12:00:00Z",
        "ltan": "11:00:00",
    },
    "ussf_178": {
        "epoch": "2026-12-15T18:00:00Z",
        "ltan": "18:00:00",
    },
}


def ltan_string_to_quantity(ltan: str) -> u.Quantity:
    """Parse 'HH:MM:SS' into an astropy Quantity in hourangle (poliastro expects this)."""
    parts = ltan.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"LTAN must be 'HH:MM:SS', got {ltan!r}")
    h, m, s = (int(parts[0]), int(parts[1]), float(parts[2]))
    return Angle(f"{h}h{m}m{s}s").to(u.hourangle)


def raan_for_launch(launch_name: str) -> Longitude:
    """RAAN in GCRS for a named launch from LAUNCHES, wrapped to [0°, 360°)."""
    if launch_name not in LAUNCHES:
        raise KeyError(f"Unknown launch {launch_name!r}; known: {list(LAUNCHES)}")
    meta = LAUNCHES[launch_name]
    epoch = Time(meta["epoch"], format="isot", scale="utc")
    ltan = ltan_string_to_quantity(meta["ltan"])
    raan = raan_from_ltan(epoch, ltan)
    return Longitude(raan).wrap_at(360 * u.deg)


def main() -> None:
    for name in LAUNCHES:
        epoch = Time(LAUNCHES[name]["epoch"], format="isot", scale="utc")
        raan_deg = raan_for_launch(name)
        print(f"{name}:")
        print(f"  epoch: {epoch.isot} ({epoch.scale})")
        print(f"  LTAN:  {LAUNCHES[name]['ltan']}")
        print(f"  RAAN:  {raan_deg.to(u.deg):.4f}")


if __name__ == "__main__":
    main()

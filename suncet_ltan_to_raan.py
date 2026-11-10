"""
Compute RAAN from LTAN for Sun-synchronous orbits using poliastro.

Launch metadata is keyed by a short launch name. The default mission entry
also holds the orbit targets used by the planning tools.
"""

from __future__ import annotations

import astropy.units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time

DEFAULT_LAUNCH_NAME = "ussf_178"
USSF_178_NET_DATE = "2027-03-15"

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
        "launch_date_net": USSF_178_NET_DATE,
        # Midnight is a planning placeholder, not injection time.
        "epoch": f"{USSF_178_NET_DATE}T00:00:00Z",
        "altitude_km": 510.0,
        "eccentricity": 0.0008,
        "inclination_deg": 97.44,
        "ltan": "18:00:00",
        "mltan_drift_target_min_per_year": 0.0,
        "mltan_drift_tolerance_min_per_year": 10.0,
    },
}
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
    for name in LAUNCHES:
        epoch = Time(LAUNCHES[name]["epoch"], format="isot", scale="utc")
        raan_deg = raan_for_launch(name)
        print(f"{name}:")
        print(f"  epoch: {epoch.isot} ({epoch.scale})")
        print(f"  LTAN:  {LAUNCHES[name]['ltan']}")
        print(f"  RAAN:  {raan_deg.to(u.deg):.4f}")


if __name__ == "__main__":
    main()

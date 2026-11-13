#!/usr/bin/env python3
"""Generate a synthetic SunCET TLE using the shared orbit configuration."""

from __future__ import annotations

import argparse
from math import atan2, cos, hypot, pi, radians, sin, sqrt
from pathlib import Path

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, GCRS, TEME
from astropy.time import Time
from astropy.utils import iers

from suncet_ltan_to_raan import DEFAULT_LAUNCH_NAME, LAUNCHES, raan_for_launch


def synthetic_tle(launch_name: str) -> str:
    """Generate a planning TLE from approximate SGP4 mean elements."""
    from scipy.optimize import brentq
    from sgp4.api import Satrec, WGS72
    from sgp4.earth_gravity import wgs72
    from sgp4.exporter import export_tle

    meta = LAUNCHES[launch_name]
    assumptions = meta.get("synthetic_tle")
    if assumptions is None:
        raise ValueError(f"No synthetic TLE settings for {launch_name}.")
    epoch = Time(meta["epoch"], format="isot", scale="utc")
    inclination = radians(meta["inclination_deg"])
    with (iers.conf.set_temp("auto_download", False),
          iers.conf.set_temp("iers_degraded_accuracy", "warn")):
        raan = raan_for_launch(launch_name).to_value(u.rad)
        # Transform the orbit normal before extracting TEME inclination and RAAN.
        normal = CartesianRepresentation(
            [sin(inclination) * sin(raan), -sin(inclination) * cos(raan), cos(inclination)] * u.km
        )
        x, y, z = GCRS(normal, obstime=epoch).transform_to(TEME(obstime=epoch)).cartesian.xyz.value
    inclination = atan2(hypot(x, y), z)
    node = atan2(x, -y) % (2 * pi)
    epoch_days = (epoch.jd1 - 2433281.5) + epoch.jd2  # UTC days since 1949-12-31
    semimajor_axis = wgs72.radiusearthkm + meta["altitude_km"]
    mean_motion = 60 * sqrt(wgs72.mu / semimajor_axis**3)
    satellite = Satrec()

    def radius_error(kozai_motion):
        # The TLE stores Kozai mean motion. Match the configured mean radius
        # after SGP4 applies its initialization correction.
        satellite.sgp4init(
            WGS72, "i", assumptions["catalog_number"], epoch_days, assumptions["bstar"],
            0.0, 0.0, meta["eccentricity"], radians(assumptions["argument_of_perigee_deg"] % 360),
            inclination, radians(assumptions["mean_anomaly_deg"] % 360), kozai_motion, node,
        )
        return satellite.a * wgs72.radiusearthkm - semimajor_axis

    radius_error(brentq(radius_error, .95 * mean_motion, 1.05 * mean_motion))
    if satellite.error:
        raise ValueError(f"SGP4 rejected the configured mean elements (error {satellite.error}).")
    line1, line2 = export_tle(satellite)
    return f"SunCET synthetic ({launch_name})\n{line1}\n{line2}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", choices=LAUNCHES, default=DEFAULT_LAUNCH_NAME,
                        help="Launch entry to use (default: %(default)s).")
    parser.add_argument("--output", type=Path, default=Path("suncet_synthetic.tle"),
                        help="Output TLE file (default: %(default)s).")
    args = parser.parse_args()
    try:
        contents = synthetic_tle(args.launch)
        args.output.write_text(contents, encoding="ascii")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

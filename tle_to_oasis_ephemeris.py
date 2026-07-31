#!/usr/bin/env python3
"""Generate an OASIS ephemeris-init script from the latest Space-Track TLE.

The generated state vector defaults to GCRS, which is the Astropy geocentric
celestial frame aligned with ICRS/J2000 axes. Position is written in km and
velocity in km/s to match the old IDL ephemeris script's J2000 ECI output.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence


OASIS_COMMAND = (
    "cmd_adcs_Refs_InitPosVelUtcGreg "
    "year $ephYear mon $ephMonth day $ephDay "
    "hour $ephHour min $ephMinute sec $ephSecond millisec 0 "
    "PosX $ephPosX PosY $ephPosY PosZ $ephPosZ "
    "VelX $ephVelX VelY $ephVelY VelZ $ephVelZ"
)


@dataclass(frozen=True)
class TLERecord:
    epoch: datetime
    line1: str
    line2: str


@dataclass(frozen=True)
class StateVector:
    epoch: datetime
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]
    frame: str


def tle_epoch_datetime(line1: str) -> datetime:
    """Parse the TLE epoch field from line 1 as a UTC datetime."""
    epoch_field = line1[18:32].strip()
    if not re.fullmatch(r"\d{2}\d{3}\.\d+", epoch_field):
        raise ValueError(f"Malformed TLE epoch in line 1: {line1!r}")

    yy = int(epoch_field[:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    day_of_year = float(epoch_field[2:])
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    return start + timedelta(days=day_of_year - 1.0)


def parse_tle_blocks(raw_text: str) -> list[TLERecord]:
    """Extract TLE line pairs from text, ignoring optional name/header lines."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    records: list[TLERecord] = []
    i = 0
    while i + 1 < len(lines):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            line1, line2 = lines[i], lines[i + 1]
            records.append(TLERecord(tle_epoch_datetime(line1), line1, line2))
            i += 2
            continue
        i += 1
    return sorted(records, key=lambda record: record.epoch)


def get_spacetrack_client():
    identity = os.environ.get("SPACETRACK_EMAIL")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        raise RuntimeError(
            "Missing Space-Track credentials. Set SPACETRACK_EMAIL and "
            "SPACETRACK_PASSWORD."
        )

    try:
        from spacetrack import SpaceTrackClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'spacetrack'. Install the repo environment from "
            "environment.yml before fetching from Space-Track."
        ) from exc

    return SpaceTrackClient(identity=identity, password=password)


def fetch_latest_tle(norad_id: int) -> TLERecord:
    client = get_spacetrack_client()
    text = client.gp(
        norad_cat_id=norad_id,
        orderby="epoch desc",
        limit=1,
        format="tle",
    )
    records = parse_tle_blocks(text)
    if not records:
        raise RuntimeError(f"No TLE returned by Space-Track for NORAD {norad_id}.")
    return records[-1]


def read_tle_file(path: Path) -> TLERecord:
    records = parse_tle_blocks(path.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError(f"No TLE line pair found in {path}.")
    return records[-1]


def parse_utc_datetime(value: str) -> datetime:
    text = value.strip()
    if text.lower() == "now":
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def rounded_to_nearest_second(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    if dt.microsecond >= 500_000:
        dt = dt + timedelta(seconds=1)
    return dt.replace(microsecond=0)


def propagate_tle(
    tle: TLERecord,
    epoch: datetime,
    frame: str,
    allow_iers_download: bool,
) -> StateVector:
    try:
        from astropy import units as u
        from astropy.coordinates import (
            GCRS,
            TEME,
            CartesianDifferential,
            CartesianRepresentation,
        )
        from astropy.time import Time
        from astropy.utils.iers import conf as iers_conf
        from sgp4.api import Satrec
    except ImportError as exc:
        raise RuntimeError(
            "Missing orbit dependency. This script needs sgp4 and astropy; "
            "install/update the repo environment from environment.yml."
        ) from exc

    iers_conf.auto_max_age = None
    iers_conf.auto_download = bool(allow_iers_download)

    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    obstime = Time(epoch)
    err_code, position_km, velocity_km_s = sat.sgp4(obstime.jd1, obstime.jd2)
    if err_code != 0:
        raise RuntimeError(f"SGP4 propagation failed with code {err_code}.")

    if frame == "teme":
        return StateVector(
            epoch=epoch,
            position_km=tuple(float(value) for value in position_km),
            velocity_km_s=tuple(float(value) for value in velocity_km_s),
            frame="TEME",
        )

    cart = CartesianRepresentation(position_km * u.km).with_differentials(
        CartesianDifferential(velocity_km_s * u.km / u.s)
    )
    gcrs = TEME(cart, obstime=obstime).transform_to(GCRS(obstime=obstime))
    gcrs_cart = gcrs.cartesian
    gcrs_vel = gcrs_cart.differentials["s"]

    return StateVector(
        epoch=epoch,
        position_km=tuple(float(value) for value in gcrs_cart.xyz.to_value(u.km)),
        velocity_km_s=tuple(float(value) for value in gcrs_vel.d_xyz.to_value(u.km / u.s)),
        frame="GCRS/J2000",
    )


def format_oasis_dn(value: float, precision: int = 9) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return f"{number:.1f}dn"
    text = f"{number:.{precision}f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return f"{text}dn"


def oasis_proc_name(output_path: Path, norad_id: int, epoch: datetime) -> str:
    if output_path.name != "-":
        stem = output_path.stem
    else:
        stem = f"set_ephemeris_norad_{norad_id}_{epoch:%Y%m%d_%H%M%S}_oasis"
    clean = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    if not clean or clean[0].isdigit():
        clean = f"proc_{clean}"
    return clean


def build_oasis_script(
    *,
    proc_name: str,
    norad_id: int,
    tle: TLERecord,
    state: StateVector,
    command_prefix: str,
    full_year: bool,
) -> str:
    epoch = state.epoch
    eph_year = epoch.year if full_year else epoch.year - 2000
    declarations: Sequence[tuple[str, float, int]] = (
        ("ephYear", eph_year, 0),
        ("ephMonth", epoch.month, 0),
        ("ephDay", epoch.day, 0),
        ("ephHour", epoch.hour, 0),
        ("ephMinute", epoch.minute, 0),
        ("ephSecond", epoch.second, 0),
        ("ephPosX", state.position_km[0], 9),
        ("ephPosY", state.position_km[1], 9),
        ("ephPosZ", state.position_km[2], 9),
        ("ephVelX", state.velocity_km_s[0], 12),
        ("ephVelY", state.velocity_km_s[1], 12),
        ("ephVelZ", state.velocity_km_s[2], 12),
    )
    declare_lines = [
        f"declare variable ${name} = {format_oasis_dn(value, precision)}"
        for name, value, precision in declarations
    ]
    command = f"{command_prefix} {OASIS_COMMAND}".strip()

    return "\n".join(
        [
            f"proc {proc_name}",
            "",
            "; Autogenerated by tle_to_oasis_ephemeris.py",
            f"; NORAD ID: {norad_id}",
            f"; Latest TLE epoch UTC: {tle.epoch.isoformat()}",
            f"; State epoch UTC: {epoch.isoformat()}",
            f"; State frame: {state.frame}",
            "; Position units: km",
            "; Velocity units: km/s",
            f"; TLE line 1: {tle.line1}",
            f"; TLE line 2: {tle.line2}",
            "",
            *declare_lines,
            "",
            "wait",
            command,
            "wait 00:00:01",
            "",
            "endproc",
            "",
        ]
    )


def default_output_path(norad_id: int, epoch: datetime) -> Path:
    return Path(f"set_ephemeris_norad_{norad_id}_{epoch:%Y%m%d_%H%M%S}_oasis.prc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a latest Space-Track TLE and write an OASIS ADCS ephemeris init PRC."
    )
    parser.add_argument("--norad-id", type=int, required=True, help="NORAD catalog ID.")
    parser.add_argument(
        "--at",
        default=None,
        help=(
            "UTC state epoch to propagate to, such as 2026-07-06T18:30:00Z, "
            "or 'now'. Default: latest TLE epoch rounded to the nearest second."
        ),
    )
    parser.add_argument(
        "--tle-file",
        type=Path,
        default=None,
        help="Read a TLE text file instead of fetching from Space-Track.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .prc path. Use '-' for stdout. Default: set_ephemeris_norad_<id>_<utc>_oasis.prc.",
    )
    parser.add_argument(
        "--frame",
        choices=["gcrs", "teme"],
        default="gcrs",
        help="Output state frame. gcrs is J2000-like ECI; teme is raw SGP4 TEME.",
    )
    parser.add_argument(
        "--command-prefix",
        default="start",
        help="Prefix before the ADCS command. Default matches OASIS examples: 'start'.",
    )
    parser.add_argument(
        "--two-digit-year",
        action="store_true",
        help="Declare ephYear as YY instead of the full Gregorian year.",
    )
    parser.add_argument(
        "--allow-iers-download",
        action="store_true",
        help="Allow Astropy to download fresh IERS tables for the TEME-to-GCRS transform.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tle = read_tle_file(args.tle_file) if args.tle_file else fetch_latest_tle(args.norad_id)
    state_epoch = parse_utc_datetime(args.at) if args.at else tle.epoch
    state_epoch = rounded_to_nearest_second(state_epoch)
    state = propagate_tle(tle, state_epoch, args.frame, args.allow_iers_download)

    output_path = args.output or default_output_path(args.norad_id, state_epoch)
    script = build_oasis_script(
        proc_name=oasis_proc_name(output_path, args.norad_id, state_epoch),
        norad_id=args.norad_id,
        tle=tle,
        state=state,
        command_prefix=args.command_prefix,
        full_year=not args.two_digit_year,
    )

    if output_path == Path("-"):
        print(script, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script, encoding="utf-8")
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parse GMAT EarthShadow.txt umbra events: plot duration vs time and print statistics."""

from __future__ import annotations

import argparse
import calendar
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

START_TIME_FMT = "%d %b %Y %H:%M:%S.%f"

# Earth gravity (km^3/s^2) and equatorial radius (km) for circular LEO period.
EARTH_MU_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137
DEFAULT_ALTITUDE_KM = 510.0
SECONDS_PER_DAY = 86400.0


def circular_orbit_period_s(altitude_km: float) -> float:
    """Kepler period for a circular orbit (same altitude as GMAT semimajor axis if circular)."""
    a_km = EARTH_RADIUS_KM + altitude_km
    return float(2.0 * np.pi * np.sqrt(a_km**3 / EARTH_MU_KM3_S2))


@dataclass(frozen=True)
class UmbraEvent:
    start_utc: datetime
    total_duration_s: float


def mission_span_s(events: list[UmbraEvent]) -> float:
    first = min(e.start_utc for e in events)
    last_end = max(e.start_utc + timedelta(seconds=e.total_duration_s) for e in events)
    return (last_end - first).total_seconds()


def default_earthshadow_path() -> Path:
    base = os.getenv("suncet_data")
    if not base:
        print(
            "Error: set environment variable suncet_data to your data directory "
            "(e.g. …/9000 Processing/data/).",
            file=sys.stderr,
        )
        sys.exit(1)
    return (
        Path(base).resolve().parent.parent
        / "2000 Systems"
        / "Systems Analyses"
        / "GMAT Daniel 2026-04-17"
        / "EarthShadow.txt"
    )


def parse_earthshadow_txt(path: Path) -> list[UmbraEvent]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[UmbraEvent] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Spacecraft:") or "Start Time (UTC)" in line:
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) != 7:
            continue
        start_s, _stop_s, _dur_s, _body, _typ, _ev, total_s = parts
        try:
            start = datetime.strptime(start_s, START_TIME_FMT)
        except ValueError:
            continue
        try:
            total = float(total_s)
        except ValueError:
            continue
        rows.append(UmbraEvent(start_utc=start, total_duration_s=total))
    rows.sort(key=lambda r: r.start_utc)
    return rows


def ym_key(dt: datetime) -> int:
    return dt.year * 12 + dt.month


def month_ranges_with_eclipses(events: list[UmbraEvent]) -> list[tuple[int, int]]:
    """Return sorted (start_ym_key, end_ym_key) for each contiguous run of calendar months."""
    keys = sorted({ym_key(e.start_utc) for e in events})
    if not keys:
        return []
    ranges: list[tuple[int, int]] = []
    run_start = keys[0]
    prev = keys[0]
    for k in keys[1:]:
        if k == prev + 1:
            prev = k
        else:
            ranges.append((run_start, prev))
            run_start = k
            prev = k
    ranges.append((run_start, prev))
    return ranges


def format_ym_key(k: int) -> str:
    # k = year * 12 + month with month in 1..12
    y = (k - 1) // 12
    m = (k - 1) % 12 + 1
    abbr = calendar.month_abbr[m]
    return f"{abbr} {y}"


def describe_seasons(events: list[UmbraEvent]) -> list[str]:
    lines: list[str] = []
    for a, b in month_ranges_with_eclipses(events):
        if a == b:
            lines.append(format_ym_key(a))
        else:
            lines.append(f"{format_ym_key(a)} – {format_ym_key(b)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot umbra total duration vs time from GMAT EarthShadow.txt and print stats."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="EarthShadow.txt path (default: from suncet_data relative path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: earthshadow_eclipse_duration.png next to the data file)",
    )
    parser.add_argument(
        "--altitude-km",
        type=float,
        default=DEFAULT_ALTITUDE_KM,
        help="Circular altitude for orbit count (default: %(default)s km)",
    )
    args = parser.parse_args()
    path = args.input if args.input is not None else default_earthshadow_path()
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    out_path = args.output if args.output is not None else path.parent / "earthshadow_eclipse_duration.png"

    events = parse_earthshadow_txt(path)
    if not events:
        print("Error: no umbra rows parsed.", file=sys.stderr)
        sys.exit(1)

    times = [e.start_utc for e in events]
    minutes = np.array([e.total_duration_s / 60.0 for e in events], dtype=float)

    median_m = float(np.median(minutes))
    mean_m = float(np.mean(minutes))
    std_m = float(np.std(minutes, ddof=0))
    max_m = float(np.max(minutes))

    period_s = circular_orbit_period_s(args.altitude_km)
    orbits_per_day = SECONDS_PER_DAY / period_s
    span_s = mission_span_s(events)
    total_orbits = span_s / period_s
    n_gt_3 = int(np.sum(minutes > 3.0))
    n_gt_15 = int(np.sum(minutes > 15.0))
    share_orbits_gt_3 = 100.0 * n_gt_3 / total_orbits if total_orbits > 0 else 0.0
    share_orbits_gt_15 = 100.0 * n_gt_15 / total_orbits if total_orbits > 0 else 0.0

    print(f"Source: {path}")
    print(f"Eclipse count: {len(events)}")
    print()
    print("Eclipse seasons (contiguous calendar months with ≥1 eclipse):")
    for s in describe_seasons(events):
        print(f"  {s}")
    print()
    print(
        f"Orbit model: circular LEO at {args.altitude_km:g} km; "
        f"period {period_s / 60.0:.2f} min; {orbits_per_day:.3f} orbits/day"
    )
    print(
        f"Mission span (first eclipse start → last eclipse end): "
        f"{span_s / SECONDS_PER_DAY:.4f} days → {total_orbits:.1f} total orbits"
    )
    print()
    print(f"Median duration: {median_m:.4f} min")
    print(f"Mean duration:   {mean_m:.4f} min")
    print(f"Std deviation:   {std_m:.4f} min")
    print(f"Max duration:    {max_m:.4f} min")
    print(
        f"Share of all orbits with eclipse > 3 min:   {share_orbits_gt_3:.2f}% "
        f"({n_gt_3} eclipses / {total_orbits:.1f} orbits)"
    )
    print(
        f"Share of all orbits with eclipse > 15 min:  {share_orbits_gt_15:.2f}% "
        f"({n_gt_15} eclipses / {total_orbits:.1f} orbits)"
    )

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.plot(times, minutes, ".", markersize=3, color="0.2")
    ax.set_ylabel("eclipse duration [minutes]")
    ax.set_title("SunCET-1 USSF-178 launch")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150)
    print()
    print(f"Wrote plot: {out_path.resolve()}")


if __name__ == "__main__":
    main()

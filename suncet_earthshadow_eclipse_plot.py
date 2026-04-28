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


def eclipse_time_in_window_s(events: list[UmbraEvent], window_start: datetime, window_end: datetime) -> float:
    """Total eclipse seconds inside [window_start, window_end), clipping partial overlaps."""
    total_s = 0.0
    for e in events:
        ev_start = e.start_utc
        ev_end = e.start_utc + timedelta(seconds=e.total_duration_s)
        overlap_start = max(ev_start, window_start)
        overlap_end = min(ev_end, window_end)
        if overlap_end > overlap_start:
            total_s += (overlap_end - overlap_start).total_seconds()
    return total_s


def per_orbit_eclipse_seconds(
    events: list[UmbraEvent], window_start: datetime, window_end: datetime, period_s: float
) -> np.ndarray:
    """Eclipse seconds for each orbit in a fixed window; zero means no eclipse that orbit."""
    n_orbits = max(1, int(np.ceil((window_end - window_start).total_seconds() / period_s)))
    eclipse_s = np.zeros(n_orbits, dtype=float)

    for e in events:
        ev_start = e.start_utc
        ev_end = e.start_utc + timedelta(seconds=e.total_duration_s)
        overlap_start = max(ev_start, window_start)
        overlap_end = min(ev_end, window_end)
        if overlap_end <= overlap_start:
            continue

        # Assign clipped event duration to the orbit containing its overlap start.
        orbit_idx = int((overlap_start - window_start).total_seconds() // period_s)
        orbit_idx = min(max(orbit_idx, 0), n_orbits - 1)
        eclipse_s[orbit_idx] += (overlap_end - overlap_start).total_seconds()

    return np.clip(eclipse_s, 0.0, period_s)


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
    duty_out_path = out_path.with_name(f"{out_path.stem}_science_duty_cycle{out_path.suffix}")

    events = parse_earthshadow_txt(path)
    if not events:
        print("Error: no umbra rows parsed.", file=sys.stderr)
        sys.exit(1)

    times = [e.start_utc for e in events]
    minutes = np.array([e.total_duration_s / 60.0 for e in events], dtype=float)

    period_s = circular_orbit_period_s(args.altitude_km)
    duty_cycle = np.clip(1.0 - (minutes * 60.0) / period_s, 0.0, 1.0)
    duty_cycle_pct = 100.0 * duty_cycle

    median_m = float(np.median(minutes))
    mean_m = float(np.mean(minutes))
    std_m = float(np.std(minutes, ddof=0))
    max_m = float(np.max(minutes))
    orbits_per_day = SECONDS_PER_DAY / period_s
    span_s = mission_span_s(events)
    total_orbits = span_s / period_s
    n_gt_3 = int(np.sum(minutes > 3.0))
    n_gt_15 = int(np.sum(minutes > 15.0))
    share_orbits_gt_3 = 100.0 * n_gt_3 / total_orbits if total_orbits > 0 else 0.0
    share_orbits_gt_15 = 100.0 * n_gt_15 / total_orbits if total_orbits > 0 else 0.0

    report_start = min(e.start_utc for e in events)
    report_end = report_start + timedelta(days=365.0)
    report_seconds = (report_end - report_start).total_seconds()
    eclipse_seconds_report = eclipse_time_in_window_s(events, report_start, report_end)
    no_eclipse_pct_report = (
        100.0 * (1.0 - eclipse_seconds_report / report_seconds) if report_seconds > 0 else 0.0
    )
    orbit_eclipse_s = per_orbit_eclipse_seconds(events, report_start, report_end, period_s)
    orbit_duty_cycle_pct = 100.0 * (1.0 - orbit_eclipse_s / period_s)
    median_dc_pct = float(np.median(orbit_duty_cycle_pct))
    mean_dc_pct = float(np.mean(orbit_duty_cycle_pct))

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
    print(f"Median science duty cycle (insolated/orbit): {median_dc_pct:.0f}%")
    print(f"Mean science duty cycle (insolated/orbit):   {mean_dc_pct:.0f}%")
    print(
        f"Share of all orbits with eclipse > 3 min:   {share_orbits_gt_3:.2f}% "
        f"({n_gt_3} eclipses / {total_orbits:.1f} orbits)"
    )
    print(
        f"Share of all orbits with eclipse > 15 min:  {share_orbits_gt_15:.2f}% "
        f"({n_gt_15} eclipses / {total_orbits:.1f} orbits)"
    )
    print(
        "No-eclipse time over 1-year window from first event "
        f"({report_start} to {report_end}): {no_eclipse_pct_report:.2f}%"
    )

    fig1, ax1 = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax1.plot(times, minutes, ".", markersize=3, color="0.2")
    ax1.set_ylabel("eclipse duration [minutes]")
    ax1.set_xlabel("UTC date")
    ax1.set_title("SunCET-1 USSF-178 launch")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax1.set_xlim(report_start, report_end)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax1.grid(True, alpha=0.3)
    fig1.savefig(out_path, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax2.plot(times, duty_cycle_pct, ".", markersize=3, color="tab:blue")
    ax2.set_ylabel("science duty cycle [%]")
    ax2.set_xlabel("UTC date")
    ax2.set_ylim(0.0, 100.0)
    ax2.set_title("SunCET-1 science duty cycle")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax2.set_xlim(report_start, report_end)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.02,
        0.06,
        f"Mean: {mean_dc_pct:.0f}%\nMedian: {median_dc_pct:.0f}%",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    fig2.savefig(duty_out_path, dpi=150)
    plt.close(fig2)
    print()
    print(f"Wrote eclipse-duration plot: {out_path.resolve()}")
    print(f"Wrote science-duty-cycle plot: {duty_out_path.resolve()}")


if __name__ == "__main__":
    main()

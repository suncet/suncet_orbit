#!/usr/bin/env python3
"""Track current and historical orbital altitude from Space-Track TLE data."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from spacetrack import SpaceTrackClient, operators as op
from sgp4.api import Satrec, jday

EARTH_RADIUS_KM = 6378.137
DEORBIT_THRESHOLD_KM = 120.0
CACHE_DIR = Path(".cache")


@dataclass(frozen=True)
class TLERecord:
    epoch: datetime
    line1: str
    line2: str

    @property
    def mean_motion_rev_per_day(self) -> float:
        parts = self.line2.split()
        if len(parts) < 8:
            raise ValueError(f"Malformed TLE line 2: {self.line2}")
        return float(parts[7])

    @property
    def epoch_altitude_km(self) -> float:
        # Estimate altitude from semi-major axis derived from mean motion.
        mu_earth_km3_s2 = 398600.4418
        n_rad_s = self.mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
        semi_major_axis_km = (mu_earth_km3_s2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
        return semi_major_axis_km - EARTH_RADIUS_KM


@dataclass(frozen=True)
class ProjectionResult:
    note: str
    projected_date: datetime | None
    fit_epochs: np.ndarray | None = None
    fit_altitudes_km: np.ndarray | None = None


def tle_epoch_datetime(sat: Satrec) -> datetime:
    # Convert SGP4 Julian date epoch to UTC datetime.
    from sgp4.conveniences import sat_epoch_datetime

    return sat_epoch_datetime(sat).astimezone(timezone.utc)


def parse_tle_blocks_safe(raw_text: str) -> list[TLERecord]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    records: list[TLERecord] = []
    i = 0
    while i + 1 < len(lines):
        if not lines[i].startswith("1 ") or not lines[i + 1].startswith("2 "):
            i += 1
            continue
        line1, line2 = lines[i], lines[i + 1]
        sat = Satrec.twoline2rv(line1, line2)
        records.append(TLERecord(epoch=tle_epoch_datetime(sat), line1=line1, line2=line2))
        i += 2
    return sorted(records, key=lambda r: r.epoch)


def get_spacetrack_client() -> SpaceTrackClient:
    identity = os.environ.get("SPACETRACK_EMAIL")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        raise RuntimeError(
            "Missing Space-Track credentials. Set SPACETRACK_EMAIL and SPACETRACK_PASSWORD."
        )
    return SpaceTrackClient(identity=identity, password=password)


def fetch_latest_tle(client: SpaceTrackClient, catnr: int) -> TLERecord:
    text = client.gp(
        norad_cat_id=catnr,
        orderby="epoch desc",
        limit=1,
        format="tle",
    )
    records = parse_tle_blocks_safe(text)
    if not records:
        raise RuntimeError(f"No latest TLE returned by Space-Track for NORAD {catnr}")
    return records[-1]


def load_cache(cache_path: Path) -> list[TLERecord]:
    if not cache_path.exists():
        return []
    data = json.loads(cache_path.read_text())
    records: list[TLERecord] = []
    for row in data:
        records.append(
            TLERecord(
                epoch=datetime.fromisoformat(row["epoch"]).astimezone(timezone.utc),
                line1=row["line1"],
                line2=row["line2"],
            )
        )
    return sorted(records, key=lambda r: r.epoch)


def save_cache(cache_path: Path, records: Iterable[TLERecord]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"epoch": r.epoch.isoformat(), "line1": r.line1, "line2": r.line2}
        for r in sorted(records, key=lambda x: x.epoch)
    ]
    cache_path.write_text(json.dumps(payload, indent=2))


def merge_records(*sets_of_records: Iterable[TLERecord]) -> list[TLERecord]:
    by_epoch: dict[str, TLERecord] = {}
    for records in sets_of_records:
        for record in records:
            key = record.epoch.isoformat()
            by_epoch[key] = record
    return sorted(by_epoch.values(), key=lambda r: r.epoch)


def parse_history_start_utc(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def downsample_tle_records(records: list[TLERecord], max_points: int) -> list[TLERecord]:
    if max_points <= 0 or len(records) <= max_points:
        return records
    if len(records) <= 2:
        return records
    idx = np.linspace(0, len(records) - 1, max_points)
    idx = np.unique(np.round(idx).astype(int))
    return [records[i] for i in idx]


def fetch_archive_tles(
    client: SpaceTrackClient,
    catnr: int,
    full_history: bool,
    history_start: datetime | None,
    history_days: int,
) -> list[TLERecord]:
    kwargs: dict[str, object] = {
        "norad_cat_id": catnr,
        "format": "tle",
    }
    if not full_history:
        end = datetime.now(timezone.utc)
        if history_start is not None:
            start = history_start
        else:
            start = end - timedelta(days=max(1, history_days))
        kwargs["epoch"] = op.inclusive_range(start, end)

    text = client.gp_history(**kwargs)
    return parse_tle_blocks_safe(text)


def fetch_spacetrack_decay_prediction(
    client: SpaceTrackClient, catnr: int
) -> tuple[datetime | None, str | None]:
    raw_json = client.decay(
        norad_cat_id=catnr,
        orderby="decay_epoch desc",
        limit=1,
        format="json",
    )
    rows = json.loads(raw_json)
    if not rows:
        return (None, None)
    row = rows[0]
    decay_text = row.get("DECAY_EPOCH")
    if not decay_text:
        return (None, row.get("MSG_TYPE"))
    decay_dt = datetime.strptime(decay_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return (decay_dt, row.get("MSG_TYPE"))


def altitude_now_km(tle: TLERecord) -> float:
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    now = datetime.now(tz=timezone.utc)
    jd, fr = jday(
        now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond / 1e6
    )
    err_code, position_km, _velocity_km_s = sat.sgp4(jd, fr)
    if err_code != 0:
        raise RuntimeError(f"SGP4 propagation failed with code {err_code}")
    radius_km = math.sqrt(sum(component * component for component in position_km))
    return radius_km - EARTH_RADIUS_KM


def estimate_deorbit_date(
    epochs: np.ndarray,
    altitudes_km: np.ndarray,
    threshold_km: float = DEORBIT_THRESHOLD_KM,
    model: str = "weighted",
    window_days: float = 45.0,
) -> ProjectionResult:
    if len(epochs) < 3:
        return ProjectionResult("Insufficient history for projection (need >=3 points).", None)

    x_days = np.array([(ep - epochs[0]).total_seconds() / 86400.0 for ep in epochs], dtype=float)
    y = np.array(altitudes_km, dtype=float)

    if np.nanmin(y) <= threshold_km:
        return ProjectionResult(
            "Object already below threshold in historical data.",
            epochs[int(np.nanargmin(y))],
        )

    chosen_model = model
    if chosen_model in {"quadratic", "auto"} and len(x_days) >= 7:
        coeffs = np.polyfit(x_days, y, deg=2)
        poly_model = np.poly1d(coeffs)
        roots = np.roots(poly_model - threshold_km)
        real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-6 and r.real >= x_days[-1]]
        if real_roots:
            hit_day = min(real_roots)
            est = epochs[0] + timedelta(seconds=float(hit_day * 86400.0))
            fit_x = np.linspace(x_days[0], hit_day, 300)
            fit_y = poly_model(fit_x)
            fit_epochs = np.array(
                [epochs[0] + timedelta(days=float(day)) for day in fit_x], dtype=object
            )
            return ProjectionResult(
                "Quadratic decay projection.",
                est,
                fit_epochs=fit_epochs,
                fit_altitudes_km=np.array(fit_y, dtype=float),
            )

    if chosen_model in {"powerlaw", "auto"}:
        # Finite-time accelerating decay:
        # h(t) = threshold + A * (t_c - t)^p, with 0 < p < 1 and t <= t_c.
        # As t -> t_c, slope magnitude grows (accelerating descent toward threshold).
        cutoff = x_days[-1] - max(float(window_days), 1.0)
        mask = x_days >= cutoff
        x_fit = x_days[mask]
        y_fit = y[mask]
        if len(x_fit) >= 3:
            age = x_fit[-1] - x_fit
            half_life_days = max(window_days / 2.0, 1.0)
            sample_w = 0.5 ** (age / half_life_days)
            sigma = 1.0 / np.sqrt(sample_w + 1e-12)
            x0 = float(x_fit[0])
            y0 = float(y_fit[0])

            def power_law_model(x: np.ndarray, a: float, p: float, t_c: float) -> np.ndarray:
                return threshold_km + a * np.power(np.maximum(t_c - x, 1e-9), p)

            t_c0 = float(x_fit[-1] + max(7.0, window_days * 0.75))
            a0 = max(y0 - threshold_km, 1.0) / np.power(max(t_c0 - x0, 1e-6), 0.5)
            p0 = 0.5

            try:
                (a_fit, p_fit, t_c_fit), _ = curve_fit(
                    power_law_model,
                    x_fit,
                    y_fit,
                    p0=(a0, p0, t_c0),
                    bounds=(
                        [1e-8, 0.05, x_fit[-1] + 1e-4],
                        [1e5, 0.99, x_fit[-1] + 3650.0],
                    ),
                    sigma=sigma,
                    absolute_sigma=False,
                    maxfev=30000,
                )
            except (RuntimeError, ValueError):
                a_fit, p_fit, t_c_fit = None, None, None

            if (
                a_fit is not None
                and p_fit is not None
                and t_c_fit is not None
                and a_fit > 0
                and 0 < p_fit < 1
                and t_c_fit >= x_days[-1]
            ):
                hit_day = float(t_c_fit)
                est = epochs[0] + timedelta(seconds=float(hit_day * 86400.0))
                fit_x = np.linspace(x_fit[0], hit_day, 300)
                fit_y = power_law_model(fit_x, a_fit, p_fit, t_c_fit)
                fit_epochs = np.array(
                    [epochs[0] + timedelta(days=float(day)) for day in fit_x],
                    dtype=object,
                )
                return ProjectionResult(
                    f"Power-law accelerating projection (p={p_fit:.3f}) over last {window_days:.0f} days.",
                    est,
                    fit_epochs=fit_epochs,
                    fit_altitudes_km=np.array(fit_y, dtype=float),
                )
        if chosen_model == "powerlaw":
            return ProjectionResult(
                "Power-law model unavailable (fit failed or non-accelerating exponent).",
                None,
            )

    if chosen_model in {"weighted", "auto"}:
        cutoff = x_days[-1] - max(float(window_days), 1.0)
        mask = x_days >= cutoff
        x_fit = x_days[mask]
        y_fit = y[mask]
        if len(x_fit) >= 3:
            # Exponential recency weighting: newest points dominate the trend.
            age = x_fit[-1] - x_fit
            half_life_days = max(window_days / 2.0, 1.0)
            weights = 0.5 ** (age / half_life_days)
            slope, intercept = np.polyfit(x_fit, y_fit, deg=1, w=weights)
            if slope < 0:
                hit_day = (threshold_km - intercept) / slope
                if hit_day >= x_days[-1]:
                    est = epochs[0] + timedelta(seconds=float(hit_day * 86400.0))
                    fit_x = np.linspace(x_fit[0], hit_day, 300)
                    fit_y = slope * fit_x + intercept
                    fit_epochs = np.array(
                        [epochs[0] + timedelta(days=float(day)) for day in fit_x], dtype=object
                    )
                    return ProjectionResult(
                        f"Weighted linear projection over last {window_days:.0f} days.",
                        est,
                        fit_epochs=fit_epochs,
                        fit_altitudes_km=np.array(fit_y, dtype=float),
                    )
        if chosen_model == "weighted":
            return ProjectionResult(
                f"Weighted model unavailable (need >=3 points in the last {window_days:.0f} days).",
                None,
            )

    slope, intercept = np.polyfit(x_days, y, deg=1)
    if slope >= 0:
        return ProjectionResult("Trend is non-decaying; cannot project de-orbit.", None)
    hit_day = (threshold_km - intercept) / slope
    if hit_day < x_days[-1]:
        return ProjectionResult("Linear model crosses threshold in the past; projection unstable.", None)
    est = epochs[0] + timedelta(seconds=float(hit_day * 86400.0))
    fit_x = np.linspace(x_days[0], hit_day, 300)
    fit_y = slope * fit_x + intercept
    fit_epochs = np.array([epochs[0] + timedelta(days=float(day)) for day in fit_x], dtype=object)
    if chosen_model == "linear":
        return ProjectionResult(
            "Linear projection.",
            est,
            fit_epochs=fit_epochs,
            fit_altitudes_km=np.array(fit_y, dtype=float),
        )
    return ProjectionResult(
        "Linear fallback projection.",
        est,
        fit_epochs=fit_epochs,
        fit_altitudes_km=np.array(fit_y, dtype=float),
    )


def plot_altitude_history(
    epochs: np.ndarray,
    altitudes_km: np.ndarray,
    out_png: Path,
    catnr: int,
    threshold_km: float,
    fit_epochs: np.ndarray | None = None,
    fit_altitudes_km: np.ndarray | None = None,
    predicted_deorbit_utc: datetime | None = None,
    spacetrack_decay_utc: datetime | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, altitudes_km, marker="o", linestyle="-", linewidth=1.2, markersize=3, label="TLE-derived altitude")
    if fit_epochs is not None and fit_altitudes_km is not None:
        ax.plot(
            fit_epochs,
            fit_altitudes_km,
            color="purple",
            linestyle="-.",
            linewidth=1.6,
            label="Model fit + projection",
        )
    ax.axhline(threshold_km, color="red", linestyle="--", linewidth=1, label=f"De-orbit threshold ({threshold_km:.0f} km)")
    ymin_data = float(np.nanmin(altitudes_km)) if len(altitudes_km) else threshold_km
    ymax_data = float(np.nanmax(altitudes_km)) if len(altitudes_km) else threshold_km
    y_span = max(ymax_data - ymin_data, 1.0)
    text_pad_km = max(0.02 * y_span, 2.0)
    box_h_km = max(0.015 * y_span, 5.0)

    if predicted_deorbit_utc is not None or spacetrack_decay_utc is not None:
        left, right = ax.get_xlim()
        for dt in (predicted_deorbit_utc, spacetrack_decay_utc):
            if dt is not None:
                dn = mdates.date2num(dt)
                left = min(left, dn)
                right = max(right, dn)
        span = right - left
        ax.set_xlim(left - 0.03 * span, right + 0.03 * span)

    left, right = ax.get_xlim()
    span_x = max(right - left, 1e-6)
    box_w_days = max(2.0, 0.012 * span_x)

    if predicted_deorbit_utc is not None:
        xn = mdates.date2num(predicted_deorbit_utc)
        rect = Rectangle(
            (xn - box_w_days / 2.0, threshold_km - box_h_km / 2.0),
            box_w_days,
            box_h_km,
            facecolor="#6a1b9a",
            edgecolor="#4a148c",
            linewidth=1.0,
            alpha=0.9,
            zorder=5,
            label="power law predicted deorbit",
        )
        ax.add_patch(rect)
        ax.text(
            xn,
            threshold_km + box_h_km / 2.0 + text_pad_km,
            f"power law predicted deorbit\n{predicted_deorbit_utc.strftime('%Y-%m-%d')}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#6a1b9a",
            zorder=6,
        )
    if spacetrack_decay_utc is not None:
        xn = mdates.date2num(spacetrack_decay_utc)
        rect = Rectangle(
            (xn - box_w_days / 2.0, threshold_km - box_h_km / 2.0),
            box_w_days,
            box_h_km,
            facecolor="#bf6516",
            edgecolor="#8d4008",
            linewidth=1.0,
            alpha=0.9,
            zorder=5,
            label="space-track predicted deorbit",
        )
        ax.add_patch(rect)
        ax.text(
            xn,
            threshold_km - box_h_km / 2.0 - text_pad_km,
            f"space-track predicted deorbit\n{spacetrack_decay_utc.strftime('%Y-%m-%d')}",
            ha="center",
            va="top",
            fontsize=8,
            color="#bf6516",
            zorder=6,
        )
    if predicted_deorbit_utc is not None or spacetrack_decay_utc is not None:
        yb, yt = ax.get_ylim()
        margin = max(0.04 * y_span, 3.0)
        need_bottom = threshold_km - box_h_km / 2.0 - text_pad_km - margin
        ax.set_ylim(min(yb, need_bottom), yt)
    ax.set_title(f"NORAD {catnr} Altitude History from TLE Epochs")
    ax.set_xlabel("Epoch (UTC)")
    ax.set_ylabel("Estimated Altitude (km)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch TLEs from Space-Track, report altitude history, and estimate de-orbit date."
    )
    parser.add_argument("--norad-id", type=int, default=51657, help="NORAD CATNR to track (default: 51657)")
    parser.add_argument(
        "--plot-file",
        type=Path,
        default=Path("altitude_history.png"),
        help="Output PNG path for altitude history plot.",
    )
    parser.add_argument(
        "--deorbit-threshold-km",
        type=float,
        default=DEORBIT_THRESHOLD_KM,
        help="Altitude threshold for de-orbit estimate.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Use cached records only and skip all network calls.",
    )
    parser.add_argument(
        "--archive-file",
        type=Path,
        default=None,
        help="Optional local TLE text file to merge as additional historical archive.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=365,
        help="Days of lookback from now when --history-start is not set (ignored by --full-history).",
    )
    parser.add_argument(
        "--history-start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="UTC start date (inclusive) for gp_history; end is now. Overrides --history-days. "
        "If omitted and --norad-id is 51657, defaults to 2022-02-15.",
    )
    parser.add_argument(
        "--max-history-points",
        type=int,
        default=2000,
        help="After fetching, keep at most this many TLEs (evenly spaced in time). 0 = no limit.",
    )
    parser.add_argument(
        "--use-history-days",
        action="store_true",
        help="Use --history-days lookback from now instead of the NORAD 51657 default --history-start.",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Request full available history from Space-Track for the object.",
    )
    parser.add_argument(
        "--skip-archive-fetch",
        action="store_true",
        help="Skip gp_history (no bulk TLE download). Uses local cache for history; still fetches latest TLE + DECAY and updates plot.",
    )
    parser.add_argument(
        "--model",
        choices=["weighted", "powerlaw", "quadratic", "linear", "auto"],
        default="weighted",
        help="Projection model. weighted/powerlaw use a rolling window; powerlaw enforces accelerating finite-time descent.",
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=45.0,
        help="Rolling window size (days) for weighted and powerlaw projections.",
    )
    return parser


def resolve_history_start(args: argparse.Namespace) -> datetime | None:
    if args.use_history_days:
        return None
    if args.history_start is not None:
        return parse_history_start_utc(args.history_start)
    if args.norad_id == 51657:
        return datetime(2022, 2, 15, tzinfo=timezone.utc)
    return None


def main() -> None:
    args = build_parser().parse_args()
    history_start = resolve_history_start(args)
    cache_path = CACHE_DIR / f"tle_archive_{args.norad_id}.json"

    cached_records = load_cache(cache_path)
    live_records: list[TLERecord] = []
    archive_records: list[TLERecord] = []
    file_archive_records: list[TLERecord] = []
    latest_record: TLERecord | None = None
    if args.archive_file is not None and args.archive_file.exists():
        file_archive_records = parse_tle_blocks_safe(args.archive_file.read_text())


    if not args.offline_only:
        if args.skip_archive_fetch and args.full_history:
            print("[warn] --skip-archive-fetch ignores --full-history.")
        client = get_spacetrack_client()
        try:
            latest_record = fetch_latest_tle(client, args.norad_id)
            live_records = [latest_record]
        except Exception as err:
            print(f"[warn] Space-Track latest TLE fetch failed: {err}")

        if not args.skip_archive_fetch:
            try:
                archive_records = fetch_archive_tles(
                    client,
                    args.norad_id,
                    full_history=args.full_history,
                    history_start=None if args.full_history else history_start,
                    history_days=args.history_days,
                )
                archive_records = downsample_tle_records(archive_records, args.max_history_points)
            except Exception as err:
                print(f"[warn] Space-Track archive fetch failed: {err}")
        else:
            print(
                "[info] Skipping gp_history archive fetch (--skip-archive-fetch); "
                "history comes from cache / --archive-file only."
            )

    merged = merge_records(cached_records, archive_records, file_archive_records, live_records)
    if not merged:
        raise RuntimeError(
            "No TLE data available. Run once online or populate the local cache first."
        )

    save_cache(cache_path, merged)
    latest = latest_record or merged[-1]

    try:
        current_alt_km = altitude_now_km(latest)
        current_alt_note = "from SGP4 propagation at current UTC"
    except Exception as err:
        current_alt_km = latest.epoch_altitude_km
        current_alt_note = f"fallback from latest TLE epoch mean motion ({err})"

    epochs = np.array([r.epoch for r in merged], dtype=object)
    altitudes = np.array([r.epoch_altitude_km for r in merged], dtype=float)

    projection = estimate_deorbit_date(
        epochs,
        altitudes,
        threshold_km=args.deorbit_threshold_km,
        model=args.model,
        window_days=args.window_days,
    )
    spacetrack_decay_date: datetime | None = None
    spacetrack_decay_type: str | None = None
    if not args.offline_only:
        try:
            spacetrack_decay_date, spacetrack_decay_type = fetch_spacetrack_decay_prediction(
                client, args.norad_id
            )
        except Exception as err:
            print(f"[warn] Space-Track decay fetch failed: {err}")

    plot_altitude_history(
        epochs,
        altitudes,
        args.plot_file,
        args.norad_id,
        args.deorbit_threshold_km,
        fit_epochs=projection.fit_epochs,
        fit_altitudes_km=projection.fit_altitudes_km,
        predicted_deorbit_utc=projection.projected_date,
        spacetrack_decay_utc=spacetrack_decay_date,
    )

    print(f"NORAD ID: {args.norad_id}")
    if args.skip_archive_fetch and not args.offline_only:
        print("Space-Track mode: quick update (latest TLE + DECAY only; gp_history skipped)")
    if args.full_history and not args.skip_archive_fetch:
        print("History window (Space-Track gp_history): full catalog history")
    elif args.skip_archive_fetch and not args.offline_only:
        print("History window: from local cache / --archive-file (gp_history not requested this run)")
    elif history_start is not None:
        print(
            f"History window (UTC): {history_start.date().isoformat()} .. now "
            f"(gp_history epoch range; --history-days ignored)"
        )
    else:
        print(
            f"History window (UTC): last {args.history_days} days .. now "
            f"(gp_history epoch range)"
        )
    if args.max_history_points > 0 and not args.skip_archive_fetch:
        print(f"Max history points (after downsample): {args.max_history_points}")
    print(f"Latest TLE epoch (UTC): {latest.epoch.isoformat()}")
    print(f"Current altitude: {current_alt_km:.2f} km ({current_alt_note})")
    print(f"Historical points used: {len(merged)}")
    print(f"Cache file: {cache_path}")
    print(f"Plot file: {args.plot_file}")
    if projection.projected_date is None:
        print(f"Projected de-orbit date: unavailable ({projection.note})")
    else:
        print(
            "Projected de-orbit date "
            f"(threshold {args.deorbit_threshold_km:.1f} km): {projection.projected_date.isoformat()} ({projection.note})"
        )
    if spacetrack_decay_date is None:
        print("Space-Track DECAY prediction: unavailable")
    else:
        pred_type = f" [{spacetrack_decay_type}]" if spacetrack_decay_type else ""
        print(f"Space-Track DECAY prediction{pred_type}: {spacetrack_decay_date.isoformat()}")


if __name__ == "__main__":
    main()

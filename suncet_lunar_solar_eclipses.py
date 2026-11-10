#!/usr/bin/env python3
"""Predict lunar occultations of the solar disk from a SunCET orbit.

Planning mode uses a Kepler orbit with secular J2 nodal precession. TLE mode
uses SGP4 and transforms TEME states to GCRS. Both include spacecraft parallax
and iterated reception light time for the Sun and Moon. All output times are UTC.
See README.md for assumptions, accuracy limits, and examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    GCRS,
    TEME,
    get_body_barycentric,
)
from astropy.time import Time
from astropy.utils import iers
from scipy.optimize import brentq, minimize_scalar, newton

from suncet_ltan_to_raan import DEFAULT_LAUNCH, DEFAULT_MLTAN_HOURS

MU_EARTH = 398600.4418  # km^3 / s^2
EARTH_RADIUS_KM = 6378.137  # equatorial; conservative spherical limb
EARTH_J2 = 1.08262668e-3
TROPICAL_YEAR_DAYS = 365.2422
SUN_RADIUS_KM = 695700.0  # IAU nominal photospheric radius
MOON_RADIUS_KM = 1737.4  # mean spherical radius
LIGHT_SPEED_KM_S = c.to_value(u.km / u.s)
ROOT_TOLERANCE_S = 0.01  # numerical precision, not prediction accuracy


def utc_time(text: str) -> Time:
    """Accept an ISO UTC date/time, including a trailing Z."""
    return Time(text.removesuffix("Z"), scale="utc")


def iso_utc(time: Time) -> str:
    time = time.utc.copy()
    time.precision = 3
    return f"{time.isot}Z"


@dataclass
class PlanningOrbit:
    epoch: Time
    altitude_km: float = DEFAULT_LAUNCH["altitude_km"]
    inclination_deg: float = DEFAULT_LAUNCH["inclination_deg"]
    ltan_hours: float = DEFAULT_MLTAN_HOURS
    phase_deg: float = 0.0  # argument of latitude at epoch: 0 = ascending node
    raan_deg: float | None = None
    eccentricity: float = DEFAULT_LAUNCH["eccentricity"]

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([
            self.altitude_km, self.inclination_deg, self.ltan_hours, self.phase_deg, self.eccentricity,
        ])):
            raise ValueError("Orbit parameters must be finite.")
        if self.altitude_km <= 0 or not 0 <= self.inclination_deg <= 180:
            raise ValueError("Altitude must be positive and inclination within [0, 180].")
        if not 0 <= self.ltan_hours < 24:
            raise ValueError("MLTAN must be within [0, 24) hours.")
        if not 0 <= self.eccentricity < 1:
            raise ValueError("Eccentricity must be within [0, 1).")
        self.semimajor_axis_km = EARTH_RADIUS_KM + self.altitude_km
        if self.semimajor_axis_km * (1 - self.eccentricity) <= EARTH_RADIUS_KM:
            raise ValueError("Perigee must be above the reference Earth radius.")
        if self.raan_deg is None:
            # Reuse the same mean-local-time convention as the existing tools.
            from poliastro.earth.util import raan_from_ltan

            self.raan_deg = float(raan_from_ltan(
                self.epoch, self.ltan_hours * u.hourangle,
            ).to_value(u.deg) % 360)
        if not np.isfinite(self.raan_deg):
            raise ValueError("RAAN must be finite.")
        self.max_radius_km = self.semimajor_axis_km * (1 + self.eccentricity)
        self.mean_motion = np.sqrt(MU_EARTH / self.semimajor_axis_km**3)
        semilatus_rectum_km = self.semimajor_axis_km * (1 - self.eccentricity**2)
        self.node_rate = (
            -1.5 * EARTH_J2 * self.mean_motion
            * (EARTH_RADIUS_KM / semilatus_rectum_km)**2
            * np.cos(np.deg2rad(self.inclination_deg))
        )
        # The mission target does not specify an argument of perigee. Set it
        # to zero so phase_deg remains the argument of latitude at the epoch.
        self._minor_axis_ratio = np.sqrt(1 - self.eccentricity**2)
        phase = np.deg2rad(self.phase_deg)
        eccentric_anomaly = np.arctan2(self._minor_axis_ratio * np.sin(phase),
                                      self.eccentricity + np.cos(phase))
        self._mean_anomaly_at_epoch = eccentric_anomaly - self.eccentricity * np.sin(eccentric_anomaly)

    def positions(self, times: Time) -> np.ndarray:
        """Geocentric position (N, 3), km, in GCRS-aligned axes."""
        seconds = np.atleast_1d((times - self.epoch).to_value(u.s))
        ecc = self.eccentricity
        mean = self._mean_anomaly_at_epoch + self.mean_motion * seconds
        anomaly = mean
        if ecc and mean.size:
            mean = (mean + np.pi) % (2 * np.pi) - np.pi
            initial = np.copysign(np.pi, mean) if ecc > .8 else mean
            anomaly = newton(lambda value: value - ecc * np.sin(value) - mean, initial,
                             fprime=lambda value: 1 - ecc * np.cos(value), tol=1e-12, maxiter=80)
        node = np.deg2rad(self.raan_deg) + self.node_rate * seconds
        ci, si = np.cos(np.deg2rad(self.inclination_deg)), np.sin(np.deg2rad(self.inclination_deg))
        x = np.cos(anomaly) - ecc
        y = self._minor_axis_ratio * np.sin(anomaly)
        cn, sn = np.cos(node), np.sin(node)
        return self.semimajor_axis_km * np.column_stack((
            cn * x - sn * y * ci, sn * x + cn * y * ci, y * si,
        ))

    def metadata(self) -> dict:
        raan_rate_deg_per_day = float(np.rad2deg(self.node_rate) * 86400)
        # A degree of node drift relative to the mean Sun is four minutes of MLTAN.
        mltan_drift = 4 * (raan_rate_deg_per_day * TROPICAL_YEAR_DAYS - 360)
        return {
            "model": "Kepler orbit plus secular J2 RAAN precession",
            "epoch_utc": iso_utc(self.epoch), "altitude_km": self.altitude_km,
            "eccentricity": self.eccentricity, "argument_of_perigee_deg": 0.0,
            "inclination_deg": self.inclination_deg, "mltan_hours": self.ltan_hours,
            "mission_mltan_drift_target_min_per_year": DEFAULT_LAUNCH["mltan_drift_target_min_per_year"],
            "mission_mltan_drift_tolerance_min_per_year": DEFAULT_LAUNCH["mltan_drift_tolerance_min_per_year"],
            "modeled_mltan_drift_min_per_year": mltan_drift,
            "raan_deg": self.raan_deg, "phase_deg": self.phase_deg,
            "raan_rate_deg_per_day": raan_rate_deg_per_day,
        }


class TLEOrbit:
    def __init__(self, path: Path):
        from sgp4.api import Satrec
        from tle_to_oasis_ephemeris import read_tle_file

        self.path = path
        self.record = read_tle_file(path)
        self.epoch = Time(self.record.epoch)
        self.satellite = Satrec.twoline2rv(self.record.line1, self.record.line2)
        # Conservative screening envelope, not the propagator itself.
        self.max_radius_km = (
            self.satellite.a * self.satellite.radiusearthkm
            * (1 + self.satellite.ecco) * 1.1 + 1000
        )

    def positions(self, times: Time) -> np.ndarray:
        from sgp4.api import SGP4_ERRORS

        times = times.reshape((-1,))
        errors, position, velocity = self.satellite.sgp4_array(times.jd1, times.jd2)
        if np.any(errors):
            code = int(errors[errors != 0][0])
            raise ValueError(f"SGP4 propagation failed: {SGP4_ERRORS.get(code, code)}")
        rep = CartesianRepresentation(position.T * u.km).with_differentials(
            CartesianDifferential(velocity.T * u.km / u.s),
        )
        return TEME(rep, obstime=times).transform_to(
            GCRS(obstime=times),
        ).cartesian.xyz.to_value(u.km).T

    def metadata(self) -> dict:
        return {
            "model": "SGP4, TEME transformed to GCRS", "file": str(self.path),
            "epoch_utc": iso_utc(self.epoch),
            "line1": self.record.line1, "line2": self.record.line2,
        }


def separation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stable angular separation in radians for arrays of vectors (N, 3)."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def disk_overlap_fraction(sun_radius, moon_radius, distance) -> np.ndarray:
    """Fraction of solar area covered by a circular lunar disk (angular units).

    Uses a tangent-plane circular-disk area formula; this is area, not EUV flux.
    """
    rs, rm, d = np.broadcast_arrays(
        np.asarray(sun_radius, dtype=float), np.asarray(moon_radius, dtype=float),
        np.asarray(distance, dtype=float),
    )
    result = np.zeros(rs.shape, dtype=float)
    contained = d <= np.abs(rs - rm)
    result[contained] = (np.minimum(rs, rm)[contained] / rs[contained])**2
    partial = (d < rs + rm) & ~contained
    a, b, x = rs[partial], rm[partial], d[partial]
    area = (
        a*a * np.arccos(np.clip((x*x + a*a - b*b) / (2*x*a), -1, 1))
        + b*b * np.arccos(np.clip((x*x + b*b - a*a) / (2*x*b), -1, 1))
        - 0.5 * np.sqrt(np.maximum(0, (-x+a+b)*(x+a-b)*(x-a+b)*(x+a+b)))
    )
    result[partial] = area / (np.pi * a*a)
    return np.clip(result, 0, 1)


class EclipseGeometry:
    def __init__(self, orbit, start: Time, ephemeris: str, earth_limb_margin_km: float = 0):
        self.orbit, self.start, self.ephemeris = orbit, start, ephemeris
        self.earth_radius_km = EARTH_RADIUS_KM + earth_limb_margin_km

    def body_vectors(self, times: Time, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reception light-time-corrected astrometric vectors from spacecraft.

        Follows Astropy get_body's light-time iteration, with the spacecraft as
        observer. Stellar aberration is omitted for the relative disk geometry,
        as in SPICE occultation searches. No geocentric apparent vectors are mixed
        with barycentric geometric ones.
        """
        observer = get_body_barycentric(
            "earth", times, ephemeris=self.ephemeris,
        ).xyz.to_value(u.km).T + positions
        vectors = []
        for body in ("sun", "moon"):
            delay = np.zeros(len(times))
            for _ in range(10):
                body_pos = get_body_barycentric(
                    body, times - delay * u.s, ephemeris=self.ephemeris,
                ).xyz.to_value(u.km).T
                vector = body_pos - observer
                new_delay = np.linalg.norm(vector, axis=-1) / LIGHT_SPEED_KM_S
                if np.max(np.abs(new_delay - delay)) < 1e-8:
                    break
                delay = new_delay
            else:
                raise RuntimeError("Body light-time iteration did not converge.")
            vectors.append(vector)
        return vectors[0], vectors[1]

    def evaluate(self, seconds) -> dict[str, np.ndarray]:
        times = self.start + np.atleast_1d(seconds) * u.s
        position = self.orbit.positions(times)
        radius = np.linalg.norm(position, axis=-1)
        if np.any(radius <= self.earth_radius_km):
            raise ValueError("Spacecraft is inside the configured Earth limb.")
        sun, moon = self.body_vectors(times, position)
        sun_dist, moon_dist = np.linalg.norm(sun, axis=-1), np.linalg.norm(moon, axis=-1)
        rs, rm = np.arcsin(SUN_RADIUS_KM / sun_dist), np.arcsin(MOON_RADIUS_KM / moon_dist)
        sep = separation(sun, moon)
        # A clear Sun means its ENTIRE disk is above the specified Earth limb.
        earth_separation = separation(sun, -position)
        earth_radius = np.arcsin(self.earth_radius_km / radius)
        clearance = earth_separation - earth_radius - rs
        return {
            "gap": sep - rs - rm, "inner_gap": sep - np.abs(rs - rm),
            "separation": sep, "sun_radius": rs, "moon_radius": rm,
            "obscuration": disk_overlap_fraction(rs, rm, sep),
            "earth_clearance": clearance,
            "earth_separation": earth_separation, "earth_radius": earth_radius,
            "earth_full_gap": earth_separation + rs - earth_radius,
        }

    @lru_cache(maxsize=16384)
    def sample(self, second: float) -> dict[str, float]:
        return {key: float(value[0]) for key, value in self.evaluate([second]).items()}

    def field(self, name: str) -> Callable:
        def evaluate(seconds):
            if np.ndim(seconds) == 0:
                return self.sample(float(seconds))[name]
            return self.evaluate(seconds)[name]
        return evaluate

    def candidate_windows(self, duration_s: float) -> list[tuple[float, float]]:
        """Screen at 10-minute cadence using a conservative parallax envelope.

        A 0.25-degree pad exceeds the Sun-Moon relative motion over one screening
        step for modern near-Earth epochs. Adjacent samples are retained as well.
        """
        grid = time_grid(0, duration_s, 600)
        times = self.start + grid * u.s
        sun, moon = self.body_vectors(times, np.zeros((len(grid), 3)))
        ds, dm = np.linalg.norm(sun, axis=-1), np.linalg.norm(moon, axis=-1)
        r = self.orbit.max_radius_km
        if r >= np.min(dm) - MOON_RADIUS_KM:
            raise ValueError("This predictor requires an Earth orbit well inside the lunar orbit.")
        limit = (
            np.arcsin(SUN_RADIUS_KM / (ds-r)) + np.arcsin(MOON_RADIUS_KM / (dm-r))
            + np.arcsin(r / ds) + np.arcsin(r / dm) + np.deg2rad(0.25)
        )
        selected = separation(sun, moon) < limit
        selected = selected | np.r_[False, selected[:-1]] | np.r_[selected[1:], False]
        runs = np.flatnonzero(np.diff(np.r_[False, selected, False].astype(int)))
        return [(float(grid[a]), float(grid[b-1])) for a, b in zip(runs[::2], runs[1::2])]


def time_grid(start: float, end: float, step: float) -> np.ndarray:
    if end <= start:
        return np.array([start])
    return np.linspace(start, end, max(1, int(np.ceil((end-start) / step))) + 1)


def local_minimum(fn: Callable, start: float, end: float) -> tuple[float, float]:
    # Optimize offsets so scipy's relative tolerance does not grow with the
    # number of seconds since the search epoch (tens of millions over a year).
    fit = minimize_scalar(lambda offset: fn(start + offset), bounds=(0, end-start),
                          method="bounded", options={"xatol": ROOT_TOLERANCE_S})
    return start + float(fit.x), float(fit.fun)


def extremum(fn: Callable, start: float, end: float, step: float) -> tuple[float, float]:
    """Find the lowest of all resolved local minima, including endpoints."""
    grid = time_grid(start, end, step)
    values = np.asarray(fn(grid))
    best = int(np.argmin(values))
    candidates = [(float(grid[best]), float(values[best]))]
    for i in range(1, len(grid)-1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            if values[i-1] == values[i] == values[i+1]:
                continue
            candidates.append(local_minimum(fn, grid[i-1], grid[i+1]))
    return min(candidates, key=lambda item: item[1])


def negative_intervals(fn: Callable, start: float, end: float, step: float) -> list[tuple[float, float]]:
    """Find fn < 0 intervals, refining local minima to capture short grazes.

    A finite cadence cannot guarantee finding arbitrarily fast sub-step structure;
    the default 30 s resolves the smooth LEO apparent-motion extrema. Contacts are
    refined independently to ROOT_TOLERANCE_S, including events at window edges.
    """
    if end <= start:
        return []
    grid = time_grid(start, end, step)
    values = np.asarray(fn(grid))
    extra = []
    for i in range(1, len(grid)-1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            if values[i-1] == values[i] == values[i+1]:
                continue
            minimum, value = local_minimum(fn, grid[i-1], grid[i+1])
            if value < 0:
                extra.append(minimum)
    if extra:
        grid = np.unique(np.r_[grid, extra])
        values = np.asarray(fn(grid))
    boundaries = [start]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if (fa < 0) != (fb < 0):
            boundaries.append(float(brentq(fn, a, b, xtol=ROOT_TOLERANCE_S)))
    boundaries.append(end)
    return [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])
            if b > a and fn((a+b)/2) < 0]


@dataclass
class EclipseEvent:
    start_utc: str
    maximum_utc: str
    end_utc: str
    duration_s: float
    kind: str
    max_obscuration_pct: float
    min_separation_arcmin: float
    sun_radius_arcmin_at_max: float
    moon_radius_arcmin_at_max: float
    sun_clear_of_earth_at_max: bool
    earth_clear_duration_s: float
    max_earth_clear_obscuration_pct: float | None
    earth_clear_maximum_utc: str | None
    start_clipped: bool
    end_clipped: bool
    earth_clear_intervals_utc: list[list[str]]
    internal_intervals_utc: list[list[str]]


def predict(geometry: EclipseGeometry, duration_s: float, step_s: float = 30) -> list[EclipseEvent]:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Search duration must be finite and positive.")
    if not np.isfinite(step_s) or not 0 < step_s <= 60:
        raise ValueError("Fine search step must be within (0, 60] seconds.")
    events = []
    stamp = lambda seconds: iso_utc(geometry.start + seconds * u.s)
    for lo, hi in geometry.candidate_windows(duration_s):
        for start, end in negative_intervals(geometry.field("gap"), lo, hi, step_s):
            # Maximizing disk area allows for changing Sun/Moon angular sizes.
            neg_area = lambda t: -geometry.field("obscuration")(t)
            peak, _ = extremum(neg_area, start, end, step_s)
            closest, min_sep = extremum(geometry.field("separation"), start, end, step_s)
            if geometry.sample(closest)["obscuration"] >= 1 - 1e-12:
                peak = closest  # unique greatest-eclipse time within a totality plateau
            sample = geometry.sample(peak)
            kind = "partial"
            if sample["inner_gap"] < 0:
                kind = "total" if sample["moon_radius"] >= sample["sun_radius"] else "annular"
            clear = negative_intervals(
                lambda t: -geometry.field("earth_clearance")(t), start, end, step_s,
            )
            inner = negative_intervals(geometry.field("inner_gap"), start, end, step_s)
            clear_max = [extremum(neg_area, a, b, step_s) for a, b in clear]
            best_clear = min(clear_max, key=lambda item: item[1]) if clear_max else None
            events.append(EclipseEvent(
                start_utc=stamp(start), maximum_utc=stamp(peak), end_utc=stamp(end),
                duration_s=end-start, kind=kind,
                max_obscuration_pct=100*sample["obscuration"],
                min_separation_arcmin=float(np.rad2deg(min_sep)*60),
                sun_radius_arcmin_at_max=float(np.rad2deg(sample["sun_radius"])*60),
                moon_radius_arcmin_at_max=float(np.rad2deg(sample["moon_radius"])*60),
                sun_clear_of_earth_at_max=sample["earth_clearance"] > 0,
                earth_clear_duration_s=sum(b-a for a, b in clear),
                max_earth_clear_obscuration_pct=-100*best_clear[1] if best_clear else None,
                earth_clear_maximum_utc=stamp(best_clear[0]) if best_clear else None,
                start_clipped=start <= ROOT_TOLERANCE_S and geometry.sample(0)["gap"] < 0,
                end_clipped=end >= duration_s-ROOT_TOLERANCE_S and geometry.sample(duration_s)["gap"] < 0,
                earth_clear_intervals_utc=[[stamp(a), stamp(b)] for a, b in clear],
                internal_intervals_utc=[[stamp(a), stamp(b)] for a, b in inner],
            ))
    return events


def resolve_ephemeris(value: str) -> str:
    if value != "auto":
        return str(Path(value).expanduser().resolve()) if Path(value).expanduser().is_file() else value
    data = os.environ.get("suncet_data")
    path = Path(data) / "spice_kernels/de440.bsp" if data else None
    return str(path.resolve()) if path and path.is_file() else "builtin"


def write_results(output_dir: Path, events: list[EclipseEvent], metadata: dict, plot: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(EclipseEvent.__dataclass_fields__)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = asdict(event)
            for key in ("earth_clear_intervals_utc", "internal_intervals_utc"):
                row[key] = json.dumps(row[key])
            writer.writerow(row)
    (output_dir / "summary.json").write_text(json.dumps({
        **metadata, "events": [asdict(event) for event in events],
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 4.5))
        for clear, label, color, marker in (
            (True, "Sun clear of Earth at maximum", "#167c80", "o"),
            (False, "Earth blocks some/all Sun at maximum", "#9b5366", "x"),
        ):
            subset = [e for e in events if e.sun_clear_of_earth_at_max == clear]
            if subset:
                ax.scatter([utc_time(e.maximum_utc).to_datetime() for e in subset],
                           [e.max_obscuration_pct for e in subset], label=label,
                           color=color, marker=marker, s=45)
        if events:
            ax.legend(loc="best", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No lunar solar-disk crossings in this interval",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(utc_time(metadata["start_utc"]).to_datetime(),
                    utc_time(metadata["end_utc"]).to_datetime())
        ax.set_ylim(0, 105)
        ax.set_ylabel("Maximum solar disk area obscured [%]")
        ax.set_xlabel("UTC date")
        ax.set_title("SunCET lunar solar eclipses — " + metadata["prediction_status"])
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "overview.png", dpi=180)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="UTC ISO date/time; default: orbit epoch.")
    parser.add_argument("--end", help="UTC ISO date/time; default: start + 365 days (7 days for TLE).")
    parser.add_argument("--tle-file", type=Path, help="Use the latest TLE in this local text file.")
    parser.add_argument("--epoch", help="Planning orbit epoch (UTC); default: shared ussf_178 epoch.")
    parser.add_argument("--altitude-km", type=float, default=DEFAULT_LAUNCH["altitude_km"],
                        help="Planning semimajor axis minus Earth's reference radius, in km.")
    parser.add_argument("--eccentricity", type=float, default=DEFAULT_LAUNCH["eccentricity"])
    parser.add_argument("--inclination-deg", type=float, default=DEFAULT_LAUNCH["inclination_deg"])
    parser.add_argument("--mltan-hours", type=float, default=DEFAULT_MLTAN_HOURS,
                        help=f"Mean LTAN in decimal hours (mission default: {DEFAULT_MLTAN_HOURS:g}).")
    parser.add_argument("--phase-deg", type=float, default=0,
                        help="Argument of latitude at planning epoch; 0 = ascending node.")
    parser.add_argument("--raan-deg", type=float, help="Override planning RAAN instead of deriving from MLTAN.")
    parser.add_argument("--ephemeris", default="auto",
                        help="auto: local suncet_data/spice_kernels/de440.bsp, else builtin; "
                             "or builtin, a JPL name (e.g. de440s), or local BSP path.")
    parser.add_argument("--step-s", type=float, default=30,
                        help="Fine search cadence, 0 < step <= 60 s; contacts refined to 0.01 s.")
    parser.add_argument("--earth-limb-margin-km", type=float, default=0,
                        help="Add an atmospheric/avoidance margin to Earth's radius (default: 0).")
    parser.add_argument("--allow-stale-tle", action="store_true",
                        help="Allow searching more than 14 days from the TLE epoch, with a warning.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lunar_solar_eclipses"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    # Deterministic offline frame transformations; no implicit IERS network calls.
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "warn"
    warnings = []
    try:
        if not np.isfinite(args.earth_limb_margin_km) or args.earth_limb_margin_km < 0:
            raise ValueError("Earth limb margin must be finite and nonnegative.")
        if args.tle_file:
            supplied = {item.split("=", 1)[0] for item in (argv if argv is not None else sys.argv[1:])}
            if any(flag in supplied for flag in (
                "--epoch", "--altitude-km", "--eccentricity", "--inclination-deg",
                "--mltan-hours", "--phase-deg", "--raan-deg",
            )):
                raise ValueError("Planning orbit options cannot be combined with --tle-file.")
            orbit = TLEOrbit(args.tle_file)
        else:
            orbit = PlanningOrbit(
                utc_time(args.epoch or DEFAULT_LAUNCH["epoch"]), args.altitude_km,
                args.inclination_deg, args.mltan_hours, args.phase_deg, args.raan_deg,
                eccentricity=args.eccentricity,
            )
            warnings.append(
                f"Planning estimate for {args.mltan_hours:05.2f} h MLTAN: provisional orbit and phase. "
                "The default epoch is midnight on the NET launch date, not a known injection time. "
                "Exact contacts and even event visibility can change "
                "with the injection orbit/phase. No drag or full J2 dynamics are modeled."
            )
        start = utc_time(args.start) if args.start else orbit.epoch
        end = utc_time(args.end) if args.end else start + (7 if args.tle_file else 365)*u.day
        duration = float((end-start).to_value(u.s))
        if args.tle_file:
            age = max(abs(float((bound-orbit.epoch).to_value(u.day))) for bound in (start, end))
            if age > 14:
                message = f"Search extends {age:.1f} days from the TLE epoch; SGP4 timing is unreliable."
                if not args.allow_stale_tle:
                    raise ValueError(message + " Shorten --start/--end or explicitly use --allow-stale-tle.")
                warnings.append(message)
        ephemeris = resolve_ephemeris(args.ephemeris)
        if ephemeris == "builtin":
            warnings.append("Using approximate built-in Sun/Moon ephemerides; use JPL DE440 for better geometry.")
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        print(f"Searching {iso_utc(start)} to {iso_utc(end)}; ephemeris: {ephemeris}", flush=True)
        geometry = EclipseGeometry(orbit, start, ephemeris, args.earth_limb_margin_km)
        events = predict(geometry, duration, args.step_s)
        metadata = {
            "prediction_status": "planning estimate" if not args.tle_file else "TLE estimate",
            "start_utc": iso_utc(start), "end_utc": iso_utc(end), "orbit": orbit.metadata(),
            "ephemeris": ephemeris, "fine_step_s": args.step_s,
            "contact_numerical_tolerance_s": ROOT_TOLERANCE_S,
            "earth_limb_margin_km": args.earth_limb_margin_km,
            "sun_radius_km": SUN_RADIUS_KM, "moon_radius_km": MOON_RADIUS_KM,
            "corrections": "spacecraft parallax and iterated reception light time; no stellar aberration",
            "visibility_definition": "entire solar disk above spherical Earth limb plus configured margin",
            "obscuration_definition": "overlap area of circular disks, not EUV irradiance loss",
            "iers_policy": "offline bundled IERS data; degraded accuracy allowed with warnings",
            "warnings": warnings,
        }
        write_results(args.output_dir, events, metadata, not args.no_plot)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    clear_count = sum(event.earth_clear_duration_s > 0 for event in events)
    print(f"Found {len(events)} lunar solar eclipses; {clear_count} have time with the Sun clear of Earth.")
    for event in events:
        visibility = f"{event.earth_clear_duration_s:.1f} s Earth-clear"
        print(f"  {event.maximum_utc}  {event.kind:7s}  {event.max_obscuration_pct:6.2f}%  "
              f"{event.duration_s:6.1f} s  ({visibility})")
    print(f"Wrote {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

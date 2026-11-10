#!/usr/bin/env python3
"""Simulate SunCET UHF link margins above each station's elevation mask."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
from astropy.constants import c, k_B
from astropy.utils import iers
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from suncet_ground_station_angles import Attitude, Station, station_geometry, unit
from suncet_ltan_to_raan import DEFAULT_LAUNCH
from suncet_lunar_solar_eclipses import PlanningOrbit, iso_utc, utc_time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


FREQUENCY_HZ = 401.2e6
DEFAULT_LINK_BPS = 9600
GROUND_POINTING_DB = 12 * (4 / 21) ** 2
SPACECRAFT_POINTING_DB = 12 * (4 / 120) ** 2
LINKS = ("downlink", "uplink")
COLORS = {"downlink": "#006eae", "uplink": "#b34c13"}
DAY_S = 86400
WEEK_S = 7 * DAY_S
CLOSURE_LINKS = (*LINKS, "both")


@dataclass(frozen=True)
class GroundRadio:
    station: Station
    gain_dbi: float
    tx_w: float
    tx_loss_db: float
    rx_loss_db: float
    rx_temperature_k: float


# Site positions and Fairbanks nominal values are from the Round 3 antenna
# definitions. Site elevation plus antenna height approximates ellipsoid height.
# Fairbanks receive line loss is borrowed from the Boulder planning workbook.
GROUND_RADIOS = (
    GroundRadio(Station("Boulder", 40 + 28.1 / 3600, -(105 + 14 / 60 + 45.2 / 3600), 1609),
                15.9, 150, 3.7135, 3.71, 584),
    GroundRadio(Station("Fairbanks", 65 + 7 / 60 + 4.2 / 3600, -(147 + 25 / 60 + 52.1 / 3600), 507),
                15.7, 150, 2, 3.71, 288),
)


@dataclass(frozen=True)
class Link:
    tx_w: float
    losses_db: float
    rx_temperature_k: float
    rate_bps: float
    required_ebn0_db: float = 10.5
    implementation_loss_db: float = 1

    def margin(self, fspl_db, ground_gain_dbi, spacecraft_gain_dbi):
        received_dbw = (10 * np.log10(self.tx_w) + ground_gain_dbi + spacecraft_gain_dbi
                        - self.losses_db - fspl_db)
        noise_dbw = 10 * np.log10(k_B.value * self.rx_temperature_k * self.rate_bps)
        return received_dbw - noise_dbw - self.required_ebn0_db - self.implementation_loss_db


def radio_links(ground, down_bps, up_bps):
    shared_losses_db = 0.1 + GROUND_POINTING_DB + SPACECRAFT_POINTING_DB
    return {
        "downlink": Link(2, shared_losses_db + 0.857 + ground.rx_loss_db,
                         ground.rx_temperature_k, down_bps),
        "uplink": Link(ground.tx_w, shared_losses_db + ground.tx_loss_db + 0.86, 732, up_bps),
    }


def load_gain_pattern(path):
    """Build an interpolator of RHCP and LHCP linear gains from the NEC export."""
    samples = {}
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 12 or fields[7] not in {"RIGHT", "LEFT", "LINEAR"}:
                continue
            theta, phi = map(float, fields[:2])
            gain_db, ratio = map(float, fields[4:6])
            if not np.isfinite([theta, phi, gain_db, ratio]).all() or not 0 <= ratio <= 1:
                raise ValueError(f"Invalid gain sample at θ={theta:g}°, φ={phi:g}°.")
            if (theta, phi) in samples:
                raise ValueError(f"Duplicate gain sample at θ={theta:g}°, φ={phi:g}°.")
            sense = {"RIGHT": 1, "LEFT": -1, "LINEAR": 0}[fields[7]]
            # Project the polarization ellipse onto the two circular bases.
            # Squared amplitudes avoid cancellation in the weaker component.
            signed_ratio = sense * ratio
            scale = 10 ** (gain_db / 10) / (2 * (1 + signed_ratio**2))
            samples[theta, phi] = (scale * (1 + signed_ratio)**2, scale * (1 - signed_ratio)**2)
    theta = sorted({t for t, _ in samples})
    phi = sorted({p for _, p in samples})
    if (not samples or theta[0] != 0 or theta[-1] != 180 or phi[0] != 0
            or phi[-1] != 360 or len(samples) != len(theta) * len(phi)):
        raise ValueError("The gain pattern must contain a complete θ=0–180°, φ=0–360° grid.")
    gain = np.array([[samples[t, p] for p in phi] for t in theta])
    if not np.allclose(gain[:, 0], gain[:, -1]) or not np.allclose(gain[[0, -1]], gain[[0, -1], :1]):
        raise ValueError("Circular gains must agree at the azimuth seam and at each pole.")
    return RegularGridInterpolator((theta, phi), gain)


def directional_gain(pattern, body):
    x, y, z = body.T
    theta = np.rad2deg(np.arctan2(np.hypot(x, y), z))
    phi = np.rad2deg(np.arctan2(y, x)) % 360
    with np.errstate(divide="ignore"):
        return 10 * np.log10(pattern(np.column_stack((theta, phi))))


@dataclass(frozen=True)
class Contact:
    pass_id: str
    aos: float
    peak: float
    los: float
    clipped: bool


def find_contacts(orbit, station, start, end_s, mask_deg):
    """Find all contacts, including short grazers and contacts at year boundaries."""
    def elevation(seconds):
        return station_geometry(orbit, station, start + seconds * u.s)[3]

    # Pad the search so contacts overlapping either boundary still have brackets.
    period_s = 2 * np.pi / orbit.mean_motion
    scan = np.arange(-period_s, end_s + period_s + 30, 30)
    el = np.concatenate([elevation(scan[i:i + 25000]) for i in range(0, len(scan), 25000)])
    peaks = np.flatnonzero((el[1:-1] >= el[:-2]) & (el[1:-1] > el[2:])) + 1
    left, right = scan[peaks - 1], scan[peaks + 1]
    # Refine the local maxima even if no grid sample exceeded the mask.
    for _ in range(22):
        third = (right - left) / 3
        a, b = left + third, right - third
        rising = elevation(a) < elevation(b)
        left, right = np.where(rising, a, left), np.where(rising, right, b)
    peak = (left + right) / 2
    peak = peak[elevation(peak) > mask_deg]
    if not len(peak):
        return []

    def crossing(left, right, rising):
        while np.max(right - left) > 0.005:
            mid = (left + right) / 2
            move_left = (elevation(mid) < mask_deg) == rising
            left, right = np.where(move_left, mid, left), np.where(move_left, right, mid)
        return (left + right) / 2

    outside = scan[el < mask_deg]
    i = np.searchsorted(outside, peak)
    bounded = (i > 0) & (i < len(outside))
    peak, i = peak[bounded], i[bounded]
    if not len(peak):
        return []
    aos = crossing(outside[i - 1], peak, rising=True)
    los = crossing(peak, outside[i], rising=False)
    contacts = []
    for a, p, b in zip(aos, peak, los):
        if b > 0 and a < end_s:
            contacts.append(Contact(f"{station.name}-{len(contacts) + 1:04d}",
                                    max(0., a), np.clip(p, 0, end_s), min(end_s, b),
                                    bool(a < 0 or b > end_s)))
    return contacts


def contact_samples(contact, roll_centers, roll_seconds):
    samples = [np.arange(contact.aos, contact.los, 1), [contact.peak, contact.los]]
    half_roll = roll_seconds / 2
    for center in roll_centers:
        lo = max(contact.aos, center - half_roll)
        hi = min(contact.los, center + half_roll)
        if hi > lo:
            samples.append(np.r_[np.arange(lo, hi, .25), hi])
        elif not roll_seconds and contact.aos < center < contact.los:
            samples.append([center - min(.001, (center - contact.aos) / 2), center])
    return np.unique(np.concatenate(samples))


def simulate_contacts(orbit, ground, start, contacts, pattern, roll_seconds, down_bps, up_bps):
    """Yield per-contact samples, sharing an attitude solution across each week."""
    links = radio_links(ground, down_bps, up_bps)
    weeks = {}
    for contact in contacts:
        weeks.setdefault(int(contact.aos // WEEK_S), []).append(contact)

    for week in weeks.values():
        first = week[0].aos
        attitude = Attitude(orbit, start + first * u.s, start + week[-1].los * u.s, roll_seconds)
        centers = first + np.asarray(attitude.flips)
        grids = [contact_samples(contact, centers, roll_seconds) for contact in week]
        seconds = np.concatenate(grids)
        times = start + seconds * u.s
        _, _, los, elevation = station_geometry(orbit, ground.station, times)
        x, y, z, rolling, _ = attitude.evaluate(times)
        direction = unit(los)
        body = np.column_stack([np.sum(direction * axis, axis=1) for axis in (x, y, z)])
        rhcp, lhcp = directional_gain(pattern, body).T
        range_km = np.linalg.norm(los, axis=1)
        fspl_db = 20 * np.log10(4 * np.pi * range_km * 1000 * FREQUENCY_HZ / c.value)
        data = {
            "seconds": seconds, "elevation_deg": elevation, "range_km": range_km,
            "angle_from_plus_y_deg": np.rad2deg(np.arccos(np.clip(body[:, 1], -1, 1))),
            "rolling": rolling, "rhcp_gain_dbi": rhcp, "lhcp_gain_dbi": lhcp,
            **{f"{name}_margin_db": link.margin(fspl_db, ground.gain_dbi, rhcp)
               for name, link in links.items()},
        }
        offset = 0
        for contact, grid in zip(week, grids):
            end = offset + len(grid)
            yield contact, {key: values[offset:end] for key, values in data.items()}
            offset = end


def closed_times(seconds, margin_db):
    """Return total and longest continuous time at or above the 0 dB threshold."""
    closed = margin_db >= 0
    edges = np.flatnonzero(closed[:-1] != closed[1:])
    # Interpolate in received power relative to the threshold. expm1 keeps
    # precision near 0 dB and also handles a zero-power null (-inf dB).
    left = np.expm1(margin_db[edges] * (np.log(10) / 10))
    right = np.expm1(margin_db[edges + 1] * (np.log(10) / 10))
    crossings = seconds[edges] + (seconds[edges + 1] - seconds[edges]) * left / (left - right)
    starts = np.r_[seconds[:1] if closed[0] else [], crossings[~closed[edges]]]
    ends = np.r_[crossings[closed[edges]], seconds[-1:] if closed[-1] else []]
    lengths = ends - starts
    return float(lengths.sum()), float(lengths.max(initial=0))


def summarize_pass(start, station, contact, data):
    seconds = data["seconds"]
    row = {
        "pass_id": contact.pass_id, "station": station, "aos_utc": iso_utc(start + contact.aos * u.s),
        "peak_utc": iso_utc(start + contact.peak * u.s), "los_utc": iso_utc(start + contact.los * u.s),
        "duration_s": contact.los - contact.aos, "clipped_at_window_boundary": contact.clipped,
        "peak_elevation_deg": float(np.max(data["elevation_deg"])),
        "roll_during_pass": bool(np.any(data["rolling"])),
    }
    for name in LINKS:
        margin = data[f"{name}_margin_db"]
        i = int(np.argmin(margin))
        row[f"min_{name}_db"] = float(margin[i])
        row[f"min_{name}_utc"] = iso_utc(start + seconds[i] * u.s)
        row[f"min_{name}_elevation_deg"] = float(data["elevation_deg"][i])
        row[f"min_{name}_during_roll"] = bool(data["rolling"][i])
    limiting = min(LINKS, key=lambda name: row[f"min_{name}_db"])
    row["limiting_link"] = limiting
    row["min_either_link_db"] = row[f"min_{limiting}_db"]
    # Both directions share the same time-varying terms. The weaker link
    # therefore determines closure at every sample in this budget.
    for name in CLOSURE_LINKS:
        margin = data[f"{limiting if name == 'both' else name}_margin_db"]
        total, longest = closed_times(seconds, margin)
        row[f"{name}_closed_s"] = total
        row[f"{name}_closed_fraction"] = min(1., total / row["duration_s"])
        row[f"{name}_longest_closed_s"] = longest
    return row


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_stations(rows):
    summary = {}
    for station in sorted({row["station"] for row in rows}):
        local = [row for row in rows if row["station"] == station]
        summary[station] = {"pass_count": len(local), "clipped_passes": sum(r["clipped_at_window_boundary"] for r in local)}
        for name in LINKS:
            values = np.array([r[f"min_{name}_db"] for r in local])
            worst = local[int(np.argmin(values))]
            summary[station][name] = {"worst_db": float(np.min(values)),
                                      "p05_db": float(np.percentile(values, 5)),
                                      "median_db": float(np.median(values)),
                                      "passes_below_0_db": int(np.sum(values < 0)),
                                      "worst_pass_id": worst["pass_id"]}
        duration = sum(row["duration_s"] for row in local)
        closure = {}
        for name in CLOSURE_LINKS:
            fractions = np.array([row[f"{name}_closed_fraction"] for row in local])
            longest = np.array([row[f"{name}_longest_closed_s"] for row in local])
            closure[name] = {
                "passes_closed_throughout": int(np.count_nonzero(fractions == 1)),
                "passes_partly_closed": int(np.count_nonzero((fractions > 0) & (fractions < 1))),
                "passes_never_closed": int(np.count_nonzero(fractions == 0)),
                "fraction_of_all_contact_time_closed": sum(row[f"{name}_closed_s"] for row in local) / duration,
                "median_pass_closed_fraction": float(np.median(fractions)),
                "median_longest_closed_s": float(np.median(longest)),
            }
        summary[station]["closure"] = closure
    return summary


def plot_passes(examples, output):
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 3.7 * len(examples) + .6),
                             squeeze=False, sharey=True)
    legend = {}
    for panels, (station, (high, low)) in zip(axes, examples.items()):
        for ax, label, (row, data) in zip(panels, ("Highest elevation", "Lowest margin"), (high, low)):
            minutes = (data["seconds"] - data["seconds"][0]) / 60
            if np.any(data["rolling"]):
                ax.fill_between(minutes, 0, 1, where=data["rolling"], color="#ebdfca", alpha=.8,
                                transform=ax.get_xaxis_transform(), label="Spacecraft roll")
            for name in LINKS:
                margins = data[f"{name}_margin_db"]
                ax.plot(minutes, margins, color=COLORS[name], label=name.capitalize(), lw=2)
                i = np.argmin(margins)
                ax.scatter(minutes[i], margins[i], color=COLORS[name], s=23, zorder=3)
            ax.axhline(0, color="#43505a", ls="--", lw=1, label="0 dB threshold")
            elevation = ax.twinx()
            elevation.plot(minutes, data["elevation_deg"], color="#77818a", lw=1.2, alpha=.8,
                           label="Elevation (right axis)")
            elevation.set_ylim(0, 90)
            elevation.set_ylabel("Elevation (°)", color="#606d78")
            elevation.tick_params(axis="y", colors="#606d78")
            ax.set(xlim=(0, minutes[-1]), xlabel="Minutes from contact start", ylabel="Margin (dB)",
                   title=f"{station} · {label}\n{row['pass_id']}  |  {row['aos_utc'][:10]} UTC  |  Peak {row['peak_elevation_deg']:.1f}°")
            ax.grid(axis="y", alpha=.22)
            for panel in (ax, elevation):
                handles, labels = panel.get_legend_handles_labels()
                legend.update(zip(labels, handles))
    order = ("Downlink", "Uplink", "Elevation (right axis)", "0 dB threshold", "Spacecraft roll")
    labels = [label for label in order if label in legend]
    fig.legend([legend[label] for label in labels], labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.subplots_adjust(left=.075, right=.925, bottom=.14, top=1 - .6 / fig.get_figheight(),
                        hspace=.52, wspace=.33)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_annual(rows, output):
    stations = sorted({row["station"] for row in rows})
    values = [row[f"min_{name}_db"] for row in rows for name in LINKS]
    bins = np.arange(min(0, np.floor(min(values))), np.ceil(max(values)) + 1)
    fig, axes = plt.subplots(len(stations), 2, figsize=(13, 3.2 * len(stations) + .8), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1.4]}, sharey=True)
    for (ax, hist), station in zip(axes, stations):
        local = [r for r in rows if r["station"] == station]
        dates = [datetime.fromisoformat(r["aos_utc"].removesuffix("Z")).replace(tzinfo=timezone.utc)
                 for r in local]
        highest_count = 0
        for name in LINKS:
            minimum = [r[f"min_{name}_db"] for r in local]
            color = COLORS[name]
            ax.scatter(dates, minimum, s=6, alpha=.45, color=color, label=name.capitalize())
            counts, edges, _ = hist.hist(minimum, bins=bins, orientation="horizontal", histtype="step",
                                         lw=1.9, color=color)
            peak = int(np.argmax(counts))
            count = counts[peak]
            highest_count = max(highest_count, count)
            low, high = edges[peak:peak + 2]
            center = (low + high) / 2
            hist.plot(count, center, "o", color=color, markersize=3)
            hist.annotate(f"{high:+.0f} dB", (count, center), xytext=(6, 0),
                          textcoords="offset points", va="center", color=color, fontsize=9)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set(title=f"{station} · {len(local):,} passes", ylabel="Minimum margin (dB)")
        hist.set(title="Distribution", xlabel="Pass count per 1 dB bin", xlim=(0, 1.5 * highest_count))
        for panel in (ax, hist):
            panel.axhline(0, color="#43505a", ls="--", lw=1)
            panel.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, markerscale=2)
    fig.subplots_adjust(left=.08, right=.98, bottom=.13, top=1 - .4 / fig.get_figheight(),
                        hspace=.42, wspace=.11)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_annual_closure(rows, mask_deg, days, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = (
        ("both_closed_fraction", 100, "Share of each pass", "Pass time with both links closed (%)"),
        ("both_longest_closed_s", 1 / 60, "Longest uninterrupted window", "Time with both links closed (min)"),
    )
    duration_limit = max(1, np.ceil(max(row["both_longest_closed_s"] for row in rows) / 60))
    colors = {"Boulder": "#006eae", "Fairbanks": "#b34c13"}
    for ax, (key, scale, title, xlabel), limit in zip(axes, panels, (100, duration_limit)):
        for station in sorted({row["station"] for row in rows}):
            values = [row[key] * scale for row in rows if row["station"] == station]
            thresholds, counts = np.unique(values, return_counts=True)
            share = 100 * np.cumsum(counts[::-1])[::-1] / len(values)
            x, y = np.r_[0, thresholds], np.r_[100, share]
            if thresholds[-1] < limit:
                x, y = np.r_[x, limit], np.r_[y, 0]
            ax.step(x, y, where="pre", color=colors[station], lw=2.2,
                    label=f"{station} · {len(values):,} passes")
        ax.set(xlim=(0, limit), ylim=(0, 103), yticks=[0, 25, 50, 75, 100], title=title, xlabel=xlabel)
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Passes meeting or exceeding x (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"UHF link closure across {days:g} days\n"
                 f"Both margins at or above 0 dB · Contacts above {mask_deg:g}° elevation", fontsize=14)
    fig.subplots_adjust(left=.09, right=.98, bottom=.22, top=.77, wspace=.14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", type=Path, required=True, help="Body-frame NEC radiation pattern export.")
    parser.add_argument("--start", default=DEFAULT_LAUNCH["epoch"], help="Planning epoch in UTC.")
    parser.add_argument("--days", type=float, default=365)
    parser.add_argument("--mask-deg", type=float, default=5)
    parser.add_argument("--roll-seconds", type=float, default=60)
    parser.add_argument("--downlink-bps", type=float, default=DEFAULT_LINK_BPS)
    parser.add_argument("--uplink-bps", type=float, default=DEFAULT_LINK_BPS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/uhf_link_passes"))
    args = parser.parse_args(argv)
    if (not np.isfinite([args.days, args.mask_deg, args.roll_seconds, args.downlink_bps, args.uplink_bps]).all()
            or min(args.days, args.downlink_bps, args.uplink_bps) <= 0 or not 0 <= args.mask_deg < 90):
        parser.error("Use finite, positive days and rates and an elevation mask from 0° to less than 90°.")

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "warn"
    start = utc_time(args.start)
    orbit = PlanningOrbit(start)
    if not 0 <= args.roll_seconds < np.pi / (2 * orbit.mean_motion):
        parser.error("Roll duration must be nonnegative and less than a quarter orbit.")
    pattern = load_gain_pattern(args.pattern)
    end_s = args.days * DAY_S
    rows, examples = [], {}
    for ground in GROUND_RADIOS:
        station = ground.station.name
        contacts = find_contacts(orbit, ground.station, start, end_s, args.mask_deg)
        print(f"{station}: {len(contacts):,} contacts above {args.mask_deg:g}°", flush=True)
        high = low = None
        tracks = simulate_contacts(orbit, ground, start, contacts, pattern,
                                   args.roll_seconds, args.downlink_bps, args.uplink_bps)
        for processed, (contact, data) in enumerate(tracks, 1):
            row = summarize_pass(start, station, contact, data)
            rows.append(row)
            if high is None or row["peak_elevation_deg"] > high[0]["peak_elevation_deg"]:
                high = row, data
            if low is None or row["min_either_link_db"] < low[0]["min_either_link_db"]:
                low = row, data
            if processed % 500 == 0:
                print(f"{station}: evaluated {processed:,} / {len(contacts):,} contacts", flush=True)
        if high is not None:
            examples[station] = (high, low)
    if not rows:
        parser.error("No contacts in the planning window. Increase --days or lower --mask-deg.")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["aos_utc"], row["station"]))
    write_csv(output / "pass_minima.csv", rows)
    for pair in examples.values():
        for row, data in pair:
            times = start + data["seconds"] * u.s
            times.precision = 3
            columns = {"utc": [f"{t}Z" for t in times.utc.isot],
                       "seconds_from_contact_start": data["seconds"] - data["seconds"][0],
                       **{key: values for key, values in data.items() if key != "seconds"}}
            write_csv(output / f"{row['pass_id']}.csv", [dict(zip(columns, vals)) for vals in zip(*columns.values())])

    summary = summarize_stations(rows)
    provenance = {
        "window_utc": [iso_utc(start), iso_utc(start + end_s * u.s)], "orbit": orbit.metadata(),
        "elevation_mask_deg": args.mask_deg, "roll_duration_s": args.roll_seconds,
        "sampling": "1 s plus contact endpoints and peak elevation, with 0.25 s samples during rolls",
        "closure": {"threshold_db": 0, "crossing_interpolation": "linear in relative received power",
                    "both": "uplink and downlink at or above threshold at the same time"},
        "frequency_hz": FREQUENCY_HZ, "polarization": "RHCP, reciprocal spacecraft pattern in both directions",
        "pattern": {"path": str(args.pattern.resolve()), "sha256": hashlib.sha256(args.pattern.read_bytes()).hexdigest(),
                    "axes": "export XYZ match SunCET body XYZ", "interpolation": "bilinear on linear circular gains"},
        "ground_radios": [asdict(ground) for ground in GROUND_RADIOS],
        "link_inputs": {ground.station.name: {key: asdict(link) for key, link in
                        radio_links(ground, args.downlink_bps, args.uplink_bps).items()} for ground in GROUND_RADIOS},
        "assumptions": [
            "Kepler planning orbit with J2 nodal precession. Epoch, phase, argument of perigee and roll timing are provisional.",
            "Round 3 antenna definitions supply site coordinates; heights approximate ellipsoidal heights.",
            "Boulder link inputs and spacecraft RF inputs come from the LASP UHF workbook. Each loss is applied once.",
            "The filing and ground radio configuration specify 401.2 MHz both ways. The workbook uplink tab lists 400.2 MHz.",
            "The same passive spacecraft antenna port is assumed for TX and RX. Feed losses are assumed external to the export.",
            "Fairbanks uses filed nominal RHCP, gain, transmitter power, transmit connector loss and system noise temperature. "
            "Its receive passive loss is borrowed from Boulder. Ground pointing loss uses the workbook for both sites.",
            "Free-space loss plus workbook atmosphere, pointing and implementation losses. "
            "The contact mask is geometric. Terrain, scheduling constraints, interference and acquisition transients are outside this budget.",
        ],
        "stations": summary,
    }
    (output / "summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot_passes(examples, output / "example_passes.png")
    plot_annual(rows, output / "annual_pass_minima.png")
    plot_annual_closure(rows, args.mask_deg, args.days, output / "annual_link_closure.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

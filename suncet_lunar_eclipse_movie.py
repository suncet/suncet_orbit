#!/usr/bin/env python3
"""Overlay predicted lunar and Earth occultations on a saved SunCET movie.

The source movie must show the full, uncropped detector with FITS origin='lower'.
An accompanying source FITS supplies the solar center, angular scale and radius.
This is an illustrative image overlay, not a rerun of telescope/detector physics.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import astropy.units as u
from astropy.io import fits
from astropy.utils import iers
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from suncet_lunar_solar_eclipses import (
    EclipseGeometry, PlanningOrbit, TLEOrbit, iso_utc, negative_intervals, utc_time,
)

ARCSEC_PER_RADIAN = u.rad.to(u.arcsec)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path: Path) -> dict:
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path),
    ]))
    return {**info["streams"][0], "duration_s": float(info["format"]["duration"])}


def calibration(path: Path, width: int, height: int) -> dict:
    header = fits.getheader(path)
    if not (header.get("CTYPE1", "").startswith("HPLN") and header.get("CTYPE2", "").startswith("HPLT")):
        raise ValueError("Source FITS must describe helioprojective detector coordinates.")
    if abs(header.get("CRVAL1", 0)) + abs(header.get("CRVAL2", 0)) > 1e-8:
        raise ValueError("Source FITS must be centered on the Sun (CRVAL1=CRVAL2=0).")
    sx, sy = width/header["NAXIS1"], height/header["NAXIS2"]
    if abs(sx/sy - 1) > 0.01:
        raise ValueError("Source movie aspect ratio differs from its FITS image.")
    scales = [u.Unit(header.get(f"CUNIT{i}", "arcsec")).to(u.arcsec) for i in (1, 2)]
    if all(f"CD{i}_{j}" in header for i in (1, 2) for j in (1, 2)):
        matrix = np.array([[header[f"CD{i}_{j}"] for j in (1, 2)] for i in (1, 2)])
        convention = "explicit CD matrix; legacy duplicate PC/CDELT cards ignored"
    else:
        pc = np.array([[header.get(f"PC{i}_{j}", int(i == j)) for j in (1, 2)] for i in (1, 2)])
        matrix = np.diag([header["CDELT1"], header["CDELT2"]]) @ pc
        convention = "CDELT times PC matrix"
    matrix = np.diag(scales) @ matrix @ np.diag([1/sx, 1/sy])
    # Pixel-center convention: FITS is 1-based; the movie has already flipped Y.
    center = [
        (header["CRPIX1"] - 0.5)*sx - 0.5,
        height - 0.5 - (header["CRPIX2"] - 0.5)*sy,
    ]
    radius = header.get("RSUN_OBS", header.get("RSUN"))
    if radius is None or float(radius) <= 0:
        raise ValueError("Source FITS needs RSUN_OBS or RSUN, in arcseconds.")
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Source FITS coordinate matrix is singular.")
    return {"center_xy": center, "matrix_arcsec_per_movie_pixel": matrix.tolist(),
            "solar_radius_arcsec": float(radius), "matrix_convention": convention,
            "movie_origin": "lower FITS origin, displayed north up for this source"}


def solar_basis(sun: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sun direction, projected solar north, and solar west in ICRS axes."""
    sun = sun / np.linalg.norm(sun, axis=1)[:, None]
    # IAU solar north pole, ICRS/J2000 (also SunPy's solar-north convention).
    ra, dec = np.deg2rad([286.13, 63.87])
    pole = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    north = pole - (sun @ pole)[:, None]*sun
    north /= np.linalg.norm(north, axis=1)[:, None]
    west = np.cross(north, sun)
    return sun, north, west


def occultation_track(geometry: EclipseGeometry, seconds: np.ndarray) -> dict[str, np.ndarray]:
    times = geometry.start + seconds*u.s
    positions = geometry.orbit.positions(times)
    sun, moon = geometry.body_vectors(times, positions)
    center, north, west = solar_basis(sun)
    direction = moon / np.linalg.norm(moon, axis=1)[:, None]
    earth = -positions / np.linalg.norm(positions, axis=1)[:, None]
    values = geometry.evaluate(seconds)
    track = {
        **values, "moon_forward": np.sum(direction*center, axis=1),
        "moon_north": np.sum(direction*north, axis=1),
        "moon_west": np.sum(direction*west, axis=1),
        "earth_forward": np.sum(earth*center, axis=1),
        "earth_north": np.sum(earth*north, axis=1),
        "earth_west": np.sum(earth*west, axis=1),
    }
    track["earth_obscuration"], track["total_obscuration"] = solar_disk_coverage(track)
    return track


def ray_grid(cal: dict, width: int, height: int, roll_deg: float = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:height, :width].astype(float)
    x, y = xx-cal["center_xy"][0], cal["center_xy"][1]-yy
    matrix = np.asarray(cal["matrix_arcsec_per_movie_pixel"])
    west = (matrix[0, 0]*x + matrix[0, 1]*y) / ARCSEC_PER_RADIAN
    north = (matrix[1, 0]*x + matrix[1, 1]*y) / ARCSEC_PER_RADIAN
    roll = np.deg2rad(roll_deg)
    west, north = np.cos(roll)*west - np.sin(roll)*north, np.sin(roll)*west + np.cos(roll)*north
    return west, north, np.sqrt(1 + west*west + north*north)


def occultor_alpha(track: dict, body: str, index: int, rays: tuple,
                   pixel_rad: float, field_radius: float | None = None):
    """Antialiased opaque disk on spherical sight lines, including Earth's wide disk."""
    west, north, norm = rays
    if field_radius is None:
        field_radius = float(np.arctan(np.max(np.hypot(west, north))))
    offset = np.arctan2(np.hypot(track[f"{body}_west"][index], track[f"{body}_north"][index]),
                        track[f"{body}_forward"][index])
    radius = track[f"{body}_radius"][index]
    if offset - radius >= field_radius + pixel_rad:
        return 0.0
    if offset + field_radius <= radius - pixel_rad:
        return 1.0
    dot = (track[f"{body}_forward"][index] + west*track[f"{body}_west"][index]
           + north*track[f"{body}_north"][index]) / norm
    distance = np.arccos(np.clip(dot, -1, 1))
    return np.clip((radius-distance) / pixel_rad + 0.5, 0, 1)


def black_mask(frame: Image.Image, alpha) -> Image.Image:
    if np.ndim(alpha) == 0:
        return frame if alpha == 0 else Image.new("RGB", frame.size, (0, 0, 0))
    pixels = np.asarray(frame).astype(float)
    return Image.fromarray(np.rint(pixels*(1-alpha[..., None])).astype(np.uint8))


def mask_moon(frame: Image.Image, track: dict, index: int, rays: tuple, pixel_rad: float) -> Image.Image:
    return black_mask(frame, occultor_alpha(track, "moon", index, rays, pixel_rad))


def mask_occultors(frame: Image.Image, track: dict, index: int, rays: tuple,
                   pixel_rad: float, field_radius: float | None = None) -> Image.Image:
    earth = occultor_alpha(track, "earth", index, rays, pixel_rad, field_radius)
    if np.ndim(earth) == 0 and earth == 1:
        return black_mask(frame, 1.0)
    moon = occultor_alpha(track, "moon", index, rays, pixel_rad, field_radius)
    return black_mask(frame, np.maximum(moon, earth))


@lru_cache(maxsize=4)
def solar_area_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform projected-area samples with fractional solar-limb pixels."""
    axis = (np.arange(size)+0.5)*2/size - 1
    x, y = np.meshgrid(axis, axis)
    weights = np.clip((1-np.hypot(x, y))*size/2 + 0.5, 0, 1)
    selected = weights > 0
    return x[selected], y[selected], weights[selected]


def solar_disk_coverage(track: dict, grid_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Earth and combined coverage of solar area; overlapping masks count once.

    Lunar-only coverage retains the analytic circular-disk result. Partial Earth
    coverage uses spherical sight lines and projected-area quadrature, avoiding
    a small-angle approximation for Earth's roughly 68-degree angular radius.
    """
    earth = np.zeros(len(track["sun_radius"]))
    total = np.asarray(track["obscuration"]).copy()
    unit_x, unit_y, weights = solar_area_grid(grid_size)
    for i, rs in enumerate(track["sun_radius"]):
        if track["earth_separation"][i] >= track["earth_radius"][i] + rs:
            continue
        if track["earth_separation"][i] + rs <= track["earth_radius"][i]:
            earth[i] = total[i] = 1.0
            continue
        radius = np.tan(rs)
        x, y = radius*unit_x, radius*unit_y
        rays = (x, y, np.sqrt(1+x*x+y*y))
        pixel = 2*radius/grid_size
        a = occultor_alpha(track, "earth", i, rays, pixel, rs+pixel)
        b = occultor_alpha(track, "moon", i, rays, pixel, rs+pixel)
        earth[i] = np.sum(a*weights)/np.sum(weights)
        total[i] = np.sum(np.maximum(a, b)*weights)/np.sum(weights)
    return earth, np.clip(total, 0, 1)


def select_interval(summary: dict, event_number: int | None, start_text: str | None,
                    end_text: str | None, padding_s: float):
    """Choose a saved lunar event or an explicit interval, including Earth eclipses."""
    if bool(start_text) != bool(end_text):
        raise ValueError("Supply both --start and --end for a custom movie interval.")
    if start_text:
        if event_number is not None:
            raise ValueError("Use either --event-number or --start/--end.")
        start, end = utc_time(start_text), utc_time(end_text)
        if end <= start:
            raise ValueError("Movie end must be later than start.")
        if start < utc_time(summary["start_utc"]) or end > utc_time(summary["end_utc"]):
            raise ValueError("Custom movie interval must lie within the supplied prediction range.")
        overlapping = [event for event in summary["events"]
                       if utc_time(event["start_utc"]) < end and utc_time(event["end_utc"]) > start]
        event = max(overlapping, key=lambda item: item["max_obscuration_pct"]) if overlapping else None
        return start, end, event
    if event_number is not None:
        if not 1 <= event_number <= len(summary["events"]):
            raise ValueError("--event-number must be a 1-based index into the saved event list.")
        event = summary["events"][event_number-1]
    else:
        complete = [event for event in summary["events"]
                    if not event["start_clipped"] and not event["end_clipped"]]
        clear = [event for event in complete if event["earth_clear_duration_s"] >= event["duration_s"]-0.05]
        if not complete:
            raise ValueError("No complete lunar event; supply --start/--end for an Earth-only movie.")
        event = max(clear or complete, key=lambda item: item["max_obscuration_pct"])
    return (utc_time(event["start_utc"])-padding_s*u.s,
            utc_time(event["end_utc"])+padding_s*u.s, event)


def rescale_background(frame: Image.Image, cal: dict, factor: float, roll_deg: float) -> Image.Image:
    """Apply the same detector roll to the solar scene as to the lunar geometry."""
    matrix = np.asarray(cal["matrix_arcsec_per_movie_pixel"]) @ np.diag([1., -1.])
    roll = np.deg2rad(roll_deg)
    rotation = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    affine = np.linalg.solve(matrix, rotation @ matrix) / factor
    center = np.asarray(cal["center_xy"])
    shift = center - affine @ center
    return frame.transform(frame.size, Image.Transform.AFFINE,
                           (affine[0, 0], affine[0, 1], shift[0],
                            affine[1, 0], affine[1, 1], shift[1]),
                           Image.Resampling.BICUBIC)


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False):
    filename = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / filename
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    from matplotlib.font_manager import findfont
    return ImageFont.truetype(findfont("DejaVu Sans"), size)


def annotated(frame: Image.Image, title: str, timestamp: str, obscuration: float) -> Image.Image:
    panel = Image.new("RGB", (1280, 1040), (5, 3, 10))
    panel.paste(frame.resize((1280, 960), Image.Resampling.LANCZOS), (0, 80))
    draw = ImageDraw.Draw(panel)
    draw.text((28, 12), title, font=font(29, True), fill=(255, 222, 182))
    draw.text((28, 48), timestamp.replace("T", "  ").replace(".000Z", " UTC"),
              font=font(21), fill=(221, 215, 226))
    draw.text((900, 48), f"Disk covered: {obscuration:5.1f}%", font=font(21), fill=(255, 222, 182))
    return panel


def titled_movie_frame(panels: list[Image.Image]) -> Image.Image:
    """Add one title above either a single detector view or the comparison."""
    if len(panels) == 1:
        frame = Image.new("RGB", (1280, 1100), (5, 3, 10))
        frame.paste(panels[0], (0, 60))
    else:
        frame = Image.new("RGB", (1920, 840), (5, 3, 10))
        for i, panel in enumerate(panels):
            frame.paste(panel.resize((960, 780), Image.Resampling.LANCZOS), (960*i, 60))
    ImageDraw.Draw(frame).text((28, 13), "SunCET eclipses", font=font(32, True), fill=(255, 222, 182))
    return frame


def encoder(path: Path, size: tuple[int, int], fps: int) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def read_frame(process: subprocess.Popen, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = process.stdout.read(remaining)
        if not chunk:
            raise RuntimeError("Source movie ended before the requested simulated interval.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="One or two predictor summary.json files.")
    parser.add_argument("--source-movie", type=Path, required=True)
    parser.add_argument("--source-fits", type=Path, required=True,
                        help="FITS frame underlying the full-frame source movie, for angular calibration.")
    parser.add_argument("--source-span-s", type=float, required=True,
                        help="Actual simulated scene time spanned by the source movie, in seconds.")
    parser.add_argument("--duration-s", type=float, default=25, help="Output playback duration (default: 25 s).")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--padding-s", type=float, default=180, help="Real seconds before/after disk contact.")
    parser.add_argument("--event-number", type=int,
                        help="Select a 1-based event index, including events that overlap Earth eclipse.")
    parser.add_argument("--start", help="Custom movie start, UTC ISO date/time; requires --end.")
    parser.add_argument("--end", help="Custom movie end, UTC ISO date/time; requires --start.")
    parser.add_argument("--roll-deg", type=float, default=0, help="Detector roll from source solar-north orientation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lunar_solar_eclipses/movies"))
    args = parser.parse_args(argv)
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "warn"
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe must be on PATH.")
    if len(args.summaries) not in (1, 2):
        parser.error("Supply one or two scenario summaries.")
    numbers = [args.source_span_s, args.duration_s, args.padding_s, args.roll_deg]
    if not np.all(np.isfinite(numbers)) or min(args.source_span_s, args.duration_s, args.fps) <= 0 or args.padding_s < 0:
        parser.error("Durations and FPS must be positive, padding nonnegative, and all inputs finite.")
    info = probe(args.source_movie)
    width, height = info["width"], info["height"]
    cal = calibration(args.source_fits, width, height)
    rays = ray_grid(cal, width, height, args.roll_deg)
    field_radius = float(np.arctan(np.max(np.hypot(rays[0], rays[1]))))
    pixel_rad = np.sqrt(abs(np.linalg.det(cal["matrix_arcsec_per_movie_pixel"]))) / ARCSEC_PER_RADIAN
    count = max(2, round(args.duration_s*args.fps))
    output_duration = count/args.fps
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios, processes, provenance = [], [], []
    try:
        for number, path in enumerate(args.summaries):
            summary = json.loads(path.read_text())
            start, end, event = select_interval(summary, args.event_number, args.start, args.end, args.padding_s)
            meta = summary["orbit"]
            if "altitude_km" in meta:
                orbit = PlanningOrbit(utc_time(meta["epoch_utc"]), meta["altitude_km"],
                                      meta["inclination_deg"], meta["mltan_hours"],
                                      meta["phase_deg"], meta["raan_deg"],
                                      eccentricity=meta.get("eccentricity", 0.0))
                minutes = round(meta["mltan_hours"]*60) % (24*60)
                ltan = f"{minutes//60:02d}:{minutes%60:02d}"
                title = f"{ltan} MLTAN"
                slug = f"mltan{minutes//60:02d}" + (f"{minutes%60:02d}" if minutes%60 else "")
            else:
                orbit = TLEOrbit(Path(meta["file"]))
                title, slug = f"TLE scenario {number+1}", f"tle{number+1}"
            if any(case["slug"] == slug for case in scenarios):
                slug += f"_scenario{number+1}"
            span = float((end-start).to_value(u.s))
            if span > args.source_span_s:
                raise ValueError("Source scene has insufficient duration for this event and padding.")
            seconds = np.linspace(0, span, count)
            geometry = EclipseGeometry(orbit, start, summary["ephemeris"], summary["earth_limb_margin_km"])
            track = occultation_track(geometry, seconds)
            stamp = lambda second: iso_utc(start + second*u.s)
            earth_intervals = negative_intervals(geometry.field("earth_clearance"), 0, span, 10)
            full_earth_intervals = negative_intervals(geometry.field("earth_full_gap"), 0, span, 10)
            source_seconds = span/args.source_span_s * info["duration_s"]
            decoder = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(args.source_movie),
                "-vf", f"setpts=(PTS-STARTPTS)*{output_duration/source_seconds:.12f},fps={args.fps}",
                "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ], stdout=subprocess.PIPE)
            writer = encoder(args.output_dir/f"{slug}.mp4", (1280, 1100), args.fps)
            processes.extend([decoder, writer])
            scenarios.append({
                "decoder": decoder, "writer": writer, "track": track, "title": title,
                "slug": slug, "geometry": geometry, "seconds": seconds, "speed": span/output_duration,
            })
            provenance.append({
                "summary": str(path.resolve()), "summary_sha256": sha256(path),
                "event": event, "start_utc": iso_utc(start), "end_utc": iso_utc(end),
                "real_seconds_per_playback_second": span/output_duration,
                "source_movie_seconds_used": source_seconds,
                "earth_limb_radius_km": geometry.earth_radius_km,
                "earth_solar_overlap_intervals_utc": [[stamp(a), stamp(b)] for a, b in earth_intervals],
                "earth_totality_intervals_utc": [[stamp(a), stamp(b)] for a, b in full_earth_intervals],
                "max_earth_disk_coverage_pct": float(np.max(track["earth_obscuration"])*100),
            })
            with (args.output_dir/f"{slug}_coverage.csv").open("w", newline="") as stream:
                writer_csv = csv.writer(stream)
                writer_csv.writerow(["utc", "moon_disk_coverage_pct", "earth_disk_coverage_pct", "combined_disk_coverage_pct"])
                for i, second in enumerate(seconds):
                    writer_csv.writerow([stamp(second), 100*track["obscuration"][i],
                                         100*track["earth_obscuration"][i], 100*track["total_obscuration"][i]])
        comparison = None
        if len(scenarios) == 2:
            comparison = encoder(args.output_dir/"mltan_comparison.mp4", (1920, 840), args.fps)
            processes.append(comparison)
        peak_indices = [int(np.argmax(case["track"]["obscuration"])) for case in scenarios]
        earth_indices = []
        for case in scenarios:
            earth_area = case["track"]["earth_obscuration"]
            if np.any(earth_area > 0):
                score = 4*earth_area*(1-earth_area) + 0.01*case["track"]["obscuration"]
                earth_indices.append(int(np.argmax(score)))
            else:
                earth_indices.append(None)
        preview_index = earth_indices[0] if earth_indices[0] is not None else peak_indices[0]
        for n in range(count):
            panels = []
            for case_index, case in enumerate(scenarios):
                pixels = read_frame(case["decoder"], width*height*3)
                frame = Image.frombytes("RGB", (width, height), pixels)
                # Match the Sun's apparent angular size at the prediction date.
                factor = case["track"]["sun_radius"][n]*ARCSEC_PER_RADIAN / cal["solar_radius_arcsec"]
                frame = rescale_background(frame, cal, factor, args.roll_deg)
                frame = mask_occultors(frame, case["track"], n, rays, pixel_rad, field_radius)
                panel = annotated(
                    frame, case["title"], iso_utc(case["geometry"].start + case["seconds"][n]*u.s),
                    100*case["track"]["total_obscuration"][n],
                )
                single = titled_movie_frame([panel])
                case["writer"].stdin.write(single.tobytes())
                panels.append(panel)
                if n == peak_indices[case_index]:
                    single.save(args.output_dir/f"{case['slug']}_maximum.png")
                if n == earth_indices[case_index]:
                    single.save(args.output_dir/f"{case['slug']}_earth_limb.png")
            if comparison:
                combined = titled_movie_frame(panels)
                comparison.stdin.write(combined.tobytes())
                if n == preview_index:
                    combined.save(args.output_dir/"comparison_preview.png")
            if n % 120 == 0:
                print(f"Rendered {n}/{count} frames", flush=True)
        for case in scenarios:
            case["writer"].stdin.close()
            case["decoder"].stdout.close()
        if comparison:
            comparison.stdin.close()
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError("Video decoding or encoding failed.")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    (args.output_dir/"movie_provenance.json").write_text(json.dumps({
        "source_movie": str(args.source_movie.resolve()), "source_movie_sha256": sha256(args.source_movie),
        "source_fits": str(args.source_fits.resolve()), "source_fits_sha256": sha256(args.source_fits),
        "source_scene_span_s": args.source_span_s, "calibration": cal, "roll_deg": args.roll_deg,
        "fps": args.fps, "frame_count": count, "playback_duration_s": output_duration,
        "annotations": {"title": "SunCET eclipses", "panel_title": "MLTAN",
                        "timestamp": True, "disk_coverage": True, "footer": False},
        "coverage_definition": "Combined projected solar-disk area hidden by Earth or Moon; overlaps counted once.",
        "coverage_method": "Analytic Moon-only area; partial Earth uses 512x512 antialiased projected-area quadrature.",
        "scenarios": provenance,
        "limitations": [
            "Provisional orbit, phase and injection date; exact events can change.",
            "Saved simulated solar scene is illustrative, not a forecast of the Sun at the eclipse date.",
            "Source movie starts at scene t=0 and advances at the same simulated cadence as lunar motion.",
            "Opaque Earth/Moon masks applied after simulator optics/readout; no new PSF, scatter, exposure-smearing or detector calculation.",
            "Earth is a sphere at the predictor's configured radius; no atmospheric absorption or refraction profile.",
            "No prediction of actual attitude/roll; source helioprojective orientation assumed.",
        ],
    }, indent=2, allow_nan=False) + "\n")
    print(f"Wrote movies to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

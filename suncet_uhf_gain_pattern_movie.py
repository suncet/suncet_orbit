#!/usr/bin/env python3
"""Render the rotating SunCET UHF gain-pattern movie for two example passes."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.request import urlopen

import astropy.units as u
from astropy.utils import iers
import numpy as np
from scipy.interpolate import PchipInterpolator
import websocket

from suncet_ground_station_angles import Attitude, station_geometry, unit
from suncet_ltan_to_raan import DEFAULT_LAUNCH
from suncet_lunar_solar_eclipses import PlanningOrbit, utc_time
from suncet_uhf_link_passes import (
    DAY_S, GROUND_RADIOS, contact_samples, directional_gain, find_contacts, load_gain_pattern,
)


ASSETS = Path(__file__).resolve().parent / "assets"
EXAMPLE_PASSES = {"fairbanks": "Fairbanks-0003", "boulder": "Boulder-0018"}
CHAPTER_LENGTHS = {"fairbanks": 34, "boulder": 23}
ROLL_SECONDS = 60
MAC_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def example_tracks(pattern):
    """Generate the same body-frame station tracks used in the interactive view."""
    epoch = utc_time(DEFAULT_LAUNCH["epoch"])
    orbit = PlanningOrbit(epoch)
    tracks = {}
    for radio in GROUND_RADIOS:
        key = radio.station.name.lower()
        pass_id = EXAMPLE_PASSES[key]
        contacts = find_contacts(orbit, radio.station, epoch, 5 * DAY_S, mask_deg=5)
        contact = next((item for item in contacts if item.pass_id == pass_id and not item.clipped), None)
        if contact is None:
            raise ValueError(f"Example contact {pass_id} is missing from the five-day planning window.")

        start, end = epoch + np.array([contact.aos, contact.los]) * u.s
        attitude = Attitude(orbit, start, end, ROLL_SECONDS)
        centers = contact.aos + np.asarray(attitude.flips)
        elapsed = contact_samples(contact, centers, ROLL_SECONDS) - contact.aos
        times = start + elapsed * u.s
        _, _, los, _ = station_geometry(orbit, radio.station, times)
        x, y, z, rolling, _ = attitude.evaluate(times)
        body = np.einsum("nij,nj->ni", np.stack([x, y, z], axis=1), unit(los))
        rhcp = directional_gain(pattern, body)[:, 0]
        roll_times = elapsed[rolling]
        tracks[key] = {
            "worst": float(elapsed[np.argmin(rhcp)]),
            "roll": [float(roll_times[0]), float(roll_times[-1])] if len(roll_times) else None,
            "samples": [[float(t), *v, bool(r)] for t, v, r in zip(elapsed, body.round(8), rolling)],
        }
    return tracks


def write_render_page(path, pattern, tracks):
    # The viewer uses a 2-degree grid for lookups and a 4-degree mesh for display.
    theta, phi = np.meshgrid(np.arange(0, 181, 2), np.arange(0, 360, 2), indexing="ij")
    power = pattern(np.column_stack((theta.ravel(), phi.ravel()))).ravel()
    # Store millidecibels. Null represents a component with zero power.
    gain = [round(10000 * np.log10(value)) if value > 0 else None for value in power]
    data = json.dumps({"gain": gain, "tracks": tracks}, separators=(",", ":"), allow_nan=False)
    viewer = (ASSETS / "uhf_gain_pattern.html").read_text(encoding="utf-8")
    renderer = (ASSETS / "uhf_gain_pattern_movie.js").read_text(encoding="utf-8")
    fragment = viewer.replace("__PATTERN_DATA__", data).replace("/* __MOVIE_RENDERER__ */", renderer)
    header = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>SunCET UHF gain movie render</title>
<style>
:root {
  color-scheme:dark;
  --background:#181818; --foreground:#fff; --muted-foreground:#aaa; --border:#555;
  --viz-series-1:#83c3ff; --viz-series-2:#f59a56;
  --font-size-base:20px;
  font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;
}
body { margin:0; }
#suncet-uhf-pattern { position:absolute; left:-2000px; width:1280px; }
#suncet-uhf-pattern .pattern-views { display:flex; }
#suncet-uhf-pattern .pattern-views figure { width:384px; }
#suncet-uhf-pattern .pattern-views canvas { width:384px; height:375px; }
</style>
<body>
"""
    path.write_text(header + fragment + "</body></html>", encoding="utf-8")


def frame_plan(tracks, fps):
    """Slow pass playback around the Fairbanks minimum and the Boulder roll."""
    fairbanks, boulder = tracks["fairbanks"], tracks["boulder"]
    fairbanks_end = fairbanks["samples"][-1][0]
    before = fairbanks["worst"]
    after = fairbanks_end - before
    # The knots are normalized to the two sides of the reference fade at 240 s.
    fairbanks_times = [0, before * 184 / 240, before * 229 / 240, before * 239 / 240,
                       before + after / 312, before + after * 15 / 312,
                       before + after * 97 / 312, fairbanks_end]
    boulder_end = boulder["samples"][-1][0]
    if boulder["roll"]:
        first, last = boulder["roll"]
        boulder_times = [0, first, last, last + (boulder_end - last) * .185,
                         last + (boulder_end - last) * .60, boulder_end]
    else:
        boulder_times = np.linspace(0, boulder_end, 6)

    scenes = (
        ("fairbanks", [0, 9, 13, 16, 19, 22, 26, 34], fairbanks_times, -26, 113, 16, 9),
        ("boulder", [0, 2, 8, 12, 17, 23], boulder_times, 160, -115, 30, 8),
    )
    for key, video, elapsed, yaw, sweep, pitch, lift in scenes:
        fraction = np.linspace(0, 1, CHAPTER_LENGTHS[key] * fps)
        timestamps = PchipInterpolator(video, elapsed)(fraction * video[-1])
        for timestamp, progress in zip(np.clip(timestamps, 0, elapsed[-1]), fraction):
            yield [key, round(timestamp, 6), yaw + sweep * progress, pitch + lift * np.sin(np.pi * progress)]


def chrome_target(chrome, profile):
    """Wait for the loopback DevTools endpoint on Chrome's assigned port."""
    marker = profile / "DevToolsActivePort"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if chrome.poll() is not None:
            raise RuntimeError(f"Chrome exited before opening the render page ({chrome.returncode}).")
        if marker.exists():
            try:
                port = marker.read_text().splitlines()[0]
                with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as response:
                    targets = json.load(response)
                for target in targets:
                    if target["type"] == "page":
                        return target["webSocketDebuggerUrl"]
            except (OSError, IndexError):
                pass
        time.sleep(.1)
    raise RuntimeError("Chrome did not expose a local DevTools page.")


def evaluator(connection):
    sequence = 0

    def evaluate(expression):
        nonlocal sequence
        sequence += 1
        connection.send(json.dumps({"id": sequence, "method": "Runtime.evaluate", "params": {
            "expression": expression, "returnByValue": True,
        }}))
        while True:
            message = json.loads(connection.recv())
            if message.get("id") == sequence:
                break
        result = message.get("result", {})
        if "error" in message or "exceptionDetails" in result:
            raise RuntimeError(f"Movie renderer failed: {message}")
        return result["result"].get("value")

    return evaluate


def encode_movie(evaluate, ffmpeg, temporary, tracks, fps):
    metadata = temporary / "chapters.txt"
    chapters = [";FFMETADATA1", "title=SunCET UHF antenna pattern"]
    elapsed = 0
    for name, length in CHAPTER_LENGTHS.items():
        chapters.extend(("[CHAPTER]", "TIMEBASE=1/1000", f"START={elapsed * 1000}",
                         f"END={(elapsed + length) * 1000}", f"title={name.title()} pass"))
        elapsed += length
    metadata.write_text("\n".join(chapters) + "\n", encoding="utf-8")

    output = temporary / "movie.mp4"
    args = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "image2pipe", "-framerate", str(fps), "-vcodec", "mjpeg", "-i", "pipe:0",
        "-f", "ffmetadata", "-i", str(metadata), "-map", "0:v", "-map_metadata", "1", "-map_chapters", "1",
        "-vf", "scale=in_range=pc:out_range=tv", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    frames = list(frame_plan(tracks, fps))
    encoder = subprocess.Popen(args, stdin=subprocess.PIPE)
    try:
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            js = json.dumps(batch, separators=(",", ":"), allow_nan=False)
            for jpeg in evaluate(f"({js}).map(frame => window.movieFrame(...frame))"):
                encoder.stdin.write(base64.b64decode(jpeg))
            if i % 240 == 0:
                print(f"Rendered {i + len(batch)}/{len(frames)} frames", flush=True)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("FFmpeg could not encode the movie.")
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    return output


def render_movie(chrome, ffmpeg, output, pattern, tracks, fps):
    with tempfile.TemporaryDirectory(prefix="suncet-uhf-gain-") as tmp:
        temporary = Path(tmp)
        page = temporary / "render.html"
        write_render_page(page, pattern, tracks)
        profile = temporary / "chrome"
        with (temporary / "chrome.log").open("w", encoding="utf-8") as log:
            browser = subprocess.Popen([
                chrome, "--headless", "--disable-gpu", "--disable-background-networking", "--no-first-run",
                "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
                "--force-device-scale-factor=1", "--window-size=1280,720", f"--user-data-dir={profile}",
                page.as_uri(),
            ], stdout=log, stderr=log)
            try:
                target = chrome_target(browser, profile)
                connection = websocket.create_connection(target, timeout=45, suppress_origin=True)
                try:
                    evaluate = evaluator(connection)
                    for _ in range(40):
                        if evaluate("typeof window.movieFrame") == "function":
                            break
                        time.sleep(.1)
                    else:
                        raise RuntimeError("The gain-pattern render page did not initialize.")
                    movie = encode_movie(evaluate, ffmpeg, temporary, tracks, fps)
                finally:
                    connection.close()
            finally:
                if browser.poll() is None:
                    browser.terminate()
                    browser.wait()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a completed encode, keeping an existing movie intact on error.
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".gain-movie-", delete=False) as staging:
            staged = Path(staging.name)
        try:
            shutil.copyfile(movie, staged)
            staged.replace(output)
        finally:
            staged.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", type=Path, required=True, help="Body-frame NEC radiation pattern export.")
    parser.add_argument("--chrome", help="Chrome or Chromium executable, if it is not found automatically.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/uhf_link_passes/movies/suncet-uhf-gain-pattern.mp4"))
    args = parser.parse_args(argv)
    if args.fps < 1:
        parser.error("--fps must be at least 1.")
    chrome = args.chrome or shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if chrome is None and MAC_CHROME.is_file():
        chrome = str(MAC_CHROME)
    if not chrome or not shutil.which(chrome):
        parser.error("Chrome or Chromium is required. Pass its executable with --chrome.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("FFmpeg is required on PATH.")

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "warn"
    pattern = load_gain_pattern(args.pattern)
    tracks = example_tracks(pattern)
    render_movie(chrome, ffmpeg, args.output, pattern, tracks, args.fps)
    print(args.output)


if __name__ == "__main__":
    main()

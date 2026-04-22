"""
Render a one-year SunCET orbit movie: orthographic 3D view toward Earth from the
Sun side of the Earth–Sun line (camera on +Z, focal point at Earth).

**Default camera** (**``--camera-frame sun_locked``**): +Z tracks the instantaneous
Earth→Sun direction each frame; **+Y** is J2000 ecliptic north projected ⊥ +Z.

Plain two-body Kepler keeps the orbital plane fixed in GCRS while the Sun vector
moves ~360°/year, which makes an SSO ring look like it tumbles. This script
applies **J2 secular RAAN precession** (on by default; disable with
``--no-j2-raan``): after each Kepler step, positions and the closed orbit ring are
rotated about GCRS +Z by ΔΩ ≈ Ω̇Δt. For this 510 km / ~97.4° mission, |Ω̇| is
~360°/year, so the ring stays nearly steady in the Sun line view while **Earth’s
texture** still advances with GCRS→ITRS (seasons / subsolar latitude).
``--no-j2-raan`` reproduces the raw Kepler + sun-locked tumble.

**``--camera-frame inertial_epoch``** freezes the plot basis at the first frame
(Sun line at t₀); the physical Sun direction then drifts in that frame.

Rendering uses **PyVista** (VTK) by default. Use ``--backend matplotlib`` for the
legacy mplot3d path (slower, approximate camera).

Optional **equirectangular Earth texture**: first ``--earth-texture`` path if given,
else ``~/.cache/suncet_orbit/earth_equirect_21k.jpg`` (Visible Earth **21600×10800**
JPEG on disk; loaded in memory resized to ``--texture-max-width``, default 8192).
Surface **mesh** resolution also limits sharpness (see ``GLOBE_MESH_U`` / ``V``).
Texture UVs use **GCRS→ITRS** at each frame time so geographic lon/lat match the
equirectangular map (the globe rotates correctly in inertial space; not full ITRF fidelity).
Falls back to wireframe if loading fails.

Uses poliastro two-body propagation plus optional J2 nodal drift (not full STK).
Defaults match the sun-vector orbit plan: 510 km SSO, i ~ 97.44°, LTAN 18:00,
epoch 2026-12-15T18:00:00Z (ussf_178), ~1 frame per day, 24 fps MP4.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
    
import numpy as np
import astropy.units as u
from astropy.coordinates import GCRS, SkyCoord, get_sun
from astropy.coordinates.builtin_frames import GeocentricMeanEcliptic, ITRS
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

try:
    import pyvista as pv  # type: ignore
except ImportError:
    pv = None  # type: ignore

# NASA 21600×10800 equirect exceeds PIL’s default “decompression bomb” threshold.
Image.MAX_IMAGE_PIXELS = None


from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from suncet_ltan_to_raan import LAUNCHES, raan_for_launch


# --- Scenario (plan defaults) ---
LAUNCH_NAME = "ussf_178"
ALTITUDE_KM = 510.0
INCLINATION_DEG = 97.44
N_FRAMES_DEFAULT = 366  # ~daily over one year
FPS_DEFAULT = 24
# ~16:9 figure; at dpi=128 this is roughly 720p-class pixel dimensions.
FIGSIZE = (12.8, 7.2)

# NASA Visible Earth — 21600×10800 equirect (same series as 5.4k; much sharper source).
EARTH_TEXTURE_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/"
    "world.topo.bathy.200412.3x21600x10800.jpg"
)
# Default cap when loading texture (float32); full NASA asset is 21600×10800.
TEXTURE_MAX_WIDTH = 8192
# Sphere mesh segments; higher = sharper globe (texture sampled per quad cell).
GLOBE_MESH_U = 128
GLOBE_MESH_V = 65

# Tighter bounds → camera closer (smaller fraction = closer).
MARGIN_FACTOR = 0.52

# Matplotlib mplot3d: rotate view so plot +Y (Earth north) maps to screen up (see view_init).
VIEW_AZIM_DEG = 90.0

SAVE_DPI_DEFAULT = 128


def _sun_unit_gkms(obstime: Time) -> np.ndarray:
    """Earth→Sun unit vector in GCRS (km direction; unitless)."""
    sun = get_sun(obstime).transform_to(GCRS(obstime=obstime))
    xyz = sun.cartesian.xyz.to(u.km).value
    n = np.linalg.norm(xyz)
    if n == 0:
        raise RuntimeError("Zero sun vector")
    return xyz / n


def _ecliptic_north_unit_gcrs(obstime: Time) -> np.ndarray:
    """J2000 ecliptic north pole as a unit vector in GCRS at ``obstime``."""
    c = SkyCoord(
        0 * u.deg,
        90 * u.deg,
        1 * u.au,
        frame=GeocentricMeanEcliptic(obstime=obstime, equinox="J2000"),
    )
    c_g = c.transform_to(GCRS(obstime=obstime))
    xyz = c_g.cartesian.xyz.to(u.km).value
    n = np.linalg.norm(xyz)
    if n == 0:
        raise RuntimeError("Zero ecliptic north vector")
    return xyz / n


def camera_basis_sun_line_ecliptic_north_up(
    obstime: Time, *, for_matplotlib: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Right-handed frame: +Z = Earth→Sun at ``obstime``; +Y = ecliptic north projected ⊥ +Z.
    If ``for_matplotlib``, flip ex/ey so mplot3d screen-up matches globe north-up.
    PyVista uses ``for_matplotlib=False`` and sets VTK ``view_up`` = ey directly.
    """
    ez = _sun_unit_gkms(obstime).astype(np.float64)
    ez = ez / np.linalg.norm(ez)
    g = _ecliptic_north_unit_gcrs(obstime)
    ey = g - np.dot(g, ez) * ez
    norm = np.linalg.norm(ey)
    if norm < 1e-10:
        ref = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(ez, ref)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0])
        ey = np.cross(ref, ez)
        norm = np.linalg.norm(ey)
    ey = ey / norm
    ex = np.cross(ey, ez)
    ex = ex / np.linalg.norm(ex)
    if for_matplotlib:
        ex, ey = -ex, -ey
    return ex, ey, ez


def _earth_sphere_wireframe(radius_km: float, n_u: int = 24, n_v: int = 12):
    u_lin = np.linspace(0, 2 * np.pi, n_u)
    v_lin = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u_lin, v_lin)
    x = radius_km * np.cos(uu) * np.sin(vv)
    y = radius_km * np.sin(uu) * np.sin(vv)
    z = radius_km * np.cos(vv)
    return x, y, z


def _default_texture_cache_path() -> Path:
    return Path.home() / ".cache" / "suncet_orbit" / "earth_equirect_21k.jpg"


def _load_texture_image(path: Path, max_width: int | None = None) -> np.ndarray:
    """RGB float32 array in [0, 1], shape (H, W, 3). Large NASA JPEGs are resized with PIL."""
    cap = TEXTURE_MAX_WIDTH if max_width is None else max_width
    with Image.open(path) as pil:
        pil = pil.convert("RGB")
        w, h = pil.size
        if w > cap:
            nh = int(round(h * cap / w))
            pil = pil.resize((cap, nh), Image.Resampling.LANCZOS)
        img = np.asarray(pil, dtype=np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return np.clip(img, 0.0, 1.0)


def load_or_fetch_earth_texture(
    custom_path: Path | None,
    texture_max_width: int | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """
    Load equirectangular Earth texture (RGB). Tries custom path, then cache,
    then downloads NASA JPEG to ~/.cache/suncet_orbit/earth_equirect_21k.jpg.
    Returns (None, None) on failure (caller falls back to wireframe).
    """
    candidates: list[Path] = []
    if custom_path is not None:
        candidates.append(custom_path)
    candidates.append(_default_texture_cache_path())

    for p in candidates:
        if p.is_file():
            try:
                arr = _load_texture_image(p, texture_max_width)
                h, w = arr.shape[:2]
                print(
                    f"Earth texture: {p} → loaded {w}×{h} "
                    f"(NASA source is 21600×10800; on-disk cache is the downloaded JPEG)",
                    file=sys.stderr,
                )
                return arr, str(p)
            except OSError:
                continue

    cache = _default_texture_cache_path()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(EARTH_TEXTURE_URL, cache)
        arr = _load_texture_image(cache, texture_max_width)
        h, w = arr.shape[:2]
        cap = texture_max_width if texture_max_width is not None else TEXTURE_MAX_WIDTH
        print(
            f"Earth texture: downloaded to {cache} → loaded {w}×{h} "
            f"(source URL is 21600×10800; resized to max width {cap})",
            file=sys.stderr,
        )
        return arr, str(cache)
    except Exception:
        return None, None


def _gcrs_to_itrs_rotation_matrix(obstime: Time) -> np.ndarray:
    """Rotation R (3,3) with r_itrs = R @ r_gcrs (unit directions; orthogonal)."""
    cols = []
    for i in range(3):
        v = np.zeros(3)
        v[i] = 1.0
        sc = SkyCoord(
            v[0] * u.km,
            v[1] * u.km,
            v[2] * u.km,
            representation_type="cartesian",
            frame=GCRS(obstime=obstime),
        )
        it = sc.transform_to(ITRS(obstime=obstime))
        raw = it.cartesian.xyz.to(u.km).value
        if raw.ndim == 2 and raw.shape[0] == 3:
            w = raw[:, 0]
        else:
            w = np.asarray(raw).ravel()
        cols.append(w / np.linalg.norm(w))
    return np.column_stack(cols)


def _gcrs_mesh_to_tex_uv_equirect(
    flat_gcrs: np.ndarray,
    obstime: Time,
    xs_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Geographic equirectangular u,v in [0,1]. ``flat_gcrs`` is (N,3) vertex positions
    in GCRS (km); lon/lat from ITRS at ``obstime`` via a fast rotation matrix.
    """
    r_mat = _gcrs_to_itrs_rotation_matrix(obstime)
    xyz = (r_mat @ flat_gcrs.T).T
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    uu = xyz / norm
    lat = np.arcsin(np.clip(uu[:, 2], -1.0, 1.0))
    lon = np.arctan2(uu[:, 1], uu[:, 0])
    u_tex = np.mod(lon + np.pi, 2 * np.pi) / (2 * np.pi)
    v_tex = (np.pi / 2 - lat) / np.pi
    sh = xs_shape
    return (
        u_tex.reshape(sh).astype(np.float64),
        np.clip(v_tex.reshape(sh), 0.0, 1.0).astype(np.float64),
    )


def _sample_texture_bilinear(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear RGB sample; u,v in [0,1]; u wraps on longitude (columns only)."""
    tex = np.asarray(tex, dtype=np.float32)
    if tex.shape[-1] == 1:
        tex = np.repeat(tex, 3, axis=-1)
    H, W = tex.shape[:2]
    u = np.clip(np.asarray(u, dtype=np.float64).squeeze(), 0.0, 1.0)
    v = np.clip(np.asarray(v, dtype=np.float64).squeeze(), 0.0, 1.0)
    col = u * (W - 1)
    row = v * (H - 1)
    c0 = (np.floor(col).astype(np.int64) % W).squeeze()
    c1 = (c0 + 1) % W
    r0 = np.clip(np.floor(row).astype(np.int64), 0, H - 2).squeeze()
    r1 = r0 + 1
    tc = np.squeeze(col - np.floor(col))
    tr = np.squeeze(row - np.floor(row))
    out = np.zeros(u.shape + (3,), dtype=np.float32)
    for k in range(3):
        ch = tex[..., k]
        a = np.squeeze(ch[r0, c0])
        b = np.squeeze(ch[r0, c1])
        c = np.squeeze(ch[r1, c0])
        d = np.squeeze(ch[r1, c1])
        out[..., k] = (
            (1 - tc) * (1 - tr) * a
            + tc * (1 - tr) * b
            + (1 - tc) * tr * c
            + tc * tr * d
        )
    return out


def earth_facecolors_for_time(
    tex: np.ndarray,
    flat_gcrs: np.ndarray,
    xs_shape: tuple[int, ...],
    obstime: Time,
) -> np.ndarray:
    """Facecolors (n_v-1, n_u-1, 3) for ``xs`` from ``_earth_sphere_wireframe`` (shape n_v×n_u)."""
    u_tex, v_tex = _gcrs_mesh_to_tex_uv_equirect(flat_gcrs, obstime, xs_shape)
    vc = _sample_texture_bilinear(tex, u_tex, v_tex)
    return 0.25 * (
        vc[:-1, :-1] + vc[1:, :-1] + vc[:-1, 1:] + vc[1:, 1:]
    )


def _vtk_structured_cell_rgb_from_facecolors(fc: np.ndarray) -> np.ndarray:
    """
    ``fc`` is (n_v-1, n_u-1, 3) matching matplotlib / numpy mesh rows=lat, cols=lon.
    VTK StructuredGrid cells are ordered with the first index (v) varying fastest;
    C-order ``fc.ravel()`` uses the second index (u) fastest — transpose before reshape.
    """
    return np.clip(np.ascontiguousarray(fc.transpose(1, 0, 2)).reshape(-1, 3), 0.0, 1.0)


def build_initial_orbit() -> Orbit:
    raan = raan_for_launch(LAUNCH_NAME)
    meta = LAUNCHES[LAUNCH_NAME]
    epoch = Time(meta["epoch"], format="isot", scale="utc")
    a = Earth.R + ALTITUDE_KM * u.km
    inc = INCLINATION_DEG * u.deg
    return Orbit.from_classical(
        Earth,
        a,
        0.0 * u.one,
        inc,
        raan,
        0.0 * u.deg,
        0.0 * u.deg,
        epoch=epoch,
    )


def _j2_raan_dot_rad_s(orbit: Orbit) -> float:
    """Secular RAAN rate from J₂ about Earth’s mean equator (GCRS z); rad/s."""
    body = orbit.attractor
    a_m = orbit.a.to(u.m).value
    ecc = float(orbit.ecc)
    inc = orbit.inc.to(u.rad).value
    mu = body.k.to(u.m**3 / u.s**2).value
    r_e = body.R.to(u.m).value
    j2 = float(body.J2)
    p = a_m * (1.0 - ecc**2)
    n = np.sqrt(mu / (a_m**3))
    return float(-1.5 * n * j2 * (r_e**2 / p**2) * np.cos(inc))


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def precompute_trajectory(
    orbit0: Orbit,
    n_frames: int,
    *,
    apply_j2_raan: bool = True,
) -> tuple[np.ndarray, Time]:
    """Positions (n_frames, 3) in km GCRS; times at each frame from epoch over 1 year."""
    epoch = orbit0.epoch
    t_end = epoch + 1 * u.year
    times = Time(np.linspace(epoch.jd, t_end.jd, n_frames), format="jd", scale=epoch.scale)
    positions = np.empty((n_frames, 3))
    omega_dot = _j2_raan_dot_rad_s(orbit0) if apply_j2_raan else 0.0
    for i in range(n_frames):
        dt = times[i] - epoch
        orb = orbit0.propagate(dt)
        r = np.asarray(orb.r.to(u.km).value, dtype=np.float64).reshape(3)
        if apply_j2_raan:
            dt_s = (times[i] - epoch).to(u.s).value
            r = _rotation_z(-omega_dot * dt_s) @ r
        positions[i, :] = r
    return positions, times


def one_revolution_ring_gkms(orbit0: Orbit, n_points: int = 200) -> np.ndarray:
    """Sample one full revolution (GCRS, km) for drawing a closed orbit ring."""
    samples = orbit0.sample(n_points)
    return np.column_stack(
        [
            samples.x.to(u.km).value,
            samples.y.to(u.km).value,
            samples.z.to(u.km).value,
        ]
    )


def run_movie_pyvista(
    output_path: str,
    n_frames: int,
    fps: float,
    earth_texture_path: Path | None = None,
    use_earth_texture: bool = True,
    margin_factor: float = MARGIN_FACTOR,
    save_dpi: float = SAVE_DPI_DEFAULT,
    texture_max_width: int | None = None,
    camera_frame: str = "sun_locked",
    apply_j2_raan: bool = True,
) -> None:
    if pv is None:
        raise RuntimeError(
            "PyVista is not installed. Install with: pip install pyvista"
        )
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    orbit0 = build_initial_orbit()
    positions_km, times = precompute_trajectory(
        orbit0, n_frames, apply_j2_raan=apply_j2_raan
    )
    r_earth = Earth.R.to(u.km).value
    ring_base_gcrs = one_revolution_ring_gkms(orbit0)
    omega_dot = _j2_raan_dot_rad_s(orbit0) if apply_j2_raan else 0.0
    if apply_j2_raan:
        deg_yr = omega_dot * (180.0 / np.pi) * 86400.0 * 365.25
        print(
            f"J2 RAAN precession: |Ω̇| ≈ {deg_yr:.1f}°/yr (nodal drift after each Kepler step).",
            file=sys.stderr,
        )

    earth_tex: np.ndarray | None = None
    if use_earth_texture:
        earth_tex, _tex_src = load_or_fetch_earth_texture(
            earth_texture_path, texture_max_width
        )
        if earth_tex is None:
            print(
                "Earth texture unavailable (offline or download failed); using wireframe.",
                file=sys.stderr,
            )

    flat_gcrs: np.ndarray | None = None
    xs = ys = zs = None
    if earth_tex is not None:
        xs, ys, zs = _earth_sphere_wireframe(r_earth, n_u=GLOBE_MESH_U, n_v=GLOBE_MESH_V)
        flat_gcrs = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)

    margin = margin_factor * (np.max(np.linalg.norm(positions_km, axis=1)) + r_earth)
    # Even pixel dimensions so libx264 + yuv420p (ffmpeg) accept the frame size.
    px_w = max(16, int(FIGSIZE[0] * save_dpi))
    px_h = max(16, int(FIGSIZE[1] * save_dpi))
    px_w = (px_w // 2) * 2
    px_h = (px_h // 2) * 2

    basis_time = times[0] if camera_frame == "inertial_epoch" else None

    def build_frame(plotter: "pv.Plotter", frame: int) -> None:
        plotter.clear()
        plotter.set_background("black")
        obstime = times[frame]
        t_basis = basis_time if basis_time is not None else obstime
        ex, ey, ez = camera_basis_sun_line_ecliptic_north_up(t_basis, for_matplotlib=False)
        r_plot = np.vstack([ex, ey, ez])

        if earth_tex is not None and flat_gcrs is not None and xs is not None:
            trans_e = (r_plot @ flat_gcrs.T).T
            wx = trans_e[:, 0].reshape(xs.shape)
            wy = trans_e[:, 1].reshape(xs.shape)
            wz = trans_e[:, 2].reshape(xs.shape)
            fc = earth_facecolors_for_time(earth_tex, flat_gcrs, xs.shape, obstime)
            grid = pv.StructuredGrid(wx, wy, wz)
            grid.cell_data["rgb"] = _vtk_structured_cell_rgb_from_facecolors(fc)
            plotter.add_mesh(
                grid,
                scalars="rgb",
                rgb=True,
                lighting=False,
                show_scalar_bar=False,
            )
        else:
            xw, yw, zw = _earth_sphere_wireframe(r_earth, 32, 16)
            flat = np.stack([xw.ravel(), yw.ravel(), zw.ravel()], axis=1)
            trans = (r_plot @ flat.T).T
            wx_w = trans[:, 0].reshape(xw.shape)
            wy_w = trans[:, 1].reshape(yw.shape)
            wz_w = trans[:, 2].reshape(zw.shape)
            wf = pv.StructuredGrid(wx_w, wy_w, wz_w)
            plotter.add_mesh(
                wf,
                style="wireframe",
                color="#2a4a6a",
                line_width=1,
                opacity=0.9,
            )

        dt_s = (obstime - orbit0.epoch).to(u.s).value
        ring_gkms = (_rotation_z(-omega_dot * dt_s) @ ring_base_gcrs.T).T
        ring_fix = (r_plot @ ring_gkms.T).T
        plotter.add_mesh(
            pv.lines_from_points(ring_fix, close=True),
            color="gray",
            line_width=2,
            render_lines_as_tubes=False,
        )

        trail_km = positions_km[: frame + 1]
        trail = (r_plot @ trail_km.T).T
        if len(trail) > 1:
            plotter.add_mesh(
                pv.lines_from_points(trail, close=False),
                color="#ff7f0e",
                line_width=2,
            )

        sc = r_plot @ positions_km[frame]
        sat = pv.Sphere(radius=r_earth * 0.02, theta_resolution=16, phi_resolution=16)
        sat.translate(sc, inplace=True)
        plotter.add_mesh(sat, color="#d62728", lighting=False)

        plotter.add_text(
            f"SunCET   •   {obstime.isot} UTC",
            position="upper_edge",
            font_size=14,
            color=(0.85, 0.85, 0.85),
        )

        cam = plotter.camera
        cam.parallel_projection = True
        cam.parallel_scale = margin
        cam.position = (0.0, 0.0, float(margin))
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.up = tuple(ey.astype(float))

    out_lower = output_path.lower()
    use_gif = out_lower.endswith(".gif")
    if not use_gif and shutil.which("ffmpeg") is None:
        alt = os.path.splitext(output_path)[0] + ".gif"
        print(
            f"ffmpeg not on PATH; writing GIF instead of video: {alt}",
            file=sys.stderr,
        )
        output_path = alt
        use_gif = True

    plotter = pv.Plotter(off_screen=True, window_size=(px_w, px_h))

    try:
        if use_gif:
            plotter.open_gif(output_path, fps=fps)
            for frame in range(n_frames):
                build_frame(plotter, frame)
                plotter.write_frame()
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                for frame in range(n_frames):
                    build_frame(plotter, frame)
                    plotter.screenshot(
                        os.path.join(tmpdir, f"frame_{frame:06d}.png"),
                        transparent_background=False,
                    )
                ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
                subprocess.run(
                    [
                        ffmpeg_cmd,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        str(fps),
                        "-i",
                        os.path.join(tmpdir, "frame_%06d.png"),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        output_path,
                    ],
                    check=True,
                )
    except Exception as e:
        alt = os.path.splitext(output_path)[0] + ".gif"
        print(f"Video save failed ({e}); writing GIF to {alt}", file=sys.stderr)
        plotter.close()
        plotter = pv.Plotter(off_screen=True, window_size=(px_w, px_h))
        plotter.open_gif(alt, fps=fps)
        for frame in range(n_frames):
            build_frame(plotter, frame)
            plotter.write_frame()
    finally:
        plotter.close()


def run_movie_matplotlib(
    output_path: str,
    n_frames: int,
    fps: float,
    earth_texture_path: Path | None = None,
    use_earth_texture: bool = True,
    margin_factor: float = MARGIN_FACTOR,
    save_dpi: float = SAVE_DPI_DEFAULT,
    texture_max_width: int | None = None,
    camera_frame: str = "sun_locked",
    apply_j2_raan: bool = True,
) -> None:
    orbit0 = build_initial_orbit()
    positions_km, times = precompute_trajectory(
        orbit0, n_frames, apply_j2_raan=apply_j2_raan
    )
    r_earth = Earth.R.to(u.km).value

    ring_base_gcrs = one_revolution_ring_gkms(orbit0)
    omega_dot = _j2_raan_dot_rad_s(orbit0) if apply_j2_raan else 0.0
    if apply_j2_raan:
        deg_yr = omega_dot * (180.0 / np.pi) * 86400.0 * 365.25
        print(
            f"J2 RAAN precession: |Ω̇| ≈ {deg_yr:.1f}°/yr (nodal drift after each Kepler step).",
            file=sys.stderr,
        )

    earth_tex: np.ndarray | None = None
    if use_earth_texture:
        earth_tex, _tex_src = load_or_fetch_earth_texture(
            earth_texture_path, texture_max_width
        )
        if earth_tex is None:
            print(
                "Earth texture unavailable (offline or download failed); using wireframe.",
                file=sys.stderr,
            )
    # GCRS sphere mesh (texture UVs); plot positions use R(t) per frame.
    flat_gcrs: np.ndarray | None = None
    xs = ys = zs = None
    if earth_tex is not None:
        xs, ys, zs = _earth_sphere_wireframe(r_earth, n_u=GLOBE_MESH_U, n_v=GLOBE_MESH_V)
        flat_gcrs = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)

    fig = plt.figure(figsize=FIGSIZE, facecolor="black")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor("black")

    margin = margin_factor * (np.max(np.linalg.norm(positions_km, axis=1)) + r_earth)

    def update(frame: int):
        ax.clear()
        ax.set_facecolor("black")
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_zlim(-margin, margin)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        ax.grid(False)
        ax.set_axis_off()
        obstime = times[frame]
        t_basis = times[0] if camera_frame == "inertial_epoch" else obstime
        ex, ey, ez = camera_basis_sun_line_ecliptic_north_up(t_basis)
        r_plot = np.vstack([ex, ey, ez])

        if earth_tex is not None and flat_gcrs is not None and xs is not None:
            trans_e = (r_plot @ flat_gcrs.T).T
            wx = trans_e[:, 0].reshape(xs.shape)
            wy = trans_e[:, 1].reshape(xs.shape)
            wz = trans_e[:, 2].reshape(xs.shape)
            fc = earth_facecolors_for_time(earth_tex, flat_gcrs, xs.shape, obstime)
            ax.plot_surface(
                wx,
                wy,
                wz,
                facecolors=fc,
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0,
                antialiased=False,
            )
        else:
            xw, yw, zw = _earth_sphere_wireframe(r_earth, 32, 16)
            flat = np.stack([xw.ravel(), yw.ravel(), zw.ravel()], axis=1)
            trans = (r_plot @ flat.T).T
            wx_w = trans[:, 0].reshape(xw.shape)
            wy_w = trans[:, 1].reshape(yw.shape)
            wz_w = trans[:, 2].reshape(zw.shape)
            ax.plot_wireframe(
                wx_w, wy_w, wz_w, color="#2a4a6a", alpha=0.9, linewidth=0.4
            )

        dt_s = (obstime - orbit0.epoch).to(u.s).value
        ring_gkms = (_rotation_z(-omega_dot * dt_s) @ ring_base_gcrs.T).T
        ring_fix = (r_plot @ ring_gkms.T).T
        ax.plot(
            ring_fix[:, 0],
            ring_fix[:, 1],
            ring_fix[:, 2],
            color="gray",
            linewidth=0.8,
            alpha=0.7,
        )

        trail_km = positions_km[: frame + 1]
        trail = (r_plot @ trail_km.T).T
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="C1", linewidth=1.2)
        sc = r_plot @ positions_km[frame]
        ax.scatter([sc[0]], [sc[1]], [sc[2]], color="C3", s=40, depthshade=False)

        ax.set_title(
            f"SunCET   •   {obstime.isot} UTC",
            fontsize=11,
            color="0.85",
        )
        # Ortho view along +Z; azim rotates so plot +Y (ecliptic north) is screen-up.
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        ax.view_init(elev=90, azim=VIEW_AZIM_DEG)
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_lower = output_path.lower()
    use_gif = out_lower.endswith(".gif")
    if not use_gif and shutil.which("ffmpeg") is None:
        alt = os.path.splitext(output_path)[0] + ".gif"
        print(
            f"ffmpeg not on PATH; writing GIF instead of video: {alt}",
            file=sys.stderr,
        )
        output_path = alt
        use_gif = True

    try:
        if use_gif:
            anim.save(
                output_path, writer=animation.PillowWriter(fps=fps), dpi=save_dpi
            )
        else:
            anim.save(
                output_path,
                writer=animation.FFMpegWriter(
                    fps=fps, metadata={"title": "SunCET sun-vector orbit"}
                ),
                dpi=save_dpi,
            )
    except Exception as e:
        alt = os.path.splitext(output_path)[0] + ".gif"
        print(f"Video save failed ({e}); writing GIF to {alt}", file=sys.stderr)
        anim.save(alt, writer=animation.PillowWriter(fps=fps), dpi=save_dpi)
    finally:
        plt.close(fig)


def run_movie(
    output_path: str,
    n_frames: int,
    fps: float,
    earth_texture_path: Path | None = None,
    use_earth_texture: bool = True,
    margin_factor: float = MARGIN_FACTOR,
    save_dpi: float = SAVE_DPI_DEFAULT,
    texture_max_width: int | None = None,
    backend: str = "pyvista",
    camera_frame: str = "sun_locked",
    apply_j2_raan: bool = True,
) -> None:
    b = backend.lower()
    if b == "pyvista":
        if pv is None:
            print(
                "PyVista not installed; using matplotlib. pip install pyvista for VTK rendering.",
                file=sys.stderr,
            )
            run_movie_matplotlib(
                output_path=output_path,
                n_frames=n_frames,
                fps=fps,
                earth_texture_path=earth_texture_path,
                use_earth_texture=use_earth_texture,
                margin_factor=margin_factor,
                save_dpi=save_dpi,
                texture_max_width=texture_max_width,
                camera_frame=camera_frame,
                apply_j2_raan=apply_j2_raan,
            )
            return
        run_movie_pyvista(
            output_path=output_path,
            n_frames=n_frames,
            fps=fps,
            earth_texture_path=earth_texture_path,
            use_earth_texture=use_earth_texture,
            margin_factor=margin_factor,
            save_dpi=save_dpi,
            texture_max_width=texture_max_width,
            camera_frame=camera_frame,
            apply_j2_raan=apply_j2_raan,
        )
    elif b == "matplotlib":
        run_movie_matplotlib(
            output_path=output_path,
            n_frames=n_frames,
            fps=fps,
            earth_texture_path=earth_texture_path,
            use_earth_texture=use_earth_texture,
            margin_factor=margin_factor,
            save_dpi=save_dpi,
            texture_max_width=texture_max_width,
            camera_frame=camera_frame,
            apply_j2_raan=apply_j2_raan,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}; use 'pyvista' or 'matplotlib'.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        default="suncet_sun_vector_orbit_movie.mp4",
        help="Output video path (.mp4 preferred)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=N_FRAMES_DEFAULT,
        help=f"Number of animation frames (default {N_FRAMES_DEFAULT})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=FPS_DEFAULT,
        help=f"Frames per second (default {FPS_DEFAULT})",
    )
    p.add_argument(
        "--earth-texture",
        type=Path,
        default=None,
        help="Optional path to equirectangular Earth image (RGB). Else cache or download.",
    )
    p.add_argument(
        "--no-earth-texture",
        action="store_true",
        help="Use plain wireframe sphere instead of image texture.",
    )
    p.add_argument(
        "--margin-factor",
        type=float,
        default=MARGIN_FACTOR,
        help=f"Tighter value = closer camera (default {MARGIN_FACTOR})",
    )
    p.add_argument(
        "--dpi",
        type=float,
        default=SAVE_DPI_DEFAULT,
        help=(
            f"Figure DPI when saving video (default {SAVE_DPI_DEFAULT}; "
            f"with default figsize ≈ {int(FIGSIZE[0] * SAVE_DPI_DEFAULT)}×"
            f"{int(FIGSIZE[1] * SAVE_DPI_DEFAULT)} px)"
        ),
    )
    p.add_argument(
        "--texture-max-width",
        type=int,
        default=None,
        help=(
            f"Max equirectangular width in pixels after load (default {TEXTURE_MAX_WIDTH}; "
            "NASA file is 21600×10800). Higher = sharper but more RAM."
        ),
    )
    p.add_argument(
        "--backend",
        choices=("pyvista", "matplotlib"),
        default="pyvista",
        help=(
            "Rendering backend: PyVista (VTK, orthographic camera) or legacy matplotlib mplot3d."
        ),
    )
    p.add_argument(
        "--camera-frame",
        choices=("sun_locked", "inertial_epoch"),
        default="sun_locked",
        help=(
            "sun_locked: +Z tracks Earth→Sun each frame (default; use with J2 RAAN for SSO). "
            "inertial_epoch: lock plot axes to the first-frame Sun line (Sun drifts in-frame)."
        ),
    )
    p.add_argument(
        "--no-j2-raan",
        action="store_true",
        help=(
            "Disable J2 secular nodal precession after Kepler (sun_locked then shows a fixed "
            "Kepler plane tumbling vs the Sun ~360°/yr)."
        ),
    )
    args = p.parse_args()

    run_movie(
        output_path=args.output,
        n_frames=max(2, args.frames),
        fps=float(args.fps),
        earth_texture_path=args.earth_texture,
        use_earth_texture=not args.no_earth_texture,
        margin_factor=float(args.margin_factor),
        save_dpi=float(args.dpi),
        texture_max_width=args.texture_max_width,
        backend=args.backend,
        camera_frame=args.camera_frame,
        apply_j2_raan=not args.no_j2_raan,
    )


if __name__ == "__main__":
    main()

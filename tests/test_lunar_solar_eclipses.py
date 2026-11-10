from pathlib import Path
import json
import sys

import astropy.units as u
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time
from astropy.utils import iers
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suncet_lunar_solar_eclipses import (
    EclipseGeometry, PlanningOrbit, TLEOrbit, disk_overlap_fraction, extremum,
    main, negative_intervals, predict, separation, utc_time,
)
from suncet_ltan_to_raan import DEFAULT_LAUNCH_NAME, LAUNCHES, raan_for_launch

iers.conf.auto_download = False
iers.conf.auto_max_age = None
iers.conf.iers_degraded_accuracy = "warn"


def test_model_and_predictor_cli_use_mission_orbit_targets(tmp_path):
    epoch = utc_time(LAUNCHES[DEFAULT_LAUNCH_NAME]["epoch"])
    orbit = PlanningOrbit(epoch)
    assert orbit.ltan_hours == 18
    assert orbit.raan_deg == pytest.approx(
        raan_for_launch(DEFAULT_LAUNCH_NAME).to_value(u.deg),
    )
    assert main([
        "--end", (epoch+60*u.s).isot, "--ephemeris", "builtin",
        "--no-plot", "--output-dir", str(tmp_path),
    ]) == 0
    summary = json.loads((tmp_path/"summary.json").read_text())
    settings = summary["orbit"]
    assert settings["epoch_utc"] == "2027-03-15T00:00:00.000Z"
    assert settings["altitude_km"] == 510
    assert settings["eccentricity"] == 0.0008
    assert settings["inclination_deg"] == 97.44
    assert settings["mltan_hours"] == 18
    assert settings["mission_mltan_drift_target_min_per_year"] == 0
    assert settings["mission_mltan_drift_tolerance_min_per_year"] == 10
    assert settings["modeled_mltan_drift_min_per_year"] == pytest.approx(0.05, abs=0.01)
    assert settings["raan_deg"] == pytest.approx(orbit.raan_deg)


def test_disk_geometry_limits_and_known_equal_circle_overlap():
    np.testing.assert_allclose(disk_overlap_fraction(1, [1, 1, 0.5, 2], [3, 0, 0, 0]),
                               [0, 1, 0.25, 1])
    assert disk_overlap_fraction(1, 1, 1) == pytest.approx(2/3 - np.sqrt(3)/(2*np.pi))
    assert disk_overlap_fraction(1, 1, 2) == 0


def test_short_graze_between_samples_and_clipped_search():
    # An eclipse only 0.4 s long; no 30 s sample lies inside it.
    fn = lambda t: (np.asarray(t)-45.1)**2 - 0.2**2
    intervals = negative_intervals(fn, 0, 120, 30)
    np.testing.assert_allclose(intervals, [(44.9, 45.3)], atol=0.01)
    np.testing.assert_allclose(negative_intervals(lambda t: np.asarray(t)-40, 0, 60, 15),
                               [(0, 40)], atol=0.01)
    assert negative_intervals(lambda t: np.ones_like(t), 0, 60, 15) == []


def test_multiple_passages_and_peak_precision_far_from_epoch():
    intervals = negative_intervals(lambda t: np.cos(np.asarray(t)), 0, 4*np.pi, 0.2)
    np.testing.assert_allclose(intervals, [(np.pi/2, 3*np.pi/2), (5*np.pi/2, 7*np.pi/2)], atol=0.01)
    center = 30_000_045.123
    at, _ = extremum(lambda t: (np.asarray(t)-center)**2, 30_000_000, 30_000_120, 30)
    assert at == pytest.approx(center, abs=0.01)


def test_planning_orbit_ascending_node_and_correct_j2_direction():
    orbit = PlanningOrbit(utc_time("2027-03-15"), raan_deg=30)
    half_period = np.pi / orbit.mean_motion
    r0, r1 = orbit.positions(orbit.epoch + np.array([0, half_period])*u.s)
    assert np.linalg.norm(r0) == pytest.approx(6888.137*(1 - 0.0008))
    assert np.linalg.norm(r1) == pytest.approx(6888.137*(1 + 0.0008))
    assert np.linalg.norm(r1) == pytest.approx(orbit.max_radius_km)
    assert r0[2] == pytest.approx(0)
    assert orbit.positions(orbit.epoch + 1*u.s)[0, 2] > 0
    assert 0.9 < np.rad2deg(orbit.node_rate)*86400 < 1.1


def test_spacecraft_light_time_matches_astropy_ground_observer():
    # Independent public Astropy API with a fixed Dallas observer during totality.
    location = EarthLocation.from_geodetic(-96.797*u.deg, 32.777*u.deg, 130*u.m)
    times = Time(["2024-04-08T18:42:40"])
    positions = location.get_gcrs_posvel(times)[0].xyz.to_value(u.km).T
    geometry = EclipseGeometry(None, times[0], "builtin")
    sun, moon = geometry.body_vectors(times, positions)
    reference_sun = get_body("sun", times, location=location, ephemeris="builtin")
    reference_moon = get_body("moon", times, location=location, ephemeris="builtin")
    actual = separation(sun, moon)[0]
    expected = reference_sun.separation(reference_moon).to_value(u.rad)[0]
    # Relative stellar aberration across these nearly coincident disks is tiny.
    assert abs(actual-expected)*u.rad.to(u.arcsec) < 0.1
    assert actual*u.rad.to(u.arcsec) < 100  # known 2024 total solar eclipse


def test_august_2027_event_contacts_and_cadence_convergence():
    orbit = PlanningOrbit(utc_time("2027-03-15"), raan_deg=262.6)
    geometry = EclipseGeometry(orbit, utc_time("2027-08-02T07:00:00"), "builtin")
    coarse = predict(geometry, 6*3600, 60)
    fine = predict(geometry, 6*3600, 10)
    assert len(coarse) == len(fine) >= 1
    for a, b in zip(coarse, fine):
        assert abs((utc_time(a.start_utc)-utc_time(b.start_utc)).to_value(u.s)) < 0.05
        assert a.max_obscuration_pct == pytest.approx(b.max_obscuration_pct, abs=0.001)
        assert 0 <= a.earth_clear_duration_s <= a.duration_s + 0.01
        for contact in (a.start_utc, a.end_utc):
            second = (utc_time(contact)-geometry.start).to_value(u.s)
            assert abs(geometry.sample(second)["gap"]) < 1e-6
    first = fine[0]
    start = utc_time(first.maximum_utc)
    clipped = predict(EclipseGeometry(orbit, start, "builtin"), 1, 0.5)
    assert len(clipped) == 1
    assert clipped[0].start_clipped and clipped[0].end_clipped
    assert clipped[0].duration_s == pytest.approx(1)


def test_tle_propagation_and_cli_staleness_guard(tmp_path):
    # Vallado's public SGP4 test satellite 00005 at its epoch.
    tle = tmp_path / "test.tle"
    tle.write_text(
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753\n"
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667\n"
    )
    orbit = TLEOrbit(tle)
    position = orbit.positions(orbit.epoch.reshape((1,)))[0]
    assert 7000 < np.linalg.norm(position) < 10000
    with pytest.raises(SystemExit, match="2"):
        main(["--tle-file", str(tle), "--end", "2001-01-01", "--no-plot"])
    with pytest.raises(SystemExit, match="2"):
        main(["--tle-file", str(tle), "--altitude-km=600"])


@pytest.mark.parametrize("step", [0, -1, 61, np.nan])
def test_invalid_search_step(step):
    with pytest.raises(ValueError):
        predict(None, 100, step)


def test_jpl_vectors_against_independent_spice():
    import os
    spice = pytest.importorskip("spiceypy")
    root = Path(os.environ.get("suncet_data", "/missing")) / "spice_kernels"
    kernel, leap_seconds = root/"de440.bsp", root/"naif0012.tls"
    if not kernel.is_file() or not leap_seconds.is_file():
        pytest.skip("Local JPL/LSK kernels are optional.")
    times = Time(["2027-08-02T11:47:46.5", "2028-01-26T12:34:52.4"])
    orbit = PlanningOrbit(utc_time("2027-03-15"), raan_deg=262.6)
    position = orbit.positions(times)
    geometry = EclipseGeometry(orbit, times[0], str(kernel))
    sun, moon = geometry.body_vectors(times, position)
    spice.furnsh(str(kernel))
    spice.furnsh(str(leap_seconds))
    try:
        for i, time in enumerate(times):
            et = float((time.tdb-Time("J2000", scale="tdb")).to_value(u.s))
            reference = []
            for body in ("SUN", "MOON"):
                state, _ = spice.spkcpo(body, et, "J2000", "OBSERVER", "CN", position[i], "EARTH", "J2000")
                reference.append(state[:3])
            expected = separation(np.array([reference[0]]), np.array([reference[1]]))[0]
            actual = separation(sun[i:i+1], moon[i:i+1])[0]
            assert abs(actual-expected)*u.rad.to(u.arcsec) < 0.001
    finally:
        spice.unload(str(leap_seconds))
        spice.unload(str(kernel))

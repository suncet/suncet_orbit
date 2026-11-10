from pathlib import Path
import sys

from astropy.io import fits
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suncet_lunar_eclipse_movie import (
    calibration, mask_moon, mask_occultors, ray_grid, rescale_background,
    select_interval, solar_basis, solar_disk_coverage,
)


def test_detector_calibration_prefers_explicit_cd_and_flips_fits_origin(tmp_path):
    # Legacy simulator headers can contain both a physical CD and stale PC.
    header = fits.Header({
        "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN", "CUNIT1": "arcsec", "CUNIT2": "arcsec",
        "CRPIX1": 5.5, "CRPIX2": 4.5, "CRVAL1": 0, "CRVAL2": 0,
        "CDELT1": 9.6, "CDELT2": 9.6, "PC1_1": 0.916, "PC2_2": 0.914,
        "CD1_1": 9.6, "CD1_2": 0, "CD2_1": 0, "CD2_2": 9.6, "RSUN": 960,
    })
    path = tmp_path/"source.fits"
    fits.writeto(path, np.zeros((8, 10)), header)
    cal = calibration(path, 20, 16)
    np.testing.assert_allclose(cal["center_xy"], [9.5, 7.5])
    np.testing.assert_allclose(cal["matrix_arcsec_per_movie_pixel"], [[4.8, 0], [0, 4.8]])
    assert "explicit CD" in cal["matrix_convention"]


def test_opaque_lunar_mask_size_direction_and_unchanged_outside():
    cal = {"center_xy": [50, 50], "matrix_arcsec_per_movie_pixel": [[100, 0], [0, 100]]}
    rays = ray_grid(cal, 101, 101)
    angle = 100/206264.806
    # Lunar center is 10 pixels west (right), radius 5 pixels.
    vector = np.array([1, 10*angle, 0]); vector /= np.linalg.norm(vector)
    track = {"moon_forward": [vector[0]], "moon_west": [vector[1]],
             "moon_north": [vector[2]], "moon_radius": [5*angle]}
    source = Image.fromarray(np.full((101, 101, 3), 200, dtype=np.uint8))
    result = np.asarray(mask_moon(source, track, 0, rays, angle))
    assert np.all(result[50, 60] == 0)
    assert np.all(result[50, 40] == 200)
    area = np.sum(1-result[:, :, 0]/200)
    assert area == pytest.approx(np.pi*25, rel=0.05)
    assert np.all(np.asarray(source) == 200)


def test_solar_north_projection_and_west_form_correct_basis():
    forward, north, west = solar_basis(np.array([[1., 0., 0.], [0., 1., 0.]]))
    np.testing.assert_allclose(np.sum(forward*north, axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(north, axis=1), 1)
    np.testing.assert_allclose(np.cross(west, north), forward, atol=1e-12)
    assert np.all(north[:, 2] > 0)


def test_background_and_moon_use_the_same_detector_roll():
    cal = {"center_xy": [50, 50], "matrix_arcsec_per_movie_pixel": [[100, 0], [0, 100]]}
    pixels = np.zeros((101, 101, 3), dtype=np.uint8)
    pixels[49:52, 59:62] = 255  # source feature west of the Sun
    rolled = np.asarray(rescale_background(Image.fromarray(pixels), cal, 1, 90))
    # In the rolled detector, west is upward in world coordinates, downward in pixels.
    west, north, _ = ray_grid(cal, 101, 101, 90)
    assert np.all(rolled[60, 50] == 255)
    assert west[60, 50] > 0
    assert abs(north[60, 50]) < 1e-12


def earth_test_track(earth_offset, moon_offset=0.002, moon_radius=0.002):
    rs, re = 0.005, np.arcsin(6378.137/6888.137)
    from suncet_lunar_solar_eclipses import disk_overlap_fraction
    return {
        "sun_radius": np.array([rs]), "earth_radius": np.array([re]),
        "earth_separation": np.array([earth_offset]),
        "earth_forward": np.array([np.cos(earth_offset)]),
        "earth_west": np.array([np.sin(earth_offset)]), "earth_north": np.array([0.]),
        "moon_radius": np.array([moon_radius]), "moon_forward": np.array([np.cos(moon_offset)]),
        "moon_west": np.array([np.sin(moon_offset)]), "moon_north": np.array([0.]),
        "obscuration": np.atleast_1d(disk_overlap_fraction(rs, moon_radius, abs(moon_offset))),
    }


def test_earth_mask_matches_independent_ray_sphere_intersection():
    cal = {"center_xy": [50, 50], "matrix_arcsec_per_movie_pixel": [[100, 0], [0, 100]]}
    rays = ray_grid(cal, 101, 101)
    re = np.arcsin(6378.137/6888.137)
    track = earth_test_track(re, moon_offset=-0.5)
    source = Image.fromarray(np.full((101, 101, 3), 200, dtype=np.uint8))
    actual = np.asarray(mask_occultors(source, track, 0, rays, 100/206264.806))[:, :, 0]
    # Solve ray-sphere intersection in camera coordinates, independent of angular masks.
    center = 6888.137*np.array([np.cos(re), np.sin(re), 0])
    forward = np.ones_like(rays[0])/rays[2]
    west, north = rays[0]/rays[2], rays[1]/rays[2]
    projection = center[0]*forward + center[1]*west + center[2]*north
    discriminant = projection**2 - (6888.137**2-6378.137**2)
    blocked = (projection > 0) & (discriminant > 0)
    # Exclude the one-pixel antialias band around the known center limb.
    keep = np.abs(np.arange(101)-50)[None, :] > 2
    np.testing.assert_array_equal(actual[np.broadcast_to(keep, actual.shape)],
                                  np.where(blocked, 0, 200)[np.broadcast_to(keep, actual.shape)])


def test_earth_totality_and_no_earth_reproduce_expected_detector_pixels():
    cal = {"center_xy": [50, 50], "matrix_arcsec_per_movie_pixel": [[100, 0], [0, 100]]}
    rays = ray_grid(cal, 101, 101)
    source = Image.fromarray(np.full((101, 101, 3), 200, dtype=np.uint8))
    assert np.all(np.asarray(mask_occultors(source, earth_test_track(0), 0, rays, 0.0005)) == 0)
    track = earth_test_track(np.pi)
    np.testing.assert_array_equal(np.asarray(mask_occultors(source, track, 0, rays, 0.0005)),
                                  np.asarray(mask_moon(source, track, 0, rays, 0.0005)))


def test_combined_coverage_counts_overlapping_earth_and_moon_once():
    re = np.arcsin(6378.137/6888.137)
    # Earth occupies almost exactly the western half of the small solar disk.
    same_side = earth_test_track(re, moon_offset=0.002)
    earth, total = solar_disk_coverage(same_side)
    assert earth[0] == pytest.approx(0.5, abs=0.001)
    assert total[0] == pytest.approx(earth[0], abs=0.001)
    opposite_side = earth_test_track(re, moon_offset=-0.002)
    earth, total = solar_disk_coverage(opposite_side)
    assert total[0] == pytest.approx(earth[0]+opposite_side["obscuration"][0], abs=0.001)
    _, fine = solar_disk_coverage(opposite_side, grid_size=1024)
    assert total[0] == pytest.approx(fine[0], abs=0.0005)
    earth, total = solar_disk_coverage(earth_test_track(0))
    assert earth[0] == total[0] == 1
    no_earth = earth_test_track(np.pi)
    earth, total = solar_disk_coverage(no_earth)
    assert earth[0] == 0
    np.testing.assert_array_equal(total, no_earth["obscuration"])


def test_custom_movie_interval_can_contain_earth_eclipse_and_no_lunar_event():
    summary = {"start_utc": "2028-01-01", "end_utc": "2028-02-01", "events": []}
    start, end, event = select_interval(summary, None, "2028-01-26T15:50:00", "2028-01-26T16:34:00", 180)
    assert event is None
    assert (end-start).sec == pytest.approx(2640)
    with pytest.raises(ValueError):
        select_interval(summary, None, "2028-01-26", None, 180)
    with pytest.raises(ValueError):
        select_interval(summary, None, "2028-01-26", "2028-01-25", 180)


def test_explicit_event_selection_does_not_filter_earth_occulted_events():
    event = {"start_utc": "2028-01-26T16:00:00", "end_utc": "2028-01-26T16:20:00",
             "max_obscuration_pct": 50., "start_clipped": False, "end_clipped": False,
             "duration_s": 1200, "earth_clear_duration_s": 0}
    _, _, selected = select_interval({"events": [event]}, 1, None, None, 180)
    assert selected is event
    # Default falls back to Earth-occulted lunar events if none are fully clear.
    _, _, selected = select_interval({"events": [event]}, None, None, None, 180)
    assert selected is event

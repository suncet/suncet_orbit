import astropy.units as u
import numpy as np
import pytest

import suncet_uhf_link_passes as link_passes
from suncet_uhf_link_passes import GROUND_RADIOS, directional_gain, load_gain_pattern, radio_links
from suncet_lunar_solar_eclipses import utc_time


@pytest.mark.parametrize("powers, expected", [
    ([.5, 1.5, 1.5, .5, 1.5], (12, 9)),
    ([1, 2, 2, 2, 1], (18, 18)),
    ([.5, .5, 1, .5, .5], (0, 0)),
    ([2, 0, 0, 0, 0], (2, 2)),
])
def test_closure_interpolates_crossings_and_keeps_separate_windows(powers, expected):
    with np.errstate(divide="ignore"):
        margin_db = 10 * np.log10(powers)
    assert link_passes.closed_times(np.array([0, 4, 10, 12, 18]), margin_db) == pytest.approx(expected)


def test_circular_gain_uses_handedness_and_body_azimuth(tmp_path):
    path = tmp_path / "pattern.txt"
    lines = []
    for theta in (0, 90, 180):
        for phi in (0, 90, 180, 270, 360):
            ratio = 1 if (theta, phi) == (90, 90) else .5
            sense = "LEFT" if (theta, phi) == (90, 270) else "RIGHT"
            lines.append(f"{theta} {phi} 0 0 0 {ratio} 0 {sense} 1 0 1 0")
    path.write_text("\n".join(lines))
    pattern = load_gain_pattern(path)

    gains = directional_gain(pattern, np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1], [1, 1, 0]]))
    assert gains[0, 0] == pytest.approx(0)
    assert gains[0, 1] == -np.inf
    assert 10 ** (gains[1, 0] / 10) == pytest.approx(.1)
    assert 10 ** (gains[2, 0] / 10) == pytest.approx(.9)
    assert 10 ** (gains[3, 0] / 10) == pytest.approx(.95)
    assert np.allclose(np.sum(10 ** (gains / 10), axis=1), 1)


def test_contact_search_refines_grazer_missed_by_sample_grid(monkeypatch):
    start = utc_time("2027-03-15")
    # The sampled peak can lie below 5° even when the true peak just exceeds it.
    class Orbit:
        mean_motion = 2 * np.pi / 600

    def geometry(orbit, station, times):
        seconds = (times - start).to_value(u.s)
        elevation = 5.01 - (seconds - 105) ** 2 / 1000
        return None, None, None, elevation

    monkeypatch.setattr(link_passes, "station_geometry", geometry)
    contacts = link_passes.find_contacts(Orbit(), GROUND_RADIOS[0].station, start, 107, 5)
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.peak == pytest.approx(105, abs=.01)
    assert contact.aos == pytest.approx(105 - np.sqrt(10), abs=.01)
    assert contact.los == 107 and contact.clipped


def test_boulder_budget_matches_workbook_after_reinstating_original_polarization_loss():
    links = radio_links(GROUND_RADIOS[0], 9600, 19200)
    # Workbook B49 differs by about 0.001 dB because it rounds the Boltzmann term.
    assert links["downlink"].margin(150.971628, 15.9, -7 - 3) == pytest.approx(2.436123646, abs=.002)
    # Workbook uplink B49 subtracts the transmit and pointing losses twice.
    expected = 15.06175664 + 3.7135 + .4353741497
    assert links["uplink"].margin(150.9499512, 15.9, -7 - 1) == pytest.approx(expected, abs=.004)

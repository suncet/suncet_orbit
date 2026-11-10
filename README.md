# suncet_orbit
Tools that deal with orbital simulation and analysis

The shared mission orbit targets are `LAUNCHES["ussf_178"]` in
`suncet_ltan_to_raan.py`. The planning tools default to **510 km** altitude,
**0.0008** eccentricity, **97.44°** inclination, and **18:00:00** MLTAN.
The MLTAN drift target is **0 min/year** with a **±10 min/year** tolerance.
The launch date is **NET March 15, 2027**.

## Ground-station angles and roll attitude

`suncet_ground_station_angles.py` plots the angle **at the spacecraft**, between
body **+Y** and the **spacecraft-to-station** line of sight for Boulder and
Fairbanks. An angle of 0° means +Y points at the station; 180° means −Y points
at it. Station elevation is measured independently at the ground station.

The axis definitions follow the earlier mission discussion: **+Z** is the
telescope boresight, detector **X** is the long dimension, detector **Y** is the
short dimension, and **−Y** is the radiator. +Z tracks the Sun. Between rolls,
either +Y or −Y follows projected solar north, using the
[IAU solar pole](https://docs.sunpy.org/en/stable/reference/sun.html).
The model selects the settled roll state that points −Y farther from Earth's
center. This produces two 180° rolls per orbit near the equator, at the
equal-preference crossings of projected solar north and nadir. They are **not
forced to exactly 0° geographic latitude**. The half-turn direction is chosen
so −Y sweeps away from Earth. The default 60 s smooth rolls and their direction
are illustrative; actual commanded roll timing is not available.
Solar north leaves Y during each finite roll. Keeping the radiator normal
away from Earth's center does not guarantee that its whole facing hemisphere
is clear of the extended Earth disk.

```sh
python suncet_ground_station_angles.py
```

The default seven-day planning search begins March 15, 2027, using the shared
Kepler orbit and J2 nodal precession model from the eclipse predictor.
It evaluates **18:00 MLTAN**, selecting the highest-elevation
complete pass for each city. The two illustrated passes have independent UTC
clocks. Midnight injection and initial orbital phase remain placeholders;
these are geometric illustrations, not operational contact predictions.
Coordinates are approximate **city centers**, not surveyed antenna locations:
Boulder 40.0150° N, 105.2705° W, 1624 m; Fairbanks 64.8378° N, 147.7164° W, 136 m.
The horizon is WGS84 geodetic, without terrain or refraction. Earth orientation
uses offline bundled IERS data; solar-pole projection neglects sub-arcminute
GCRS/ICRS differences. No antenna gain, link budget, or spacecraft blockage is modeled.

Outputs in `outputs/ground_station_angles/` include the 24 s movie
`ground_station_angles.mp4`, two-pass plot `pass_angles.png`, full-orbit roll
explanation `roll_cycle.png`, one-second geometry CSVs (plus exact contacts and
culmination), all complete passes peaking at least 10° in `all_passes.csv`, and
`provenance.json` recording assumptions and selected passes. The movie shows
the station direction on a unit sphere fixed to the spacecraft; the blue line
and green arc retain the true 3D angular separation. Its dashed orange ray is
the Earth-center direction. The two cities play through their own passes at
the same fractional progress; they are not simultaneous.

Use `--start`, `--days`, `--mltan-hours` (default 18), `--roll-seconds`,
`--duration-s` (per MLTAN), `--fps`, `--no-movie`, and `--output-dir` to change
the illustration. The tool uses the shared mission planning model.

## UHF link margin during passes

Run a 365-day planning simulation for Boulder and Fairbanks with the engineer's
body-frame radiation pattern:

```sh
python suncet_uhf_link_passes.py --pattern /path/to/suncet_uhf_radiation_pattern.txt
```

The simulation uses the shared 18:00 MLTAN orbit, the modeled 60-second rolls,
the filed ground antenna positions and a 5° elevation limit. It applies the
spacecraft pattern's RHCP gain to both links at 401.2 MHz, assuming the same
passive spacecraft antenna port for transmit and receive. Both links default
to 9,600 bit/s. The LASP workbook supplies Boulder and
spacecraft link inputs. Fairbanks uses nominal filing values and the Boulder
receive line loss as an estimate. The pass times use a provisional epoch and
orbit phase.

`outputs/uhf_link_passes/` contains `example_passes.png` with both links and
ground-station elevation for the highest-elevation and weakest pass at each site,
and `annual_link_closure.png` with the share of passes meeting targets for
fraction of contact time and longest uninterrupted window with both margins
at or above 0 dB. The `pass_minima.csv` includes each link's minimum and the
total time, fraction, and longest window for each link and for both together.
Contact time starts and ends at the elevation limit. Threshold crossings are
interpolated in received power between samples. The plotted tracks are available
as CSVs. The `annual_pass_minima.png` plot shows per-pass minima over time and
labels the most common margin bin for each link. The `summary.json` file records
assumptions and aggregate results.
Use `--days` for a shorter run, `--mask-deg` to change the elevation limit,
`--roll-seconds` to change slew duration, or `--uplink-bps` and `--downlink-bps`
to change rates.

## Lunar solar eclipses seen by SunCET

`suncet_lunar_solar_eclipses.py` predicts the Moon crossing the solar disk from
the **spacecraft's** viewpoint. The older `suncet_earthshadow_eclipse_plot.py`
only analyzes GMAT Earth-shadow events and does not predict lunar crossings.

The model treats the 510 km altitude as the semimajor axis minus Earth's
equatorial radius. Midnight UTC on the NET date, initial argument of latitude
0° (ascending node), and argument of perigee 0° are planning assumptions.
A changed launch time, date, or orbital phase can change contact times and
which events are visible.

Use the existing conda environment (`conda activate suncet_orbit`), or create it
from `environment.yml`. Run the 18:00 orbit for 365 days from the planning epoch:

```sh
python suncet_lunar_solar_eclipses.py --output-dir outputs/lunar_solar_eclipses/mltan18
```

The directory receives `events.csv`, `summary.json` (including parameters,
warnings, and events), and `overview.png`. Generated products are gitignored.
All dates/times are **UTC**. Override the search with `--start` and `--end`,
and the orbit with `--epoch`, `--altitude-km`, `--eccentricity`, `--inclination-deg`,
`--mltan-hours`, and `--phase-deg`. `--raan-deg` overrides the MLTAN-derived RAAN.

Ephemerides default to a local `$suncet_data/spice_kernels/de440.bsp`, when
available, otherwise approximate Astropy built-in ephemerides. `--ephemeris`
accepts a local planetary BSP, `builtin`, or a JPL name such as `de440s` (which
may download a kernel). JPL kernels require `jplephem`, included in the environment.
The spacecraft is propagated separately; a planetary BSP is not a spacecraft orbit.

For a five-year search from the provisional launch date, run:

```sh
python suncet_lunar_solar_eclipses.py --end 2032-03-15 --output-dir outputs/lunar_solar_eclipses/five_years/mltan18
```

This holds the planning semimajor axis fixed for seasonal geometry. Future UTC values extrapolate the bundled
leap-second table and can trigger ERFA warnings. Eclipse seasons recur about
173.3 days apart, making an eclipse year about 346.6 days; their calendar timing
therefore drifts earlier. Individual event dates also depend on new Moon and
spacecraft visibility. See [NASA's eclipse glossary](https://science.nasa.gov/eclipses/glossary/).

For an actual orbit, supply a current local TLE and a short date range:

```sh
python suncet_lunar_solar_eclipses.py --tle-file suncet.tle --start 2027-08-02 --end 2027-08-03
```

TLE mode selects the latest pair in the text file and defaults to a seven-day
search from its epoch. It refuses searches more than 14 days from that epoch
unless `--allow-stale-tle` is explicitly set. This guard is not a guarantee of
14-day accuracy: fresh TLEs and updated Earth-orientation data are needed for
observing schedules. No Space-Track credentials or network fetch is needed.

### Event definitions and numerical method

The model uses observer-dependent Sun/Moon distances, spacecraft parallax, and
iterated reception light time in consistent barycentric axes. Planning orbits
have Kepler motion and prograde J2 nodal precession. The model derives the
precession rate from the orbit elements. For the default targets this gives
approximately +0.05 min/year of MLTAN drift relative to the mean Sun. Run metadata
lists the modeled rate, mission target, and tolerance separately. TLE mode uses SGP4, converting
TEME positions to GCRS. Offline bundled IERS data are used; future epochs can
produce degraded-accuracy warnings.

An event occurs when the angular separation is smaller than the sum of the
solar and lunar angular radii. A conservative geocentric parallax envelope
screens at ten-minute cadence, followed by a 30-second spacecraft search.
Local minima are refined, so an event shorter than a sample step can still be
detected. Reduce `--step-s` to check very close grazes; arbitrarily fast unresolved
structure is not guaranteed. Contacts are solved to 0.01 s **numerical tolerance**,
which does not imply comparable physical timing accuracy.

`start_utc`/`end_utc` are external contacts (C1/C4) or the clipped search boundaries;
`start_clipped`/`end_clipped` identify the latter. `maximum_utc` maximizes area
obscuration; in totality it uses the closest disk-center alignment.
`internal_intervals_utc` gives C2/C3 intervals, if any, within the search window.
`kind` is partial, annular, or total at maximum.

`max_obscuration_pct` is circular-disk **area**, not a prediction of lost EUV
flux. The reference solar disk uses the nominal photospheric radius 695,700 km;
the Moon uses a spherical mean radius 1,737.4 km. The Moon can also obscure
coronal emission outside this reference solar disk; that is visible in the
movies but is not counted as a solar-disk crossing in the event table.

Events are retained even when Earth blocks the Sun. `earth_clear_intervals_utc`
and `earth_clear_duration_s` identify intervals when the **entire** solar disk
is above a spherical, equatorial-radius Earth limb. The maximum restricted to
these intervals is reported separately. `--earth-limb-margin-km` adds an optional
atmospheric/avoidance margin (default 0). This is a geometric visibility check,
not a wavelength-dependent atmospheric transmission or attitude model.

References: [JPL DE440 planetary and lunar ephemerides](https://ssd.jpl.nasa.gov/doc/de440_de441.html),
[Astropy solar-system ephemerides](https://docs.astropy.org/en/stable/coordinates/solarsystem.html),
[NAIF occultation searches and light-time conventions](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/gfoclt_c.html),
[IAU nominal solar radius](https://arxiv.org/abs/1510.07674).

### Validation

```sh
python -m pytest -q tests
```

Tests cover disk overlap, sub-cadence grazes, multiple events, boundary clipping,
cadence convergence, J2 direction, SGP4 input handling, the April 2024 Dallas
eclipse using Astropy's independent public observer API, and movie plate-scale,
orientation and opaque-mask behavior. When local kernels are available, the
light-time-corrected Sun/Moon geometry is also checked against SPICE `spkcpo`.
Earth masks are checked against an independent ray–sphere intersection, and
combined coverage is checked for totality, overlap accounting and convergence.

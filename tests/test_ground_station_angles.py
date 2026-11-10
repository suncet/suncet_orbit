"""Independent geometry and physical constraints for station-angle planning."""
import csv
import json

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
from astropy.time import Time
import numpy as np
import pytest

from suncet_ground_station_angles import (
    Attitude, PlanningOrbit, STATIONS, angle_deg, find_passes, make_track,
    main, station_geometry, sun_north_basis, unit, utc_time,
)


def test_default_ground_station_run_simulates_only_18_mltan(tmp_path):
    main(["--days", "1", "--no-movie", "--output-dir", str(tmp_path)])
    meta=json.loads((tmp_path/"provenance.json").read_text())
    assert len(meta["cases"]) == 1
    assert meta["cases"][0]["orbit"]["mltan_hours"] == 18
    with (tmp_path/"all_passes.csv").open() as stream:
        passes=list(csv.DictReader(stream))
    assert passes and {int(row["mltan_hours"]) for row in passes} == {18}
    assert {path.name for path in tmp_path.glob("*_mltan*.csv")} == {
        "boulder_mltan18.csv", "fairbanks_mltan18.csv",
    }


@pytest.mark.parametrize("raan", [262.0067,82.0067])
def test_attitude_is_right_handed_sun_pointing_and_thermally_preferred(raan):
    start=utc_time("2027-03-15")
    orbit=PlanningOrbit(start,raan_deg=raan)
    period=2*np.pi/orbit.mean_motion
    times=start+np.linspace(0,period,2000)*u.s
    att=Attitude(orbit,times[0],times[-1],60)
    x,y,z,rolling,_=att.evaluate(times)
    frame=np.stack([x,y,z],axis=-1)
    assert np.allclose(np.linalg.det(frame),1,atol=1e-12)
    assert np.allclose(np.cross(x,y),z,atol=1e-12)
    north=sun_north_basis(orbit,times)[1]
    assert np.allclose(abs(np.sum(y[~rolling]*north[~rolling],axis=-1)),1)
    # Includes the slew: a wrong half-turn sense points -Y straight at Earth.
    assert np.all(angle_deg(-y,unit(-orbit.positions(times)))>=90-1e-8)
    assert len([t for t in att.flips if 0<t<period])==2
    # Resolve a roll at 0.01 s cadence: no instantaneous sign change allowed.
    flip=next(t for t in att.flips if 0<t<period)
    dense=start+np.linspace(flip-31,flip+31,6201)*u.s
    yd=att.evaluate(dense)[1]
    assert np.max(angle_deg(yd[:-1],yd[1:]))<.05
    assert angle_deg(yd[0],yd[-1])>179.99


def test_elevation_agrees_with_independent_earth_fixed_local_horizon():
    times=Time(["2027-03-15T06:30:00","2027-03-15T06:35:00"])
    orbit=PlanningOrbit(utc_time("2027-03-15"),raan_deg=262.0067)
    station=STATIONS[0]
    position,ground,los,elevation=station_geometry(orbit,station,times)
    fixed=GCRS(CartesianRepresentation(position.T*u.km),obstime=times).transform_to(ITRS(obstime=times))
    topocentric=(fixed.cartesian-station.location().get_itrs(times).cartesian).xyz.to_value(u.km).T
    lat,lon=np.deg2rad([station.latitude_deg,station.longitude_deg])
    up_fixed=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    expected=np.rad2deg(np.arcsin(unit(topocentric)@up_fixed))
    assert np.allclose(elevation,expected,atol=1e-5)


def test_complete_pass_contacts_and_exported_body_angle():
    start=utc_time("2027-03-15")
    orbit=PlanningOrbit(start,raan_deg=262.0067)
    passes=find_passes(orbit,STATIONS[1],start,1)
    event=max(passes,key=lambda p:p["max_elevation_deg"])
    track=make_track(orbit,STATIONS[1],start,event,60)
    assert abs(track["elevation"][0])<1e-4
    assert abs(track["elevation"][-1])<1e-4
    assert np.all(track["elevation"][1:-1]>0)
    assert event["aos_s"]<event["peak_s"]<event["los_s"]
    assert np.allclose(track["theta"],np.rad2deg(np.arccos(np.clip(track["body"][:,1],-1,1))),atol=1e-9)
    assert np.allclose(np.linalg.norm(track["body"],axis=1),1)


def test_plus_y_angle_uses_spacecraft_to_ground_direction():
    y=np.array([0.,1.,0.])
    assert angle_deg(y,y)==0
    assert angle_deg(y,-y)==180
    assert angle_deg(y,np.array([0.,0.,1.]))==90

#!/usr/bin/env python3
"""Plot and animate station line-of-sight angles from SunCET's +Y body axis.

Planning illustration: +Z points at the Sun, detector Y (short dimension) is
parallel to projected solar north, and the settled roll state puts radiator -Y
farther from Earth center. Rolls are centered on equal thermal preference, NOT
on prescribed geographic equator crossings. Their duration is illustrative.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from astropy.coordinates import CartesianRepresentation, EarthLocation, GCRS, ITRS, get_sun
from astropy.time import Time
from astropy.utils import iers
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from suncet_ltan_to_raan import DEFAULT_LAUNCH, DEFAULT_MLTAN_HOURS
from suncet_lunar_solar_eclipses import PlanningOrbit, iso_utc, utc_time

iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"

POLE_SOURCE = "https://docs.sunpy.org/en/stable/reference/sun.html"
RA, DEC = np.deg2rad([286.13, 63.87])
SOLAR_POLE = np.array([np.cos(DEC)*np.cos(RA), np.cos(DEC)*np.sin(RA), np.sin(DEC)])
BG, FG, MUTED = "#0b1220", "#edf3fb", "#9caec4"
BLUE, ORANGE, GREEN = "#64bcff", "#ffb65c", "#80dfac"


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_deg(a, b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                np.sum(a*b, axis=-1)))


@dataclass(frozen=True)
class Station:
    name: str
    latitude_deg: float
    longitude_deg: float
    height_m: float

    def location(self):
        return EarthLocation.from_geodetic(self.longitude_deg*u.deg,
                                          self.latitude_deg*u.deg, self.height_m*u.m)


STATIONS = [Station("Boulder", 40.0150, -105.2705, 1624),
            Station("Fairbanks", 64.8378, -147.7164, 136)]


def station_geometry(orbit, station, times):
    """WGS84 station, GCRS positions and geodetic horizon; no refraction."""
    position = orbit.positions(times)
    # Transform the geodetic surface normal with the same ITRS->GCRS rotation.
    # The difference of locations 1 km apart along ellipsoid normal removes
    # the frame translation without confusing geocentric with geodetic up.
    loc = station.location()
    upper = EarthLocation.from_geodetic(station.longitude_deg*u.deg,
                                       station.latitude_deg*u.deg,
                                       (station.height_m+1000)*u.m)
    ground = loc.get_gcrs(times).cartesian.xyz.to_value(u.km).T
    up = unit(upper.get_gcrs(times).cartesian.xyz.to_value(u.km).T-ground)
    to_ground = ground-position
    elevation = np.rad2deg(np.arcsin(np.clip(np.sum(-unit(to_ground)*up, axis=-1), -1, 1)))
    return position, ground, to_ground, elevation


def sun_north_basis(orbit, times):
    position = orbit.positions(times)
    sun = get_sun(times).cartesian.xyz.to_value(u.km).T
    z = unit(sun-position)
    north = unit(SOLAR_POLE-(z@SOLAR_POLE)[:, None]*z)
    x = unit(np.cross(north, z))
    # Positive score: north is the +Y branch nearest nadir, radiator points away.
    score = np.sum(north*unit(-position), axis=-1)
    return x, north, z, score


class Attitude:
    def __init__(self, orbit, start, end, roll_seconds=60):
        self.orbit, self.start, self.roll_seconds = orbit, start, roll_seconds
        period = 2*np.pi/orbit.mean_motion
        if not np.isfinite(roll_seconds) or not 0 <= roll_seconds < period/4:
            raise ValueError("Roll duration must be finite, nonnegative and less than a quarter orbit.")
        lo, hi = -period, float((end-start).to_value(u.s))+period
        grid = np.arange(lo, hi+period/100, period/100)
        score = sun_north_basis(orbit, start+grid*u.s)[3]
        self.initial_phase = 0 if score[0] >= 0 else np.pi
        self.flips = []
        self.turns = []
        phase_before = self.initial_phase
        for a, b, sa, sb in zip(grid[:-1], grid[1:], score[:-1], score[1:]):
            if sa*sb < 0:
                root = brentq(lambda s: sun_north_basis(orbit, start+np.array([s])*u.s)[3][0],
                              a, b, xtol=.001)
                self.flips.append(root)
                # Of the two half-turn paths, use the one whose midpoint puts
                # +Y toward Earth, so -Y sweeps away. The endpoint states alone
                # do not determine the thermally appropriate slew direction.
                root_time = start+np.array([root])*u.s
                x_at_root = sun_north_basis(orbit, root_time)[0][0]
                nadir_at_root = unit(-orbit.positions(root_time))[0]
                turn = -np.pi*np.sign(np.cos(phase_before)*np.dot(x_at_root,nadir_at_root))
                self.turns.append(turn)
                phase_before += turn

    def evaluate(self, times):
        x0, north, z, score = sun_north_basis(self.orbit, times)
        seconds = np.atleast_1d((times-self.start).to_value(u.s))
        phase = np.full(len(seconds), self.initial_phase, dtype=float)
        rolling = np.zeros(len(seconds), dtype=bool)
        for flip, turn in zip(self.flips,self.turns):
            if self.roll_seconds:
                q = np.clip((seconds-flip)/self.roll_seconds+.5, 0, 1)
                # Smooth finite roll, zero rate at either end.
                phase += turn*(q*q*(3-2*q))
                rolling |= (q > 0) & (q < 1)
            else:
                phase += turn*(seconds >= flip)
        cp, sp = np.cos(phase)[:, None], np.sin(phase)[:, None]
        x, y = cp*x0+sp*north, cp*north-sp*x0
        return x, y, z, rolling, score


def find_passes(orbit, station, start, days, minimum_peak=10):
    seconds = np.arange(0, days*86400+30, 30.)
    elevation = station_geometry(orbit, station, start+seconds*u.s)[3]
    visible = elevation > 0
    rises = np.where(~visible[:-1] & visible[1:])[0]
    sets = np.where(visible[:-1] & ~visible[1:])[0]
    passes = []
    def el(s):
        return station_geometry(orbit, station, start+np.array([s])*u.s)[3][0]
    for rise in rises:
        subsequent = sets[sets > rise]
        if not len(subsequent):
            continue  # Do not label a clipped pass as complete.
        setting = subsequent[0]
        aos = brentq(el, seconds[rise], seconds[rise+1], xtol=.01)
        los = brentq(el, seconds[setting], seconds[setting+1], xtol=.01)
        peak = minimize_scalar(lambda s: -el(s), bounds=(aos, los), method="bounded")
        if -peak.fun >= minimum_peak:
            passes.append({"aos_s": aos, "los_s": los, "peak_s": peak.x,
                           "max_elevation_deg": -peak.fun})
    return passes


def make_track(orbit, station, start, event, roll_seconds):
    # Include exact rise, culmination and set in the exported samples.
    seconds = np.unique(np.r_[np.arange(event["aos_s"], event["los_s"], 1.),
                                 event["peak_s"], event["los_s"]])
    times = start+seconds*u.s
    position, ground, los, elevation = station_geometry(orbit, station, times)
    attitude = Attitude(orbit, times[0], times[-1], roll_seconds)
    x, y, z, rolling, score = attitude.evaluate(times)
    direction = unit(los)
    body = np.column_stack([np.sum(direction*axis, axis=-1) for axis in (x, y, z)])
    theta = angle_deg(y, direction)
    nadir = unit(-position)
    return {"station": station, "event": event, "times": times,
            "seconds": seconds, "minutes": (seconds-event["peak_s"])/60,
            "elevation": elevation, "theta": theta, "body": body,
            "range_km": np.linalg.norm(los, axis=-1), "rolling": rolling,
            "nadir_body": np.column_stack([np.sum(nadir*axis, axis=-1) for axis in (x,y,z)]),
            "radiator_nadir_deg": angle_deg(-y, nadir)}


def style_axis(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#314157")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(color="#29374b", linewidth=.6, alpha=.65)


def shade_rolls(ax,track,annotate=False):
    idx=np.flatnonzero(track["rolling"])
    if not len(idx): return
    for run in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
        left,right=track["minutes"][[run[0],run[-1]]]
        ax.axvspan(left,right,color=FG,alpha=.15)
        if annotate:
            ax.text((left+right)/2,173,"180° roll",color=FG,fontsize=8,ha="center")


def write_overview(cases, out):
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 4*len(cases)), facecolor=BG, squeeze=False)
    for row, case in enumerate(cases):
        for col, track in enumerate(case["tracks"]):
            ax = axes[row, col]
            style_axis(ax)
            ax.plot(track["minutes"], track["theta"], color=BLUE, lw=2.5)
            shade_rolls(ax,track,annotate=True)
            ax.axhline(90, color=MUTED, lw=.8, ls=":")
            ax.set_ylim(0,180)
            ax.set_yticks([0,45,90,135,180])
            ax.set_ylabel("Angle from spacecraft +Y (deg)")
            ax.set_xlabel("Minutes from maximum elevation")
            other = ax.twinx()
            other.plot(track["minutes"], track["elevation"], color=ORANGE, lw=1.5, ls="--")
            other.set_ylim(0,90)
            other.set_ylabel("Station elevation (deg)", color=ORANGE)
            other.tick_params(colors=ORANGE, labelsize=9)
            for spine in other.spines.values(): spine.set_visible(False)
            idx = int(np.argmin(abs(track["seconds"]-track["event"]["peak_s"])))
            ax.scatter([0], [track["theta"][idx]], color=BLUE, s=25)
            ax.set_title(f"{track['station'].name}  |  {case['mltan']:02d}:00 MLTAN\n"
                         f"{iso_utc(track['times'][idx])[:19]} UTC  |  peak {track['elevation'][idx]:.1f}°  |  +Y angle {track['theta'][idx]:.1f}°",
                         color=FG, fontsize=10)
    fig.suptitle("SunCET ground passes: +Y to station line of sight", color=FG, fontsize=19, y=.98)
    fig.text(.5,.915,"Blue: spacecraft angle    ·    Dashed orange: elevation seen from the station", ha="center",color=MUTED)
    fig.text(.5,.025,"Highest complete pass per city in the 7-day planning search. Each panel has its own UTC. City-center coordinates.",ha="center",color=MUTED,fontsize=9)
    fig.subplots_adjust(left=.075,right=.92,top=.84,bottom=.11,hspace=.52,wspace=.42)
    fig.savefig(out/"pass_angles.png",dpi=150,facecolor=BG)
    plt.close(fig)


def display_coords(body):
    """Right-handed plotting coordinates: (body X, -body Z, body Y)."""
    return np.asarray(body)[..., [0,2,1]]*np.array([1,-1,1])


def sphere_panel(ax, track):
    ax.set_facecolor(BG)
    az, lat = np.meshgrid(np.linspace(0,2*np.pi,37),np.linspace(-np.pi/2,np.pi/2,19))
    ax.plot_wireframe(np.cos(lat)*np.cos(az),np.cos(lat)*np.sin(az),np.sin(lat),
                      color="#304057",linewidth=.35,alpha=.45,rstride=3,cstride=3)
    for body, label, color in [([0,1,0],"+Y",GREEN),([0,-1,0],"−Y radiator",ORANGE),
                                ([0,0,1],"+Z to Sun",MUTED),([1,0,0],"+X",MUTED)]:
        p=display_coords(body)
        ax.plot(*np.array([[0,0,0],p]).T,color=color,lw=1.6)
        ax.text(*(p*1.17),label,color=color,fontsize=10,ha="center")
    path=display_coords(track["body"])
    ax.plot(*path.T,color=BLUE,lw=2,alpha=.5)
    current,=ax.plot([],[],[],"o",color=BLUE,ms=7)
    sightline,=ax.plot([],[],[],color=BLUE,lw=2.5)
    arc,=ax.plot([],[],[],color=GREEN,lw=2)
    nadir,=ax.plot([],[],[],color=ORANGE,lw=1,ls="--",alpha=.7)
    ax.set(xlim=(-1.25,1.25),ylim=(-1.25,1.25),zlim=(-1.25,1.4))
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=17,azim=-65)
    ax.dist=7.5
    ax.set_axis_off()
    return current,sightline,arc,nadir


def render_movie(cases,out,duration=24,fps=24):
    fig=plt.figure(figsize=(12.8,8),facecolor=BG)
    grid=fig.add_gridspec(2,2,height_ratios=[1.7,1],left=.075,right=.91,
                         top=.84,bottom=.15,hspace=.05,wspace=.4)
    spheres=[fig.add_subplot(grid[0,c],projection="3d") for c in range(2)]
    axes=[fig.add_subplot(grid[1,c]) for c in range(2)]
    title=fig.text(.5,.955,"",ha="center",color=FG,fontsize=19)
    fig.text(.5,.911,"Spacecraft frame: blue line points toward the station; green arc is its true 3D angle from +Y.",
             ha="center",color=MUTED,fontsize=10)
    fig.text(.5,.88,"Dashed orange ray: direction toward Earth's center. Gray shading: assumed 180° roll.",
             ha="center",color=MUTED,fontsize=9)
    fig.text(.5,.065,"+Z → Sun   ·   short detector axis Y ∥ solar north between rolls   ·   −Y radiator favors the direction away from Earth",
             ha="center",color=MUTED,fontsize=9)
    fig.text(.5,.035,f"Planning orbit, {DEFAULT_LAUNCH['altitude_km']:g} km / {DEFAULT_LAUNCH['inclination_deg']:g}°. "
                     "Highest complete pass per city. Flight roll timing is unconfirmed.",
             ha="center",color=MUTED,fontsize=9)
    writer=FFMpegWriter(fps=fps,codec="libx264",extra_args=["-pix_fmt","yuv420p","-crf","19","-movflags","+faststart"])
    with writer.saving(fig,str(out/"ground_station_angles.mp4"),100):
        for case in cases:
            title.set_text(f"SunCET +Y overflight angles  |  {case['mltan']:02d}:00 MLTAN")
            artists=[]
            for c,track in enumerate(case["tracks"]):
                spheres[c].clear()
                sphere_art=sphere_panel(spheres[c],track)
                ax=axes[c]
                ax.clear()
                style_axis(ax)
                ax.plot(track["minutes"],track["theta"],color=BLUE,lw=2)
                shade_rolls(ax,track)
                # Elevation shares numerical degrees, with both curves labeled.
                ax.plot(track["minutes"],track["elevation"],color=ORANGE,lw=1.5,ls="--")
                ax.axhline(90,color=MUTED,lw=.7,ls=":")
                ax.set(ylim=(0,180),xlim=(track["minutes"][0],track["minutes"][-1]),
                       yticks=[0,45,90,135,180],xlabel="Minutes from maximum elevation",ylabel="Degrees")
                cursor=ax.axvline(track["minutes"][0],color=FG,lw=1,alpha=.6)
                dot,=ax.plot([],[],"o",color=BLUE,ms=6)
                label=ax.text(.02,1.07,"",transform=ax.transAxes,color=FG,fontsize=11)
                clock=spheres[c].text2D(.5,1.01,"",transform=spheres[c].transAxes,ha="center",color=FG,fontsize=11)
                ax.text(.02,.96,"+Y angle",transform=ax.transAxes,color=BLUE,fontsize=9,va="top")
                ax.text(.98,.96,"Station elevation - -",transform=ax.transAxes,color=ORANGE,fontsize=9,va="top",ha="right")
                artists.append((sphere_art,cursor,dot,label,clock))
            for f in range(round(duration*fps)):
                fraction=f/(round(duration*fps)-1)
                for track,artist in zip(case["tracks"],artists):
                    # Normalize elapsed fraction only for playback; real UTC is explicit.
                    target=track["seconds"][0]+fraction*np.ptp(track["seconds"])
                    k=int(np.argmin(abs(track["seconds"]-target)))
                    (current,line,arc,nadir),cursor,dot,label,clock=artist
                    p=display_coords(track["body"][k])
                    current.set_data_3d(*p[:,None])
                    line.set_data_3d(*np.array([[0,0,0],p]).T)
                    n=display_coords(track["nadir_body"][k])
                    nadir.set_data_3d(*np.array([[0,0,0],n]).T)
                    theta=np.deg2rad(track["theta"][k])
                    tangent=p-np.array([0,0,1])*np.cos(theta)
                    tangent=unit(tangent) if np.linalg.norm(tangent)>1e-10 else np.array([1.,0.,0.])
                    t=np.linspace(0,theta,50)
                    points=.42*(np.cos(t)[:,None]*[0,0,1]+np.sin(t)[:,None]*tangent)
                    arc.set_data_3d(*points.T)
                    minute=track["minutes"][k]
                    cursor.set_xdata([minute,minute])
                    dot.set_data([minute],[track["theta"][k]])
                    flag="  ·  ROLL" if track["rolling"][k] else ""
                    label.set_text(f"+Y angle {track['theta'][k]:5.1f}°     Elevation {max(0,track['elevation'][k]):4.1f}°{flag}")
                    clock.set_text(f"{track['station'].name}  ·  {iso_utc(track['times'][k])[:19]} UTC")
                writer.grab_frame()
    plt.close(fig)


def write_roll_cycle(cases,out,start,roll_seconds):
    fig,axes=plt.subplots(2,len(cases),figsize=(6.5*len(cases),7),facecolor=BG,squeeze=False)
    records=[]
    for col,case in enumerate(cases):
        orbit=case["orbit"]
        period=2*np.pi/orbit.mean_motion
        times=start+np.linspace(0,period,1200)*u.s
        att=Attitude(orbit,times[0],times[-1],roll_seconds)
        x,y,z,rolling,score=att.evaluate(times)
        nadir=unit(-orbit.positions(times))
        minutes=(times-start).to_value(u.min)
        style_axis(axes[0,col]);style_axis(axes[1,col])
        axes[0,col].plot(minutes,angle_deg(y,nadir),color=BLUE,label="+Y to Earth center")
        axes[0,col].plot(minutes,angle_deg(-y,nadir),color=ORANGE,label="−Y radiator to Earth center")
        axes[0,col].axhline(90,color=MUTED,ls=":",lw=.8)
        axes[0,col].set(ylim=(0,180),ylabel="Angle (deg)",title=f"{case['mltan']:02d}:00 MLTAN")
        axes[0,col].title.set_color(FG)
        axes[0,col].legend(facecolor=BG,labelcolor=FG,fontsize=8,loc="upper right")
        axes[1,col].plot(minutes,np.sum(y*sun_north_basis(orbit,times)[1],axis=-1),color=GREEN)
        axes[1,col].set(ylim=(-1.15,1.15),yticks=[-1,0,1],yticklabels=["−Y north","rolling","+Y north"],
                        xlabel="Minutes from planning epoch",ylabel="Solar-north alignment")
        for flip in att.flips:
            if 0 <= flip <= period:
                for ax in axes[:,col]:
                    ax.axvspan((flip-roll_seconds/2)/60,(flip+roll_seconds/2)/60,color=FG,alpha=.16)
                when=start+flip*u.s
                rep=CartesianRepresentation(orbit.positions(Time([when]))[0]*u.km)
                fixed=GCRS(rep,obstime=when).transform_to(ITRS(obstime=when))
                lat=EarthLocation.from_geocentric(*fixed.cartesian.xyz).lat.value
                records.append({"mltan_hours":case["mltan"],"utc":iso_utc(when),
                                "geodetic_latitude_deg":float(lat)})
    fig.suptitle("Two 180° rolls per orbit preserve detector-Y alignment and favor radiator Earth avoidance",color=FG,fontsize=14,y=.98)
    fig.text(.5,.025,f"Shading: assumed {roll_seconds:g} s rolls. Settled −Y maximizes Earth-center separation; this does not guarantee the entire Earth disk is behind the radiator.",ha="center",color=MUTED,fontsize=9)
    fig.subplots_adjust(left=.11,right=.97,top=.88,bottom=.13,hspace=.26,wspace=.3)
    fig.savefig(out/"roll_cycle.png",dpi=150,facecolor=BG)
    plt.close(fig)
    return records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",default=DEFAULT_LAUNCH["epoch"])
    p.add_argument("--days",type=float,default=7)
    p.add_argument("--mltan-hours",type=int,nargs="+",default=[int(DEFAULT_MLTAN_HOURS)],
                   help=f"MLTAN scenarios in whole hours; default: confirmed {DEFAULT_MLTAN_HOURS:g}:00 orbit only.")
    p.add_argument("--roll-seconds",type=float,default=60)
    p.add_argument("--duration-s",type=float,default=24,help="Playback seconds per MLTAN")
    p.add_argument("--fps",type=int,default=24)
    p.add_argument("--no-movie",action="store_true")
    p.add_argument("--output-dir",type=Path,default=Path("outputs/ground_station_angles"))
    args=p.parse_args(argv)
    if not np.isfinite(args.days) or args.days <= 0 or args.duration_s <= 0 or args.fps <= 0 or round(args.duration_s*args.fps)<2:
        p.error("Search days, playback duration and fps must be positive; movie needs at least two frames.")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    start=utc_time(args.start)
    cases=[]
    metadata={"definition":"Angle at spacecraft between +Y and spacecraft-to-station line of sight, 0..180 degrees",
              "stations":[asdict(s) for s in STATIONS],"search_start_utc":iso_utc(start),"search_days":args.days,
              "attitude":"+Z Sun pointing; +/-Y along projected solar north; settled -Y chosen farther from Earth center; half-turn sense chosen so -Y sweeps away from Earth",
              "solar_pole_source":POLE_SOURCE,"roll_seconds_assumed":args.roll_seconds,
              "selection":"Highest-elevation complete pass for each city in search; all passes >=10 deg also exported",
              "limitations":["Planning epoch and orbital phase are placeholders; not an operational pass schedule.",
                             "City centers, not surveyed antenna locations; geometric ellipsoid horizon, no refraction or terrain.",
                             "Roll centers minimize radiator Earth-center alignment; actual flight trigger, duration and slew sense unconfirmed.",
                             "Y follows solar north between rolls; cannot remain aligned during a finite roll.",
                             "Offline bundled Earth orientation; projected IAU solar pole in GCRS-aligned axes, neglecting sub-arcminute frame corrections.",
                             "No antenna pattern, link budget, spacecraft blockage, attitude errors, or guarantee of Sun visibility in Earth eclipse."],
              "cases":[]}
    all_passes=[]
    for mltan in args.mltan_hours:
        orbit=PlanningOrbit(start,ltan_hours=mltan)
        case={"mltan":mltan,"orbit":orbit,"tracks":[]}
        meta={"orbit":orbit.metadata(),"selected_passes":[]}
        for station in STATIONS:
            passes=find_passes(orbit,station,start,args.days)
            if not passes: raise RuntimeError(f"No complete pass >=10° for {station.name}; increase --days.")
            event=max(passes,key=lambda event:event["max_elevation_deg"])
            track=make_track(orbit,station,start,event,args.roll_seconds)
            case["tracks"].append(track)
            for event_row in passes:
                all_passes.append({"mltan_hours":mltan,"station":station.name,
                                   "aos_utc":iso_utc(start+event_row["aos_s"]*u.s),
                                   "maximum_utc":iso_utc(start+event_row["peak_s"]*u.s),
                                   "los_utc":iso_utc(start+event_row["los_s"]*u.s),
                                   "max_elevation_deg":event_row["max_elevation_deg"]})
            k=int(np.argmin(abs(track["seconds"]-event["peak_s"])))
            meta["selected_passes"].append({"station":station.name,"aos_utc":iso_utc(track["times"][0]),
                "los_utc":iso_utc(track["times"][-1]),"maximum_utc":iso_utc(track["times"][k]),
                "max_elevation_deg":float(track["elevation"][k]),"plus_y_angle_at_max_deg":float(track["theta"][k]),
                "plus_y_angle_min_deg":float(track["theta"].min()),"plus_y_angle_max_deg":float(track["theta"].max()),
                "roll_during_pass":bool(np.any(track["rolling"]))})
            path=out/f"{station.name.lower()}_mltan{mltan:02d}.csv"
            with path.open("w",newline="") as stream:
                w=csv.writer(stream)
                w.writerow(["utc","minutes_from_max_elevation","plus_y_angle_deg","elevation_deg","range_km","radiator_to_earth_center_deg","rolling","los_body_x","los_body_y","los_body_z"])
                w.writerows(zip([iso_utc(t) for t in track["times"]],track["minutes"],track["theta"],track["elevation"],
                                track["range_km"],track["radiator_nadir_deg"],track["rolling"],*track["body"].T))
            print(f"{station.name}, MLTAN {mltan}: peak {track['elevation'][k]:.1f}°, +Y angle {track['theta'][k]:.1f}°",flush=True)
        metadata["cases"].append(meta)
        cases.append(case)
    with (out/"all_passes.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(all_passes[0]));w.writeheader();w.writerows(all_passes)
    write_overview(cases,out)
    metadata["first_orbit_roll_centers"]=write_roll_cycle(cases,out,start,args.roll_seconds)
    (out/"provenance.json").write_text(json.dumps(metadata,indent=2)+"\n")
    if not args.no_movie:
        render_movie(cases,out,args.duration_s,args.fps)
    print(out.resolve(),flush=True)


if __name__ == "__main__":
    main()

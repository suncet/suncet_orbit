import os
import numpy as np
import astropy.units as u
from astropy.constants import G, M_earth, R_earth
from astropy.time import Time
import pandas as pd
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import spiceypy as sp
import wmm2020
from suncet_get_spatial_average_atmospheric_density import average_atmospheric_density

# User input (main things that will be exposed by eventual function)
altitude = 450 * u.km
solar_conditions = 'max'
hamburger_or_hotdog = 'hotdog' # Which configuration is the dual deploy solar panel in

# Constants -- really these are tuneable inputs too, but don't expect to change them much for SunCET
coefficient_of_drag = 2.5 * u.dimensionless_unscaled # depends on shape but 2.5 is typical
asymmetric_surface_area = 0.144 * u.m**2 # area normal to velocity vector, e.g., the solar array sailing against the wind imparting torque in one direction about the spacecraft
if hamburger_or_hotdog == 'hotdog': 
    torque_lever_arm = (0.36 + (0.1 - 0.061)) * u.m # distance from centroid of the asymmetric protrusion (e.g., the solar arrays) to center of mass. 0.36 m [one panel out to get to array center] + (0.1 - 0.061) m [from +Y wall to deployed cg based on CDR Bus slide 13]
elif hamburger_or_hotdog == 'hamburger':
    torque_lever_arm = 0.203 * u.m # distance from centroid of the asymmetric protrusion (e.g., the solar arrays) to center of mass. Evan's idea to dual deploy by first panel normal, second panel unfolds in perpindicular direction

# Input data from SPICE
kernels = [
    os.getenv('suncet_data') + 'spice_kernels/naif0012.tls',
    os.getenv('suncet_data') + 'spice_kernels/pck00010.tpc',
    os.getenv('suncet_data') + 'spice_kernels/de440.bsp',
    os.getenv('suncet_data') + 'spice_kernels/suncet_ephemeris_from_stk.bsp'
]

for kernel in kernels:
    sp.furnsh(kernel)

# Function to compute orbital velocity
def calculate_orbital_velocity(altitude):
    altitude_m = altitude.to(u.m)
    r = R_earth + altitude_m # Distance from center of the Earth
    v = np.sqrt(G * M_earth / r)
    return v

# Function to compute angles using SPICE
def compute_angles(start_time, end_time, interval, spacecraft_id):
    times = sp.str2et([start_time, end_time])
    ephemeris_times = np.linspace(times[0], times[1], interval)

    orbit_sun_vector_to_velocity_vector_angle = []
    local_mag_field_vector_to_body_axes_angles = {'X': [], 'Y': [], 'Z': []}

    for ephemeris_time in ephemeris_times:
        # Get spacecraft state relative to Earth
        state_sc, lt = sp.spkezr(spacecraft_id, ephemeris_time, 'J2000', 'NONE', 'EARTH')
        position_sc = state_sc[:3]
        velocity_sc = state_sc[3:]

        # Get Sun state relative to Earth
        state_sun, lt = sp.spkezr('SUN', ephemeris_time, 'J2000', 'NONE', 'EARTH')
        position_sun = state_sun[:3]

        # Vector from spacecraft to Sun
        sc_to_sun_vector = position_sun - position_sc

        # Angle between sc_to_sun_vector and velocity_sc
        cos_angle = np.dot(sc_to_sun_vector, velocity_sc) / (np.linalg.norm(sc_to_sun_vector) * np.linalg.norm(velocity_sc))
        sun_velocity_angle = np.degrees(np.arccos(cos_angle))
        orbit_sun_vector_to_velocity_vector_angle.append(sun_velocity_angle)

        # Get the magnetic field vector at the spacecraft's position
        mag_field_vector = get_mag_field_vector(position_sc, ephemeris_time)

        # Angles between magnetic field vector and body axes
        for axis, direction in zip(['X', 'Y', 'Z'], np.eye(3)):
            cos_angle = np.dot(mag_field_vector, direction) / np.linalg.norm(mag_field_vector)
            angle = np.degrees(np.arccos(cos_angle))
            local_mag_field_vector_to_body_axes_angles[axis].append(angle)

    return ephemeris_times, orbit_sun_vector_to_velocity_vector_angle, local_mag_field_vector_to_body_axes_angles

def ecef_to_geodetic(x, y, z):
    # WGS-84 ellipsoid parameters
    a = 6378.137  # Earth's radius in kilometers
    e2 = 6.69437999014e-3  # Earth's eccentricity squared

    # Compute longitude
    lon = np.arctan2(y, x)

    # Iterative computation of latitude
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    lat_prev = 0
    while np.abs(lat - lat_prev) > 1e-10:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    # Compute altitude
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    
    return lat, lon, alt

# Function to get the magnetic field vector using the IGRF model
def get_mag_field_vector(position_sc, ephemeris_time):
    # Convert position to geographic coordinates
    x, y, z = position_sc
    lat, lon, alt = ecef_to_geodetic(x, y, z)
    
    # Convert ephemeris time to UTC year
    iso = sp.et2utc(ephemeris_time, 'ISOC', 0)
    time = Time(iso, format='isot', scale='utc')
    year = time.decimalyear

    # Compute the magnetic field vector using the geomag package
    mag_data = wmm2020.wmm(lat, lon, alt, time.decimalyear)
    Bx = mag_data['north'].values[0][0]
    By = mag_data['east'].values[0][0]
    Bz = mag_data['down'].values[0][0]

    # Ensure B_ned only contains the North, East, and Down components
    B_ned = np.array([Bx, By, Bz])

    # Convert NED (North-East-Down) to ECEF (Earth-Centered, Earth-Fixed)
    B_ecef = ned_to_ecef(B_ned, lat, lon)

    return B_ecef

# Function to convert NED to ECEF
def ned_to_ecef(B_ned, lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    # Transformation matrix from NED to ECEF
    R = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])
    B_ecef = R.T @ B_ned  # Apply the transformation
    return B_ecef

# Calculate required data
start_time = '2025-03-20T18:02:00'
end_time = '2025-03-21T18:02:00'
interval = 1440  # 1-minute intervals
spacecraft_id = '-200000'

ephemeris_times, orbit_sun_vector_to_velocity_vector_angles, local_mag_field_vector_to_body_axes_angles = compute_angles(start_time, end_time, interval, spacecraft_id)

# Create dataframes similar to the CSV files previously used
orbit_sun_vector_to_velocity_vector_angles = pd.DataFrame({
    'Time (UTC)': sp.et2utc(ephemeris_times, 'ISOC', 0),
    'Angle (deg)': orbit_sun_vector_to_velocity_vector_angles
})

local_mag_field_vector_to_body_axes_angles = pd.DataFrame({
    'Time (UTC)': sp.et2utc(ephemeris_times, 'ISOC', 0),
    'Angle_X (deg)': local_mag_field_vector_to_body_axes_angles['X'],
    'Angle_Y (deg)': local_mag_field_vector_to_body_axes_angles['Y'],
    'Angle_Z (deg)': local_mag_field_vector_to_body_axes_angles['Z']
})


# XACT-15 specs
adcs_momentum_storage = 0.015 * u.N * u.m * u.s # amount of angular momentum the system can store. The spec sheet says that this is for the whole system, though it's not clear if each wheel can hold this much or if each is less and it knows how to transition the momentum between them. 
torque_rod_dipole_moment = 0.2 * u.A * u.m**2 # basically the "strength" of each torque rod
earth_mag_field_strength_in_leo = 50e-6 * u.T # it actually varies but this is a typical value
torque_rod_duty_cycle = 0.85 # private communication but also based on actual observed performance of XACT-15 on orbit (MinXSS-1). Highest duty cycle is 85%.

# Function to extract closest angles
def extract_closest_angles(df):
    # Vectorized approach to calculate the differences and choose the closest angle
    diff_y = np.abs(df['Angle_Y (deg)'] - 90)
    diff_z = np.abs(df['Angle_Z (deg)'] - 90)
    
    # Adjust for angles closer to 270 by considering 180 degrees complementary
    diff_y = np.minimum(diff_y, 180 - diff_y)
    diff_z = np.minimum(diff_z, 180 - diff_z)
    
    # Use np.where to select the angle value that is closer to 90 or 270 degrees
    closest_angles = np.where(diff_y < diff_z, df['Angle_Y (deg)'], df['Angle_Z (deg)'])
    
    return closest_angles * u.deg

# Calculate intermediate quantities
atmospheric_density = average_atmospheric_density(altitude.value, solar_conditions) * u.kg / u.m**3
velocity = calculate_orbital_velocity(altitude)
asymmetric_surface_area_projected_vs_time = asymmetric_surface_area * np.cos(np.radians(orbit_sun_vector_to_velocity_vector_angles['Angle (deg)'].values))
minutes_since_start = (ephemeris_times - ephemeris_times[0]) / 60

torque_rod_mag_field_angle = extract_closest_angles(local_mag_field_vector_to_body_axes_angles)
torque_rod_torque = (torque_rod_dipole_moment * earth_mag_field_strength_in_leo * np.sin(torque_rod_mag_field_angle.to(u.radian))).to(u.N * u.m)

# Calculate main values of interest
torque_drag_peak = (1/2 * atmospheric_density * velocity**2 * coefficient_of_drag * asymmetric_surface_area * torque_lever_arm).to(u.N * u.m)
time_to_saturate_wheels_if_always_peak_drag = (adcs_momentum_storage / torque_drag_peak).to(u.minute) # From wheel at 0 speed, how long until it saturates with disturbance torques above?
torque_drag_vs_time = (1/2 * atmospheric_density * velocity**2 * coefficient_of_drag * asymmetric_surface_area_projected_vs_time * torque_lever_arm).to(u.N * u.m)
cumulative_torque_drag = cumtrapz(torque_drag_vs_time, (minutes_since_start * 60), initial=0) * torque_drag_vs_time.unit * u.s

cumulative_torque_rod = np.zeros_like(cumulative_torque_drag)
net_momentum = np.zeros_like(cumulative_torque_drag)
current_cumulative_torque_rod = 0
for i in range(1, len(minutes_since_start)):
    if net_momentum[i-1] < 0: 
        direction_of_torque_rod = 1
    else: 
        direction_of_torque_rod = -1
    cumulative_torque_rod[i] = direction_of_torque_rod * torque_rod_torque[i] * torque_rod_duty_cycle * ((minutes_since_start[i] - minutes_since_start[i-1]) * u.minute).to(u.s)
    cumulative_torque_rod[i] += cumulative_torque_rod[i-1] # since drag is cumulative, also need rods to be cumulative in order to compute the net
    net_momentum[i] = cumulative_torque_drag[i] + cumulative_torque_rod[i]

target_index = np.argmax(net_momentum >= adcs_momentum_storage)
target_time = minutes_since_start[target_index] if net_momentum[target_index] >= adcs_momentum_storage else None

# Plot the torques
fig, axs = plt.subplots(4, 1, figsize=(8, 13))
axs[0].plot(minutes_since_start, torque_drag_vs_time)
axs[0].set_title('SunCET {} km, noon-midnight, {}, solar panel {} config'.format(altitude.value, solar_conditions, hamburger_or_hotdog))
axs[0].set_ylabel('drag torque [Nm]')
axs[0].grid(True)

axs[1].plot(minutes_since_start, cumulative_torque_drag)
axs[1].set_ylabel('cumulative drag momentum [Nms]')
axs[1].grid(True)

axs[2].plot(minutes_since_start, cumulative_torque_rod)
axs[2].set_ylabel('cumulative torque rod momentum [Nms]')
axs[2].grid(True)

axs[3].plot(minutes_since_start, net_momentum)
axs[3].set_xlabel('time [minutes since start]')
axs[3].set_ylabel('net momentum [Nms]')
axs[3].grid(True)

axs[3].axhline(y=adcs_momentum_storage.value, color='r', linestyle='--', label=f'XACT-15 angular momentum storage: {adcs_momentum_storage.value} Nms')
axs[3].axhline(y=-adcs_momentum_storage.value, color='r', linestyle='--')
axs[3].set_ylim([-0.020, 0.020])
axs[3].legend()

plt.tight_layout()

if np.max(net_momentum) < adcs_momentum_storage: 
    print('The system momentum never exceeds limits. Hooray!')
else: 
    print('The disturbance torque is greater than what the system can handle. The wheels will saturate in {:.2f} minutes (assuming best case that they started from rest).'.format(target_time))


pass
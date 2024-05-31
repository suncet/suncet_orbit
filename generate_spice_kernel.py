import spiceypy as sp
import numpy as np
import os

def create_ephemeris(altitude_km, ltan_hours, start_time, end_time, interval_seconds, path_filename=None):
    times, positions, velocities = generate_ephemeris_data(altitude_km, ltan_hours, start_time, end_time, interval_seconds)
    
    if path_filename is None:
        ephemeris_path = os.getenv('suncet_data') + 'spice_kernels/suncet_{}km_{}ltanHour.bsp'.format(altitude_km, ltan_hours)
        ck_path = os.getenv('suncet_data') + 'spice_kernels/suncet_{}km_{}ltanHour.ck'.format(altitude_km, ltan_hours)
    write_spice_kernel(times, positions, velocities, ephemeris_path)
    create_ck_file(times, positions, ck_path)


def generate_ephemeris_data(altitude_km, ltan_hours, start_time, end_time, interval_seconds):
    altitude_m = altitude_km * 1000.0
    mu = 398600.4418 * 1e9
    R_earth = 6378.137 * 1000.0
    a = R_earth + altitude_m
    n = np.sqrt(mu / a**3)
    T = 2 * np.pi / n
    times = np.arange(start_time, end_time, interval_seconds)
    positions = []
    velocities = []
    RAAN_initial = (ltan_hours / 24.0) * 360.0

    for t in times:
        M = n * t
        E = M
        nu = E
        x = a * np.cos(nu)
        y = a * np.sin(nu)
        z = 0
        RAAN = RAAN_initial + 360.0 * (t / T)
        RAAN_rad = np.radians(RAAN)
        position = np.array([
            x * np.cos(RAAN_rad) - y * np.sin(RAAN_rad),
            x * np.sin(RAAN_rad) + y * np.cos(RAAN_rad),
            z
        ])
        vx = -a * np.sin(nu) * n / (1 - 0 * np.cos(nu))
        vy = a * np.cos(nu) * n / (1 - 0 * np.cos(nu))
        vz = 0
        velocity = np.array([
            vx * np.cos(RAAN_rad) - vy * np.sin(RAAN_rad),
            vx * np.sin(RAAN_rad) + vy * np.cos(RAAN_rad),
            vz
        ])
        positions.append(position)
        velocities.append(velocity)

    return times, np.array(positions), np.array(velocities)


def write_spice_kernel(times, positions, velocities, path_filename=None):
    if os.path.exists(path_filename):
        os.remove(path_filename)
    handle = sp.spkopn(path_filename, "New SPICE Kernel", 0)
    step = times[1] - times[0]  # Compute the step size in seconds
    states = np.hstack((positions, velocities))
    sp.spkw08(
        handle,
        -200000,  # Spacecraft ID
        399,      # ID for Earth
        'J2000',  # Reference frame
        times[0], # Start time of interval covered by segment
        times[-1], # End time of interval covered by segment
        'Ephemeris Data', # Segment identifier
        3, # Degree of interpolating polynomials
        len(states), # Number of states
        states, # State records: positions and velocities
        times[0], # Epoch of first state in states array
        step # Time step separating epochs of states
    )
    sp.spkcls(handle)


def create_ck_file(times, positions, path_filename=None):
    if os.path.exists(path_filename):
        os.remove(path_filename)
    handle = sp.ckopn(path_filename, "CK Kernel", 0)
    av = [0, 0, 0]  # not changing angular velocity here
    av_array = np.array([av, av])

    for i in range(len(times) - 1):
        position_sc = positions[i]
        position_sun, _ = sp.spkezr('SUN', times[i], 'J2000', 'NONE', 'EARTH')
        quat = compute_attitude_quaternion(position_sc, position_sun[:3])

        start_time = float(times[i])
        stop_time = float(times[i + 1])
        position_sc = positions[i]
        position_sun, _ = sp.spkezr('SUN', start_time, 'J2000', 'NONE', 'EARTH')
        quat = compute_attitude_quaternion(position_sc, position_sun[:3])

        # Convert to contiguous arrays
        start_times = np.ascontiguousarray([start_time, stop_time], dtype=np.float64)
        stop_times = np.ascontiguousarray([start_time, stop_time], dtype=np.float64)
        quats = np.ascontiguousarray([quat, quat], dtype=np.float64)

        print(f"Start Time: {start_time}, Stop Time: {stop_time}")
        print(f"Quaternions: {quats}")
        print(f"Angular Velocity: {av_array}")

        sp.ckw02(
            handle,
            start_time,  # The beginning encoded SCLK of the segment
            stop_time,  # The ending encoded SCLK of the segment
            -200000,  # NAIF instrument ID code (same as spacecraft ID here)
            "J2000",  # Reference frame
            "SunCET Attitude",  # Segment identifier
            2,  # Number of pointing records (start and end times)
            start_times,  # Encoded SCLK interval start times
            stop_times,  # Encoded SCLK interval stop times
            quats,  # Quaternions representing instrument pointing
            av_array,  # Angular velocity vectors
            np.array([1.0, 1.0], dtype=np.float64)  # Rates: number of seconds per tick for each interval
        )
    sp.ckcls(handle)


def compute_attitude_quaternion(position_sc, position_sun):
    # Calculate the -Y body vector alignment with the sun
    y_body_vector = (position_sun - position_sc) / np.linalg.norm(position_sun - position_sc)
    
    # Assuming the solar north vector in the J2000 frame
    solar_north_vector = np.array([0, 0, 1])  # Adjust as necessary if using a different frame
    
    # Compute the -Z body vector to align with the solar north direction
    z_body_vector = solar_north_vector / np.linalg.norm(solar_north_vector)
    
    # Recompute the -Y body vector to be orthogonal to -Z and along the sun direction
    y_body_vector = y_body_vector - np.dot(y_body_vector, z_body_vector) * z_body_vector
    y_body_vector /= np.linalg.norm(y_body_vector)
    
    # Compute the X body vector to ensure a right-handed coordinate system
    x_body_vector = np.cross(y_body_vector, z_body_vector)
    
    # Create the rotation matrix
    rotation_matrix = np.vstack((x_body_vector, y_body_vector, z_body_vector)).T
    
    # Convert the rotation matrix to a quaternion
    rotation_matrix = np.ascontiguousarray(rotation_matrix)
    quat = sp.m2q(rotation_matrix)
    return quat


def load_kernels():
    kernels = [
        os.getenv('suncet_data') + 'spice_kernels/naif0012.tls',
        os.getenv('suncet_data') + 'spice_kernels/pck00010.tpc',
        os.getenv('suncet_data') + 'spice_kernels/de440.bsp',
    ]
    for kernel in kernels:
        sp.furnsh(kernel)


if __name__ == "__main__":
    load_kernels()
    start_time = sp.str2et('2025-03-21T00:00:00')
    end_time = sp.str2et('2025-03-22T00:00:00')
    interval_seconds = 60
    altitude_km = 400
    ltan_hours = 12

    create_ephemeris(altitude_km, ltan_hours, start_time, end_time, interval_seconds)

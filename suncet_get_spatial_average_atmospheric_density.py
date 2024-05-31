import numpy as np
from msise00 import rungtd1d
from datetime import datetime

def average_atmospheric_density(altitude_km, solar_conditions='mean'):
    # Define solar activity levels for min, mean, and max conditions
    solar_parameters = {
        'min': {'f107': 70, 'f107a': 70, 'f107s': 70, 'Ap': 4},
        'mean': {'f107': 150, 'f107a': 150, 'f107s': 150, 'Ap': 12},
        'max': {'f107': 200, 'f107a': 200, 'f107s': 200, 'Ap': 20}
    }

    # Extract parameters based on solar conditions input
    solar_indices = solar_parameters[solar_conditions]

    # Sample range of latitudes and longitudes
    lat_range = np.linspace(-90, 90, 19)
    lon_range = np.linspace(-180, 180, 19)
    time = datetime.now()

    # List to store densities
    densities = []

    # Iterate over latitudes and longitudes
    for lat in lat_range:
        for lon in lon_range:
            # Compute atmospheric density using msise00
            result = rungtd1d(time, altkm=altitude_km, glat=lat, glon=lon, indices=solar_indices)
            
            # Extract total mass density
            density = result.squeeze()['Total'].item()
            
            # Store the density
            densities.append(density)

    # Calculate the average density
    average_density = np.mean(densities)

    print(f"The average atmospheric density at {altitude_km} km altitude under {solar_conditions} solar conditions is {average_density:.2e} kg/m^3")
    return average_density

if __name__ == "__main__":
    altitude_km = 400  # Altitude in kilometers
    solar_conditions = 'max'  # 'min', 'mean', or 'max'
    average_density = average_atmospheric_density(altitude_km, solar_conditions)

    print(f"The average atmospheric density at {altitude_km} km altitude under {solar_conditions} solar conditions is {average_density:.2e} kg/m^3")

import os
import requests
import spiceypy as sp

# Function to download generic kernels
def download_generic_kernels():
    save_path = os.getenv('suncet_data') + 'spice_kernels/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # URLs for the kernels
    lsk_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls'
    pck_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc'
    spk_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp'

    # Download and save the Leap Seconds Kernel
    lsk_path = os.path.join(save_path, 'naif0012.tls')
    with requests.get(lsk_url) as response:
        response.raise_for_status()  # Ensure we notice bad responses
        with open(lsk_path, 'wb') as file:
            file.write(response.content)
    
    # Download and save the Planetary Constants Kernel (PCK)
    pck_path = os.path.join(save_path, 'pck00010.tpc')
    with requests.get(pck_url) as response:
        response.raise_for_status()  # Ensure we notice bad responses
        with open(pck_path, 'wb') as file:
            file.write(response.content)

    # Download and save the DE440 SPK Kernel
    spk_path = os.path.join(save_path, 'de440.bsp')
    with requests.get(spk_url) as response:
        response.raise_for_status()  # Ensure we notice bad responses
        with open(spk_path, 'wb') as file:
            file.write(response.content)
    
    return lsk_path, pck_path, spk_path

# Download the kernels
lsk_path, pck_path, spk_path = download_generic_kernels()

print(f"Leap Seconds Kernel saved to: {lsk_path}")
print(f"PCK Kernel saved to: {pck_path}")
print(f"SPK Kernel saved to: {spk_path}")

# Load the kernels to verify they work
sp.furnsh(lsk_path)
sp.furnsh(pck_path)
sp.furnsh(spk_path)

# Verify the kernels can be loaded
print(f"Kernels can be loaded. These are the total loaded kernels: {sp.ktotal('ALL')}")

# CIE Colorimetry

A Python package for calculating CIE colorimetry parameters from emission spectra in photon counts.

## Installation

```bash
pip install cie_colorimetry
```

## Usage

```python
from cie_colorimetry import (
    calculate_cie_coordinates, 
    calculate_dominant_wavelength, 
    calculate_saturation,
    plot_cie_coordinates
)
from cie_colorimetry.utils import load_spectrum_from_file

# Load spectrum from file (wavelength in first column, photon counts in second)
wavelengths, photon_counts = load_spectrum_from_file("spectrum.csv")

# Or specify columns by name
wavelengths, photon_counts = load_spectrum_from_file(
    "spectrum.csv",
    wavelength_col="Wavelength (nm)",
    counts_col="Counts"
)

# Calculate CIE coordinates
x, y = calculate_cie_coordinates(wavelengths, photon_counts)

# Calculate dominant wavelength
dominant_wavelength = calculate_dominant_wavelength(x, y)

# Calculate saturation
saturation = calculate_saturation(x, y, dominant_wavelength)

# Print results
print(f"CIE coordinates: ({x:.3f}, {y:.3f})")
print(f"Dominant wavelength: {dominant_wavelength:.1f} nm")
print(f"Saturation: {saturation:.1f}%")

# Plot results
plot_cie_coordinates(x, y, label=f"λd={dominant_wavelength:.1f}nm\nS={saturation:.1f}%")
```

## Data Format

The spectrum file should be a CSV file with two columns:
1. Wavelength (in nanometers, micrometers, or meters)
2. Photon counts (arbitrary units)

The package will automatically:
- Convert wavelengths to nanometers if needed
- Convert photon counts to energy units for color calculations
- Handle negative or NaN values
- Interpolate the spectrum to match CIE wavelength points

## Required Data Files

The package requires two CSV files in the `cie_colorimetry/data/` directory:

1. `cie_1931.csv`: CIE 1931 2-degree color matching functions
2. `spectral_locus.csv`: Spectral locus data

Both files should be included in the package distribution.
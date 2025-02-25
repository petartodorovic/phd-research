# CIE Colorimetry

A Python package for calculating CIE colorimetry parameters from emission spectra and visualizing them on the CIE 1931 chromaticity diagram.

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

# Load spectrum from file
wavelengths, intensities = load_spectrum_from_file("spectrum.csv")

# Calculate CIE coordinates
x, y = calculate_cie_coordinates(wavelengths, intensities)

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

## Required Data Files

The package requires two CSV files in the `cie_colorimetry/data/` directory:

1. `cie_1931.csv`: CIE 1931 2-degree color matching functions
2. `spectral_locus.csv`: Spectral locus data

Both files should be included in the package distribution.
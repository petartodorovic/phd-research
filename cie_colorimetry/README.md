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

# Load spectrum from a CSV or Excel file (wavelength in first column, photon counts in second)
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

### All-in-one analysis

`analyze_spectrum` returns every parameter as a `ColorimetryResult`. It accepts a `Spectrum` (wavelengths stored in nm) or plain wavelength and count arrays:

```python
from cie_colorimetry import Spectrum, analyze_spectrum

spectrum = Spectrum.from_file("spectrum.csv")
result = analyze_spectrum(spectrum)  # or analyze_spectrum(wavelengths, photon_counts)

print(f"CIE coordinates: ({result.x:.3f}, {result.y:.3f})")
print(f"Peak wavelength: {result.peak_wavelength:.1f} nm")
if result.dominant_wavelength is not None:
    print(f"Dominant wavelength: {result.dominant_wavelength:.1f} nm")
print(f"Saturation: {result.saturation:.1f}%")
```

`dominant_wavelength` is `None` when the ray from the white point (Illuminant E, `WHITE_POINT`) through the sample does not reach the visible spectral locus, for example for purple colours.

### 3D plots

Both plots take `Spectrum` objects, mark each spectrum's dominant (◆) and peak (▲) wavelengths, and return the `ColorimetryResult` for each spectrum.

```python
from cie_colorimetry import Spectrum, plot_cie_3d, plot_spectra_waterfall

samples = [Spectrum.from_file(path) for path in ["blue.csv", "green.csv", "red.csv"]]

# Spectra standing on the CIE 1931 diagram, with the ray from the white point
# through each sample to its dominant wavelength (up to 8 spectra)
plot_cie_3d(samples, labels=["Blue", "Green", "Red"],
            colors=["#2a78d6", "#008300", "#e34948"], save_path="cie_3d.png")

# An ordered series (e.g. a temperature sweep) one behind another, with trend
# lines showing how the dominant and peak wavelengths shift
sweep = [Spectrum.from_file(f"sample_{t}K.csv") for t in (100, 150, 200, 250)]
plot_spectra_waterfall(sweep, labels=["100 K", "150 K", "200 K", "250 K"],
                       save_path="waterfall.png")
```

Each spectrum is normalized to a maximum of 1 unless `normalize=False`. Use `elev` and `azim` to change the camera angle, and pass `ax` (a 3D axis) to draw into an existing figure.

## Data Format

The spectrum file can be a CSV (`.csv`) or Excel (`.xls`, `.xlsx`, or `.xlsm`) file with two columns:
1. Wavelength (in nanometers, micrometers, or meters)
2. Photon counts (arbitrary units)

The package will automatically:
- Convert wavelengths to nanometers if needed
- Convert photon counts to energy units for color calculations
- Handle negative or NaN values
- Interpolate the spectrum to match CIE wavelength points

CSV files may use comma, tab, or whitespace delimiters. Excel files are read from their first worksheet.

## Required Data Files

The package requires two reference files in the `cie_colorimetry/data/` directory:

1. `cie_1931.csv`: CIE 1931 2-degree color matching functions
2. `spectral_locus.xlsx`: Spectral locus data

Both files should be included in the package distribution.

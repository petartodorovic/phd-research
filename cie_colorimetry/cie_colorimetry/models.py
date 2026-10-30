from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import load_spectrum_from_file, to_nanometers


@dataclass(frozen=True)
class Spectrum:
    """
    An emission spectrum with wavelengths in nanometers.

    Attributes:
        wavelengths (ndarray): Wavelength values in nm. Values given in meters or
            micrometers are converted on construction.
        counts (ndarray): Photon counts (arbitrary units) at each wavelength
    """
    wavelengths: np.ndarray
    counts: np.ndarray

    def __post_init__(self):
        wavelengths = np.asarray(self.wavelengths, dtype=float)
        counts = np.asarray(self.counts, dtype=float)

        if wavelengths.ndim != 1 or counts.ndim != 1:
            raise ValueError("Wavelengths and counts must be one-dimensional")
        if len(wavelengths) != len(counts):
            raise ValueError("Wavelengths and counts must have the same length")
        if len(wavelengths) == 0:
            raise ValueError("Spectrum must contain at least one point")

        wavelengths = to_nanometers(wavelengths)

        # Frozen dataclasses need object.__setattr__ to store the normalized arrays
        object.__setattr__(self, 'wavelengths', wavelengths)
        object.__setattr__(self, 'counts', counts)

    @classmethod
    def from_file(cls, file_path, wavelength_col=0, counts_col=1):
        """Load a spectrum from a CSV or Excel file (see load_spectrum_from_file)"""
        return cls(*load_spectrum_from_file(file_path, wavelength_col, counts_col))

    @property
    def peak_wavelength(self) -> float:
        """Wavelength (nm) of the highest photon count"""
        return float(self.wavelengths[np.argmax(self.counts)])


@dataclass(frozen=True)
class ColorimetryResult:
    """
    CIE 1931 colorimetry parameters for a spectrum.

    Attributes:
        x, y (float): CIE 1931 chromaticity coordinates
        dominant_wavelength (Optional[float]): Dominant wavelength in nm, or None when
            the ray from the white point does not reach the visible spectral locus
        saturation (float): Colour saturation in percent (0 when there is no dominant wavelength)
        peak_wavelength (float): Wavelength (nm) of the highest photon count
    """
    x: float
    y: float
    dominant_wavelength: Optional[float]
    saturation: float
    peak_wavelength: float

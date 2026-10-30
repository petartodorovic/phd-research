import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from functools import lru_cache
from pathlib import Path

from .utils import read_spectrum_table

DATA_DIR = Path(__file__).resolve().parent / 'data'

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def _load_reference_table(data_dir, stem):
    """Load a packaged CSV or Excel reference table by filename stem."""
    for suffix in ('.csv', '.xlsx', '.xlsm', '.xls'):
        path = Path(data_dir) / f'{stem}{suffix}'
        if path.exists():
            return read_spectrum_table(path)
    raise FileNotFoundError(f"Could not find packaged reference data '{stem}'")

class CIECalculator:
    def __init__(self):
        # Load CIE 1931 color matching functions
        data_dir = DATA_DIR
        cie_data = _load_reference_table(data_dir, 'cie_1931')
        self.wavelengths = cie_data['wavelength'].values
        self.x_bar = cie_data['x_bar'].values
        self.y_bar = cie_data['y_bar'].values
        self.z_bar = cie_data['z_bar'].values
        
        # Load spectral locus data
        self.spectral_locus = _load_reference_table(data_dir, 'spectral_locus')
        
        # Rename columns to match expected format
        if 'Wavelength (nm)' in self.spectral_locus.columns:
            self.spectral_locus = self.spectral_locus.rename(columns={'Wavelength (nm)': 'wavelength'})
        
        # Create x, y coordinates if they don't exist (based on CIE standard)
        if 'x' not in self.spectral_locus.columns or 'y' not in self.spectral_locus.columns:
            self.generate_spectral_locus_coordinates()

    def generate_spectral_locus_coordinates(self):
        """Generate x,y coordinates for the spectral locus from the colour matching functions"""
        wavelengths = self.spectral_locus['wavelength'].values

        # Find closest matching wavelength in CIE data for each locus point
        idx = np.abs(self.wavelengths[None, :] - wavelengths[:, None]).argmin(axis=1)
        X, Y, Z = self.x_bar[idx], self.y_bar[idx], self.z_bar[idx]

        # Convert to chromaticity coordinates, using (0, 0) where X + Y + Z is zero
        sum_XYZ = X + Y + Z
        safe_sum = np.where(sum_XYZ > 0, sum_XYZ, 1)
        self.spectral_locus['x'] = np.where(sum_XYZ > 0, X / safe_sum, 0)
        self.spectral_locus['y'] = np.where(sum_XYZ > 0, Y / safe_sum, 0)
    
    def convert_photon_counts_to_energy(self, wavelengths, photon_counts):
        """Convert photon counts to energy/power units"""
        return photon_counts / wavelengths
    
    def interpolate_spectrum(self, wavelengths, intensities):
        """Interpolate spectrum to match CIE wavelength points"""
        # Convert wavelengths to nanometers; check the smallest unit first
        if np.median(wavelengths) < 0.1:  # Assume wavelengths are in meters
            wavelengths = wavelengths * 1e9
        elif np.median(wavelengths) < 100:  # Assume wavelengths are in micrometers
            wavelengths = wavelengths * 1000
            
        # Convert photon counts to energy
        energy_intensities = self.convert_photon_counts_to_energy(wavelengths, intensities)
        
        # Create interpolation function
        f = interp1d(wavelengths, energy_intensities, bounds_error=False, fill_value=0)
        return f(self.wavelengths)
    
    def calculate_xyz(self, wavelengths, photon_counts):
        """Calculate XYZ tristimulus values from photon count spectrum"""
        # Interpolate spectrum to match CIE wavelength points
        spectrum = self.interpolate_spectrum(wavelengths, photon_counts)
        
        # Calculate tristimulus values
        X = _trapezoid(spectrum * self.x_bar, self.wavelengths)
        Y = _trapezoid(spectrum * self.y_bar, self.wavelengths)
        Z = _trapezoid(spectrum * self.z_bar, self.wavelengths)
        
        return X, Y, Z
    
    def calculate_xy(self, X, Y, Z):
        """Calculate xy chromaticity coordinates"""
        sum_XYZ = X + Y + Z
        if sum_XYZ == 0:
            return 0, 0
        
        x = X / sum_XYZ
        y = Y / sum_XYZ
        
        return x, y
    
    def find_dominant_wavelength(self, x, y):
        """Calculate dominant wavelength using the spectral locus.

        The dominant wavelength is where the ray from the reference white point
        through the sample chromaticity intersects the visible spectral locus.
        Locus points beyond 700 nm are nearly coincident in chromaticity, so they
        are excluded from the intersection search.
        """
        # Reference white point (Illuminant E)
        xw, yw = 0.3333, 0.3333
        
        # Get spectral locus data
        spectral_x = self.spectral_locus['x'].values
        spectral_y = self.spectral_locus['y'].values
        spectral_wavelengths = self.spectral_locus['wavelength'].values
        
        # Ray from the white point through the sample point: p + t*r, t > 0.
        p = np.array([xw, yw], dtype=float)
        r = np.array([x - xw, y - yw], dtype=float)
        if np.linalg.norm(r) < 1e-12:
            return None

        def cross(a, b):
            return a[0] * b[1] - a[1] * b[0]

        intersections = []
        for i in range(len(spectral_x) - 1):
            wavelength_start = spectral_wavelengths[i]
            wavelength_end = spectral_wavelengths[i + 1]
            if not (380 <= wavelength_start <= 700 and 380 <= wavelength_end <= 700):
                continue

            q = np.array([spectral_x[i], spectral_y[i]], dtype=float)
            s = np.array([
                spectral_x[i + 1] - spectral_x[i],
                spectral_y[i + 1] - spectral_y[i],
            ])
            denominator = cross(r, s)
            if abs(denominator) < 1e-12:
                continue

            offset = q - p
            ray_distance = cross(offset, s) / denominator
            segment_position = cross(offset, r) / denominator
            if ray_distance > 1e-12 and 0 <= segment_position <= 1:
                wavelength = wavelength_start + segment_position * (wavelength_end - wavelength_start)
                intersections.append((ray_distance, wavelength))

        if not intersections:
            return None
        return min(intersections, key=lambda hit: hit[0])[1]
    
    def calculate_saturation(self, x, y, dominant_wavelength):
        """Calculate color saturation"""
        if dominant_wavelength is None:
            return 0
            
        # Reference white point (Illuminant E)
        xw, yw = 0.3333, 0.3333
        
        # Find spectral locus point for dominant wavelength
        # Use nearest wavelength if exact match not found
        wl_diff = np.abs(self.spectral_locus['wavelength'].values - dominant_wavelength)
        closest_idx = np.argmin(wl_diff)
        xs = self.spectral_locus['x'].values[closest_idx]
        ys = self.spectral_locus['y'].values[closest_idx]
        
        # Calculate distances
        color_distance = np.sqrt((x - xw)**2 + (y - yw)**2)
        total_distance = np.sqrt((xs - xw)**2 + (ys - yw)**2)
        
        # Calculate saturation as percentage
        if total_distance < 1e-10:  # Avoid division by zero
            return 0
            
        saturation = (color_distance / total_distance) * 100
        
        return saturation

@lru_cache(maxsize=None)
def get_calculator():
    """Return a shared CIECalculator so reference data is loaded from disk only once"""
    return CIECalculator()

def calculate_cie_coordinates(wavelengths, photon_counts):
    """Calculate CIE xy coordinates from wavelengths and photon counts"""
    calculator = get_calculator()
    X, Y, Z = calculator.calculate_xyz(wavelengths, photon_counts)
    x, y = calculator.calculate_xy(X, Y, Z)
    return x, y

def calculate_dominant_wavelength(x, y):
    """Calculate dominant wavelength from CIE coordinates"""
    return get_calculator().find_dominant_wavelength(x, y)

def calculate_saturation(x, y, dominant_wavelength):
    """Calculate color saturation"""
    return get_calculator().calculate_saturation(x, y, dominant_wavelength)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pkg_resources

class CIECalculator:
    def __init__(self):
        # Load CIE 1931 color matching functions
        data_path = pkg_resources.resource_filename('cie_colorimetry', 'data/cie_1931.csv')
        cie_data = pd.read_csv(data_path)
        self.wavelengths = cie_data['wavelength'].values
        self.x_bar = cie_data['x_bar'].values
        self.y_bar = cie_data['y_bar'].values
        self.z_bar = cie_data['z_bar'].values
        
        # Load spectral locus data
        locus_path = pkg_resources.resource_filename('cie_colorimetry', 'data/spectral_locus.csv')
        self.spectral_locus = pd.read_excel(locus_path, engine='openpyxl')
        
        # Rename columns to match expected format
        if 'Wavelength (nm)' in self.spectral_locus.columns:
            self.spectral_locus = self.spectral_locus.rename(columns={'Wavelength (nm)': 'wavelength'})
        
        # Create x, y coordinates if they don't exist (based on CIE standard)
        if 'x' not in self.spectral_locus.columns or 'y' not in self.spectral_locus.columns:
            # Using wavelength to compute standard CIE xy coordinates
            # This is an approximation - replace with precise values when available
            self.generate_spectral_locus_coordinates()
    
    def generate_spectral_locus_coordinates(self):
        """Generate x,y coordinates for the spectral locus based on wavelengths"""
        wavelengths = self.spectral_locus['wavelength'].values
        x_coords = []
        y_coords = []
        
        # Standard approximation for spectral locus coordinates
        for wl in wavelengths:
            # Map visible wavelength range to appropriate xy chromaticity coordinates
            # Based on standard CIE 1931 color space
            if 380 <= wl <= 700:
                # Find closest matching wavelength in CIE data
                idx = np.argmin(np.abs(self.wavelengths - wl))
                X = self.x_bar[idx]
                Y = self.y_bar[idx]
                Z = self.z_bar[idx]
                
                # Convert to chromaticity coordinates
                sum_XYZ = X + Y + Z
                if sum_XYZ > 0:
                    x = X / sum_XYZ
                    y = Y / sum_XYZ
                else:
                    x = 0
                    y = 0
                
                x_coords.append(x)
                y_coords.append(y)
            else:
                # Outside visible range
                x_coords.append(0.3333)
                y_coords.append(0.3333)
        
        # Add to dataframe
        self.spectral_locus['x'] = x_coords
        self.spectral_locus['y'] = y_coords
    
    def convert_photon_counts_to_energy(self, wavelengths, photon_counts):
        """Convert photon counts to energy/power units"""
        return photon_counts / wavelengths
    
    def interpolate_spectrum(self, wavelengths, intensities):
        """Interpolate spectrum to match CIE wavelength points"""
        # Convert wavelengths to nanometers if in meters
        if np.median(wavelengths) < 100:  # Assume wavelengths are in micrometers
            wavelengths = wavelengths * 1000
        elif np.median(wavelengths) < 0.1:  # Assume wavelengths are in meters
            wavelengths = wavelengths * 1e9
            
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
        X = np.trapz(spectrum * self.x_bar, self.wavelengths)
        Y = np.trapz(spectrum * self.y_bar, self.wavelengths)
        Z = np.trapz(spectrum * self.z_bar, self.wavelengths)
        
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
        """Calculate dominant wavelength using spectral locus"""
        # Reference white point (Illuminant E)
        xw, yw = 0.3333, 0.3333
        
        # Get spectral locus data
        spectral_x = self.spectral_locus['x'].values
        spectral_y = self.spectral_locus['y'].values
        spectral_wavelengths = self.spectral_locus['wavelength'].values
        
        # Find intersection of line from white point through sample point to spectral locus
        dx = x - xw
        dy = y - yw
        
        # Calculate intersections with all line segments of spectral locus
        for i in range(len(spectral_x) - 1):
            x1, y1 = spectral_x[i], spectral_y[i]
            x2, y2 = spectral_x[i + 1], spectral_y[i + 1]
            
            # Line intersection calculation
            denominator = (x2 - x1) * dy - (y2 - y1) * dx
            if abs(denominator) < 1e-10:  # Avoid division by zero
                continue
                
            ua = ((x2 - x1) * (yw - y1) - (y2 - y1) * (xw - x1)) / denominator
            ub = (dx * (yw - y1) - dy * (xw - x1)) / denominator
            
            if 0 <= ub <= 1 and ua >= 0:
                # Interpolate wavelength at intersection point
                wavelength = spectral_wavelengths[i] + ub * (spectral_wavelengths[i + 1] - spectral_wavelengths[i])
                return wavelength
        
        return None
    
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

def calculate_cie_coordinates(wavelengths, photon_counts):
    """Calculate CIE xy coordinates from wavelengths and photon counts"""
    calculator = CIECalculator()
    X, Y, Z = calculator.calculate_xyz(wavelengths, photon_counts)
    x, y = calculator.calculate_xy(X, Y, Z)
    return x, y

def calculate_dominant_wavelength(x, y):
    """Calculate dominant wavelength from CIE coordinates"""
    calculator = CIECalculator()
    return calculator.find_dominant_wavelength(x, y)

def calculate_saturation(x, y, dominant_wavelength):
    """Calculate color saturation"""
    calculator = CIECalculator()
    return calculator.calculate_saturation(x, y, dominant_wavelength)
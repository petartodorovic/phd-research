import pandas as pd
import numpy as np
from pathlib import Path


def to_nanometers(wavelengths):
    """Convert wavelengths to nanometers, inferring the unit from their median value"""
    wavelengths = np.asarray(wavelengths, dtype=float)
    # Check the smallest unit first
    if np.median(wavelengths) < 0.1:  # Assume wavelengths are in meters
        return wavelengths * 1e9
    if np.median(wavelengths) < 100:  # Assume wavelengths are in micrometers
        return wavelengths * 1000
    return wavelengths

def read_spectrum_table(file_path):
    """Read a CSV or Excel table, including legacy mislabeled files."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {'.xls', '.xlsx', '.xlsm'}:
        return pd.read_excel(path, engine='openpyxl')
    if suffix in {'.csv', '.txt', '.dat'}:
        try:
            return pd.read_csv(path, sep=None, engine='python')
        except (UnicodeDecodeError, pd.errors.ParserError):
            # Keep compatibility with the repository's former Excel file
            # that was distributed with a .csv extension.
            return pd.read_excel(path, engine='openpyxl')

    raise ValueError(
        f"Unsupported spectrum file format '{path.suffix}'. "
        "Use CSV, XLS, XLSX, or XLSM."
    )

def load_spectrum_from_file(file_path, wavelength_col=0, counts_col=1):
    """
    Load spectrum from a CSV or Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the spectrum file
    wavelength_col : int or str
        Column index or name for wavelength data
    counts_col : int or str
        Column index or name for photon counts data
        
    Returns:
    --------
    wavelengths : numpy array
        Wavelength values
    photon_counts : numpy array
        Photon count values
    """
    try:
        # Read space-delimited file with headers
        data = read_spectrum_table(file_path)
        
        # Handle column selection
        if isinstance(wavelength_col, int):
            wavelengths = data.iloc[:, wavelength_col].values
        else:
            wavelengths = data[wavelength_col].values
            
        if isinstance(counts_col, int):
            photon_counts = data.iloc[:, counts_col].values
        else:
            photon_counts = data[counts_col].values
        
        # Convert to numeric, handling any string values
        wavelengths = pd.to_numeric(wavelengths, errors='coerce')
        photon_counts = pd.to_numeric(photon_counts, errors='coerce')
        
        # Remove any negative or NaN values
        valid_mask = (photon_counts >= 0) & (~np.isnan(photon_counts)) & (~np.isnan(wavelengths))
        wavelengths = wavelengths[valid_mask]
        photon_counts = photon_counts[valid_mask]
        
        return wavelengths, photon_counts
        
    except Exception as e:
        raise ValueError(f"Error processing spectrum file: {str(e)}")

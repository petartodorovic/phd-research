import pandas as pd
import numpy as np

def load_spectrum_from_file(file_path, wavelength_col=0, counts_col=1):
    """
    Load spectrum from space-delimited file
    
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
        data = pd.read_excel(file_path, engine='openpyxl')
        
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
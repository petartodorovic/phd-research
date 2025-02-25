import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pkg_resources

def plot_cie_coordinates(x, y, label=None, save_path=None, show_plot=True):
    """
    Plot CIE coordinates on the 1931 chromaticity diagram.
    
    Parameters:
    -----------
    x, y : float
        CIE x and y coordinates to plot
    label : str, optional
        Label for the plotted point
    save_path : str, optional
        Path to save the plot
    show_plot : bool, optional
        Whether to display the plot (default: True)
    """
    # Load spectral locus data
    locus_path = pkg_resources.resource_filename('cie_colorimetry', 'data/spectral_locus.csv')
    spectral_locus = pd.read_excel(locus_path, engine='openpyxl')
    
    # Check if we need to generate CIE coordinates and rename columns
    if 'Wavelength (nm)' in spectral_locus.columns:
        spectral_locus = spectral_locus.rename(columns={'Wavelength (nm)': 'wavelength'})
    
    # Generate x, y coordinates if they don't exist
    if 'x' not in spectral_locus.columns or 'y' not in spectral_locus.columns:
        print("Generating CIE coordinates from wavelengths...")
        # Load CIE 1931 color matching functions
        data_path = pkg_resources.resource_filename('cie_colorimetry', 'data/cie_1931.csv')
        cie_data = pd.read_csv(data_path)
        
        # Prepare interpolation functions
        wavelengths = cie_data['wavelength'].values
        x_bar = cie_data['x_bar'].values
        y_bar = cie_data['y_bar'].values
        z_bar = cie_data['z_bar'].values
        
        # Compute chromaticity coordinates for each wavelength
        spec_x = []
        spec_y = []
        
        for wl in spectral_locus['wavelength'].values:
            # Find closest matching wavelength in CIE data
            idx = np.argmin(np.abs(wavelengths - wl))
            X = x_bar[idx]
            Y = y_bar[idx]
            Z = z_bar[idx]
            
            # Convert to chromaticity coordinates
            sum_XYZ = X + Y + Z
            if sum_XYZ > 0:
                x_coord = X / sum_XYZ
                y_coord = Y / sum_XYZ
            else:
                x_coord = 0
                y_coord = 0
            
            spec_x.append(x_coord)
            spec_y.append(y_coord)
        
        # Add to dataframe
        spectral_locus['x'] = spec_x
        spectral_locus['y'] = spec_y
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot spectral locus
    plt.plot(spectral_locus['x'], spectral_locus['y'], 'k-', linewidth=1)
    
    # Connect ends of spectral locus (purple line)
    plt.plot([spectral_locus['x'].iloc[0], spectral_locus['x'].iloc[-1]], 
             [spectral_locus['y'].iloc[0], spectral_locus['y'].iloc[-1]], 
             'k-', linewidth=1)
    
    # Plot wavelength markers (only if they exist in the spectral locus data)
    wavelength_markers = [380, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 
                         560, 570, 580, 590, 600, 620, 700]
    for wl in wavelength_markers:
        # Find closest wavelength if exact value doesn't exist
        wl_diff = np.abs(spectral_locus['wavelength'].values - wl)
        closest_idx = np.argmin(wl_diff)
        
        # Only annotate if it's reasonably close
        if wl_diff[closest_idx] < 5:  # within 5nm
            plt.annotate(f'{wl}', 
                        (spectral_locus['x'].iloc[closest_idx], 
                         spectral_locus['y'].iloc[closest_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    # Plot white point (Illuminant E)
    plt.plot(1/3, 1/3, 'k+', markersize=10, label='White Point (E)')
    
    # Plot sample point
    plt.plot(x, y, 'ro', markersize=8, label='Sample')
    
    # Add label if provided
    if label:
        plt.annotate(label, (x, y), xytext=(10, 10), 
                    textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Customize plot
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('CIE x')
    plt.ylabel('CIE y')
    plt.title('CIE 1931 Chromaticity Diagram')
    plt.axis([0, 0.8, 0, 0.9])
    plt.legend()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
import matplotlib.pyplot as plt
import numpy as np

from .colorimetry import get_calculator

def plot_cie_coordinates(x, y, label=None, save_path=None, show_plot=True, ax=None):
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
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If omitted, a new figure and axis are created.
    """
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Spectral locus data is shared with (and cached by) the colorimetry calculator
    spectral_locus = get_calculator().spectral_locus

    # Plot spectral locus
    ax.plot(spectral_locus['x'], spectral_locus['y'], 'k-', linewidth=1)
    
    # Connect ends of spectral locus (purple line)
    ax.plot([spectral_locus['x'].iloc[0], spectral_locus['x'].iloc[-1]],
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
            ax.annotate(f'{wl}',
                        (spectral_locus['x'].iloc[closest_idx], 
                         spectral_locus['y'].iloc[closest_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    # Plot white point (Illuminant E)
    ax.plot(1/3, 1/3, 'k+', markersize=10, label='White Point (E)')
    
    # Plot sample point
    ax.plot(x, y, 'ro', markersize=8, label='Sample')
    
    # Add label if provided
    if label:
        ax.annotate(label, (x, y), xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Customize plot
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('CIE x')
    ax.set_ylabel('CIE y')
    ax.set_title('CIE 1931 Chromaticity Diagram')
    ax.axis([0, 0.8, 0, 0.9])
    ax.legend()
    
    # Save plot if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    elif owns_figure:
        plt.close(fig)

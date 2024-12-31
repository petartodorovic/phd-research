########################################################## 
# Code for Generating an Ideal Gaussian Emission Profile #
##########################################################

# Written by Petar Todorović
# Updated - December 31, 2024

# Import all of the packages and dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

# Create a class which will let the user generate normally distributed profiles of luminescence

## @TODO - consider different units of wavelengths, and if one is actually provided

# Adding a DataClass for Plot Settings which are initialized at the beginning
@dataclass
class PlotSettings:
    figsize: Tuple[int, int] = (8,6)
    font_size: Dict[str, int] = field(default_factory = lambda: {
        'ticks': 14,
        'labels': 18,
        'title': 20
    })
    y_limits: Tuple[float, float] = (-0.001, 1.05) # lower/upper bounds of graph

class LuminescenceSpectra:
    """
    A class representing an ideal, symmetric Gaussian-like emission spectra (i.e., if you were to observe an ideal luminescence profile).
    """
    # Creating a dictionary of Spectral Ranges for colour matching post spectra generation for the visible region
    # @TODO Consider outside of the visible spectrum 
    SPECTRAL_RANGES = {
        'violet': (380, 450),
        'blue': (450, 475),
        'cyan': (475, 495),
        'green': (495, 570),
        'yellow': (570, 590),
        'orange': (590, 620),
        'red': (620, 800)
    }
    # 
    def __init__(self, 
                    center_wv: float, 
                    fwhm: float = 25, 
                    wavelength_range: Tuple[float, float] = (390, 830),
                    num_points: int = 441):
        """
        Initialise the LuminescenceSpectra object.
        Assumption 1: A full-width at half-maximum of 25 (units in nm)
        Assumption 2: The generated luminescence spectra follows a Gaussian normal like distribution

        """
        
        self._validate_inputs(center_wv, fwhm, wavelength_range)
        self.center_wv = center_wv

        # Setting the FWHM and calculating the standard deviation using the formula below
        self.fwhm = fwhm
        self.std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # Setting the linear x-range to use for plotting
        self.xvals = np.linspace(*wavelength_range, num_points)
        
        # Creating the ideal dataframe attribute 
        self.df_ideal = None
        
        print(f"Generating spectra at {center_wv} nm with FWHM of {fwhm} nm")
    
    # Validate that the actual inputs fall within the wavelength range and return an error if it fails
    @staticmethod
    def _validate_inputs(center_wv: float, 
                        fwhm: float, 
                        wavelength_range: Tuple[float, float]) -> None:
        
        if not (wavelength_range[0] < center_wv < wavelength_range[1]):
            raise ValueError("Center wavelength must be within wavelength range")

        if fwhm <= 0:
            raise ValueError("FWHM must be positive")

        if wavelength_range[0] >= wavelength_range[1]:
            raise ValueError("Invalid wavelength range")

    def generate_spectra(self) -> pd.DataFrame:
        distribution = norm(loc=self.center_wv, scale=self.std_dev)
        counts = distribution.pdf(self.xvals)
        
        self.df_ideal = pd.DataFrame({
            'Wavelength (nm)': self.xvals,
            'Counts': counts / np.max(counts)  # Normalize to 1
        })
        return self.df_ideal

    def calculate_roots(self) -> Tuple[float, float]:
        """Calculate the roots of the spectra at FWHM points"""
        if self.df_ideal is None:
            self.generate_spectra()
        
        yvals = self.df_ideal['Counts'] - 0.5  # Half maximum for normalized data
        spline = UnivariateSpline(self.xvals, yvals, s=0)
        roots = spline.roots()
        return roots[0], roots[1]
    
    def get_plot_colour(self) -> str:
        for color, (min_wave, max_wave) in self.SPECTRAL_RANGES.items():
            if min_wave <= self.center_wv <= max_wave:
                return color
        raise ValueError(f"Wavelength {self.center_wv} nm is outside visible spectrum")

    def plot_spectra(self, r1: float, r2: float, 
                    settings: Optional[PlotSettings] = None,
                    save_fig: bool = False) -> None:
        if self.df_ideal is None:
            self.generate_spectra()
            
        settings = settings or PlotSettings()
        
        with sns.axes_style('whitegrid', {
            'ytick.left': True,
            'axes.edgecolor': 'black',
            'font.family': 'Times New Roman'
        }):
            plt.figure(figsize=settings.figsize)
            plot_color = self.get_plot_colour()
            plt.plot(self.xvals, self.df_ideal['Counts'], 
                    color=plot_color, lw=2)
            
            plt.axvspan(r1, r2, facecolor=plot_color, alpha=0.2)
            plot_range = self.center_wv + np.array([-60, 60])
            plt.xlim(plot_range)
            plt.ylim(settings.y_limits)
            
            plt.vlines(x=self.center_wv, ls='--', 
                    ymin=settings.y_limits[0], 
                    ymax=settings.y_limits[1], 
                    color='k')
            
            self._set_plot_styling(settings)
            
            if save_fig:
                output_dir = os.path.join(os.getcwd(), 'outputs')
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, 
                                f"generated_luminescence_{self.center_wv}nm.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.show()

    def _set_plot_styling(self, settings: PlotSettings) -> None:
        plt.xticks(fontsize=settings.font_size['ticks'])
        plt.yticks(fontsize=settings.font_size['ticks'])
        plt.ylabel('Normalized Intensity', fontsize=settings.font_size['labels'])
        plt.xlabel('Wavelength (nm)', fontsize=settings.font_size['labels'])
        plt.legend([f"λ = {self.center_wv} nm"])
        plt.title("Generated Gaussian Spectra", fontsize=settings.font_size['title'])
    
if __name__ == "__main__":
    # Basic usage
    spectra = LuminescenceSpectra(center_wv=475)  # Creates spectra at 475nm
    df = spectra.generate_spectra()  # Generate the data
    r1, r2 = spectra.calculate_roots()  # Get FWHM points
    spectra.plot_spectra(r1, r2, save_fig=False)  # Plot with default settings


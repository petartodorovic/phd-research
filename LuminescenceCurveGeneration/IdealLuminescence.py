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
    """
   Configuration settings for plot visualization.

   Attributes:
       figsize (Tuple[int, int]): Figure dimensions (width, height) in inches. Defaults to (8,6).
       font_size (Dict[str, int]): Font sizes for plot elements:
           'ticks': Axis tick labels (default: 14)
           'labels': Axis labels (default: 18)
           'title': Plot title (default: 20)
       y_limits (Tuple[float, float]): Y-axis limits (min, max). Defaults to (-0.001, 1.05).
   """
    figsize: Tuple[int, int] = (8,6)
    font_size: Dict[str, int] = field(default_factory = lambda: {
        'ticks': 14,
        'labels': 18,
        'title': 20
    })
    y_limits: Tuple[float, float] = (-0.001, 1.05) # lower/upper bounds of graph

class LuminescenceSpectra:
    """
    Generates and analyzes ideal Gaussian emission spectra.

    Attributes:
       SPECTRAL_RANGES (Dict[str, Tuple[float, float]]): Wavelength ranges (nm) mapped to colors,
            spanning from 100nm to 1500nm across UV, visible, and IR regions.
    """
    # Creating a dictionary of Spectral Ranges for colour matching post spectra generation for the visible region
    # @TODO Consider outside of the visible spectrum 
    SPECTRAL_RANGES = {
        'indigo': (100, 200),
        'darkviolet': (200, 280),
        'blueviolet': (380, 315),
        'darkorchid': (315, 380),
        'violet': (380, 450),
        'blue': (450, 475),
        'cyan': (475, 495),
        'green': (495, 570),
        'yellow': (570, 590),
        'orange': (590, 620),
        'red': (620, 800),
        'indianred': (800, 1500)
    }
    # 
    def __init__(self, 
                    center_wv: float, 
                    fwhm: float = 25, 
                    wavelength_range: Tuple[float, float] = (100, 1500),
                    num_points: int = 441):
        """
        Initialize a LuminescenceSpectra object for generating Gaussian emission profiles.

        Args:
            center_wv (float): Center wavelength (nm) of the emission peak
            fwhm (float, optional): Full Width at Half Maximum in nm. Defaults to 25.
            wavelength_range (Tuple[float, float], optional): Min and max wavelength range (nm). 
                Defaults to (100, 1500).
            num_points (int, optional): Number of points to generate in wavelength range. 
                Defaults to 441.

        Attributes:
            center_wv (float): Center wavelength of emission
            fwhm (float): Full Width at Half Maximum
            std_dev (float): Standard deviation calculated from FWHM
            xvals (ndarray): Array of wavelength points
            df_ideal (DataFrame): Generated spectra data (None until generate_spectra called)

        Note:
            Assumes Gaussian (normal) distribution for emission profile
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
        """
        Validates the input parameters for the LuminescenceSpectra class.

        Args:
            center_wv (float): Center wavelength (nm)
            fwhm (float): Full Width at Half Maximum (nm)
            wavelength_range (Tuple[float, float]): Min and max wavelength values (nm)

        Raises:
            ValueError: If center_wv is outside wavelength_range, fwhm is non-positive,
                        or wavelength range is invalid (min >= max)
        """
        if not (wavelength_range[0] < center_wv < wavelength_range[1]):
            raise ValueError("Center wavelength must be within wavelength range")

        if fwhm <= 0:
            raise ValueError("FWHM must be positive")

        if wavelength_range[0] >= wavelength_range[1]:
            raise ValueError("Invalid wavelength range")

    def generate_spectra(self) -> pd.DataFrame:
        """
        Generates normalized Gaussian emission spectra using the object's parameters.
        
        Creates a normal distribution using the center wavelength and standard deviation,
        calculates probability density function values across the wavelength range,
        and normalizes intensities to maximum of 1.

        Returns:
            pd.DataFrame: DataFrame containing wavelength values (nm) and normalized intensity counts
                Columns:
                    'Wavelength (nm)': Wavelength values from self.xvals
                    'Counts': Normalized intensity values (0 to 1)
        """
        # Create the normalized distribution given the center wavelength, and the standard deviation measure 
        distribution = norm(loc=self.center_wv, scale=self.std_dev)
        
        # Obtain the PDF of the given distribution for normalization purposes in the proceeding calculation 
        counts = distribution.pdf(self.xvals)
        
        self.df_ideal = pd.DataFrame({
            'Wavelength (nm)': self.xvals,
            'Counts': counts / np.max(counts)  # Normalize to 1
        })

        return self.df_ideal

    def calculate_roots(self) -> Tuple[float, float]:
        """
        Calculates the wavelength values at Full Width at Half Maximum (FWHM) points.
        
        Generates spectra if not already generated, subtracts half maximum from normalized
        intensity values, and finds roots using UnivariateSpline interpolation.

        Returns:
            Tuple[float, float]: Lower and upper wavelength values (nm) at half maximum intensity
        """
        if self.df_ideal is None:
            self.generate_spectra()
        
        yvals = self.df_ideal['Counts'] - 0.5  # Half maximum for normalized data
        spline = UnivariateSpline(self.xvals, yvals, s=0)
        roots = spline.roots()
        return roots[0], roots[1]
    
    def get_plot_colour(self) -> str:
        """
        Determines plot color based on center wavelength.

        Returns:
            str: Color name from SPECTRAL_RANGES dictionary

        Raises:
            ValueError: If center wavelength is outside defined spectral ranges
        """
        for color, (min_wave, max_wave) in self.SPECTRAL_RANGES.items():
            if min_wave <= self.center_wv <= max_wave:
                return color
        raise ValueError(f"Wavelength {self.center_wv} nm is outside visible spectrum")

    def plot_spectra(self, r1: float, r2: float, 
                    settings: Optional[PlotSettings] = None,
                    save_fig: bool = False) -> None:
        
        """
        Plots normalized emission spectra with FWHM range highlighted.

        Args:
            r1 (float): Lower FWHM wavelength value
            r2 (float): Upper FWHM wavelength value
            settings (Optional[PlotSettings]): Plot configuration settings
            save_fig (bool): Whether to save plot to outputs directory
        """

        # Check if the instance of the object had a df_ideal already generated previously, if not call the method and generate the spectra
        if self.df_ideal is None:
            self.generate_spectra()
        
        # Take default settings or those custom ones optionally declared in the function
        settings = settings or PlotSettings()
        
        # Set the standard styling for seaborn plots
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
        """
        Applies styling to plot elements.

        Args:
            settings (PlotSettings): Plot configuration settings
        """
        plt.xticks(fontsize=settings.font_size['ticks'])
        plt.yticks(fontsize=settings.font_size['ticks'])
        plt.ylabel('Normalized Intensity', fontsize=settings.font_size['labels'])
        plt.xlabel('Wavelength (nm)', fontsize=settings.font_size['labels'])
        plt.legend([f"λ = {self.center_wv} nm"])
        plt.title("Generated Gaussian Spectra", fontsize=settings.font_size['title'])
    
if __name__ == "__main__":
    # Basic usage
    spectra = LuminescenceSpectra(center_wv=150)  # Creates spectra at 475nm
    df = spectra.generate_spectra()  # Generate the data
    r1, r2 = spectra.calculate_roots()  # Get FWHM points
    spectra.plot_spectra(r1, r2, save_fig=False)  # Plot with default settings


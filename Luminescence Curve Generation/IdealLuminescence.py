########################################################## 
# Code for Generating an Ideal Gaussian Emission Profile #
##########################################################

# Written by Petar Todorović

# Import all of the packages and dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import integrate, interpolate
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
import seaborn as sns

# Create a class which will let the user generate normally distributed profiles of luminescence

## @TODO - consider different units of wavelengths, and if one is actually provided

class LuminescenceSpectra:
    """
    A class representing an ideal, symmetric Gaussian-like emission spectra (i.e., if you were to observe an ideal luminescence profile).
    """

    # 
    def __init__(self, center_wv, fwhm = 25):
        """
        Initialise the LuminescenceSpectra object.
        Assumption 1: A full-width at half-maximum of 25 (units in nm)
        Assumption 2: The generated luminescence spectra follows a Gaussian normal like distribution

        """
        print(f"You are generating a spectra now at {center_wv} nm with a FWHM of {fwhm} nm.")
        
        self.center_wv = center_wv

        # Setting the FWHM and calculating the standard deviation using the formula below
        self.fwhm = fwhm
        self.std_dev = fwhm/(2*np.sqrt(2*np.log(2)))

        # Setting the linear x-range to use for plotting
        self.xvals = np.linspace(390, 830, 441)

        # Creating a dictionary of colors to map color to a range for plotting the generated spectra
        self.dict_colours = {'violet': [380, 450],
                'blue': [450, 475],
                'cyan': [475, 495],
                'green': [495, 570],
                'yellow': [570, 590],
                'orange': [590, 620],
                'red': [620, 800]}


    def generate_spectra(self):
        # Generate the spectra by using the stats.norm package
        self.ideal_norm_dist = norm(loc = self.center_wv, 
                                    scale = self.std_dev)
        # Create a dictionary of the data points, and then return a pandas dataframe
        self.dict_data = {'Wavelength (nm)': self.xvals, 
                         'Counts': self.ideal_norm_dist.pdf(self.xvals)}
        self.df_ideal = pd.DataFrame(self.dict_data)
        return self.df_ideal

    def calculate_roots(self):
        # Calculate the roots of the spectra to obtain the x1 and x2 coordinates which exist at the FWHM points
        yvals = self.df_ideal['Counts']-np.max(self.df_ideal['Counts'])/2
        spline_data = UnivariateSpline(self.xvals, yvals, s=0)
        r1, r2 = spline_data.roots()
        return r1, r2
    

    def get_plot_colour(self):
        # Iterate over the color and wavelength ranges, selecting the appropriate color for the given range
        for c, w in self.dict_colours.items():
            min_wave = w[0]
            max_wave = w[1]
            if min_wave <= self.center_wv <= max_wave:
                plot_color = c
                break 
        return plot_color

    def plot_spectra(self, r1, r2, save_fig = True):
        with sns.axes_style('whitegrid', 
                            {'ytick.left': True, 
                            'axes.edgecolor':'black',
                            'font.serif': ['Times New Roman'],
                            'font.family': 'Times New Roman'}):
            # Plotting Figure 
            plt.figure(figsize=(8,6))
            plt.plot(self.xvals, self.df_ideal['Counts'], color=self.get_plot_colour(), lw=2)
        
            plt.axvspan(r1, r2, facecolor='green', alpha=0.2)
            l_bound, u_bound = self.center_wv-60, self.center_wv+60
            plt.xlim([l_bound,u_bound])
            y_min, y_max = -0.001, 0.04
            plt.vlines(x = self.center_wv, ls = '--', ymin= y_min, ymax = y_max, color = 'k')
            plt.ylim([y_min, y_max])
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            #plt.setp(ax.get_yticklabels(), visible=False)

            # Figure Aesthetics
            plt.ylabel('Normalized Intensity', fontsize=18)
            plt.xlabel('Wavelength (nm)', fontsize=18)
            plt.legend([f"$\lambda$ = {self.center_wv} nm"])
            plt.title("Generated Gaussian Spectra", fontsize = 20)
            if save_fig == False:
                path_save = os.path.join(os.getcwd(), f"generated_luminescence_{self.center_wv}nm")
                plt.savefig(f"{path_save}.png")
            plt.show()
        return

if __name__ == "__main__":
    spectra_2 = LuminescenceSpectra(540, 40)
    df_ideal_550 = spectra_2.generate_spectra()
    root1, root2 = spectra_2.calculate_roots()
    spectra_2.plot_spectra(root1, root2)

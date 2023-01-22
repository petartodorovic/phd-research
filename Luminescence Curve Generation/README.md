# Explanation of the Ideal Luminescence Script

In this code, an ideal luminescence curve may be synthesized using the scipy.stats module/package. 

The first step is to instantiate a LuminanceSpectra object and providing a center wavelength which will have a default preset full-width at half-maximum of 25 nm. The input center wavelength variable should be in nanometers.

The full-width at half-maximum is the range where the peak amplitude at the centered wavelength is 50% of the original value. There will be two values associated with this which can be calculated by an interpolation and splining method in the scipy package. The roots are calculated, and provide the necessary range to be then used for plotting the resulting spectra and the FWHM. 

The formula for calculating the standard deviation which is a required parameter to generate the ideal Gaussian profile is as follows:

$$\sigma = \frac{FWHM}  {2\sqrt{(2\times log(2))}}$$


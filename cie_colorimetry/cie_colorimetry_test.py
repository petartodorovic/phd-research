#!/usr/bin/env python3
# cie_colorimetry_test.py - Test script for the CIE colorimetry package

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the colorimetry functions
try:
    from cie_colorimetry import (
        calculate_cie_coordinates,
        calculate_dominant_wavelength,
        calculate_saturation,
        plot_cie_coordinates
    )
    from cie_colorimetry.utils import load_spectrum_from_file
    print("Successfully imported cie_colorimetry package")
except ImportError as e:
    print(f"Error importing cie_colorimetry: {e}")
    exit(1)

def create_test_spectrum(output_path="test_spectrum.csv"):
    """Create a test spectrum file for testing"""
    # Create a Gaussian spectrum centered at 550 nm (green)
    wavelengths = np.arange(380, 780, 5)
    # Create a Gaussian peak centered at 550nm
    intensities = 1000 * np.exp(-((wavelengths - 550) ** 2) / (2 * 30 ** 2))
    
    # Save to CSV file
    df = pd.DataFrame({
        "Wavelength (nm)": wavelengths,
        "Counts": intensities
    })
    df.to_csv(output_path, index=False)
    print(f"Created test spectrum file: {output_path}")
    return output_path

def create_red_spectrum(output_path="red_spectrum.csv"):
    """Create a test red spectrum file"""
    wavelengths = np.arange(380, 780, 5)
    # Create a Gaussian peak centered at 650nm (red)
    intensities = 1000 * np.exp(-((wavelengths - 650) ** 2) / (2 * 30 ** 2))
    
    df = pd.DataFrame({
        "Wavelength (nm)": wavelengths,
        "Counts": intensities
    })
    df.to_csv(output_path, index=False)
    print(f"Created red spectrum file: {output_path}")
    return output_path

def create_blue_spectrum(output_path="blue_spectrum.csv"):
    """Create a test blue spectrum file"""
    wavelengths = np.arange(380, 780, 5)
    # Create a Gaussian peak centered at 450nm (blue)
    intensities = 1000 * np.exp(-((wavelengths - 450) ** 2) / (2 * 30 ** 2))
    
    df = pd.DataFrame({
        "Wavelength (nm)": wavelengths,
        "Counts": intensities
    })
    df.to_csv(output_path, index=False)
    print(f"Created blue spectrum file: {output_path}")
    return output_path

def test_spectrum_loading():
    """Test loading a spectrum from a file"""
    try:
        file_path = create_test_spectrum()
        wavelengths, counts = load_spectrum_from_file(
            file_path,
            wavelength_col="Wavelength (nm)",
            counts_col="Counts"
        )
        print(f"Successfully loaded spectrum file")
        print(f"Wavelength range: {wavelengths.min():.1f} to {wavelengths.max():.1f} nm")
        print(f"Maximum intensity: {counts.max():.1f}")
        return wavelengths, counts
    except Exception as e:
        print(f"Error loading spectrum: {e}")
        return None, None

def test_colorimetry_calculation(wavelengths, counts):
    """Test calculating CIE coordinates"""
    try:
        x, y = calculate_cie_coordinates(wavelengths, counts)
        dominant_wavelength = calculate_dominant_wavelength(x, y)
        saturation = calculate_saturation(x, y, dominant_wavelength)
        
        print("\nColorimetry Results:")
        print(f"CIE coordinates: (x={x:.4f}, y={y:.4f})")
        print(f"Dominant wavelength: {dominant_wavelength:.1f} nm")
        print(f"Saturation: {saturation:.1f}%")
        
        return x, y, dominant_wavelength, saturation
    except Exception as e:
        print(f"Error calculating colorimetry parameters: {e}")
        return None, None, None, None

def test_multiple_spectra():
    """Test processing multiple spectra"""
    # Create different test spectra
    green_path = create_test_spectrum("green_spectrum.csv")
    red_path = create_red_spectrum()
    blue_path = create_blue_spectrum()
    
    results = []
    
    # Process each spectrum
    for name, path in [("Green", green_path), ("Red", red_path), ("Blue", blue_path)]:
        try:
            wavelengths, counts = load_spectrum_from_file(
                path,
                wavelength_col="Wavelength (nm)",
                counts_col="Counts"
            )
            
            x, y = calculate_cie_coordinates(wavelengths, counts)
            dom_wl = calculate_dominant_wavelength(x, y)
            sat = calculate_saturation(x, y, dom_wl)
            
            results.append({
                "name": name,
                "x": x,
                "y": y,
                "dominant_wavelength": dom_wl,
                "saturation": sat
            })
        except Exception as e:
            print(f"Error processing {name} spectrum: {e}")
    
    # Print comparison
    print("\nComparison of Different Spectra:")
    print("-" * 60)
    print(f"{'Color':<10} {'x':<10} {'y':<10} {'Dom. WL (nm)':<15} {'Saturation (%)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<10} {r['x']:<10.4f} {r['y']:<10.4f} {r['dominant_wavelength']:<15.1f} {r['saturation']:<15.1f}")
    
    return results

def test_plotting(results):
    """Test plotting functionality"""
    try:
        # Create a figure with subplots for comparison
        plt.figure(figsize=(10, 8))
        
        for i, r in enumerate(results):
            # Plot each result
            subplot_idx = 131 + i
            plt.subplot(subplot_idx)
            
            plot_cie_coordinates(
                r['x'], r['y'],
                label=f"{r['name']}\nλd={r['dominant_wavelength']:.1f}nm\nS={r['saturation']:.1f}%",
                show_plot=False
            )
            plt.title(f"{r['name']} Spectrum")
            
        # Adjust layout and show
        plt.tight_layout()
        plt.suptitle("CIE 1931 Chromaticity Results for Different Spectra", y=1.02, fontsize=16)
        
        # Save the plot
        plt.savefig("colorimetry_results.png", dpi=300, bbox_inches='tight')
        print("\nSaved combined plot to 'colorimetry_results.png'")
        
        # Show the plot
        plt.show()
    except Exception as e:
        print(f"Error plotting results: {e}")

def cleanup_test_files():
    """Remove test files created during testing"""
    test_files = [
        "test_spectrum.csv", 
        "green_spectrum.csv", 
        "red_spectrum.csv", 
        "blue_spectrum.csv"
    ]
    
    for file in test_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed test file: {file}")
        except Exception as e:
            print(f"Error removing test file {file}: {e}")

def main():
    """Main test function"""
    print("=" * 50)
    print("CIE Colorimetry Module Test")
    print("=" * 50)
    
    # Test loading spectrum
    wavelengths, counts = test_spectrum_loading()
    
    # Test colorimetry calculations
    if wavelengths is not None and counts is not None:
        test_colorimetry_calculation(wavelengths, counts)
    
    # Test with multiple spectra
    results = test_multiple_spectra()
    
    # Test plotting
    if results:
        test_plotting(results)
    
    # Clean up test files
    cleanup_test_files()
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
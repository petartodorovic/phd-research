import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .colorimetry import WHITE_POINT, analyze_spectrum, get_calculator
from .models import Spectrum

# Categorical colours for distinct samples, assigned in this fixed order (validated
# for colour-vision deficiency on a light background). Never cycled.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#6250d6', '#e34948']

# Single-hue blue ramp (light -> dark) for ordered series such as parameter sweeps
SEQUENTIAL_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
                   '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281']

TEXT_COLOR = '#0b0b0b'
MUTED_COLOR = '#52514e'
LOCUS_COLOR = '#8a8984'

# Marker shapes distinguish the two wavelengths without relying on colour
DOMINANT_MARKER = 'D'
PEAK_MARKER = '^'

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
    ax.plot(*WHITE_POINT, 'k+', markersize=10, label='White Point (E)')
    
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


def _as_spectrum_list(spectra):
    """Accept a single Spectrum or a sequence of them"""
    if isinstance(spectra, Spectrum):
        spectra = [spectra]
    spectra = list(spectra)
    if not spectra:
        raise ValueError("At least one spectrum is required")
    if not all(isinstance(s, Spectrum) for s in spectra):
        raise TypeError("Spectra must be Spectrum objects")
    return spectra


def _resolve_labels(labels, count):
    if labels is None:
        return [f"Sample {i + 1}" for i in range(count)]
    labels = list(labels)
    if len(labels) != count:
        raise ValueError("Provide one label per spectrum")
    return labels


def _intensities(spectrum, normalize):
    counts = spectrum.counts
    if normalize and counts.max() > 0:
        return counts / counts.max()
    return counts


def _finish_figure(fig, owns_figure, save_path, show_plot):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    elif owns_figure:
        plt.close(fig)


def _new_3d_axes(ax, figsize):
    owns_figure = ax is None
    if owns_figure:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
    elif getattr(ax, 'name', None) != '3d':
        raise ValueError("ax must be a 3D axis, e.g. fig.add_subplot(projection='3d')")
    return ax.figure, ax, owns_figure


def _locus_xy(wavelengths):
    """Chromaticity of monochromatic light, interpolating the 5 nm colour matching functions linearly"""
    calculator = get_calculator()
    X = np.interp(wavelengths, calculator.wavelengths, calculator.x_bar)
    Y = np.interp(wavelengths, calculator.wavelengths, calculator.y_bar)
    Z = np.interp(wavelengths, calculator.wavelengths, calculator.z_bar)
    total = X + Y + Z
    return X / total, Y / total


def _wavelength_legend_handles():
    return [
        Line2D([], [], linestyle='none', marker=DOMINANT_MARKER, markersize=7,
               markerfacecolor=TEXT_COLOR, markeredgecolor='white', label='Dominant wavelength'),
        Line2D([], [], linestyle='none', marker=PEAK_MARKER, markersize=8,
               markerfacecolor=TEXT_COLOR, markeredgecolor='white', label='Peak wavelength'),
    ]


def plot_cie_3d(spectra, labels=None, colors=None, normalize=True, elev=25, azim=-60,
                save_path=None, show_plot=True, ax=None):
    """
    Plot spectra as ridges standing on the CIE 1931 chromaticity diagram.

    Each spectrum's intensity is drawn above the spectral locus point of its
    wavelength. The floor shows the white point, each sample's chromaticity and
    the ray from the white point through it to the dominant wavelength, so the
    gap between peak and dominant wavelength is visible.

    Parameters:
    -----------
    spectra : Spectrum or sequence of Spectrum
        Spectra to plot (up to 8; 4 or fewer stays readable)
    labels : sequence of str, optional
        One name per spectrum (default: "Sample 1", "Sample 2", ...)
    colors : sequence of str, optional
        One matplotlib colour per spectrum (default: a fixed colour-blind-safe order)
    normalize : bool, optional
        Scale each spectrum to a maximum of 1 (default: True)
    elev, azim : float, optional
        Camera elevation and azimuth in degrees
    save_path : str, optional
        Path to save the plot
    show_plot : bool, optional
        Whether to display the plot (default: True)
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing 3D axis to draw on. If omitted, a new figure and axis are created.

    Returns:
    --------
    list of ColorimetryResult, one per spectrum
    """
    spectra = _as_spectrum_list(spectra)
    if colors is None:
        if len(spectra) > len(SERIES_COLORS):
            raise ValueError(f"plot_cie_3d supports up to {len(SERIES_COLORS)} spectra; "
                             "use plot_spectra_waterfall for longer series")
        colors = SERIES_COLORS[:len(spectra)]
    elif len(colors) != len(spectra):
        raise ValueError("Provide one colour per spectrum")
    labels = _resolve_labels(labels, len(spectra))
    fig, ax, owns_figure = _new_3d_axes(ax, figsize=(11, 9))

    # Floor: spectral locus, closing purple line and white point
    locus_wl = get_calculator().spectral_locus['wavelength'].values
    locus_x, locus_y = _locus_xy(locus_wl)
    ax.plot(locus_x, locus_y, 0, color=LOCUS_COLOR, linewidth=1)
    ax.plot([locus_x[0], locus_x[-1]], [locus_y[0], locus_y[-1]], 0,
            color=LOCUS_COLOR, linewidth=1, linestyle=':')
    for wl in [460, 480, 500, 520, 540, 560, 580, 600, 620]:
        # Nudge labels outward, away from the white point, so they clear sample markers
        lx, ly = _locus_xy(wl)
        ax.text(lx + 0.12 * (lx - WHITE_POINT[0]), ly + 0.12 * (ly - WHITE_POINT[1]), 0,
                f'{wl}', color=MUTED_COLOR, fontsize=7, ha='center', va='center')
    ax.scatter(*WHITE_POINT, 0, marker='+', s=80, color=TEXT_COLOR, depthshade=False)
    ax.text(*WHITE_POINT, 0, '  E', color=MUTED_COLOR, fontsize=8)

    results = []
    for spectrum, label, color in zip(spectra, labels, colors):
        result = analyze_spectrum(spectrum)
        results.append(result)
        intensities = _intensities(spectrum, normalize)

        # Ridge: intensity above the locus point of each wavelength. Faint tails are
        # dropped so they don't trace over the floor locus.
        visible = ((spectrum.wavelengths >= locus_wl[0]) & (spectrum.wavelengths <= locus_wl[-1])
                   & (intensities >= 0.01 * intensities.max()))
        ridge_x, ridge_y = _locus_xy(spectrum.wavelengths[visible])
        ax.plot(ridge_x, ridge_y, intensities[visible], color=color, linewidth=2)

        # Sample chromaticity on the floor
        ax.scatter(result.x, result.y, 0, color=color, s=40,
                   edgecolors='white', linewidths=1, depthshade=False)

        # Peak wavelength on the ridge, labelled with both wavelengths
        peak_x, peak_y = _locus_xy(result.peak_wavelength)
        peak_z = intensities.max()
        ax.scatter(peak_x, peak_y, peak_z, marker=PEAK_MARKER, s=70, color=color,
                   edgecolors='white', linewidths=1, depthshade=False)
        dominant_text = ('n/a' if result.dominant_wavelength is None
                         else f'{result.dominant_wavelength:.1f} nm')
        ax.text(peak_x, peak_y, peak_z * 1.06,
                f'{label}\npeak {result.peak_wavelength:.0f} nm · λd {dominant_text}',
                color=TEXT_COLOR, fontsize=8, ha='center')

        if result.dominant_wavelength is None:
            continue

        # Ray from the white point through the sample to the dominant wavelength
        dom_x, dom_y = _locus_xy(result.dominant_wavelength)
        ax.plot([WHITE_POINT[0], dom_x], [WHITE_POINT[1], dom_y], 0,
                color=color, linewidth=1, linestyle='--')

        # Stem from the floor up to the ridge at the dominant wavelength
        dom_z = np.interp(result.dominant_wavelength, spectrum.wavelengths, intensities)
        ax.plot([dom_x, dom_x], [dom_y, dom_y], [0, dom_z], color=color, linewidth=1)
        ax.scatter(dom_x, dom_y, dom_z, marker=DOMINANT_MARKER, s=45, color=color,
                   edgecolors='white', linewidths=1, depthshade=False)

    series_handles = [Line2D([], [], color=color, linewidth=2, label=label)
                      for label, color in zip(labels, colors)]
    ax.legend(handles=series_handles + _wavelength_legend_handles(),
              loc='upper left', frameon=False)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_zlim(0, None)
    ax.set_xlabel('CIE x')
    ax.set_ylabel('CIE y')
    ax.set_zlabel('Normalized intensity' if normalize else 'Photon counts')
    ax.set_title('Spectra on the CIE 1931 Chromaticity Diagram')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=0.9)  # keep the z-axis label inside the figure

    _finish_figure(fig, owns_figure, save_path, show_plot)
    return results


def plot_spectra_waterfall(spectra, labels=None, normalize=True, wavelength_range=(380, 780),
                           elev=25, azim=-65, save_path=None, show_plot=True, ax=None):
    """
    Plot an ordered series of spectra as a 3D waterfall.

    Spectra are placed one behind another in the given order (e.g. a temperature
    or composition sweep). Each curve marks its dominant and peak wavelengths, and
    dashed trend lines connect those markers across the series.

    Parameters:
    -----------
    spectra : Spectrum or sequence of Spectrum
        Spectra in series order
    labels : sequence of str, optional
        One name per spectrum, shown on the series axis
    normalize : bool, optional
        Scale each spectrum to a maximum of 1 (default: True)
    wavelength_range : tuple of float, optional
        Wavelength axis limits in nm (default: 380 to 780)
    elev, azim : float, optional
        Camera elevation and azimuth in degrees
    save_path : str, optional
        Path to save the plot
    show_plot : bool, optional
        Whether to display the plot (default: True)
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing 3D axis to draw on. If omitted, a new figure and axis are created.

    Returns:
    --------
    list of ColorimetryResult, one per spectrum
    """
    spectra = _as_spectrum_list(spectra)
    labels = _resolve_labels(labels, len(spectra))
    fig, ax, owns_figure = _new_3d_axes(ax, figsize=(12, 8))

    # Spread the ramp across the series so first and last are always the lightest and darkest steps
    ramp_idx = np.linspace(0, len(SEQUENTIAL_RAMP) - 1, len(spectra)).round().astype(int)
    colors = [SEQUENTIAL_RAMP[i] for i in ramp_idx]

    lo, hi = wavelength_range
    results = []
    dominant_points = []
    peak_points = []
    for position, (spectrum, color) in enumerate(zip(spectra, colors)):
        result = analyze_spectrum(spectrum)
        results.append(result)
        intensities = _intensities(spectrum, normalize)

        in_range = (spectrum.wavelengths >= lo) & (spectrum.wavelengths <= hi)
        wl = spectrum.wavelengths[in_range]
        z = intensities[in_range]
        if len(wl) == 0:
            continue

        # Filled curve, drawn back to front so nearer spectra overlap farther ones
        verts = [list(zip(wl, np.full_like(wl, position), z))
                 + [(wl[-1], position, 0), (wl[0], position, 0)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=0.25, edgecolors='none'))
        ax.plot(wl, np.full_like(wl, position), z, color=color, linewidth=2)

        if lo <= result.peak_wavelength <= hi:
            peak_points.append((result.peak_wavelength, position, intensities.max()))
        if result.dominant_wavelength is not None and lo <= result.dominant_wavelength <= hi:
            dom_z = np.interp(result.dominant_wavelength, spectrum.wavelengths, intensities)
            dominant_points.append((result.dominant_wavelength, position, dom_z))

    for points, marker, size in [(dominant_points, DOMINANT_MARKER, 45), (peak_points, PEAK_MARKER, 70)]:
        if not points:
            continue
        px, py, pz = np.array(points).T
        ax.scatter(px, py, pz, marker=marker, s=size, color=TEXT_COLOR,
                   edgecolors='white', linewidths=1, depthshade=False)
        if len(points) > 1:
            ax.plot(px, py, pz, color=TEXT_COLOR, linewidth=1, linestyle='--', alpha=0.6)

    ax.legend(handles=_wavelength_legend_handles(), loc='upper left', frameon=False)

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.5, len(spectra) - 0.5)
    ax.set_zlim(0, None)
    ax.set_yticks(range(len(spectra)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_zlabel('Normalized intensity' if normalize else 'Photon counts')
    ax.set_title('Spectral Series: Dominant and Peak Wavelengths')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=0.9)  # keep the z-axis label inside the figure

    _finish_figure(fig, owns_figure, save_path, show_plot)
    return results

"""MRI denoising and visualization utilities for Module 2.

This module contains helper functions for loading the coursework knee k-space
data, transforming it to image space, combining coils, applying denoising
methods, and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_wavelet
from skimage.metrics import structural_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
ASSETS_DIR = BASE_DIR / 'assets'

########################
### Helper Functions ###
########################

def _save_figure(save_filename):
    """Save the current Matplotlib figure into the coursework assets folder."""

    if not save_filename:
        return

    ASSETS_DIR.mkdir(exist_ok=True)
    plt.savefig(ASSETS_DIR / save_filename, bbox_inches='tight')


def _plot_coil_grid(images, save_filename=None, suptitle=None, cmap='gray'):
    """Plot a 2x3 grid of coil images with consistent formatting."""

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flat:
        ax.axis('off')

    for index, image in enumerate(images):
        axes.flat[index].imshow(image, cmap=cmap)
        axes.flat[index].set_title(f'Coil {index + 1}')
        axes.flat[index].axis('off')

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)

    plt.tight_layout()
    _save_figure(save_filename)
    plt.show()

#####################
### Excercise 2.1 ###
#####################

# Excercise 2.1 a
def load_kspace_data():
    """Load the coursework knee k-space array from disk.

    Returns:
        numpy.ndarray: Complex-valued k-space array.
    """

    return np.load(DATA_DIR / 'knee.npy')


# Excercise 2.1 b
def get_kspace_coil_mags(data):
    """Compute log-scaled k-space magnitudes for each coil.

    Args:
        data (numpy.ndarray): Loaded k-space array.

    Returns:
        list[numpy.ndarray]: Display-ready magnitude arrays.
    """

    return [np.log1p(np.abs(coil)) for coil in data]


# Excercise 2.1 b
def plot_kspace_coil_mags(kspace_coil_mags, save_filename=None):
    """Plot k-space magnitude images for each coil.

    Args:
        kspace_coil_mags (list[numpy.ndarray]): Magnitude images to plot.
    """

    _plot_coil_grid(kspace_coil_mags, save_filename=save_filename, cmap='grey')


# Excercise 2.1 c
def kspace_to_image_space(data):
    """Transform each coil from k-space to image space.

    Args:
        data (numpy.ndarray): K-space array.

    Returns:
        numpy.ndarray: Complex image-space data with shape `(coils, H, W)`.
    """

    return np.fft.ifft2(data, axes=(-2, -1))


# Excercise 2.1 c
def plot_one_coil_mag_phase(complex_im, save_filename=None):
    """Plot magnitude and phase for a single coil image.

    Args:
        complex_im (numpy.ndarray): Complex image-space coil data.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.rot90(np.abs(complex_im)), cmap='gray')
    axes[0].set_title('Magnitude')
    axes[1].imshow(np.rot90(np.angle(complex_im)), cmap='twilight_shifted')
    axes[1].set_title('Phase')
    plt.tight_layout()
    _save_figure(save_filename)
    plt.show()


# Excercise 2.1 d
def plot_all_coil_magnitudes(image_space_data, save_filename=None):
    """Plot magnitude images for all coils.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.
    """

    coil_images = [np.rot90(np.abs(coil)) for coil in image_space_data]
    _plot_coil_grid(coil_images, save_filename=save_filename)


# Excercise 2.1 e
def combine_coils(image_space_data):
    """Combine the coil images with the root-sum-of-squares method.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.

    Returns:
        numpy.ndarray: Combined magnitude image.
    """

    return combine_coil_magnitudes(get_coil_magnitudes(image_space_data))


# Excercise 2.1 e
def plot_combined(combined_im, save_filename=None):
    """Plot the combined coil image.

    Args:
        combined_im (numpy.ndarray): Combined image to display.
    """

    plt.figure(figsize=(6, 6))
    plt.imshow(np.rot90(combined_im), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    _save_figure(save_filename)
    plt.show()


#####################
### Excercise 2.2 ###
#####################

########################
### Helper Functions ###
########################

def get_coil_magnitudes(image_space_data):
    """Return the per-coil image magnitudes.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.

    Returns:
        numpy.ndarray: Magnitude array with shape `(coils, H, W)`.
    """

    return np.abs(image_space_data).astype(np.float64)


def combine_coil_magnitudes(coil_magnitudes):
    """Combine per-coil magnitude images with root-sum-of-squares.

    Args:
        coil_magnitudes (numpy.ndarray): Magnitude images with shape `(coils, H, W)`.

    Returns:
        numpy.ndarray: Combined magnitude image.
    """

    return np.sqrt(np.sum(coil_magnitudes ** 2, axis=0)).real


# Excercise 2.2 a
def denoise_coils_gaussian(image_space_data, sigma=1.0):
    """Denoise each coil magnitude image with a Gaussian filter.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.
        sigma (float, optional): Gaussian smoothing parameter.

    Returns:
        numpy.ndarray: Denoised magnitude images.
    """

    mag = get_coil_magnitudes(image_space_data)
    out = np.empty_like(mag)
    for index, coil_magnitude in enumerate(mag):
        out[index] = gaussian_filter(coil_magnitude, sigma=sigma, mode='nearest')
    return out


# Excercise 2.2 a
def denoise_coils_bilateral(image_space_data, sigma_spatial=1, sigma_color=None):
    """Denoise each coil magnitude image with a bilateral filter.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.
        sigma_spatial (float, optional): Spatial smoothing parameter.
        sigma_color (float | None, optional): Intensity-domain smoothing parameter.

    Returns:
        numpy.ndarray: Denoised magnitude images.
    """

    mag = get_coil_magnitudes(image_space_data)
    out = np.empty_like(mag)
    for index, coil_magnitude in enumerate(mag):
        out[index] = denoise_bilateral(
            coil_magnitude,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            mode='reflect',
        )
    return out


# Excercise 2.2 a
def denoise_coils_wavelet(image_space_data, sigma=None, method='BayesShrink'):
    """Denoise each coil magnitude image with wavelet thresholding.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.
        sigma (float | None, optional): Estimated noise level.
        method (str, optional): Wavelet thresholding method.

    Returns:
        numpy.ndarray: Denoised magnitude images.
    """

    mag = get_coil_magnitudes(image_space_data)
    out = np.empty_like(mag)
    for index, coil_magnitude in enumerate(mag):
        out[index] = denoise_wavelet(
            coil_magnitude,
            sigma=sigma,
            method=method,
            mode='soft',
            rescale_sigma=True,
        )
    return out


# Excercise 2.2 a
def plot_denoised_coils(denoised_magnitudes, title='Denoised', save_filename=None):
    """Plot denoised magnitude images for all coils.

    Args:
        denoised_magnitudes (numpy.ndarray): Denoised coil magnitudes.
        title (str, optional): Figure title.
    """

    coil_images = [np.rot90(coil) for coil in denoised_magnitudes]
    _plot_coil_grid(coil_images, save_filename=save_filename, suptitle=title)


# Excercise 2.2 b
def butterworth_lowpass_filter(shape, D0=30, n=2):
    """Create a low-pass Butterworth filter mask.

    Args:
        shape (tuple[int, int]): Filter shape.
        D0 (float, optional): Cutoff frequency.
        n (int, optional): Filter order.

    Returns:
        numpy.ndarray: Butterworth filter mask.
    """

    P, Q = shape[0], shape[1]
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / D0) ** (2 * n))
    return H


# Excercise 2.2 b
def butterworth_first_coil_image(kspace_data, D0=30, n=2, coil_num=1):
    """Apply a Butterworth low-pass filter to the first coil in k-space.

    Args:
        kspace_data (numpy.ndarray): K-space array.
        D0 (float, optional): Cutoff frequency.
        n (int, optional): Filter order.

    Returns:
        numpy.ndarray: Filtered complex image-space data for the first coil.
    """
    first_coil = kspace_data[coil_num-1]
    
    H = butterworth_lowpass_filter(first_coil.shape, D0=D0, n=n)
    
    filtered_kspace = first_coil * H
    
    filtered_image = np.fft.ifft2(filtered_kspace)
    
    return filtered_image


#############################
### Denoising Metrics 2.2 ###
#############################

def estimate_snr(magnitude_image, signal_roi, noise_roi):
    """Estimate SNR from a magnitude image using signal and noise ROIs.

    Args:
        magnitude_image (numpy.ndarray): 2D magnitude image.
        signal_roi (tuple): ``(row_start, row_end, col_start, col_end)`` for signal.
        noise_roi (tuple): ``(row_start, row_end, col_start, col_end)`` for noise.

    Returns:
        float: Estimated SNR (linear).
    """
    signal_mean = np.mean(
        magnitude_image[signal_roi[0]:signal_roi[1], signal_roi[2]:signal_roi[3]]
    )
    noise_std = np.std(
        magnitude_image[noise_roi[0]:noise_roi[1], noise_roi[2]:noise_roi[3]]
    )
    if noise_std == 0:
        return float('inf')
    return signal_mean / noise_std


def compute_denoising_metrics(original_mags, denoised_mags,
                              signal_roi=None, noise_roi=None):
    """Compute per-coil SNR and SSIM comparing original and denoised magnitudes.

    Default ROIs are chosen for the 280x280 knee images: a background corner
    patch for noise and a central anatomy patch for signal.

    Args:
        original_mags (numpy.ndarray): Original magnitudes, ``(N, H, W)`` or ``(H, W)``.
        denoised_mags (numpy.ndarray): Denoised magnitudes, same shape.
        signal_roi (tuple | None): ``(row_start, row_end, col_start, col_end)``.
        noise_roi (tuple | None): ``(row_start, row_end, col_start, col_end)``.

    Returns:
        list[dict]: One dict per coil with keys ``coil``, ``snr_original``,
        ``snr_denoised``, ``noise_std_original``, ``noise_std_denoised``, ``ssim``.
    """
    if signal_roi is None:
        signal_roi = (120, 160, 120, 160)
    if noise_roi is None:
        noise_roi = (5, 35, 5, 35)

    single = original_mags.ndim == 2
    if single:
        original_mags = original_mags[np.newaxis]
        denoised_mags = denoised_mags[np.newaxis]

    results = []
    for i in range(original_mags.shape[0]):
        orig = original_mags[i]
        den = denoised_mags[i]

        snr_orig = estimate_snr(orig, signal_roi, noise_roi)
        snr_den = estimate_snr(den, signal_roi, noise_roi)
        noise_std_orig = np.std(orig[noise_roi[0]:noise_roi[1],
                                     noise_roi[2]:noise_roi[3]])
        noise_std_den = np.std(den[noise_roi[0]:noise_roi[1],
                                    noise_roi[2]:noise_roi[3]])

        data_range = orig.max() - orig.min()
        ssim_val = structural_similarity(orig, den, data_range=data_range)

        results.append({
            'coil': 'combined' if single else i + 1,
            'snr_original': snr_orig,
            'snr_denoised': snr_den,
            'noise_std_original': noise_std_orig,
            'noise_std_denoised': noise_std_den,
            'ssim': ssim_val,
        })

    return results

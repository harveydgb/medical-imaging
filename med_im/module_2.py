"""MRI denoising and visualization utilities for Module 2.

This module contains helper functions for loading the coursework knee k-space
data, transforming it to image space, combining coils, applying denoising
methods, and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_wavelet





#####################
### Excercise 2.1 ###
#####################

def load_kspace_data():
    """Load the coursework knee k-space array from disk.

    Returns:
        numpy.ndarray: Complex-valued k-space array.
    """

    knee_data = np.load(r'''../data/knee.npy''')
    return knee_data


def coil_dimension(data):
    """Return the coil dimension index.

    Args:
        data (numpy.ndarray): Loaded k-space array.

    Returns:
        int: Coil axis index.
    """

    return 0


def get_kspace_coil_mags(data):
    """Compute log-scaled k-space magnitudes for each coil.

    Args:
        data (numpy.ndarray): Loaded k-space array.

    Returns:
        list[numpy.ndarray]: Display-ready magnitude arrays.
    """

    return [np.log1p(np.abs(data[i])) for i in range(6)]


def plot_kspace_coil_mags(kspace_coil_mags):
    """Plot k-space magnitude images for each coil.

    Args:
        kspace_coil_mags (list[numpy.ndarray]): Magnitude images to plot.
    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, mag in enumerate(kspace_coil_mags):
        axes.flat[i].imshow(mag, cmap='gray')
        axes.flat[i].set_title(f'Coil {i + 1}')
        axes.flat[i].axis('off')
    plt.tight_layout()
    plt.show()


def kspace_to_image_space(data):
    """Transform each coil from k-space to image space.

    Args:
        data (numpy.ndarray): K-space array.

    Returns:
        numpy.ndarray: Complex image-space data with shape `(coils, H, W)`.
    """

    return np.fft.ifft2(data, axes=(-2, -1))

def rotate_image_anticlockwise_90(image):
    """Rotate a 2D image by 90 degrees anticlockwise."""

    return np.rot90(image, k=1)

def plot_one_coil_mag_phase(complex_im):
    """Plot magnitude and phase for a single coil image.

    Args:
        complex_im (numpy.ndarray): Complex image-space coil data.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rotate_image_anticlockwise_90(np.abs(complex_im)), cmap='gray')
    axes[0].set_title('Magnitude')
    axes[1].imshow(rotate_image_anticlockwise_90(np.angle(complex_im)), cmap='gray')
    axes[1].set_title('Phase')
    plt.tight_layout()
    plt.show()


def plot_all_coil_magnitudes(image_space_data):
    """Plot magnitude images for all coils.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.
    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(6):
        axes.flat[i].imshow(rotate_image_anticlockwise_90(np.abs(image_space_data[i])), cmap='gray')
        axes.flat[i].set_title(f'Coil {i + 1}')
        axes.flat[i].axis('off')
    plt.tight_layout()
    plt.show()


def combine_coils(image_space_data):
    """Combine the coil images with the root-sum-of-squares method.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.

    Returns:
        numpy.ndarray: Combined magnitude image.
    """

    return np.sqrt(np.sum(np.abs(image_space_data) ** 2, axis=0)).real


def plot_combined(combined_im):
    """Plot the combined coil image.

    Args:
        combined_im (numpy.ndarray): Combined image to display.
    """

    plt.figure(figsize=(6, 6))
    plt.imshow(rotate_image_anticlockwise_90(combined_im), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


#####################
### Exercise 2.2 ###
#####################

def get_coil_magnitudes(image_space_data):
    """Return the per-coil image magnitudes.

    Args:
        image_space_data (numpy.ndarray): Complex image-space data.

    Returns:
        numpy.ndarray: Magnitude array with shape `(coils, H, W)`.
    """

    return np.abs(image_space_data).astype(np.float64)


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
    for i in range(6):
        out[i] = gaussian_filter(mag[i], sigma=sigma, mode='nearest')
    return out


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
    for i in range(6):
        out[i] = denoise_bilateral(
            mag[i],
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            mode='reflect',
        )
    return out


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
    for i in range(6):
        out[i] = denoise_wavelet(mag[i], sigma=sigma, method=method, mode='soft', rescale_sigma=True)
    return out


def plot_denoised_coils(denoised_magnitudes, title='Denoised'):
    """Plot denoised magnitude images for all coils.

    Args:
        denoised_magnitudes (numpy.ndarray): Denoised coil magnitudes.
        title (str, optional): Figure title.
    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(6):
        axes.flat[i].imshow(rotate_image_anticlockwise_90(denoised_magnitudes[i]), cmap='gray')
        axes.flat[i].set_title(f'Coil {i + 1}')
        axes.flat[i].axis('off')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


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


def butterworth_first_coil_image(kspace_data, D0=30, n=2):
    """Apply a Butterworth low-pass filter to the first coil in k-space.

    Args:
        kspace_data (numpy.ndarray): K-space array.
        D0 (float, optional): Cutoff frequency.
        n (int, optional): Filter order.

    Returns:
        numpy.ndarray: Filtered complex image-space data for the first coil.
    """

    first_coil = kspace_data[0]
    first_coil_shift = np.fft.fftshift(first_coil)
    H = butterworth_lowpass_filter(first_coil.shape, D0=D0, n=n)
    filtered_shift = first_coil_shift * H
    filtered_kspace = np.fft.ifftshift(filtered_shift)
    filtered_image = np.fft.ifft2(filtered_kspace)
    return filtered_image


def denoise_combined_gaussian(combined_image, sigma=1.0):
    """Denoise the combined image with a Gaussian filter.

    Args:
        combined_image (numpy.ndarray): Combined image.
        sigma (float, optional): Gaussian smoothing parameter.

    Returns:
        numpy.ndarray: Smoothed combined image.
    """

    return gaussian_filter(combined_image, sigma=sigma, mode='nearest')

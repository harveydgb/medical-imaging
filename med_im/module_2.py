"""Module 2: MRI Image Denoising"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_wavelet


#####################
### Excercise 2.1 ###
#####################

# Excercise 2.1) (a)

def load_kspace_data():
    knee_data = np.load(r'''../data/knee.npy''')
    return knee_data

def coil_dimension(data):
    """Return the coil dimension (0 = first axis, size 6)."""
    return 0


def get_kspace_coil_mags(data):
    """Magnitude of k-space per coil, log1p scaled for display."""
    return [np.log1p(np.abs(data[i])) for i in range(6)]


def plot_kspace_coil_mags(kspace_coil_mags):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, mag in enumerate(kspace_coil_mags):
        axes.flat[i].imshow(mag, cmap='gray')
        axes.flat[i].set_title(f'Coil {i + 1}')
        axes.flat[i].axis('off')
    plt.tight_layout()
    plt.show()


def kspace_to_image_space(data):
    """Inverse FFT each coil from k-space to image space. Returns (6, H, W) complex."""
    return np.fft.ifft2(data, axes=(-2, -1))


def plot_one_coil_mag_phase(complex_im):
    """Magnitude and phase of one coil in image space."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.abs(complex_im), cmap='gray')
    axes[0].set_title('Magnitude')
    axes[1].imshow(np.angle(complex_im), cmap='gray')
    axes[1].set_title('Phase')
    plt.tight_layout()
    plt.show()


def plot_all_coil_magnitudes(image_space_data):
    """Magnitude images from all coils in image space."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(6):
        axes.flat[i].imshow(np.abs(image_space_data[i]), cmap='gray')
        axes.flat[i].set_title(f'Coil {i + 1}')
        axes.flat[i].axis('off')
    plt.tight_layout()
    plt.show()


def combine_coils(image_space_data):
    """Root-sum-of-squares: sqrt(sum of squared magnitudes over coils)."""
    return np.sqrt(np.sum(np.abs(image_space_data) ** 2, axis=0)).real


def plot_combined(combined_im):
    plt.figure(figsize=(6, 6))
    plt.imshow(combined_im, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


#####################
### Exercise 2.2 ###
#####################

# Excercise 2.2) (a)


def get_coil_magnitudes(image_space_data):
    """Return (6, H, W) float array of per-coil magnitudes."""
    return np.abs(image_space_data).astype(np.float64)


def denoise_coils_gaussian(image_space_data, sigma=1.0):
    """Denoise each coil magnitude with Gaussian filter. Returns (6, H, W) float."""
    mag = get_coil_magnitudes(image_space_data)
    out = np.empty_like(mag)
    for i in range(6):
        out[i] = gaussian_filter(mag[i], sigma=sigma, mode='nearest')
    return out


def denoise_coils_bilateral(image_space_data, sigma_spatial=1, sigma_color=None):
    """Denoise each coil magnitude with bilateral filter. Returns (6, H, W) float."""
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
    """Denoise each coil magnitude with wavelet denoising. Returns (6, H, W) float."""
    mag = get_coil_magnitudes(image_space_data)
    out = np.empty_like(mag)
    for i in range(6):
        out[i] = denoise_wavelet(mag[i], sigma=sigma, method=method, mode='soft', rescale_sigma=True)
    return out


def plot_denoised_coils(denoised_magnitudes, title='Denoised'):
    """Plot 2x3 grid of denoised coil magnitudes."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(6):
        axes.flat[i].imshow(denoised_magnitudes[i], cmap='gray')
        axes.flat[i].set_title(f'Coil {i + 1}')
        axes.flat[i].axis('off')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# Excercise 2.2) (b)

def butterworth_lowpass_filter(shape, D0=30, n=2):
    P, Q = shape[0], shape[1]
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / D0) ** (2 * n))
    return H

def butterworth_first_coil_image(kspace_data, D0=30, n=2):
    """
    Apply a low-pass Butterworth filter in k-space to the first coil
    and return the filtered image (complex) in image space.
    """
    first_coil = kspace_data[0]
    first_coil_shift = np.fft.fftshift(first_coil)
    H = butterworth_lowpass_filter(first_coil.shape, D0=D0, n=n)
    filtered_shift = first_coil_shift * H
    filtered_kspace = np.fft.ifftshift(filtered_shift)
    filtered_image = np.fft.ifft2(filtered_kspace)
    return filtered_image


def denoise_combined_gaussian(combined_image, sigma=1.0):
    """Denoise the combined (root-sum-of-squares) image with a Gaussian filter."""
    return gaussian_filter(combined_image, sigma=sigma, mode='nearest')

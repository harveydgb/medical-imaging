"""Tomographic reconstruction utilities for Module 1.

This module contains helper functions for loading the coursework CT image,
simulating noisy sinograms, reconstructing images with filtered backprojection
and gradient-descent-style updates, and comparing reconstruction variants.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import iradon
from skimage.transform import radon

#####################
### Excercise 1.1 ###
#####################

# Excercise 1.1) (a)
def load_process_image():
    """Load the CT coursework image and convert it to grayscale.

    Returns
    -------
    numpy.ndarray
        Two-dimensional grayscale image scaled to a smaller range.
    """

    image = imread("../data/CT_exercise_1.png")
    image = rgb2gray(image[:,:,:3])
    image = image/100

    return image


def image_outputs():
    """Display basic image information and plot the loaded CT image.

    Returns
    -------
    numpy.ndarray
        Loaded image array.
    """

    image = load_process_image()

    print("Image shape:",image.shape)
    print("Min value:", image.min())
    print("Max value:", image.max())
    
    plt.figure()
    plt.imshow(image,cmap='grey')
    plt.show()

    return image

# Excercise 1.1) (b)

def create_noisy_sinograms(image, angle_range=360, seed=None):
    """Create noisy sinograms across the required angle and dose settings.

    Parameters
    ----------
    image : numpy.ndarray
        Input CT image.
    angle_range : int, optional
        Total angular coverage in degrees.
    seed : int | None, optional
        Random seed for deterministic noise generation.

    Returns
    -------
    dict[tuple[int, float], numpy.ndarray]
        Mapping from ``(angles, I0)`` to noisy sinogram arrays.
    """

    rng = np.random.default_rng(seed)
    noisy_sinogram_dict = {}
    for angles in [20, 90, 360]:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in [1e2, 1e3, 1e5]:
            sinogram = radon(image, theta)
            I = I0 * np.exp(-sinogram)
            I_noisy = rng.poisson(I) + rng.normal(0, 0.05, I.shape)
            I_noisy = np.maximum(I_noisy, 1e-10)
            sinogram_noisy = -np.log(I_noisy / I0)
            sinogram_noisy = np.clip(sinogram_noisy, 0, None)
            noisy_sinogram_dict[angles, I0] = sinogram_noisy
    return noisy_sinogram_dict


def plot_sinogram_dict(sinogram_dict, suptitle=None):
    """Plot a 3x3 grid of sinograms or reconstructions.

    Parameters
    ----------
    sinogram_dict : dict
        Dictionary keyed by ``(angles, I0)``.
    suptitle : str | None, optional
        Figure title.
    """

    angles_list = [20, 90, 360]
    I0_list = [1e2, 1e3, 1e5]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for i, angles in enumerate(angles_list):
        for j, I0 in enumerate(I0_list):
            arr = sinogram_dict[(angles, I0)]
            im = axes[i, j].imshow(arr, cmap='gray', aspect='auto')
            axes[i, j].set_title(r'$N_\theta = {}$, $I_0 = 10^{}$'.format(angles, int(np.log10(I0))))
            cbar = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


# Excercise 1.1) (c)

def FBP_backprojection(sinogram_dict, angle_range=360):
    """Reconstruct sinograms using filtered backprojection.

    Parameters
    ----------
    sinogram_dict : dict
        Noisy sinograms keyed by ``(angles, I0)``.
    angle_range : int, optional
        Total angular coverage in degrees.

    Returns
    -------
    dict[tuple[int, float], numpy.ndarray]
        Reconstructed images.
    """

    backprojection_dict = {}
    for angles in [20, 90, 360]:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in [1e2, 1e3, 1e5]:
            backprojection = iradon(sinogram_dict[angles, I0], theta, filter_name='ramp')
            backprojection = np.clip(backprojection, 0, None)
            backprojection_dict[angles, I0] = backprojection
    return backprojection_dict

def GD_backprojection(sinogram_dict, angles, I0, theta, max_iter, gamma):
    """Run a simple gradient-descent-style iterative reconstruction.

    Parameters
    ----------
    sinogram_dict : dict
        Sinogram dictionary keyed by ``(angles, I0)``.
    angles : int
        Number of projection angles.
    I0 : float
        Incident intensity level.
    theta : numpy.ndarray
        Projection angles.
    max_iter : int
        Number of iterations.
    gamma : float
        Update step size.

    Returns
    -------
    numpy.ndarray
        Reconstructed image.
    """

    gd = np.zeros((512, 512))
    for _ in range(max_iter):
        residual = sinogram_dict[angles, I0] - radon(gd, theta)
        gd = gd + gamma * iradon(residual, theta, filter_name=None)
    return np.clip(gd, 0, None)


def GD_backprojection_compare(sinogram_dict, angle_range=360):
    """Reconstruct all configured sinograms with the GD-based method.

    Parameters
    ----------
    sinogram_dict : dict
        Sinogram dictionary keyed by ``(angles, I0)``.
    angle_range : int, optional
        Total angular coverage in degrees.

    Returns
    -------
    dict[tuple[int, float], numpy.ndarray]
        Reconstructed images.
    """

    backprojection_dict = {}
    max_iter = 50
    gamma = 0.001
    for angles in [20, 90, 360]:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in [1e2, 1e3, 1e5]:
            backprojection_dict[angles, I0] = GD_backprojection(sinogram_dict, angles, I0, theta, max_iter, gamma)
    return backprojection_dict

#####################
### Excercise 1.3 ###
#####################

# Excercise 1.3) (a)

def FBP_compare_filters(sinogram_dict, angles=20, I0=1e2, angle_range=360):
    """Compare the standard and alternative FBP filters for one setting.

    Parameters
    ----------
    sinogram_dict : dict
        Sinogram dictionary keyed by ``(angles, I0)``.
    angles : int, optional
        Number of projection angles.
    I0 : float, optional
        Incident intensity level.
    angle_range : int, optional
        Total angular coverage in degrees.

    Returns
    -------
    list[numpy.ndarray]
        Reconstructions for the selected filters.
    """
    
    filters = ["ramp", "shepp-logan", "cosine"]
    theta = np.linspace(0, angle_range, angles, endpoint=False)
    sinogram = sinogram_dict[angles, I0]
    reconstructions = [np.clip(iradon(sinogram, theta, filter_name=f), 0, None) for f in filters]
    return reconstructions

def plot_filter_comparison(reconstructions):
    """Plot filter comparison reconstructions side by side.

    Parameters
    ----------
    reconstructions : list[numpy.ndarray]
        Reconstruction arrays to plot.
    """

    filters = ["ramp", "shepp-logan", "cosine"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    nice_names = {"ramp": "Ram-Lak", "shepp-logan": "Shepp–Logan", "cosine": "Cosine"}
    for ax, recon, f in zip(axes, reconstructions, filters):
        im = ax.imshow(recon, cmap="gray")
        ax.set_title(rf"{nice_names[f]} filter")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


# Excercise 1.3) (c)

def GD_backprojection_single(sinogram_dict, angles, I0, angle_range=360, max_iter=50, gamma=0.001):
    """Run the GD-based reconstruction for a single acquisition setting."""

    theta = np.linspace(0, angle_range, angles, endpoint=False)
    return GD_backprojection(sinogram_dict, angles, I0, theta, max_iter, gamma)

def OS_SART_reconstruct(sinogram_dict, angles, I0, angle_range=360, max_iter=50, gamma=0.001, n_subsets=5):
    """Run an ordered-subsets reconstruction for a single acquisition setting.

    Parameters
    ----------
    sinogram_dict : dict
        Sinogram dictionary keyed by ``(angles, I0)``.
    angles : int
        Number of projection angles.
    I0 : float
        Incident intensity level.
    angle_range : int, optional
        Total angular coverage in degrees.
    max_iter : int, optional
        Number of iterations.
    gamma : float, optional
        Update step size.
    n_subsets : int, optional
        Number of ordered subsets.

    Returns
    -------
    numpy.ndarray
        Reconstructed image.
    """

    theta = np.linspace(0, angle_range, angles, endpoint=False)
    sinogram = sinogram_dict[angles, I0]
    n_angles = len(theta)
    subset_indices = np.array_split(np.arange(n_angles), n_subsets)
    x = np.zeros((512, 512))
    for _ in range(max_iter):
        for ind in subset_indices:
            theta_b = theta[ind]
            sinogram_b = sinogram[:, ind]
            residual_b = sinogram_b - radon(x, theta_b)
            x = x + gamma * iradon(residual_b, theta_b, filter_name=None)
    return np.clip(x, 0, None)


def plot_compare_SIRT_OS_SART(sinogram_dict, angles=20, I0=1e2, angle_range=360, max_iter=50, gamma=0.001, n_subsets=5):
    """Plot SIRT and OS-SART reconstructions for one experiment setting."""

    sirt = GD_backprojection_single(sinogram_dict, angles, I0, angle_range, max_iter, gamma)
    os_sart = OS_SART_reconstruct(sinogram_dict, angles, I0, angle_range, max_iter, gamma, n_subsets)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, recon, title in zip(axes, [sirt, os_sart], ["SIRT (gradient descent)", "OS-SART"]):
        im = ax.imshow(recon, cmap="gray")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

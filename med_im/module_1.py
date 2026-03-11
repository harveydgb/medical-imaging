"""Module 1: Tomographic Reconstruction"""

#importing modules
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.transform import finite_radon_transform
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon
from skimage.transform import iradon

#####################
### Excercise 1.1 ###
#####################

# Excercise 1.1) (a)
def load_process_image():

    image = imread("../data/CT_exercise_1.png")
    image = rgb2gray(image[:,:,:3])
    image = image/1000

    return image


def image_outputs():
    image = load_process_image()

    print("Image shape:",image.shape)
    print("Min value:", image.min())
    print("Max value:", image.max())
    
    plt.figure()
    plt.imshow(image,cmap='grey')
    plt.show()

    return image

# Excercise 1.1) (b)

def create_noisy_sinograms(image, angle_range=360):
    noisy_sinogram_dict = {}
    for angles in [20, 90, 360]:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in [1e2, 1e3, 1e5]:
            sinogram = radon(image, theta)
            I = I0 * np.exp(-sinogram)
            I_noisy = np.random.poisson(I) + np.random.normal(0, 0.05, I.shape)
            I_noisy = np.maximum(I_noisy, 1e-10)
            sinogram_noisy = -np.log(I_noisy / I0)
            sinogram_noisy = np.clip(sinogram_noisy, 0, None)
            noisy_sinogram_dict[angles, I0] = sinogram_noisy
    return noisy_sinogram_dict


def plot_sinogram_dict(sinogram_dict, suptitle=None):
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
    backprojection_dict = {}
    for angles in [20, 90, 360]:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in [1e2, 1e3, 1e5]:
            backprojection = iradon(sinogram_dict[angles, I0], theta, filter_name='ramp')
            backprojection = np.clip(backprojection, 0, None)
            backprojection_dict[angles, I0] = backprojection
    return backprojection_dict


def GD_backprojection(sinogram_dict, angle_range=360):
    backprojection_dict = {}
    max_iter = 50
    gamma = 0.001
    for angles in [20, 90, 360]:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in [1e2, 1e3, 1e5]:
            gd = np.zeros((512, 512))
            for _ in range(max_iter):
                residual = sinogram_dict[angles, I0] - radon(gd, theta)
                update = gamma * iradon(residual, theta, filter_name=None)
                gd = gd + update
            backprojection_dict[angles, I0] = gd
    return backprojection_dict


def FBP_compare_filters(sinogram_dict, angles=20, I0=1e2, angle_range=360):
    
    filters = ["ramp", "shepp-logan", "cosine"]
    theta = np.linspace(0, angle_range, angles, endpoint=False)
    sinogram = sinogram_dict[angles, I0]
    recons = [np.clip(iradon(sinogram, theta, filter_name=f), 0, None) for f in filters]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    nice_names = {"ramp": "Ram-Lak", "shepp-logan": "Shepp–Logan", "cosine": "Cosine"}
    for ax, recon, f in zip(axes, recons, filters):
        im = ax.imshow(recon, cmap="gray")
        ax.set_title(rf"{nice_names[f]} filter")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

"""Module 1: Tomographic Reconstruction"""

#importing modules
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.transform import finite_radon_transform
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon

#####################
### Excercise 1.1 ###
#####################

# Excercise 1.1) (a)
def load_process_image():

    image = imread("../data/CT_exercise_1.png")
    image = rgb2gray(image[:,:,:3])

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

def create_noisy_sinograms(image):

    sinogram_dict = {}

    for angles in [20, 90, 360]:
        theta = np.linspace(0,angles,angles,endpoint=False)
        
        for I0 in [10e2, 10e3, 10e5]:

            sinogram = radon(image, theta)
            I = I0 * np.exp(-sinogram)
            I_noisy = np.random.poisson(I) + np.random.normal(0,0.05,I.shape)
            sinogram_dict[angles, I0] = I_noisy

    return sinogram_dict


def plot_sinogram_dict(sinogram_dict):
    """Plot all sinograms in sinogram_dict in a 3x3 grid."""
    angles_list = [20, 90, 360]
    I0_list = [10e2, 10e3, 10e5]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, angles in enumerate(angles_list):
        for j, I0 in enumerate(I0_list):
            axes[i, j].imshow(sinogram_dict[(angles, I0)], cmap='gray')
            axes[i, j].set_title(f'angles={angles}, I0={I0:.0e}')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


# Excercise 1.1) (b)

def simulate_gaussian_noise():
    """"""

# Excercise 1.1) (c)

# Excercise 1.1) (d)
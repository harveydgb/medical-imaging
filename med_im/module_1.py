"""Tomographic reconstruction utilities for Module 1.

This module contains helper functions for loading the coursework CT image,
simulating noisy sinograms, reconstructing images with filtered backprojection
and gradient-descent-style updates, and comparing reconstruction variants.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.transform import iradon
from skimage.transform import radon


def _resolve_parameter_list(values, default_values):
    """Return a concrete parameter list while preserving default behaviour."""

    if values is None:
        return list(default_values)
    return list(values)


def _get_experiment_grid(experiment_dict):
    """Infer the angle and intensity grid from dictionary keys."""

    angles_list = sorted({key[0] for key in experiment_dict})
    i0_list = sorted({key[1] for key in experiment_dict})
    return angles_list, i0_list


def _format_i0_label(I0):
    """Format dose values neatly for plot titles."""

    exponent = np.log10(I0)
    if np.isclose(exponent, round(exponent)):
        return f"10^{{{int(round(exponent))}}}"
    return f"{I0:.0e}"

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


def image_outputs(save_filename=None):
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
    if save_filename:
        import os
        os.makedirs('../assets', exist_ok=True)
        plt.savefig(f'../assets/{save_filename}', bbox_inches='tight')
    plt.show()

    return image

# Excercise 1.1) (b)

def create_noisy_sinograms(image, angle_range=360, seed=None, angles_list=None, I0_list=None):
    """Create noisy sinograms across the required angle and dose settings.

    Parameters
    ----------
    image : numpy.ndarray
        Input CT image.
    angle_range : int, optional
        Total angular coverage in degrees.
    seed : int | None, optional
        Random seed for deterministic noise generation.
    angles_list : sequence[int] | None, optional
        Projection counts to simulate. Defaults to ``[20, 90, 360]``.
    I0_list : sequence[float] | None, optional
        Incident intensity levels to simulate. Defaults to ``[1e2, 1e3, 1e5]``.

    Returns
    -------
    dict[tuple[int, float], numpy.ndarray]
        Mapping from ``(angles, I0)`` to noisy sinogram arrays.
    """

    rng = np.random.default_rng(seed)
    noisy_sinogram_dict = {}
    angles_list = _resolve_parameter_list(angles_list, [20, 90, 360])
    I0_list = _resolve_parameter_list(I0_list, [1e2, 1e3, 1e5])

    for angles in angles_list:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in I0_list:
            sinogram = radon(image, theta)
            I = I0 * np.exp(-sinogram)
            I_noisy = rng.poisson(I) + rng.normal(0, 0.05, I.shape)
            I_noisy = np.maximum(I_noisy, 1e-10)
            sinogram_noisy = -np.log(I_noisy / I0)
            sinogram_noisy = np.clip(sinogram_noisy, 0, None)
            noisy_sinogram_dict[angles, I0] = sinogram_noisy
    return noisy_sinogram_dict


def plot_sinogram_dict(sinogram_dict, suptitle=None, save_filename=None, panel_title_fn=None):
    """Plot a grid of sinograms or reconstructions.

    Parameters
    ----------
    sinogram_dict : dict
        Dictionary keyed by ``(angles, I0)``.
    suptitle : str | None, optional
        Figure title.
    save_filename : str | None, optional
        Filename to save the figure to the assets folder.
    panel_title_fn : callable | None, optional
        Function that formats subplot titles from ``(primary_key, I0)``.
    """

    angles_list, I0_list = _get_experiment_grid(sinogram_dict)
    n_rows = len(angles_list)
    n_cols = len(I0_list)

    if panel_title_fn is None:
        panel_title_fn = lambda angles, I0: r'$N_\theta = {}$, $I_0 = {}$'.format(angles, _format_i0_label(I0))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 10.0 * n_rows / max(n_rows, 3)),
        squeeze=False,
    )

    for i, angles in enumerate(angles_list):
        for j, I0 in enumerate(I0_list):
            arr = sinogram_dict[(angles, I0)]
            im = axes[i, j].imshow(arr, cmap='gray', aspect='auto')
            axes[i, j].set_title(panel_title_fn(angles, I0))
            cbar = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    if suptitle is not None:
        fig.suptitle(suptitle)

    if suptitle is not None:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if save_filename:
        import os
        os.makedirs('../assets', exist_ok=True)
        plt.savefig(f'../assets/{save_filename}', bbox_inches='tight')
    plt.show()


def plot_limited_angle_sinograms(
    image,
    angle_ranges,
    projection_count=180,
    I0_list=None,
    seed=None,
    save_filename=None,
):
    """Plot noisy sinograms over angular range and dose for a fixed ``N_theta``.

    Parameters
    ----------
    image : numpy.ndarray
        Ground-truth image used to simulate the sinograms.
    angle_ranges : sequence[int]
        Angular ranges in degrees to compare.
    projection_count : int, optional
        Fixed number of projection angles used for each angular range.
    I0_list : sequence[float] | None, optional
        Incident intensity levels to simulate. Defaults to ``[1e2, 1e3, 1e5]``.
    seed : int | None, optional
        Base random seed. Each angular range adds its value to this seed.
    save_filename : str | None, optional
        Filename to save the figure to the assets folder.

    Returns
    -------
    dict[tuple[int, float], numpy.ndarray]
        Dictionary keyed by ``(angle_range, I0)``.
    """

    I0_list = _resolve_parameter_list(I0_list, [1e2, 1e3, 1e5])
    limited_angle_sinograms = {}

    for angle_range in angle_ranges:
        experiment_seed = None if seed is None else seed + angle_range
        limited_sinogram_dict = create_noisy_sinograms(
            image,
            angle_range=angle_range,
            seed=experiment_seed,
            angles_list=[projection_count],
            I0_list=I0_list,
        )
        for I0 in I0_list:
            limited_angle_sinograms[(angle_range, I0)] = limited_sinogram_dict[(projection_count, I0)]

    plot_sinogram_dict(
        limited_angle_sinograms,
        suptitle=rf'Noisy sinograms with $N_\theta = {projection_count}$',
        save_filename=save_filename,
        panel_title_fn=lambda angle_range, I0: (
            rf'Angular range ${angle_range}^\circ$, $I_0 = {_format_i0_label(I0)}$'
        ),
    )

    return limited_angle_sinograms


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
    angles_list, I0_list = _get_experiment_grid(sinogram_dict)
    for angles in angles_list:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in I0_list:
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
    angles_list, I0_list = _get_experiment_grid(sinogram_dict)
    for angles in angles_list:
        theta = np.linspace(0, angle_range, angles, endpoint=False)
        for I0 in I0_list:
            backprojection_dict[angles, I0] = GD_backprojection(sinogram_dict, angles, I0, theta, max_iter, gamma)
    return backprojection_dict


def reconstruction_metrics(reference, reconstruction):
    """Compute simple image-quality metrics for one reconstruction.

    Parameters
    ----------
    reference : numpy.ndarray
        Ground-truth image used as the comparison target.
    reconstruction : numpy.ndarray
        Reconstructed image.

    Returns
    -------
    dict[str, float]
        Dictionary containing reconstruction error and image-quality metrics.
    """

    error = reconstruction - reference
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))

    data_range = float(reference.max() - reference.min())
    if data_range == 0:
        data_range = 1.0

    psnr = peak_signal_noise_ratio(reference, reconstruction, data_range=data_range)
    ssim = structural_similarity(reference, reconstruction, data_range=data_range)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


def reconstruction_metrics_dict(reference, reconstruction_dict):
    """Compute metrics for a full grid of reconstructions."""

    metrics_dict = {}
    for key, reconstruction in reconstruction_dict.items():
        metrics_dict[key] = reconstruction_metrics(reference, reconstruction)
    return metrics_dict


def print_metrics_table(metrics_dict, method_name=None):
    """Print a compact metrics table for the coursework experiment grid."""

    if method_name is not None:
        print(f"{method_name} metrics")

    header = f"{'angles':>6} {'I0':>8} {'rmse':>10} {'psnr':>10} {'ssim':>10}"
    print(header)
    print("-" * len(header))

    angles_list, I0_list = _get_experiment_grid(metrics_dict)
    for angles in angles_list:
        for I0 in I0_list:
            metrics = metrics_dict[(angles, I0)]
            print(
                f"{angles:>6} {I0:>8.0e} {metrics['rmse']:>10.4f} "
                f"{metrics['psnr']:>10.2f} {metrics['ssim']:>10.4f}"
            )


def print_limited_angle_metrics_table(metrics_dict, method_name=None):
    """Print a compact metrics table for fixed-``N_theta`` limited-angle runs."""

    if method_name is not None:
        print(f"{method_name} metrics")

    header = f"{'range':>6} {'I0':>8} {'rmse':>10} {'psnr':>10} {'ssim':>10}"
    print(header)
    print("-" * len(header))

    angle_ranges, I0_list = _get_experiment_grid(metrics_dict)
    for angle_range in angle_ranges:
        for I0 in I0_list:
            metrics = metrics_dict[(angle_range, I0)]
            print(
                f"{angle_range:>6} {I0:>8.0e} {metrics['rmse']:>10.4f} "
                f"{metrics['psnr']:>10.2f} {metrics['ssim']:>10.4f}"
            )


def _load_reference_if_available(reference=None):
    """Return the provided reference image or load the coursework image if possible."""

    if reference is not None:
        return reference

    try:
        return load_process_image()
    except (FileNotFoundError, OSError):
        return None


def print_named_metrics_table(metrics_dict, metric_keys=None, runtimes=None, title=None):
    """Print a compact metrics table for named reconstructions."""

    metric_keys = list(metric_keys or ["rmse", "ssim"])
    label_map = {
        "rmse": "RMSE",
        "mae": "MAE",
        "mse": "MSE",
        "psnr": "PSNR",
        "ssim": "SSIM",
    }
    formatter_map = {
        "rmse": lambda value: f"{value:.4f}",
        "mae": lambda value: f"{value:.4f}",
        "mse": lambda value: f"{value:.6f}",
        "psnr": lambda value: f"{value:.2f}",
        "ssim": lambda value: f"{value:.4f}",
        "runtime": lambda value: f"{value:.2f}",
    }

    if title is not None:
        print(title)

    header = f"{'method':<14}"
    for metric_key in metric_keys:
        header += f" {label_map.get(metric_key, metric_key.upper()):>10}"
    if runtimes is not None:
        header += f" {'runtime_s':>10}"

    print(header)
    print("-" * len(header))

    for name, metrics in metrics_dict.items():
        row = f"{name:<14}"
        for metric_key in metric_keys:
            row += f" {formatter_map[metric_key](metrics[metric_key]):>10}"
        if runtimes is not None:
            row += f" {formatter_map['runtime'](runtimes[name]):>10}"
        print(row)


def compare_reconstruction_methods(reference, sinogram_dict, angle_range=360, save_prefix=None):
    """Run the full FBP versus GD comparison used in Exercise 1.1(c).

    Parameters
    ----------
    reference : numpy.ndarray
        Ground-truth image.
    sinogram_dict : dict
        Noisy sinograms keyed by ``(angles, I0)``.
    angle_range : int, optional
        Total angular coverage in degrees.
    save_prefix : str | None, optional
        Prefix used when saving reconstruction figures.

    Returns
    -------
    dict[str, dict]
        Reconstruction images, metrics, and runtime for each method.
    """

    results = {}
    methods = [
        ("FBP", FBP_backprojection),
        ("GD", GD_backprojection_compare),
    ]

    for index, (method_name, reconstruction_function) in enumerate(methods):
        start = time.perf_counter()
        reconstruction_dict = reconstruction_function(sinogram_dict, angle_range=angle_range)
        runtime = time.perf_counter() - start

        reconstruction_save_filename = None
        if save_prefix is not None:
            reconstruction_save_filename = f'{save_prefix}_{method_name}.png'

        plot_sinogram_dict(reconstruction_dict, save_filename=reconstruction_save_filename)

        metrics_dict = reconstruction_metrics_dict(reference, reconstruction_dict)
        print(f'{method_name} total runtime: {runtime:.2f} s')
        print_metrics_table(metrics_dict, method_name=method_name)

        if index < len(methods) - 1:
            print()

        results[method_name] = {
            'reconstructions': reconstruction_dict,
            'metrics': metrics_dict,
            'runtime': runtime,
        }

    return results


def compare_limited_angle_reconstruction_methods(
    reference,
    angle_ranges,
    projection_count=180,
    I0_list=None,
    seed=None,
    save_prefix=None,
):
    """Compare FBP and GD reconstructions across limited angular ranges.

    Parameters
    ----------
    reference : numpy.ndarray
        Ground-truth image used for simulation and metric evaluation.
    angle_ranges : sequence[int]
        Angular ranges in degrees to compare.
    projection_count : int, optional
        Fixed number of projection angles used for each angular range.
    I0_list : sequence[float] | None, optional
        Incident intensity levels to simulate. Defaults to ``[1e2, 1e3, 1e5]``.
    seed : int | None, optional
        Base random seed. Each angular range adds its value to this seed.
    save_prefix : str | None, optional
        Prefix used when saving reconstruction figures.

    Returns
    -------
    dict[str, dict]
        Reconstruction images, metrics, and runtime for each method.
    """

    I0_list = _resolve_parameter_list(I0_list, [1e2, 1e3, 1e5])
    results = {
        'FBP': {'reconstructions': {}, 'metrics': {}, 'runtime': 0.0},
        'GD': {'reconstructions': {}, 'metrics': {}, 'runtime': 0.0},
    }
    methods = [
        ('FBP', FBP_backprojection),
        ('GD', GD_backprojection_compare),
    ]

    for angle_range in angle_ranges:
        experiment_seed = None if seed is None else seed + angle_range
        sinogram_dict = create_noisy_sinograms(
            reference,
            angle_range=angle_range,
            seed=experiment_seed,
            angles_list=[projection_count],
            I0_list=I0_list,
        )

        for method_name, reconstruction_function in methods:
            start = time.perf_counter()
            reconstruction_dict = reconstruction_function(sinogram_dict, angle_range=angle_range)
            results[method_name]['runtime'] += time.perf_counter() - start

            for I0 in I0_list:
                source_key = (projection_count, I0)
                target_key = (angle_range, I0)
                reconstruction = reconstruction_dict[source_key]
                results[method_name]['reconstructions'][target_key] = reconstruction
                results[method_name]['metrics'][target_key] = reconstruction_metrics(reference, reconstruction)

    for index, method_name in enumerate(['FBP', 'GD']):
        reconstruction_save_filename = None
        if save_prefix is not None:
            reconstruction_save_filename = f'{save_prefix}_{method_name}.png'

        plot_sinogram_dict(
            results[method_name]['reconstructions'],
            suptitle=rf'{method_name} reconstructions with $N_\theta = {projection_count}$',
            save_filename=reconstruction_save_filename,
            panel_title_fn=lambda angle_range, I0: (
                rf'Angular range ${angle_range}^\circ$, $I_0 = {_format_i0_label(I0)}$'
            ),
        )

        print(f"{method_name} total runtime: {results[method_name]['runtime']:.2f} s")
        print_limited_angle_metrics_table(results[method_name]['metrics'], method_name=method_name)

        if index < len(methods) - 1:
            print()

    return results

#####################
### Excercise 1.3 ###
#####################

# Excercise 1.3) (a)

def FBP_compare_filters(
    sinogram_dict,
    angles=20,
    I0=1e2,
    angle_range=360,
    reference=None,
    metric_keys=None,
):
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
    reference : numpy.ndarray | None, optional
        Ground-truth image used for reporting metrics. If omitted, the coursework
        image is loaded when available.
    metric_keys : sequence[str] | None, optional
        Metrics to display in the printed comparison table. Defaults to
        ``["rmse", "ssim"]``.

    Returns
    -------
    list[numpy.ndarray]
        Reconstructions for the selected filters.
    """
    
    filters = ["ramp", "shepp-logan", "cosine"]
    theta = np.linspace(0, angle_range, angles, endpoint=False)
    sinogram = sinogram_dict[angles, I0]
    reconstructions = [np.clip(iradon(sinogram, theta, filter_name=f), 0, None) for f in filters]

    reference = _load_reference_if_available(reference)
    if reference is not None:
        nice_names = {"ramp": "Ram-Lak", "shepp-logan": "Shepp-Logan", "cosine": "Cosine"}
        metrics_dict = {
            nice_names[filter_name]: reconstruction_metrics(reference, reconstruction)
            for filter_name, reconstruction in zip(filters, reconstructions)
        }
        print_named_metrics_table(
            metrics_dict,
            metric_keys=metric_keys or ["rmse", "ssim"],
            title=f"Filter comparison metrics for N_theta={angles}, I0={I0:.0e}",
        )

    return reconstructions

def plot_filter_comparison(reconstructions, save_filename=None):
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
    if save_filename:
        import os
        os.makedirs('../assets', exist_ok=True)
        plt.savefig(f'../assets/{save_filename}', bbox_inches='tight')
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


def plot_compare_SIRT_OS_SART(
    sinogram_dict,
    angles=20,
    I0=1e2,
    angle_range=360,
    max_iter=50,
    sirt_gamma=0.001,
    sart_gamma=0.0001,
    n_subsets=5,
    save_filename=None,
    reference=None,
    metric_keys=None,
):
    """Plot SIRT and OS-SART reconstructions for one experiment setting."""

    start = time.perf_counter()
    sirt = GD_backprojection_single(sinogram_dict, angles, I0, angle_range, max_iter, sirt_gamma)
    sirt_runtime = time.perf_counter() - start

    start = time.perf_counter()
    os_sart = OS_SART_reconstruct(sinogram_dict, angles, I0, angle_range, max_iter, sart_gamma, n_subsets)
    os_sart_runtime = time.perf_counter() - start

    reference = _load_reference_if_available(reference)
    metrics_dict = None
    if reference is not None:
        metrics_dict = {
            "SIRT": reconstruction_metrics(reference, sirt),
            "OS-SART": reconstruction_metrics(reference, os_sart),
        }
        print_named_metrics_table(
            metrics_dict,
            metric_keys=metric_keys or ["rmse", "ssim"],
            runtimes={"SIRT": sirt_runtime, "OS-SART": os_sart_runtime},
            title=(
                f"SIRT vs OS-SART metrics for N_theta={angles}, I0={I0:.0e}, "
                f"iterations={max_iter}, subsets={n_subsets}"
            ),
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, recon, title in zip(axes, [sirt, os_sart], ["SIRT (gradient descent)", "OS-SART"]):
        im = ax.imshow(recon, cmap="gray")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_filename:
        import os
        os.makedirs('../assets', exist_ok=True)
        plt.savefig(f'../assets/{save_filename}', bbox_inches='tight')
    plt.show()

    return {
        "reconstructions": {"SIRT": sirt, "OS-SART": os_sart},
        "metrics": metrics_dict,
        "runtimes": {"SIRT": sirt_runtime, "OS-SART": os_sart_runtime},
    }

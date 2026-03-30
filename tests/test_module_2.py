from pathlib import Path

import numpy as np

from med_im import module_2


def test_load_kspace_data_reads_coursework_array(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root / "notebooks")

    data = module_2.load_kspace_data()

    assert data.ndim == 3
    assert data.shape[0] == 6


def test_kspace_to_image_space_preserves_shape():
    kspace = np.ones((6, 8, 8), dtype=np.complex128)

    image_space = module_2.kspace_to_image_space(kspace)

    assert image_space.shape == kspace.shape
    assert np.iscomplexobj(image_space)


def test_combine_coils_returns_2d_image():
    image_space = np.ones((6, 8, 8), dtype=np.complex128)

    combined = module_2.combine_coils(image_space)

    assert combined.shape == (8, 8)
    assert np.all(combined >= 0)


def test_denoising_helpers_preserve_coil_shape():
    image_space = np.ones((6, 8, 8), dtype=np.complex128)

    gaussian = module_2.denoise_coils_gaussian(image_space, sigma=0.5)
    bilateral = module_2.denoise_coils_bilateral(image_space, sigma_spatial=1, sigma_color=0.1)
    wavelet = module_2.denoise_coils_wavelet(image_space, sigma=0.1)

    assert gaussian.shape == (6, 8, 8)
    assert bilateral.shape == (6, 8, 8)
    assert wavelet.shape == (6, 8, 8)


def test_butterworth_filter_matches_requested_shape():
    filt = module_2.butterworth_lowpass_filter((8, 8), D0=3, n=2)

    assert filt.shape == (8, 8)
    assert np.all(filt > 0)
    assert np.all(filt <= 1)
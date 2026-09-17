import numpy as np

from med_im import module_2


def test_load_kspace_data_reads_packaged_array():
    data = module_2.load_kspace_data()
    expected = np.load(module_2.DATA_DIR / 'knee.npy')

    assert data.shape == expected.shape
    assert np.array_equal(data, expected)
    assert np.iscomplexobj(data)


def test_kspace_helpers_match_numpy_transforms():
    kspace = np.arange(32, dtype=np.float64).reshape(2, 4, 4) + 1j

    coil_mags = module_2.get_kspace_coil_mags(kspace)
    image_space = module_2.kspace_to_image_space(kspace)

    assert len(coil_mags) == 2
    np.testing.assert_allclose(coil_mags[0], np.log1p(np.abs(kspace[0])))
    np.testing.assert_allclose(image_space, np.fft.ifft2(kspace, axes=(-2, -1)))


def test_coil_magnitude_helpers_use_root_sum_of_squares():
    image_space = np.array(
        [
            [[1 + 1j, 2 + 0j], [0 + 3j, 4 + 0j]],
            [[2 + 0j, 1 + 1j], [1 + 0j, 0 + 2j]],
        ],
        dtype=np.complex128,
    )
    expected_magnitudes = np.abs(image_space).astype(np.float64)
    expected_combined = np.sqrt(np.sum(expected_magnitudes ** 2, axis=0))

    magnitudes = module_2.get_coil_magnitudes(image_space)
    combined = module_2.combine_coil_magnitudes(magnitudes)

    np.testing.assert_allclose(magnitudes, expected_magnitudes)
    np.testing.assert_allclose(combined, expected_combined)
    np.testing.assert_allclose(module_2.combine_coils(image_space), expected_combined)


def test_denoising_helpers_preserve_shape_and_return_real_values():
    rng = np.random.default_rng(0)
    image_space = rng.normal(size=(2, 8, 8)) + 1j * rng.normal(size=(2, 8, 8))

    gaussian = module_2.denoise_coils_gaussian(image_space, sigma=0.5)
    bilateral = module_2.denoise_coils_bilateral(image_space, sigma_spatial=1, sigma_color=0.1)
    wavelet = module_2.denoise_coils_wavelet(image_space, sigma=0.1)

    for denoised in [gaussian, bilateral, wavelet]:
        assert denoised.shape == image_space.shape
        assert np.all(np.isfinite(denoised))
        assert np.all(denoised >= 0)
        assert np.isrealobj(denoised)


def test_butterworth_lowpass_filter_has_expected_profile():
    filt = module_2.butterworth_lowpass_filter((9, 9), D0=3, n=2)

    assert filt.shape == (9, 9)
    assert np.all(filt > 0)
    assert np.all(filt <= 1)
    assert np.isclose(filt[4, 4], 1.0)
    assert filt[0, 0] < filt[4, 4]


def test_butterworth_first_coil_image_filters_requested_coil():
    kspace_data = np.arange(48, dtype=np.float64).reshape(3, 4, 4) + 1j
    expected = np.fft.ifft2(
        kspace_data[1] * module_2.butterworth_lowpass_filter((4, 4), D0=5, n=2)
    )

    filtered = module_2.butterworth_first_coil_image(kspace_data, D0=5, n=2, coil_num=2)

    np.testing.assert_allclose(filtered, expected)
import numpy as np

from med_im import module_1


def test_create_noisy_sinograms_is_seeded_and_complete():
    image = np.zeros((512, 512), dtype=float)

    first = module_1.create_noisy_sinograms(image, seed=123)
    second = module_1.create_noisy_sinograms(image, seed=123)

    expected_keys = {
        (20, 1e2),
        (20, 1e3),
        (20, 1e5),
        (90, 1e2),
        (90, 1e3),
        (90, 1e5),
        (360, 1e2),
        (360, 1e3),
        (360, 1e5),
    }

    assert set(first) == expected_keys
    for key in expected_keys:
        assert first[key].shape == second[key].shape
        assert np.allclose(first[key], second[key])


def test_fbp_backprojection_returns_expected_grid():
    image = np.zeros((512, 512), dtype=float)
    sinograms = module_1.create_noisy_sinograms(image, seed=7)

    reconstructions = module_1.FBP_backprojection(sinograms)

    assert set(reconstructions) == set(sinograms)
    for reconstruction in reconstructions.values():
        assert reconstruction.shape == (512, 512)
        assert np.all(reconstruction >= 0)


def test_filter_comparison_returns_three_reconstructions():
    image = np.zeros((512, 512), dtype=float)
    sinograms = module_1.create_noisy_sinograms(image, seed=99)

    reconstructions = module_1.FBP_compare_filters(sinograms, angles=20, I0=1e2)

    assert len(reconstructions) == 3
    for reconstruction in reconstructions:
        assert reconstruction.shape == (512, 512)


def test_os_sart_reconstruct_preserves_output_shape():
    image = np.zeros((512, 512), dtype=float)
    sinograms = module_1.create_noisy_sinograms(image, seed=5)

    reconstruction = module_1.OS_SART_reconstruct(
        sinograms,
        angles=20,
        I0=1e2,
        max_iter=1,
        gamma=0.001,
        n_subsets=2,
    )

    assert reconstruction.shape == (512, 512)
    assert np.all(reconstruction >= 0)
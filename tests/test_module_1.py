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


def test_custom_fixed_angle_grid_is_supported():
    image = np.zeros((512, 512), dtype=float)
    sinograms = module_1.create_noisy_sinograms(
        image,
        angle_range=120,
        seed=11,
        angles_list=[180],
        I0_list=[1e2, 1e3, 1e5],
    )

    assert set(sinograms) == {(180, 1e2), (180, 1e3), (180, 1e5)}

    reconstructions = module_1.FBP_backprojection(sinograms, angle_range=120)

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


def test_reconstruction_metrics_dict_returns_expected_structure():
    reference = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)
    reconstructions = {
        (20, 1e2): reference + 0.05,
        (90, 1e3): reference + 0.10,
    }

    metrics_dict = module_1.reconstruction_metrics_dict(reference, reconstructions)

    assert set(metrics_dict) == set(reconstructions)
    for metrics in metrics_dict.values():
        assert set(metrics) == {"mse", "rmse", "mae", "psnr", "ssim"}
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["psnr"] > 0
        assert 0 <= metrics["ssim"] < 1


def test_compare_reconstruction_methods_returns_both_methods(monkeypatch):
    reference = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)

    def full_grid(scale):
        return {
            (angles, I0): reference * scale
            for angles in [20, 90, 360]
            for I0 in [1e2, 1e3, 1e5]
        }

    monkeypatch.setattr(module_1, "FBP_backprojection", lambda sinogram_dict, angle_range=360: full_grid(1.0))
    monkeypatch.setattr(module_1, "GD_backprojection_compare", lambda sinogram_dict, angle_range=360: full_grid(0.5))
    monkeypatch.setattr(module_1, "plot_sinogram_dict", lambda *args, **kwargs: None)

    results = module_1.compare_reconstruction_methods(reference, {}, save_prefix="exercise_1_1_c")

    assert set(results) == {"FBP", "GD"}
    assert set(results["FBP"]) == {"reconstructions", "metrics", "runtime"}
    assert results["FBP"]["metrics"][(20, 1e2)]["rmse"] == 0.0
    assert results["GD"]["metrics"][(20, 1e2)]["rmse"] > 0
    assert results["FBP"]["runtime"] >= 0
    assert results["GD"]["runtime"] >= 0


def test_plot_limited_angle_sinograms_groups_by_range(monkeypatch):
    reference = np.zeros((8, 8), dtype=float)

    monkeypatch.setattr(
        module_1,
        "create_noisy_sinograms",
        lambda image, angle_range=360, seed=None, angles_list=None, I0_list=None: {
            (angles_list[0], I0): np.full(reference.shape, angle_range + I0 / 1e5)
            for I0 in I0_list
        },
    )
    monkeypatch.setattr(module_1, "plot_sinogram_dict", lambda *args, **kwargs: None)

    sinograms = module_1.plot_limited_angle_sinograms(
        reference,
        angle_ranges=[180, 120, 40],
        projection_count=180,
        I0_list=[1e2, 1e3, 1e5],
        seed=123,
        save_filename="exercise_1_2_a.png",
    )

    expected_keys = {
        (180, 1e2),
        (180, 1e3),
        (180, 1e5),
        (120, 1e2),
        (120, 1e3),
        (120, 1e5),
        (40, 1e2),
        (40, 1e3),
        (40, 1e5),
    }

    assert set(sinograms) == expected_keys


def test_compare_limited_angle_reconstruction_methods_groups_by_range(monkeypatch):
    reference = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)

    monkeypatch.setattr(
        module_1,
        "create_noisy_sinograms",
        lambda image, angle_range=360, seed=None, angles_list=None, I0_list=None: {
            (angles_list[0], I0): np.full(reference.shape, angle_range + I0 / 1e5)
            for I0 in I0_list
        },
    )
    monkeypatch.setattr(
        module_1,
        "FBP_backprojection",
        lambda sinogram_dict, angle_range=360: {key: value for key, value in sinogram_dict.items()},
    )
    monkeypatch.setattr(
        module_1,
        "GD_backprojection_compare",
        lambda sinogram_dict, angle_range=360: {key: value * 0.5 for key, value in sinogram_dict.items()},
    )
    monkeypatch.setattr(module_1, "plot_sinogram_dict", lambda *args, **kwargs: None)

    results = module_1.compare_limited_angle_reconstruction_methods(
        reference,
        angle_ranges=[180, 120, 40],
        projection_count=180,
        I0_list=[1e2, 1e3, 1e5],
        seed=123,
        save_prefix="exercise_1_2_b",
    )

    expected_keys = {
        (180, 1e2),
        (180, 1e3),
        (180, 1e5),
        (120, 1e2),
        (120, 1e3),
        (120, 1e5),
        (40, 1e2),
        (40, 1e3),
        (40, 1e5),
    }

    assert set(results) == {"FBP", "GD"}
    assert set(results["FBP"]["reconstructions"]) == expected_keys
    assert set(results["GD"]["reconstructions"]) == expected_keys
    assert set(results["FBP"]["metrics"]) == expected_keys
    assert set(results["GD"]["metrics"]) == expected_keys
    assert results["FBP"]["runtime"] >= 0
    assert results["GD"]["runtime"] >= 0


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
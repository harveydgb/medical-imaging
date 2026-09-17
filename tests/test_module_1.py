import numpy as np

from med_im import module_1


def _checkerboard(shape=(8, 8)):
    return (np.indices(shape).sum(axis=0) % 2).astype(float)


def _full_grid(image, scale):
    return {
        (angles, I0): image * scale
        for angles in [8, 12]
        for I0 in [1e2, 1e3]
    }


def test_load_process_image_returns_scaled_grayscale_image():
    image = module_1.load_process_image()

    assert image.ndim == 2
    assert image.size > 0
    assert np.issubdtype(image.dtype, np.floating)
    assert np.all(np.isfinite(image))
    assert image.min() >= 0
    assert image.max() < 0.05


def test_create_noisy_sinograms_is_seeded_on_custom_grid():
    image = np.zeros((16, 16), dtype=float)
    angles_list = [8, 12]
    I0_list = [1e2, 1e3]

    first = module_1.create_noisy_sinograms(
        image,
        seed=123,
        angles_list=angles_list,
        I0_list=I0_list,
    )
    second = module_1.create_noisy_sinograms(
        image,
        seed=123,
        angles_list=angles_list,
        I0_list=I0_list,
    )

    expected_keys = {(8, 1e2), (8, 1e3), (12, 1e2), (12, 1e3)}

    assert set(first) == expected_keys
    for key in expected_keys:
        assert first[key].shape == second[key].shape
        assert first[key].shape[1] == key[0]
        assert np.all(first[key] >= 0)
        np.testing.assert_allclose(first[key], second[key])


def test_fbp_and_iterative_reconstructions_return_nonnegative_images():
    image = np.zeros((16, 16), dtype=float)
    image[4:12, 4:12] = _checkerboard((8, 8))
    sinograms = module_1.create_noisy_sinograms(
        image,
        seed=7,
        angles_list=[8],
        I0_list=[1e3],
    )
    theta = np.linspace(0, 360, 8, endpoint=False)

    fbp = module_1.FBP_backprojection(sinograms)
    gd = module_1.GD_backprojection(sinograms, angles=8, I0=1e3, theta=theta, max_iter=2, gamma=0.001)
    os_sart = module_1.OS_SART_reconstruct(
        sinograms,
        angles=8,
        I0=1e3,
        max_iter=1,
        gamma=0.001,
        n_subsets=2,
    )

    for reconstruction in [fbp[(8, 1e3)], gd, os_sart]:
        assert reconstruction.ndim == 2
        assert reconstruction.shape[0] == reconstruction.shape[1]
        assert np.all(reconstruction >= 0)


def test_reconstruction_metric_helpers_return_expected_keys():
    reference = _checkerboard()
    reconstruction = reference + 0.1

    metrics = module_1.reconstruction_metrics(reference, reconstruction)
    metrics_dict = module_1.reconstruction_metrics_dict(
        reference,
        {(20, 1e2): reconstruction},
    )

    assert set(metrics) == {"mse", "rmse", "mae", "psnr", "ssim"}
    assert metrics["mse"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["psnr"] > 0
    assert 0 <= metrics["ssim"] < 1
    assert set(metrics_dict) == {(20, 1e2)}
    assert np.isclose(metrics_dict[(20, 1e2)]["rmse"], metrics["rmse"])


def test_compare_reconstruction_methods_aggregates_results(monkeypatch):
    reference = _checkerboard()

    monkeypatch.setattr(module_1, "FBP_backprojection", lambda sinogram_dict, angle_range=360: _full_grid(reference, 0.9))
    monkeypatch.setattr(module_1, "GD_backprojection_compare", lambda sinogram_dict, angle_range=360: _full_grid(reference, 0.5))
    monkeypatch.setattr(module_1, "plot_sinogram_dict", lambda *args, **kwargs: None)

    results = module_1.compare_reconstruction_methods(reference, {}, save_prefix="exercise_1_1_c")

    assert set(results) == {"FBP", "GD"}
    assert set(results["FBP"]) == {"reconstructions", "metrics", "runtime"}
    assert results["FBP"]["metrics"][(8, 1e2)]["rmse"] > 0
    assert results["FBP"]["metrics"][(8, 1e2)]["rmse"] < results["GD"]["metrics"][(8, 1e2)]["rmse"]
    assert results["GD"]["metrics"][(8, 1e2)]["rmse"] > 0
    assert results["FBP"]["runtime"] >= 0
    assert results["GD"]["runtime"] >= 0


def test_plot_limited_angle_sinograms_rekeys_by_angle_range(monkeypatch):
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
        angle_ranges=[180, 120],
        projection_count=12,
        I0_list=[1e2, 1e3],
        seed=10,
    )

    assert set(sinograms) == {(180, 1e2), (180, 1e3), (120, 1e2), (120, 1e3)}
    assert np.all(sinograms[(180, 1e2)] == 180 + 1e2 / 1e5)


def test_compare_limited_angle_reconstruction_methods_aggregates_per_range(monkeypatch):
    reference = _checkerboard()

    monkeypatch.setattr(
        module_1,
        "create_noisy_sinograms",
        lambda image, angle_range=360, seed=None, angles_list=None, I0_list=None: {
            (angles_list[0], I0): np.full(reference.shape, angle_range + I0 / 1e5)
            for I0 in I0_list
        },
    )
    monkeypatch.setattr(module_1, "FBP_backprojection", lambda sinogram_dict, angle_range=360: dict(sinogram_dict))
    monkeypatch.setattr(
        module_1,
        "GD_backprojection_compare",
        lambda sinogram_dict, angle_range=360: {key: value * 0.5 for key, value in sinogram_dict.items()},
    )
    monkeypatch.setattr(module_1, "plot_sinogram_dict", lambda *args, **kwargs: None)

    results = module_1.compare_limited_angle_reconstruction_methods(
        reference,
        angle_ranges=[180, 120],
        projection_count=12,
        I0_list=[1e2, 1e3],
        seed=11,
    )

    expected_keys = {(180, 1e2), (180, 1e3), (120, 1e2), (120, 1e3)}

    assert set(results) == {"FBP", "GD"}
    assert set(results["FBP"]["reconstructions"]) == expected_keys
    assert set(results["GD"]["reconstructions"]) == expected_keys
    assert set(results["FBP"]["metrics"]) == expected_keys
    assert set(results["GD"]["metrics"]) == expected_keys
    assert results["FBP"]["runtime"] >= 0
    assert results["GD"]["runtime"] >= 0


def test_fbp_compare_filters_returns_one_reconstruction_per_filter(monkeypatch):
    values = {"ramp": 1.0, "shepp-logan": 2.0, "cosine": 3.0}

    monkeypatch.setattr(module_1, "_load_reference_image", lambda reference: None)
    monkeypatch.setattr(
        module_1,
        "iradon",
        lambda sinogram, theta, filter_name=None: np.full(
            (sinogram.shape[0], sinogram.shape[0]),
            values[filter_name],
        ),
    )

    reconstructions = module_1.FBP_compare_filters(
        {(6, 1e2): np.ones((8, 6), dtype=float)},
        angles=6,
        I0=1e2,
    )

    assert len(reconstructions) == 3
    assert [reconstruction[0, 0] for reconstruction in reconstructions] == [1.0, 2.0, 3.0]
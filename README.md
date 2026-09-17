# Medical Imaging

CT reconstruction from noisy sinograms and multi-coil MRI denoising — a documented,
tested Python package with a development notebook and a written report.

**Data Science for Medical Imaging · MPhil in Data Intensive Science, University of Cambridge (2025/26)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--image](https://img.shields.io/badge/scikit--image-radon%20%7C%20wavelet-orange)
![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)
![Docker](https://img.shields.io/badge/Docker-reproducible-2496ed)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Two imaging modalities, each posing the same underlying question from a different direction:
how much signal can be recovered once the measurement has been deliberately degraded?

**Module 1 — Tomographic reconstruction.** X-ray dose is reduced by lowering photon count and
by restricting the angular range, then images are reconstructed from the resulting noisy and
incomplete sinograms. Filtered backprojection is compared against iterative methods (SIRT,
SART, OS-SART) and across reconstruction filters, scored on PSNR and SSIM.

**Module 2 — MRI denoising.** Raw multi-coil k-space data is transformed to image space,
combined by root-sum-of-squares, and denoised with Gaussian, bilateral and wavelet filters.
A Butterworth low-pass filter is applied in the frequency domain, and SNR is estimated to
quantify what each method actually buys.

**Module 3** is written discussion only; see the report.

## Repository structure

```
.
├── med_im/
│   ├── module_1.py           # Tomographic reconstruction
│   └── module_2.py           # MRI coil combination and denoising
├── notebooks/
│   └── development.ipynb     # Runs every experiment end to end
├── tests/                    # 14 unit tests over the core functions
├── docs/                     # Sphinx documentation source
├── data/                     # CT image and multi-coil knee k-space
├── assets/                   # Generated figures
├── report/report.pdf         # Written answers to all modules
├── A2_Coursework.pdf         # Assessed report
├── Dockerfile
└── pyproject.toml
```

## Notebook contents

| Exercise | Content |
| --- | --- |
| 1.1 | Dose reduction — noisy sinogram simulation, FBP and gradient-descent reconstruction, error maps |
| 1.2 | Limited-angle acquisition across 40°, 120° and 180° ranges |
| 1.3 | Reconstruction filters and iterative methods (SIRT, SART, OS-SART) compared |
| 2.1 | k-space loading, coil magnitudes and phase, root-sum-of-squares combination |
| 2.2 | Gaussian, bilateral and wavelet denoising; Butterworth low-pass; SNR and metric comparison |

## Setup

```bash
pip install -e .
jupyter notebook notebooks/development.ipynb
```

Run the import cell first, then Module 1 and Module 2 cells in order. Paths resolve relative to
the repository root, so the notebook works from any working directory.

### Docker

```bash
docker build -t medical-imaging .
docker run --rm -p 8888:8888 medical-imaging
```

## Tests

```bash
pytest tests -v
```

Fourteen tests cover the parts most likely to break silently: that the k-space helpers match
the equivalent NumPy transforms, that coil combination really is root-sum-of-squares, that
denoisers preserve shape and return real values, that the Butterworth filter has the expected
frequency profile, that seeded sinogram generation is reproducible, and that the comparison
routines aggregate their metrics correctly.

## Documentation

Sphinx-compatible docstrings throughout `med_im/`, with sources in `docs/`:

```bash
sphinx-build -b html docs docs/_build
```

## Reproducibility

The stochastic CT sinogram simulations take an explicit seed, which the notebook passes into
every call to `create_noisy_sinograms`. Repeated runs therefore reproduce exactly; change the
seed for a different but equally repeatable realisation.

## Use of Generative Tools

This project has utilised auto-generative tools in the development of the repo and the code.

Example prompts used for this project:

- Generate code for a plot
- Create a general README.md template structure for this project
- Generate a doc-string for this function
- Review overall project structure for completeness, consistency and best practice
- Please create a test suite for the functions in my modules
- Present this text data in a clear table below the cell

## Author

Harvey Bermingham — MPhil in Data Intensive Science, University of Cambridge

## License

Released under the MIT License. See [LICENSE](LICENSE).

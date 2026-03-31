# Medical Imaging Coursework

This repository contains the Python code, notebook workflow, tests, and documentation used for Modules 1 and 2 of the medical imaging coursework aswell as the final report constaining the written answers to all questions.

## Project layout

- `med_im/`: reusable Python scripts for Module 1 and Module 2.
- `notebooks/`: development notebook used to run the experiments.
- `tests/`: unit tests for the core reconstruction and denoising functions.
- `docs/`: Sphinx-compatible documentation source.
- `data/`: coursework input data used by the code.
- `report/`: final report PDF.

## Module mapping

- Module 1 code: `med_im/module_1.py`
- Module 2 code: `med_im/module_2.py`
- Development notebook: `notebooks/development.ipynb`
- Module 3: report discussion only, no Python code required

## How to run the notebook

The notebook is designed to run relative to the repository files in `med_im/` and `data/`.

1. Open `notebooks/development.ipynb`.
2. Use the Python environment that has the coursework dependencies installed.
3. Run the import cell first.
4. Run the Module 1 cells in order.
5. Run the Module 2 cells in order.

The notebook prepends the repository root to `sys.path`, so no package installation step is required.

## Reproducibility

The stochastic CT sinogram simulations use explicit random seeds. The notebook passes a fixed seed into the relevant noise-generation calls so repeated runs are reproducible.

If you want a different repeatable run, change the seed value in the notebook cells that call `create_noisy_sinograms`.

## Running tests

From the `hb747/` directory:

```bash
pytest tests -v
```

The tests cover deterministic sinogram generation, reconstruction output structure, MRI k-space utilities, and denoising helper outputs.

## Building the documentation

From the `hb747/` directory:

```bash
sphinx-build -b html docs docs/_build/html
```

The generated HTML documentation will be written to `docs/_build/html`.

## Docker usage

Build the image from the `hb747/` directory:

```bash
docker build -t med-im-coursework .
```

Run the default test command:

```bash
docker run --rm med-im-coursework
```

## Notes

- This repository is used as a script-and-notebook project rather than an installed Python package.
- The functions in `med_im/` are documented with Sphinx-compatible docstrings.

## Use of Generative Tools

This project has utilised auto-generative tools in the development of the repo and the code.

Example prompts used for this project:
- Generate code for a plot
- Create a general README.md template structure for this project
- Generate a doc-string for this function
- Review overall project structure for completeness, consistency and best practice
- Present this text data in a clear table below the cell
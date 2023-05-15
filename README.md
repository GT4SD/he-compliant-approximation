# HELA: Homomorphic Encryption Learnable Approximations


|  |  |
|------------| ----- |
| License    | [![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://opensource.org/licenses/MIT)|
| Repository | ![Maintenance](https://img.shields.io/badge/maintained-yes-brightgreen) |
| Code       | [![Style: black](https://img.shields.io/badge/style-black-blue)](https://github.com/psf/black) [![Linter: flake8](https://img.shields.io/badge/linter-flake8-blue)](https://github.com/pycqa/flake8) [![Imports: isort](https://img.shields.io/badge/imports-isort-blue)](https://pycqa.github.io/isort/) |

The **HELA** (Homomorphic Encryption Learnable Approximations) is an open-source library to **accelerate the design of homomorphic encryption compliant neural networks**. This is possible by:

* substituting the neural network's modules.
* customizing the behaviour of approximated modules.
* organizing network training in a customizable pipeline, eventually with more than one approximation steps.
* saving training pipeline logs and checkpoints in a single tidy experiment folder.


## Installation guide

The package can be installed, for local development, with:
```bash
pip install -e .[dev,rdkit]
```

To avoid the installation of the `RDKit` dependency:
```bash
pip install -e .[dev]
```
Eventually, the `RDKit` dependency can be installed via Conda or Pypi:
```bash
# Install RDKit from Conda
conda install -c conda-forge rdkit

# Install RDKit from Pypi
pip install rdkit
# for Python<3.7
# pip install rdkit-pypi
```
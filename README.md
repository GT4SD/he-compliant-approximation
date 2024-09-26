# HEnets: a Framework for Homomorphic Encryption Compliant Neural Networks

<p align="center">
<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow"></a>
<a href="https://opensource.org/licenses/MIT"><img alt="Maintained: yes" src="https://img.shields.io/badge/maintained-yes-brightgreen"></a>
</p>
<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/pycqa/flake8"><img alt="Code linter: flake8" src="https://img.shields.io/badge/code%20linter-flake8-blue"></a>
<a href="https://pycqa.github.io/isort/"><img alt="Imports: isort" src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336"></a>
<a href="https://mypy-lang.org/"><img alt="Typing: mypy" src="https://img.shields.io/badge/typing-mypy-blue"></a>
<a href="https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings"><img alt="Doctrings: google" src="https://img.shields.io/badge/doctrings-google-blue"></a>
</p>

The **HEnets** is an open-source library to **accelerate the design of homomorphic encryption compliant neural networks**. This is possible by:

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
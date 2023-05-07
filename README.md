# HELA: Homomorphic Encryption Learnable Approximations

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
.. HECA documentation master file, created by
   sphinx-quickstart on Fri Mar 22 19:32:42 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HELA
====

.. raw:: html

   <p align="center">
   <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow"></a>
   <a href="https://github.com/GT4SD/he-compliant-approximation/graphs/commit-activity"><img alt="Maintained: yes" src="https://img.shields.io/badge/maintained-yes-brightgreen"></a>
   </p>
   <p align="center">
   <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
   <a href="https://github.com/pycqa/flake8"><img alt="Code linter: flake8" src="https://img.shields.io/badge/code%20linter-flake8-blue"></a>
   <a href="https://pycqa.github.io/isort/"><img alt="Imports: isort" src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336"></a>
   <a href="https://mypy-lang.org/"><img alt="Typing: mypy" src="https://img.shields.io/badge/typing-mypy-blue"></a>
   <a href="https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings"><img alt="Doctrings: google" src="https://img.shields.io/badge/doctrings-google-blue"></a>
   </p>

**HELA** (Homomorphic Encryption Learnable Approximations) is an open-source library to **accelerate the design of homomorphic encryption compliant neural networks**. This is possible by:

   * substituting the neural network's modules.
   * customizing the behaviour of approximated modules.
   * organizing network training in a customizable pipeline, eventually with more than one approximation steps.
   * saving training pipeline logs and checkpoints in a single tidy experiment folder.

.. toctree::
   :maxdepth: 5
   :caption: Contents:

   installation
   cli
   resources
   approximation_resources
   repository_structure

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

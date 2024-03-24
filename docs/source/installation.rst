Installation
============

.. autosummary::


By cloning the github repository on your device with:

.. code-block:: console

    $ git clone https://github.com/GT4SD/he-compliant-approximation.git

the package can then be installed for local development:

.. code-block:: console

    $ pip install -e .[dev]


To avoid the installation of the **RDKit** dependency, use instead:

.. code-block:: console

    $ pip install -e .[dev]

Eventually, the **RDKit** dependency can be installed via **Conda** or **Pypi**:

.. code-block:: console

    $ conda install -c conda-forge rdkit

Install **RDKit** from **Pypi**

.. code-block:: console

    $ pip install rdkit

For **Python < 3.7**

.. code-block:: console

    $ pip install rdkit-pypi

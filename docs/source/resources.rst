Internal Resources 
==================

The internal resources of the package are divided into: :ref:`approximation resources<approximation>` and :ref:`models' resources<models>`. 
These include approximation aliases, and other assets essential for the models approximations through the pipeline.

.. _approximation:

Approximation resources
-----------------------

Aliases
~~~~~~~

Approximation aliases define mappings between function names and their approximations. These aliases are used for approximating certain operations in models.

- **Approximation Aliases JSON**: Found at `src/hela/resources/approximation/aliases.json`, this JSON file lists aliases for various neural network modules, including their default approximation types and any approximation dependencies.

  Example entry:

  .. code-block:: json

    {
        "name": "relu",
        "aliases": ["torch.nn.modules.activation.ReLU", "torch.nn.functional.relu"],
        "default_approximation_type": "quadratic",
        "dependencies": []
    }

.. _models:

Models resources
----------------

Tokenizers
~~~~~~~~~~

The project includes tokenizers for processing and transforming input data. One such tokenizer is for SMILES (Simplified Molecular Input Line Entry System) vocabulary, which is crucial for chemical informatics applications.

- **SMILES Vocabulary**: Located at `src/hela/resources/models/tokenizers/smiles_vocab.txt`, this file contains a comprehensive list of SMILES tokens used for tokenizing chemical compound representations.


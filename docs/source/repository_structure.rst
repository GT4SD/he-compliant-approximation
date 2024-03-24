Repository structure
--------------------

.. raw:: html

   <style>
   .repository-tree{
       margin: auto;
       width: fit-content;
       text-align: left;
   }
   </style>

.. raw:: html

   <div class="repository-tree">
       <pre>
       <code>
       ./src/hela
        ├── approximation
        |   ├── <a href="./approximation_resources.html#module-aliases">aliases.py</a>
        |   ├── approximators
        |   |   └── ...
        |   ├── controller.py
        |   ├── <a href="./approximation_resources.html#module-to-approximate">module_to_approximate.py</a>
        |   ├── pipeline
        |   |   └── ...
        |   └── <a href="./approximation_resources.html#approximation-pipeline-steps">pipeline_steps.py</a>
        ├── <a href="./cli.html">cli</a>
        ├── models
        |   └── ...
        ├── pytorch_lightning
        │   ├── datasets
        │   │   └── ...
        │   └── models
        │       └── ...
        └── <a href="./resources.html">resources</a>
            ├── <a href="./resources.html#approximation-resources">approximation</a>
            │   └── ...
            └── <a href="./resources.html#models-resources">models</a>
                └── ...
            
       </code>
       </pre>
   </div>

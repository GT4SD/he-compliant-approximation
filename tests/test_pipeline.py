"""Testing the approximation pipeline."""

import tempfile

from henets.approximation.pipeline.core import Pipeline
from henets.models.vanilla_transformer.configuration import VanillaTransformerConfig
from henets.models.vanilla_transformer.model import VanillaTransformer
from henets.pytorch_lightning.models.language.vanilla_transformer import (
    LitApproximatedVanillaTransformer,
)

PIPELINE_STEPS_FILE = "./pipeline_steps/vanilla_transformer/without_approximations.json"


def test_init_approximation_pipeline():
    """Tests the initialization of an approximation pipeline."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        steps_path = PIPELINE_STEPS_FILE

        trainer_args = {}
        trainer_args["experiment_log_dir"] = tmpdirname
        trainer_args["experiment_name"] = "try"

        model = VanillaTransformer(VanillaTransformerConfig())

        pipeline = Pipeline(
            model=model,
            lightning_model_class=LitApproximatedVanillaTransformer,
            lightning_model_args={},
            trainer_args=trainer_args,
            pipeline_steps_path=steps_path,
        )

        # ASSERTS

        assert isinstance(
            pipeline, Pipeline
        ), "The pipeline object is not an instance of Pipeline."
        assert isinstance(
            pipeline.lightning_model, LitApproximatedVanillaTransformer
        ), "The pipeline's lightning model is not an instance of LitApproximatedVanillaTransformer."

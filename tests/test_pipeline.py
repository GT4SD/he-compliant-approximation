"""Testing the approximation pipeline."""

import tempfile

from hela.approximation.pipeline.core import ApproximationPipeline
from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig
from hela.models.vanilla_transformer.model import VanillaTransformer
from hela.pytorch_lightning.models.approximations.vanilla_transformer import (
    LitApproximatedVanillaTransformer,
)


def test_init_approximation_pipeline():
    """Tests the initialization of an approximation pipeline."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        steps_path = "./pipeline_steps/without_approximations.json"

        trainer_args = {}
        trainer_args["experiment_log_dir"] = tmpdirname
        trainer_args["experiment_name"] = "try"

        model = VanillaTransformer(VanillaTransformerConfig())

        pipeline = ApproximationPipeline(
            model=model,
            lightning_model_class=LitApproximatedVanillaTransformer,
            trainer_args=trainer_args,
            pipeline_steps_path=steps_path,
        )

        # ASSERTS

        assert isinstance(pipeline, ApproximationPipeline)
        assert isinstance(pipeline.lightning_model, LitApproximatedVanillaTransformer)

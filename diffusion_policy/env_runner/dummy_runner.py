from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class DummyRunner(BaseImageRunner, BaseLowdimRunner):
    """
    Generic dummy runner for evaluation that can handle both lowdim and image
    policies.
    """

    def __init__(self, output_dir, **kwargs):
        # Call both parent constructors
        BaseImageRunner.__init__(self, output_dir)
        BaseLowdimRunner.__init__(self, output_dir)

    def run(self, policy):
        # Return minimal placeholder log data
        return {}

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class Sim2RealLowdimRunner(BaseLowdimRunner):
    """
    Minimal dummy runner for sim2real lowdim evaluation.
    """

    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)

    def run(self, policy: BaseLowdimPolicy):
        # Return minimal placeholder log data
        return {}

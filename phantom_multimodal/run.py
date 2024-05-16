import hydra
from omegaconf import DictConfig, OmegaConf

from phantom_multimodal.utils.wandb import login_wandb


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="tasks",
)
def run(config: DictConfig) -> None:
    """.

    Args:
        config (DictConfig): .
    """
    print(OmegaConf.to_yaml(config))

    from phantom_multimodal.utils import TaskRunner

    runner = TaskRunner(config)
    login_wandb()
    runner.run_task()


if __name__ == "__main__":
    run()


from dataclasses import dataclass
from omegaconf import DictConfig

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from phantom_multimodal.utils.misc import seed_all
from phantom_multimodal.utils.lightning import instantiate_trainer, set_batch_size_and_num_workers

TORCH_COMPILE_MINIMUM_CUDA_VERSION = 7


class TaskRunner():
    """``project`` :class:`.BaseDataModule`.

    Args:
        config: See :class: DictConfig.

    Attributes:
        train_val_split (`tuple[float, float]`): The train/validation\
            split (sums to `1`).
        transform (:class:`~transforms.Compose`): The\
            :mod:`torchvision` dataset transformations.
    """

    def __init__(
        self: "TaskRunner",
        config: DictConfig,
    ) -> None:
        self.config = config
        self.datamodule = build_datamodule(self.config)
        self.litmodule = build_litmodule(self.config)
        self.logger = build_logger(self.config)


    def run_task(
        self: "TaskRunner",
    ) -> None:

        if self.config.task_type is "train":
            self.train()
        elif self.config.task_type is "test":
            self.test()
        else:
            error_msg = (
                "`task_type` must be defined as either `train` "
                "or `test` in the task config file."
            )
            raise NameError(error_msg)


    def train(
        self: "TaskRunner",
    ) -> None:
        """"
        Adapted from https://github.com/courtois-neuromod/cneuromax/blob/main/cneuromax/fitting/deeplearning/train.py
        """
        seed_all(self.config.seed)

        trainer: Trainer = instantiate_trainer(
            trainer=trainer,  # TODO: how is this variable specified?
            logger=self.logger,
            device=self.config.device,
            output_dir=self.config.output_dir,
            save_every_n_epochs=self.config.save_every_n_epochs,
        )

        set_batch_size_and_num_workers(
            trainer=trainer,
            datamodule=datamodule,
            litmodule=litmodule,
            device=config.device,
            output_dir=config.output_dir,
        )
        if (
            self.config.compile
            and self.config.device == "gpu"
            and torch.cuda.get_device_capability()[0]
            >= TORCH_COMPILE_MINIMUM_CUDA_VERSION
        ):
            self.litmodule = torch.compile(  # type: ignore [assignment]
                self.litmodule,  # mypy: `torch.compile`` not typed for `BaseLitModule`.
            )
        trainer.fit(model=litmodule, datamodule=datamodule, ckpt_path="last")
        """TODO: Add logic for HPO"""
        return trainer.validate(model=litmodule, datamodule=datamodule)[0][
            "val/loss"
        ]


    def test(
        self: "TaskRunner",
    ) -> None:
        pass

"""."""

import os
from pathlib import Path

import wandb


def login_wandb() -> None:
    """Logs into W&B using the key stored in ``WANDB_KEY.txt``."""
    wandb_key_path = Path(
        f"../wandb/WANDB_KEY.txt",
    ).resolve()
    if wandb_key_path.exists():
        with wandb_key_path.open(mode="r") as f:
            key = f.read().strip()
        wandb.login(key=key)
    else:
        error_msg = (
            "W&B key not found. You can retrieve your key from"
            "`https://wandb.ai/settings` and store it in a file named "
            "`WANDB_KEY.txt` under phantom_multimodal/wandb."
        )
        raise FileNotFoundError(error_msg)

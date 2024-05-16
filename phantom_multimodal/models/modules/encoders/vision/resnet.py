"""
Adapted from https://github.com/courtois-neuromod/video_transformer

https://github.com/courtois-neuromod/video_transformer/blob/main/scripts/apply_resnet_ridge_encoding.py
https://pytorch.org/vision/stable/models.html

"""

import torch
from torchvision.models import resnet152
from torchvision.models import ResNet152_Weights


class ResNet_Encoder():
    def __init__(
        self: "ResNet_Encoder",
        config: DictConfig,
    ) -> None:
        self.config = config

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet152(weights="IMAGENET1K_V2")
        self.model.to(device)

        torch.set_grad_enabled(False)
        self.model.eval()

        self.encode = ResNet152_Weights.IMAGENET1K_V2.transforms()




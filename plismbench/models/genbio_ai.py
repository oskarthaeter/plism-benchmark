"""Models from GenBio AI company."""

from __future__ import annotations

from typing import Any

import numpy as np
import timm
import torch
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class GenBioPathFM(Extractor):
    """GenBio_PathFM model developped by GenBio AI available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/genbio-ai/genbio-pathfm

    Parameters
    ----------
    device: int | list[int] | None = DEFAULT_DEVICE,
        Compute resources to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.
    mixed_precision: bool = True
        Whether to use mixed_precision.

    """

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.output_dim = 4608
        self.mixed_precision = mixed_precision

        try:
            from genbio_pathfm.model import GenBio_PathFM_Inference as build_model
        except ImportError:
            raise ImportError(
                "In order to use GenBio-PathFM, please run the following: 'pip install git+https://github.com/genbio-ai/genbio-pathfm.git --no-deps'"
            )
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(
            repo_id="genbio-ai/genbio-pathfm",
            filename="model.pth",
        )
        # Model    
        feature_extractor = build_model(weights_path, device="cpu")

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

    @property  # type: ignore
    def transform(self) -> transforms.Compose:
        """Transform method to apply element wise."""
        return transforms.Compose(
            [
                transforms.ToTensor(),  # swap axes and normalize
                transforms.Normalize(
                    mean=(0.697, 0.575, 0.728),
                    std=(0.188, 0.240, 0.187),
                ),
            ]
        )

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Compute and return features.

        Parameters
        ----------
        images: torch.Tensor
            Input of size (n_tiles, n_channels, dim_x, dim_y).

        Returns
        -------
            torch.Tensor: Tensor of size (n_tiles, features_dim).
        """
        features = self.feature_extractor(images.to(self.device))
        return features.cpu().numpy()
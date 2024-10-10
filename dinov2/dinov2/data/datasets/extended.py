# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torchvision.datasets import VisionDataset
from .trajectory_base import TrajectoryDatasetBase

from .decoders import TargetDecoder, ImageDataDecoder
import numpy as np


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:

            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError


class ExtendedTrajectoryDataset(TrajectoryDatasetBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_npy_data(self, index: int) -> np.ndarray:
        """
        Load the .npy file at the given index.
        """
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        """
        Load the target for the given sample.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Any]:
        """
        Fetches the .npy file and its target. Applies transformations if provided.
        """
        try:
            npy_data = self.get_npy_data(index)
        except Exception as e:
            raise RuntimeError(f"Cannot read .npy file for sample {index}") from e
        
        target = self.get_target(index)

        if self.transforms is not None:
            #print("self.transforms,", self.transforms)
            npy_data, target = self.transforms(npy_data, target)

        return npy_data, target

    def __len__(self) -> int:
        """
        Return the size of the dataset.
        """
        raise NotImplementedError
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
import numpy as np
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler


from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_percent=80,
        local_crops_percent=20,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_percent = global_crops_percent
        self.local_crops_percent = local_crops_percent

        print("local_crops_scale, local_crops_scale, local_crops_number, global_crops_percent, local_crops_percent", local_crops_scale, local_crops_scale, local_crops_number, global_crops_percent, local_crops_percent)
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_percent: {global_crops_percent}")
        logger.info(f"local_crops_percent: {local_crops_percent}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()


        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        return output


class TrajectoryDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        num_4hr_slots,
        #global_crops_percent=80,
        #local_crops_percent=20,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        #self.global_crops_percent = global_crops_percent
        #self.local_crops_percent = local_crops_percent

        self.data_points = int(num_4hr_slots)  # Number of data points in trajectory

        print("global_crops_scale, local_crops_scale, local_crops_number", self.global_crops_scale, self.local_crops_scale, local_crops_number)
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {self.global_crops_scale}")
        logger.info(f"local_crops_scale: {self.local_crops_scale}")
        logger.info(f"local_crops_number: {self.local_crops_number}")
        logger.info(f"number of 4 hour slots: {self.data_points}")
        logger.info("###################################")

        # normalization
        self.normalize = transforms.Compose(
            [
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([self.normalize])
        self.global_transfo2 = transforms.Compose([self.normalize])
        self.local_transfo = transforms.Compose([self.normalize])


    def _crop_data(self, data: np.ndarray, crops_scale: int) -> np.ndarray:
        """Crop data along the first dimension."""
        start = np.random.randint(0, self.data_points - crops_scale + 1)
        cropped_data = data[start:start + crops_scale, :, :]
        # print("Cropped data shape",cropped_data.shape)
        return torch.from_numpy(cropped_data).permute(2, 0, 1)

    def __call__(self, data: np.ndarray) -> dict:
        output = {}

        # Crop global context
        global_crops = [self.global_transfo1(self._crop_data(data, self.global_crops_scale)) for _ in range(2)]
        output["global_crops"] = global_crops

        # Crop local context
        local_crops = [self.local_transfo(self._crop_data(data, self.local_crops_scale)) for _ in range(self.local_crops_number)]
        output["local_crops"] = local_crops

        

        # DO THE AUGMENTATION OF UNIX TO TIME FORMAT LOADING HERE.

        # Normalize the dataset features
        # Assuming the dataset returns a tensor, convert to numpy for normalization
        #normalized_dataset = [normalize_data(item.numpy()) for item in dataset]

        # If you need to convert it back to tensor
        # normalized_dataset = [torch.tensor(item) for item in normalized_dataset]


        return output

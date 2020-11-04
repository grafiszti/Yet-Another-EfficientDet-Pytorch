from torch.utils.data import DataLoader

from efficientdet.dataset import CocoDataset
from train import get_training_transform_weak, get_training_transform

import cv2

import numpy as np

test_dataset_dir = "test/test_coco_dataset"


def get_test_train_dataloader(transform, albu_transform: bool):
    return DataLoader(
        CocoDataset(
            root_dir=test_dataset_dir,
            set="train",
            transform=transform,
            is_albu_transform=albu_transform,
        )
    )


class TestAugmentations:
    def test_training_augmentations_weak(self):
        weak_transform = get_training_transform_weak(
            input_size=200, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        for data in get_test_train_dataloader(weak_transform, albu_transform=False):
            image = data["img"]
            annotations = data["annot"]
            assert len(annotations) > 0

    def test_training_augmentations_strong(self):
        transform = get_training_transform(
            input_size=200, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        for data in get_test_train_dataloader(transform, albu_transform=True):
            assert data["img"].cpu().numpy().shape == (1, 200, 200, 3)

    def test_generate_some_test_images(self):
        transform = get_training_transform(
            input_size=300, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        for i, data in enumerate(
            get_test_train_dataloader(transform, albu_transform=True)
        ):
            image = (data["img"].cpu().numpy() * 255).astype(np.int)
            cv2.imwrite(image, f"test/image{i}.jpg")
            # assert data["img"].cpu().numpy().shape == (1, 200, 200, 3)



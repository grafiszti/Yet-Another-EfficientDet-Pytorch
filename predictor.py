from typing import List, Tuple

import os

import torch
import numpy as np

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from config import input_sizes
from consts import INVALID_STATE_DICT_LOAD_ERROR
from utils.utils import postprocess, preprocess_images


class CocoPredictor:
    def __init__(
        self,
        model_path: str,
        compound_coef: int,
        device: str,
        gpu: int,
        classes_number: int,
        anchor_ratios: List[Tuple],
        anchor_scales: List[float],
        use_float16: bool,
    ):
        assert device == "cpu" or device == "cuda"
        assert 0 <= compound_coef <= 8
        assert classes_number > 0
        if not os.path.exists(model_path):
            raise ValueError(f"Model on path: {model_path}, does not exists.")

        self.device = device
        self.use_float16 = use_float16
        self.compound_coef = compound_coef
        self.max_size = input_sizes[self.compound_coef]

        self.model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=classes_number,
            ratios=anchor_ratios,
            scales=anchor_scales,
        )

        loaded_weights = torch.load(model_path)

        try:
            self.model.load_state_dict(loaded_weights)
        except RuntimeError as e:
            print(INVALID_STATE_DICT_LOAD_ERROR.format(e))

        self.model.requires_grad_(False)
        self.model.eval()

        if device == "cuda":
            self.model = self.model.cuda(gpu)
        if use_float16:
            self.model = self.model.half()

    def predict(self, image: np.array, threshold: float, iou_threshold: float = 0.2):
        with torch.no_grad():
            _, framed_imgs, _ = preprocess_images([image], max_size=self.max_size)

            if self.device == "cuda":
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(
                0, 3, 1, 2
            )

            features, regression, classification, anchors = self.model(x)

            # prediction is run for single image at the time
            out = postprocess(
                x,
                anchors,
                regression,
                classification,
                BBoxTransform(),
                ClipBoxes(),
                threshold,
                iou_threshold,
            )[0]

            return out["rois"], out["class_ids"], out["scores"]

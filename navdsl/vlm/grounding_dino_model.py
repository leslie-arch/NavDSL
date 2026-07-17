# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F

from navdsl.vlm.detections import ObjectDetections

file_dir = os.path.dirname(os.path.abspath(__file__))
GroundingDINO_dir = os.path.abspath(os.path.join(file_dir, "../../../GroundingDINO"))

from groundingdino.util.inference import load_model, predict

GROUNDING_DINO_CONFIG = os.path.join(
    GroundingDINO_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class GroundingDINO:
    """
    Grounding DINO (Detection with Input and Output)模型类
    用于基于文本提示的对象检测
    """

    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(
            model_config_path=config_path, model_checkpoint_path=weights_path
        ).to(device)
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(
        self, image: np.ndarray, caption: Optional[str] = None
    ) -> ObjectDetections:
        image_tensor = F.to_tensor(image)
        image_transformed = F.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("Caption:", caption_to_use)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)

        classes = caption_to_use[: -len(" .")].split(" . ")
        detections.filter_by_class(classes)

        return detections

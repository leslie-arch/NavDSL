# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch

from navdsl.vlm.coco_classes import COCO_CLASSES
from navdsl.vlm.detections import ObjectDetections

file_dir = os.path.dirname(os.path.abspath(__file__))
yolov7_dir = os.path.abspath(os.path.join(file_dir, "../../../yolov7"))

sys.path.insert(0, yolov7_dir)
try:
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import (
        check_img_size,
        non_max_suppression,
        scale_coords,
    )
    from utils.torch_utils import TracedModel
except Exception:
    print(
        "vlm.yolov7: Could not import yolov7. This is OK if you are only using the client."
    )
sys.path.pop(0)


class YOLOv7:
    def __init__(
        self, weights: str, image_size: int = 640, half_precision: bool = True
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.half_precision = self.device.type != "cpu" and half_precision
        self.model = attempt_load(weights, map_location=self.device)
        stride = int(self.model.stride.max())
        self.image_size = check_img_size(image_size, s=stride)
        self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half_precision:
            self.model.half()

        if self.device.type != "cpu":
            dummy_img = torch.rand(
                1, 3, int(self.image_size * 0.7), self.image_size
            ).to(self.device)
            if self.half_precision:
                dummy_img = dummy_img.half()
            for i in range(3):
                self.model(dummy_img)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        orig_shape = image.shape

        img = cv2.resize(
            image,
            (self.image_size, int(self.image_size * 0.7)),
            interpolation=cv2.INTER_AREA,
        )
        img = letterbox(img, new_shape=self.image_size)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_precision else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.inference_mode():
            pred = self.model(img)[0]

        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
        )[0]

        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig_shape).round()
        pred[:, 0] /= orig_shape[1]
        pred[:, 1] /= orig_shape[0]
        pred[:, 2] /= orig_shape[1]
        pred[:, 3] /= orig_shape[0]

        boxes = pred[:, :4]
        logits = pred[:, 4]
        phrases = [COCO_CLASSES[int(i)] for i in pred[:, 5]]

        detections = ObjectDetections(
            boxes, logits, phrases, image_source=image, fmt="xyxy"
        )

        return detections

# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_FIBER import GLIPDemo

from vlfm.vlm.detections import ObjectDetections

DEFAULT_CONFIG = "FIBER/fine_grained/configs/refcocog.yaml"
DEFAULT_WEIGHTS = "FIBER/fine_grained/models/fiber_refcocog.pth"


class FIBER:
    """
    FIBER (Fine-grained Instance-level Binding and Expression Recognition)模型类
    用于基于自然语言表达式的细粒度目标检测和识别
    """

    def __init__(
        self, config_file: str = DEFAULT_CONFIG, weights: str = DEFAULT_WEIGHTS
    ):
        cfg.merge_from_file(config_file)
        cfg.num_gpus = 1
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.TEST.IMS_PER_BATCH = 1
        cfg.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM = -1
        cfg.TEST.EVAL_TASK = "grounding"
        cfg.MODEL.ATSS.PRE_NMS_TOP_N = 3000
        cfg.MODEL.ATSS.DETECTIONS_PER_IMG = 100
        cfg.MODEL.ATSS.INFERENCE_TH = 0.0
        cfg.MODEL.WEIGHT = weights

        cfg.freeze()

        self.fiber = GLIPDemo(cfg, confidence_threshold=0.2)

    def detect(
        self, image: np.ndarray, phrase: str, visualize: bool = False
    ) -> ObjectDetections:
        result = self.fiber.inference(image, phrase)
        normalized_bbox = result.bbox / torch.tensor(
            [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        )

        dets = ObjectDetections(
            image_source=image,
            boxes=normalized_bbox,
            logits=result.extra_fields["scores"],
            phrases=[phrase],
            fmt="xyxy",
        )

        return dets

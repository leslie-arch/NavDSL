# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import functools
from typing import Any, List, Optional

import numpy as np
import torch

try:
    from mobile_sam import SamPredictor, sam_model_registry
except ModuleNotFoundError:
    print("Could not import mobile_sam. This is OK if you are only using the client.")


class MobileSAM:
    """
    移动版本的Segment Anything Model (SAM)
    用于图像分割任务，特别是针对边界框内的对象
    """

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_t",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        original_torch_load = torch.load
        torch.load = functools.partial(original_torch_load, weights_only=False)

        try:
            mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            mobile_sam.to(device=device)
            mobile_sam.eval()
            self.predictor = SamPredictor(mobile_sam)
        finally:
            torch.load = original_torch_load

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        with torch.inference_mode():
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(
                box=np.array(bbox), multimask_output=False
            )

        return masks[0]

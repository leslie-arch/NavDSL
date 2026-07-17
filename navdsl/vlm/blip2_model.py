# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

try:
    from lavis.models import load_model_and_preprocess
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")


class BLIP2:
    """
    BLIP2模型类
    用于图像理解、描述生成和视觉问答任务
    BLIP: Bootstrapping Language-Image Pre-training
    """

    def __init__(
        self,
        name: str = "blip2_t5",
        model_type: str = "pretrain_flant5xxl",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        self.device = device

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        pil_img = Image.fromarray(image)
        with torch.inference_mode():
            processed_image = (
                self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
            )
            if prompt is None or prompt == "":
                out = self.model.generate({"image": processed_image})[0]
            else:
                out = self.model.generate({"image": processed_image, "prompt": prompt})[
                    0
                ]

        return out

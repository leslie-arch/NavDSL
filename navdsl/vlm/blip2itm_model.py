# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

try:
    from lavis.models import load_model_and_preprocess
    from lavis.models import model_zoo
    from lavis.models import BlipCaption
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

file_dir = os.path.dirname(os.path.abspath(__file__))
chk_point_path = os.path.join("../../data/blip2_pretrained.pth")


class BLIP2ITM:
    """BLIP 2 Image-Text Matching model.

    BLIP 2 图像-文本匹配模型。
    用于评估图像和文本之间的相似度。
    """

    def __init__(
        self,
        name: str = "blip_image_text_matching",
        model_type: str = "base",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name=name,
                model_type=model_type,
                is_eval=True,
                device=device,
            )
        )
        self.device = device

    def cosine(self, image: np.ndarray, txt: str) -> float:
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            cosine = self.model(
                {"image": img, "text_input": txt}, match_head="itc"
            ).item()

        return cosine

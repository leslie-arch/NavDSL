# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import numpy as np

from navdsl.vlm.detections import ObjectDetections
from .server_wrapper import send_request


class GroundingDINOClient:
    """
    Grounding DINO模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """

    def __init__(self, port: int = 8080):
        self.url = f"http://localhost:{port}/gdino"

    def predict(
        self, image_numpy: np.ndarray, caption: Optional[str] = ""
    ) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections

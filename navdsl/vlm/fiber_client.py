# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from vlfm.vlm.detections import ObjectDetections
from .server_wrapper import send_request


class FIBERClient:
    """
    FIBER模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """

    def __init__(self, port: int = 8080):
        self.url = f"http://localhost:{port}/fiber"

    def detect(
        self, image: np.ndarray, phrase: str, visualize: bool = False
    ) -> ObjectDetections:
        response = send_request(self.url, image=image, phrase=phrase)["response"]
        detections = ObjectDetections.from_json(response, image_source=image)
        return detections

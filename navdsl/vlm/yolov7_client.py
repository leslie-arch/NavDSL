# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from navdsl.vlm.detections import ObjectDetections
from .server_wrapper import send_request


class YOLOv7Client:
    """
    YOLOv7客户端类，用于与YOLOv7服务器通信
    """

    def __init__(self, port: int = 8080):
        self.url = f"http://localhost:{port}/yolov7"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections

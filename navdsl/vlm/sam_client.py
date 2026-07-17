# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List

import numpy as np

from .server_wrapper import send_request, str_to_bool_arr


class MobileSAMClient:
    """
    移动版SAM模型的客户端，
    通过HTTP请求与服务器端的模型通信
    """

    def __init__(self, port: int = 8080):
        self.url = f"http://localhost:{port}/mobile_sam"

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        response = send_request(self.url, image=image, bbox=bbox)
        cropped_mask_str = response["cropped_mask"]
        cropped_mask = str_to_bool_arr(cropped_mask_str, shape=tuple(image.shape[:2]))
        return cropped_mask

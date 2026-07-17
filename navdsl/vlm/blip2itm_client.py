# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from .server_wrapper import send_request


class BLIP2ITMClient:
    """
    BLIP2图像-文本匹配模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """

    def __init__(self, port: int = 8080):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])

# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import numpy as np

from .server_wrapper import send_request


class BLIP2Client:
    """
    BLIP2模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """

    def __init__(self, port: int = 8080):
        self.url = f"http://localhost:{port}/blip2"

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = ""
        response = send_request(self.url, image=image, prompt=prompt)
        return response["response"]

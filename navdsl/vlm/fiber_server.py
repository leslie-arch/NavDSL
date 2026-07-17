# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from .fiber_model import FIBER
from .server_wrapper import ServerMixin, host_model, str_to_image


class FIBERServer(ServerMixin, FIBER):
    """
    FIBER模型的服务器实现
    继承ServerMixin和FIBER类
    """

    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        return {"response": self.detect(image, payload["phrase"]).to_json()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9080)
    parser.add_argument("--ip", type=str, default="localhost")
    args = parser.parse_args()

    print("Loading model...")
    fiber = FIBERServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(fiber, name="fiber", port=args.port, ip=args.ip)

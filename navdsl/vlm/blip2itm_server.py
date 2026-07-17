# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from .blip2itm_model import BLIP2ITM
from .server_wrapper import ServerMixin, host_model, str_to_image


class BLIP2ITMServer(ServerMixin, BLIP2ITM):
    """
    BLIP2ITM模型的服务器实现
    继承ServerMixin和BLIP2ITM类
    """

    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        return {"response": self.cosine(image, payload["txt"])}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--ip", type=str, default="localhost")
    args = parser.parse_args()

    print("Loading model...")
    blip = BLIP2ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port, ip=args.ip)

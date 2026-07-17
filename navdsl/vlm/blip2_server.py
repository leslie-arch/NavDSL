# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from .blip2_model import BLIP2
from .server_wrapper import ServerMixin, host_model, str_to_image


class BLIP2Server(ServerMixin, BLIP2):
    """
    BLIP2模型的服务器实现
    继承ServerMixin和BLIP2类
    """

    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        return {"response": self.ask(image, payload.get("prompt"))}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--ip", type=str, default="localhost")
    args = parser.parse_args()

    print("Loading model...")
    blip = BLIP2Server(name="blip2_t5", model_type="pretrain_flant5xl")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2", port=args.port, ip=args.ip)

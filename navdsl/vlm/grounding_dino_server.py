# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from .grounding_dino_model import GroundingDINO
from .server_wrapper import ServerMixin, host_model, str_to_image


class GroundingDINOServer(ServerMixin, GroundingDINO):
    """
    Grounding DINO模型的服务器实现
    继承ServerMixin和GroundingDINO类
    """

    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        return self.predict(image, caption=payload["caption"]).to_json()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    parser.add_argument("--ip", type=str, default="localhost")
    args = parser.parse_args()

    print("Loading model...")
    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port, ip=args.ip)

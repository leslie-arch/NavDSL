# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from .yolov7_model import YOLOv7
from .server_wrapper import ServerMixin, host_model, str_to_image


class YOLOv7Server(ServerMixin, YOLOv7):
    """
    YOLOv7服务器类，处理接收到的请求并返回检测结果
    """

    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        return self.predict(image).to_json()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    parser.add_argument("--ip", type=str, default="localhost")
    args = parser.parse_args()

    print("Loading model...")
    yolov7 = YOLOv7Server("data/yolov7-e6e.pt")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolov7, name="yolov7", port=args.port, ip=args.ip)

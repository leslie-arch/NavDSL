# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

from .sam_model import MobileSAM
from .server_wrapper import ServerMixin, bool_arr_to_str, host_model, str_to_image


class MobileSAMServer(ServerMixin, MobileSAM):
    """
    Mobile SAM模型的服务器实现，
    继承ServerMixin和MobileSAM类
    """

    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        cropped_mask = self.segment_bbox(image, payload["bbox"])
        cropped_mask_str = bool_arr_to_str(cropped_mask)
        return {"cropped_mask": cropped_mask_str}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12183)
    parser.add_argument("--ip", type=str, default="localhost")
    args = parser.parse_args()

    print("Loading model...")
    mobile_sam = MobileSAMServer(
        sam_checkpoint=os.environ.get("MOBILE_SAM_CHECKPOINT", "data/mobile_sam.pt")
    )
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(mobile_sam, name="mobile_sam", port=args.port, ip=args.ip)

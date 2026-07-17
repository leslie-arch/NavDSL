"""
VLM统一服务入口

将多个VLM模型托管在同一个Flask进程中，通过路由区分。

用法:
    # 启动所有模型
    python -m navdsl.vlm.server --all --ip 0.0.0.0 --port 8080

    # 启动指定模型（逗号分隔）
    python -m navdsl.vlm.server --models blip2,yolov7,sam --port 8080

    # 可用模型名称: blip2, yolov7, sam, gdino, blip2itm, fiber
"""

import argparse
import sys

from .server_wrapper import host_models

# 支持的模型名称 -> (Server类导入路径, 默认构造参数工厂函数)
SUPPORTED_MODELS = {
    "blip2": {
        "module": ".blip2_server",
        "class_name": "BLIP2Server",
        "route": "blip2",
        "factory": lambda args: {"name": "blip2_t5", "model_type": "pretrain_flant5xl"},
    },
    "yolov7": {
        "module": ".yolov7_server",
        "class_name": "YOLOv7Server",
        "route": "yolov7",
        "factory": lambda args: ("data/yolov7-e6e.pt",),
    },
    "sam": {
        "module": ".sam_server",
        "class_name": "MobileSAMServer",
        "route": "mobile_sam",
        "factory": lambda args: {"sam_checkpoint": args.sam_checkpoint},
    },
    "gdino": {
        "module": ".grounding_dino_server",
        "class_name": "GroundingDINOServer",
        "route": "gdino",
        "factory": lambda args: {},
    },
    "blip2itm": {
        "module": ".blip2itm_server",
        "class_name": "BLIP2ITMServer",
        "route": "blip2itm",
        "factory": lambda args: {},
    },
    "fiber": {
        "module": ".fiber_server",
        "class_name": "FIBERServer",
        "route": "fiber",
        "factory": lambda args: {},
    },
}


def _load_class(module_path: str, class_name: str) -> type:
    """动态导入模型Server类"""
    import importlib

    mod = importlib.import_module(module_path, package=__package__)
    return getattr(mod, class_name)


def build_models(args: argparse.Namespace) -> list:
    """
    根据命令行参数构建模型列表。

    Args:
        args: 命令行参数

    Returns:
        [(route_name, model_instance), ...] 的列表
    """
    models = []
    model_names = [m.strip() for m in args.models.split(",")] if args.models else []

    for name in model_names:
        if name not in SUPPORTED_MODELS:
            print(f"Error: unsupported model '{name}'", file=sys.stderr)
            print(
                f"Supported models: {', '.join(sorted(SUPPORTED_MODELS.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)

        cfg = SUPPORTED_MODELS[name]
        cls = _load_class(cfg["module"], cfg["class_name"])
        params = cfg["factory"](args)

        if isinstance(params, tuple):
            model = cls(*params)
        else:
            model = cls(**params)

        models.append((cfg["route"], model))

    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM统一服务入口")
    parser.add_argument(
        "--ip", type=str, default="localhost", help="监听地址 (默认: localhost)"
    )
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="逗号分隔的模型名: blip2,yolov7,sam,gdino,blip2itm,fiber",
    )
    parser.add_argument("--all", action="store_true", help="启动所有模型")
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="data/mobile_sam.pt",
        help="MobileSAM模型权重路径",
    )
    args = parser.parse_args()

    if not args.models and not args.all:
        parser.error("请指定 --models 或 --all")

    if args.all:
        args.models = ",".join(sorted(SUPPORTED_MODELS.keys()))

    models = build_models(args)
    print(f"Loaded {len(models)} models: {[n for n, _ in models]}")
    host_models(models, ip=args.ip, port=args.port)


if __name__ == "__main__":
    main()

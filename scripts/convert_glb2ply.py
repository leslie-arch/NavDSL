#!/bin/python3

import os
import glob
import argparse
# import aspose.threed as a3d  # commercial product
import trimesh
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root", type=str, default='.', help="root dataset path")

def convert_glb2ply(args):
    target_path = args.root
    if os.path.isfile(target_path):
        raise RuntimeError("Parameter must be a directory.")
    if not os.path.isabs(target_path):
        target_path = os.path.join(os.getcwd(), target_path)

    name_tail = "_semantic.glb"
    name_tail_len = len(name_tail)
    files_list = glob.glob(os.path.join(target_path, "**/*.semantic.glb"), recursive=True)
    for f_source in files_list:
        f_target = f"{f_source[:-name_tail_len]}_semantic.ply"
        print(f"convert {f_source} -> {f_target}")
        scene = trimesh.load(f_source)
        if isinstance(scene, trimesh.Scene):
            scene = scene.dump(concatenate=True)
        # 【核心修复】：强制转换所有数值属性为 float32
        # 转换顶点坐标
        scene.vertices = scene.vertices.astype(np.float32)

        # 转换纹理坐标 (UV)
        if hasattr(scene.visual, 'uv'):
            scene.visual.uv = scene.visual.uv.astype(np.float32)

        # 转换顶点颜色（如果有）为 uint8
        if hasattr(scene.visual, 'vertex_colors'):
            scene.visual.vertex_colors = scene.visual.vertex_colors.astype(np.uint8)
        scene.export(f_target)

    print(f"convert [{len(files_list)}] semantic glb files Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    convert_glb2ply(args)


#!/bin/python3

import os
import glob
import argparse
# import aspose.threed as a3d  # commercial product
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root", type=str, default='.', help="root dataset path")

def convert_hm3d_semantic(src, dst):
    with open(src, "r") as f:
        lines = f.readlines()

    with open(dst, "w") as f:
        for line in lines:
            if line.startswith("HM3D"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                obj_id = parts[0]
                category = parts[2].strip('"')
                f.write(f"{obj_id},{category}\n")


def convert_semantic_descripter(args):
    target_path = args.root
    if os.path.isfile(target_path):
        raise RuntimeError("Parameter must be a directory.")
    if not os.path.isabs(target_path):
        target_path = os.path.join(os.getcwd(), target_path)

    name_tail = "_semantic.txt"
    name_tail_len = len(name_tail)
    files_list = glob.glob(os.path.join(target_path, "**/*.semantic.txt"), recursive=True)
    for f_source in files_list:
        f_modified = f"{f_source[:-name_tail_len]}_semantic.txt"
        shutil.copy(f_source, f"{f_source}.bak")
        convert_hm3d_semantic(f_modified, f_source)
        print(f"convert {f_modified} -> {f_source}")

    print(f"convert [{len(files_list)}] semantic txt files Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    convert_semantic_descripter(args)



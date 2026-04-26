#!/usr/bin/python3

import shutil
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root", type=str, default='.', help="root dataset path")

def rename_semantic_scns(args):
    target_path = args.root
    if os.path.isfile(target_path):
        raise RuntimeError("Parameter must be a directory.")
    if not os.path.isabs(target_path):
        target_path = os.path.join(os.getcwd(), target_path)

    name_tail = "_semantic.txt"
    name_tail_len = len(name_tail)
    files_list = glob.glob(os.path.join(target_path, "**/*.semantic.txt"), recursive=True)
    for f_source in files_list:
        f_target = f"{f_source[:-name_tail_len]}_semantic.txt"
        print(f"cp {f_source} -> {f_target}")
        shutil.copy(f_source, f_target)

    print(f"rename [{len(files_list)}] semantic scn files Done.")

if __name__ == "__main__":
    args = parser.parse_args()
    rename_semantic_scns(args)


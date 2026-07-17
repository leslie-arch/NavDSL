#!/usr/bin/env python3
"""Split the monolithic HM3D-AutoVLN LMDB files into per-scan LMDB files.

Source (single large LMDB per feature type):
  features/view_timm_imagenet_vitb16/data.mdb         (9.2 GB)
  features/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16/data.mdb  (3.8 GB)

Target (per-scan LMDB, combined view + obj):
  features/per_scan/train/00000-kfPV7w3FaU5.lmdb      (~14 MB each)
  features/per_scan/val/00800-TEEsavR23oF.lmdb

Each per-scan LMDB contains:
  key 'view_{vp}' -> msgpack ndarray (36, 1768)
  key 'obj_{vp}'  -> msgpack dict {fts, obj_ids, obj_names, bboxes, centers, ...}

Split convention: scene_id numeric prefix < 800 -> train, >= 800 -> val
(matches HM3DAutoVLN/step00_create_nav_graphs.py:18-23 and our converter).

Processing strategy:
  LMDB cursor iterates in key-sorted order, so all entries for the same scan
  are contiguous. We process scan-by-scan: open env when first entry for a
  scan appears, close env when scan changes. This keeps only 1 env open at a
  time, avoiding both:
    - 'Too many open files' (ulimit -n)
    - Excessive memory from cached env handles

Run on the remote host:
  python -m navdsl.data_adapter.split_lmdb_per_scan \
      --source-view-lmdb /path/to/NAV_GRAPH/features/view_timm_imagenet_vitb16 \
      --source-obj-lmdb /path/to/NAV_GRAPH/features/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16 \
      --output-dir /path/to/NAV_GRAPH/features/per_scan
"""
import argparse
import os
import sys
import time

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter([])


TRAIN_SCENE_MAX = 800  # scene_id numeric prefix < this goes to train
DEFAULT_MAP_SIZE = 256 * 1024 * 1024  # 256 MB; sparse on modern filesystems


def split_of_scan(scan: str) -> str:
    try:
        prefix = int(scan.split("-")[0])
    except (ValueError, IndexError):
        return "val"
    return "train" if prefix < TRAIN_SCENE_MAX else "val"


def open_per_scan_env(out_root: str, split: str, scan: str,
                      map_size: int = DEFAULT_MAP_SIZE):
    """Create / open a per-scan LMDB at {out_root}/{split}/{scan}.lmdb.

    Uses subdir=False so each scan is a single file (not a directory containing
    data.mdb). This matches the goal of per-scan file granularity for
    maintainability — one .lmdb per scan, mirroring the HM3D mesh layout
    (one directory per scan).
    """
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, f"{scan}.lmdb")
    import lmdb
    return lmdb.open(path, map_size=map_size, lock=False, subdir=False)


def copy_split_scan_by_scan(source_path: str, kind: str, out_root: str):
    """Iterate the source LMDB in key-sorted order; for each scan, open a
    per-scan env, write all of its entries, then close the env.

    This keeps at most 1 write env open at a time.

    Returns dict {scan: count} for stats.
    """
    import lmdb

    print(f"Reading source: {source_path}", file=sys.stderr)
    env = lmdb.open(source_path, readonly=True, lock=False)

    scan_counts: dict = {}
    cur_scan = None
    cur_split = None
    cur_env = None

    with env.begin() as txn:
        n_total = txn.stat()["entries"]
        print(f"  total entries: {n_total}", file=sys.stderr)

        for key, value in tqdm(txn.cursor(), total=n_total, desc=f"split {kind}", file=sys.stderr):
            key_str = key.decode("ascii")
            try:
                scan, vp = key_str.rsplit("_", 1)
            except ValueError:
                print(f"  skip malformed key: {key_str}", file=sys.stderr)
                continue

            # Detect scan change → close previous env, open new one
            if scan != cur_scan:
                if cur_env is not None:
                    cur_env.close()
                    cur_env = None
                cur_scan = scan
                cur_split = split_of_scan(scan)
                cur_env = open_per_scan_env(out_root, cur_split, scan)

            # Write entry
            with cur_env.begin(write=True) as wtxn:
                wtxn.put(f"{kind}_{vp}".encode("ascii"), value)

            scan_counts[scan] = scan_counts.get(scan, 0) + 1

    # Close the last env
    if cur_env is not None:
        cur_env.close()
    env.close()
    return scan_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-view-lmdb", required=True,
        help="Path to source view feature LMDB *directory* (e.g., view_timm_imagenet_vitb16)",
    )
    ap.add_argument(
        "--source-obj-lmdb", required=True,
        help="Path to source object feature LMDB *directory*",
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Output root directory; will create {train,val}/{scan}.lmdb inside",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    view_counts = copy_split_scan_by_scan(args.source_view_lmdb, "view", args.output_dir)
    obj_counts = copy_split_scan_by_scan(args.source_obj_lmdb, "obj", args.output_dir)

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s: "
        f"{len(view_counts)} scans got view features, "
        f"{len(obj_counts)} scans got object features.",
        file=sys.stderr,
    )
    print(f"Output: {args.output_dir}/{{train,val}}/<scan>.lmdb", file=sys.stderr)

    # Print split breakdown
    train_n = sum(1 for s in view_counts if split_of_scan(s) == "train")
    val_n = sum(1 for s in view_counts if split_of_scan(s) == "val")
    print(f"  train: {train_n} scans", file=sys.stderr)
    print(f"  val:   {val_n} scans", file=sys.stderr)


if __name__ == "__main__":
    main()

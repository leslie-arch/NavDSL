#!/usr/bin/env python3
"""Offline converter: HM3D-AutoVLN speaker output -> habitat-lab VLN episodes.

Inputs:
  --speaker-jsonl    : ade20k_pseudo3d_depth2_epoch_94_beam0.jsonl (one entry per
                       generated instruction, contains path + instruction + pos_vps).
  --connectivity-dir : directory of {scene}_connectivity.json files from step00.

Output:
  {output_dir}/{split}/{split}.json.gz  with habitat-lab-compatible episodes.

Episode schema (consumed by HM3DAutoVLNDatasetV1):
  episode_id, scene_id, scene_scan_id, start_position, start_rotation,
  goals, reference_path, reference_viewpoints, instruction, trajectory_id,
  start_viewpoint_id, goal_viewpoint_id, target_object_id,
  target_visible_viewpoints.
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Optional progress bars — fall back to no-op if tqdm is missing.
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else iter([])

# Numeric prefix of scene_id determines train/val split (step00_create_nav_graphs.py:18-23)
TRAIN_SCENE_MAX = 800


def parse_pose(pose: List[float]) -> Tuple[float, float, float]:
    """4x4 transform matrix flattened to 16 elements. Habitat-sim Y-up.

    Per step00_create_nav_graphs.py:241, the translation entries are:
      pose[3]  = X
      pose[7]  = -Z  (negated)
      pose[11] = Y
    """
    return float(pose[3]), float(pose[11]), float(-pose[7])


def heading_to_quaternion(dx: float, dz: float) -> List[float]:
    """Habitat-sim agent default faces -Z. To face direction (dx, dz):
    rotation around Y by theta = atan2(dx, -dz). Return [qx,qy,qz,qw].
    """
    theta = math.atan2(dx, -dz)
    return [0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)]


def load_connectivity(connectivity_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Scan all {scene}_connectivity.json into {scan: {vp_id: {position, neighbors, included}}}."""
    cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    files = [
        f for f in os.listdir(connectivity_dir)
        if f.endswith("_connectivity.json")
    ]
    print(f"Loading {len(files)} connectivity files...", file=sys.stderr)
    for fname in tqdm(files, file=sys.stderr):
        scan = fname.replace("_connectivity.json", "")
        with open(os.path.join(connectivity_dir, fname)) as f:
            data = json.load(f)
        vp_map: Dict[str, Dict[str, Any]] = {}
        # First pass: positions
        for entry in data:
            if not entry["included"]:
                continue
            x, y, z = parse_pose(entry["pose"])
            vp_map[entry["image_id"]] = {
                "position": [x, y, z],
                "neighbors": [],
            }
        # Second pass: adjacency (only between included nodes)
        for i, entry in enumerate(data):
            if not entry["included"]:
                continue
            vp_id = entry["image_id"]
            for j, conn in enumerate(entry["unobstructed"]):
                if conn and data[j]["included"]:
                    neighbor_id = data[j]["image_id"]
                    vp_map[vp_id]["neighbors"].append(neighbor_id)
        cache[scan] = vp_map
    return cache


def split_of_scene(scan: str) -> str:
    """HM3DAutoVLN convention: scene_id < 800 -> train, else val."""
    try:
        prefix = int(scan.split("-")[0])
    except (ValueError, IndexError):
        return "val"  # Conservative: unknown -> val
    return "train" if prefix < TRAIN_SCENE_MAX else "val"


def build_episode(
    item: Dict[str, Any],
    connectivity: Dict[str, Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Convert one speaker jsonl entry to a habitat episode dict. None on failure."""
    scan = item["scan"]
    path_vps: List[str] = item["path"]
    if len(path_vps) < 2:
        return None
    vp_map = connectivity.get(scan)
    if vp_map is None:
        return None

    # All viewpoint ids in path must exist in connectivity
    for vp in path_vps:
        if vp not in vp_map:
            return None

    start_vp = path_vps[0]
    goal_vp = path_vps[-1]
    start_pos = vp_map[start_vp]["position"]
    goal_pos = vp_map[goal_vp]["position"]

    # start_rotation: face from start_vp toward path[1]
    next_pos = vp_map[path_vps[1]]["position"]
    dx = next_pos[0] - start_pos[0]
    dz = next_pos[2] - start_pos[2]
    start_rot = heading_to_quaternion(dx, dz)

    reference_path = [vp_map[vp]["position"] for vp in path_vps]

    return {
        "episode_id": item["instr_id"],
        "scene_scan_id": scan,
        # scene_id is relative to scenes_dir (which already ends in /hm3d/).
        # Mesh file naming follows HM3D convention: directory is `NNNNN-ID/`
        # but the .glb inside is named `ID.basis.glb` (no numeric prefix).
        "scene_id": f"{'train' if split_of_scene(scan) == 'train' else 'val'}/{scan}/{scan.split('-', 1)[1]}.basis.glb",
        "start_position": start_pos,
        "start_rotation": start_rot,
        "goals": [{"position": goal_pos, "radius": 1.0}],
        "reference_path": reference_path,
        "reference_viewpoints": path_vps,
        "start_viewpoint_id": start_vp,
        "goal_viewpoint_id": goal_vp,
        "target_object_id": str(item.get("objid", "")),
        "target_visible_viewpoints": list(item.get("pos_vps", [])),
        "instruction": {
            "instruction_text": item["instruction"],
            "instruction_tokens": list(item["instr_encoding"]),
        },
        "trajectory_id": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speaker-jsonl", required=True,
        help="Path to ade20d_pseudo3d_depth2_epoch_94_beam0.jsonl",
    )
    parser.add_argument(
        "--connectivity-dir", required=True,
        help="Directory containing {scene}_connectivity.json files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output root; will create {train,val}/{split}.json.gz inside",
    )
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="Limit number of items (for testing); -1 = all",
    )
    args = parser.parse_args()

    connectivity = load_connectivity(args.connectivity_dir)
    print(f"Loaded {len(connectivity)} scenes from connectivity", file=sys.stderr)

    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0

    # Read jsonl manually (one JSON object per line) — no jsonlines dependency.
    items: List[Dict[str, Any]] = []
    with open(args.speaker_jsonl) as f:
        for line in tqdm(f, desc="Reading speaker jsonl", file=sys.stderr):
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if args.limit > 0:
        items = items[: args.limit]

    for item in tqdm(items, desc="Converting", file=sys.stderr):
        ep = build_episode(item, connectivity)
        if ep is None:
            skipped += 1
            continue
        split = split_of_scene(item["scan"])
        by_split[split].append(ep)

    print(
        f"Converted: train={len(by_split['train'])}, "
        f"val={len(by_split['val'])}, skipped={skipped}",
        file=sys.stderr,
    )

    for split, episodes in by_split.items():
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(split_dir, f"{split}.json.gz")
        with gzip.open(out_path, "wt") as f:
            json.dump({"episodes": episodes}, f)
        print(f"Wrote {len(episodes)} episodes -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

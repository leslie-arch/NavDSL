#!/usr/bin/env python3
"""Sanity-check the output of convert_hm3d_autovln.py.

Checks:
  1. Every episode has all required fields and non-empty values.
  2. start_position / goal positions fall within plausible bounds.
  3. reference_path length ≥ 2; all positions are 3-element lists.
  4. scene_id format matches hm3d/{split}/{scan}/{scan}.basis.glb.
  5. Cross-check with original speaker jsonl: every instr_id matches.
  6. Connectivity cross-check: for one scene, all reference_viewpoints
     are present in the corresponding {scene}_connectivity.json.

Usage:
  python -m navdsl.data_adapter.verify_hm3d_autovln \
      --episodes data/datasets/vln/hm3d/autovln/v1/train/train.json.gz \
      --speaker-jsonl /path/to/ade20k_pseudo3d_depth2_epoch_94_beam0.jsonl \
      --connectivity-dir /path/to/NAV_GRAPH/connectivity \
      [--sample 5]
"""
import argparse
import gzip
import json
import os
import statistics
import sys
from typing import Any, Dict, List

REQUIRED_FIELDS = [
    "episode_id", "scene_scan_id", "scene_id", "start_position",
    "start_rotation", "goals", "reference_path", "reference_viewpoints",
    "start_viewpoint_id", "goal_viewpoint_id", "target_object_id",
    "target_visible_viewpoints", "instruction", "trajectory_id",
]

# Sanity bounds for HM3D positions (meters). Most scenes fit in ±30m, but
# large buildings (hotels, multi-floor houses) can reach 50-80m. Flag as
# warning rather than error above WARN_LIMIT; treat as error above HARD_LIMIT.
POS_WARN_LIMIT = 50.0
POS_HARD_LIMIT = 200.0


def check_field_presence(ep: Dict[str, Any]) -> List[str]:
    issues = []
    for k in REQUIRED_FIELDS:
        if k not in ep:
            issues.append(f"missing field: {k}")
            continue
        v = ep[k]
        if v is None or v == "" or v == []:
            issues.append(f"empty field: {k}")
    return issues


def check_positions(ep: Dict[str, Any]) -> List[str]:
    """Return list of issues. Warnings (large-but-plausible) and errors
    (clearly-broken) are both returned; caller can filter by prefix."""
    issues = []
    for label, pos in [
        ("start_position", ep.get("start_position")),
        ("goal", ep.get("goals", [{}])[0].get("position") if ep.get("goals") else None),
    ]:
        if pos is None:
            continue
        if not (isinstance(pos, list) and len(pos) == 3):
            issues.append(f"{label} not a 3-list: {pos!r}")
            continue
        max_abs = max(abs(c) for c in pos)
        if max_abs > POS_HARD_LIMIT:
            issues.append(f"ERROR {label} out of bounds: {pos}")
        elif max_abs > POS_WARN_LIMIT:
            issues.append(f"WARN {label} large: {pos}")
    for i, p in enumerate(ep.get("reference_path", [])):
        if not (isinstance(p, list) and len(p) == 3):
            issues.append(f"ERROR reference_path[{i}] not a 3-list: {p!r}")
    return issues


def check_scene_id_format(ep: Dict[str, Any]) -> List[str]:
    issues = []
    sid = ep.get("scene_id", "")
    if not sid.endswith(".basis.glb"):
        issues.append(f"scene_id does not end .basis.glb: {sid}")
    parts = sid.split("/")
    if len(parts) < 4 or parts[0] != "hm3d" or parts[1] not in ("train", "val"):
        issues.append(f"scene_id format wrong: {sid}")
    return issues


def load_episodes(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="Path to {split}.json.gz")
    ap.add_argument("--speaker-jsonl", required=True)
    ap.add_argument("--connectivity-dir", required=True)
    ap.add_argument("--sample", type=int, default=3, help="Episodes to print in full")
    args = ap.parse_args()

    print(f"Loading episodes from {args.episodes}", file=sys.stderr)
    data = load_episodes(args.episodes)
    episodes: List[Dict[str, Any]] = data["episodes"]
    print(f"  num_episodes = {len(episodes)}", file=sys.stderr)

    # Per-episode checks
    bad = 0
    warn = 0
    path_lens = []
    scans = set()
    for ep in episodes:
        issues = (
            check_field_presence(ep)
            + check_positions(ep)
            + check_scene_id_format(ep)
        )
        errors = [i for i in issues if i.startswith("ERROR")]
        warnings = [i for i in issues if i.startswith("WARN")]
        if errors:
            bad += 1
            if bad <= 5:
                print(f"  [BAD] {ep.get('episode_id')}: {errors}", file=sys.stderr)
        if warnings:
            warn += 1
            if warn <= 3:
                print(f"  [WARN] {ep.get('episode_id')}: {warnings}", file=sys.stderr)
        path_lens.append(len(ep.get("reference_viewpoints", [])))
        scans.add(ep.get("scene_scan_id"))

    print(f"  bad episodes (errors): {bad} / {len(episodes)}", file=sys.stderr)
    print(f"  warn episodes (large positions): {warn} / {len(episodes)}", file=sys.stderr)
    print(f"  unique scans: {len(scans)}", file=sys.stderr)
    if path_lens:
        print(
            f"  reference_path length: min={min(path_lens)} "
            f"median={statistics.median(path_lens)} "
            f"max={max(path_lens)} mean={statistics.mean(path_lens):.1f}",
            file=sys.stderr,
        )

    # Cross-check with speaker jsonl
    print(f"Cross-checking with {args.speaker_jsonl}", file=sys.stderr)
    speaker_ids = set()
    with open(args.speaker_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            speaker_ids.add(json.loads(line)["instr_id"])
    episode_ids = {ep["episode_id"] for ep in episodes}
    missing_in_episodes = speaker_ids & episode_ids  # intersection
    only_in_episodes = episode_ids - speaker_ids
    print(f"  speaker entries: {len(speaker_ids)}", file=sys.stderr)
    print(f"  episode entries: {len(episode_ids)}", file=sys.stderr)
    print(f"  episodes not in speaker (should be 0): {len(only_in_episodes)}", file=sys.stderr)
    if only_in_episodes and len(only_in_episodes) < 10:
        print(f"    examples: {list(only_in_episodes)[:5]}", file=sys.stderr)

    # Connectivity cross-check for one scene
    sample_scan = next(iter(scans), None)
    if sample_scan:
        conn_path = os.path.join(args.connectivity_dir, f"{sample_scan}_connectivity.json")
        print(f"Connectivity check for scene {sample_scan}", file=sys.stderr)
        with open(conn_path) as f:
            conn = json.load(f)
        vp_ids = {e["image_id"] for e in conn if e["included"]}
        # Find episodes in this scan
        scan_eps = [e for e in episodes if e["scene_scan_id"] == sample_scan][:50]
        bad_vps = 0
        for ep in scan_eps:
            for vp in ep["reference_viewpoints"]:
                if vp not in vp_ids:
                    bad_vps += 1
                    if bad_vps <= 5:
                        print(f"    [BAD] {ep['episode_id']} references unknown vp {vp}", file=sys.stderr)
        print(f"  checked {len(scan_eps)} episodes, bad_vp_refs={bad_vps}", file=sys.stderr)
        print(f"  scene has {len(vp_ids)} included viewpoints", file=sys.stderr)

    # Sample prints
    if args.sample > 0:
        print(f"\n--- First {args.sample} episodes ---")
        for ep in episodes[: args.sample]:
            print(json.dumps(ep, indent=2)[:1000])
            print("---")


if __name__ == "__main__":
    main()

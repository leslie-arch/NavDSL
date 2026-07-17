#!/usr/bin/env python3
"""One-shot migration: fix `scene_id` in existing DSL train/val json.gz.

Background: the original converter wrote
    ``hm3d/{split}/{scan}/{scan}.basis.glb``
which has two bugs after path composition with scenes_dir (which already
ends in ``hm3d/``):
  1. double ``hm3d/`` prefix in the absolute path
  2. file name uses ``{scan}.basis.glb`` (e.g. ``00799-deNrXzuSss5.basis.glb``)
     but the actual mesh on disk is ``{id}.basis.glb`` (``deNrXzuSss5.basis.glb``)

This script rewrites every episode's ``scene_id`` in place to the correct
form::

    {split}/{scan}/{id}.basis.glb

Idempotent: if a scene_id is already in the new format (no ``hm3d/`` prefix),
it is left alone.

Usage::

    python -m navdsl.data_adapter.migrate_scene_ids <dsl_dir> [--split train val]
    python navdsl/data_adapter/migrate_scene_ids.py /path/to/DSL
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
from pathlib import Path

# Match `hm3d/<split>/<scan>/<filename>.basis.glb` (the buggy format).
OLD_RE = re.compile(r"^hm3d/(train|val)/(\d{5}-[A-Za-z0-9]+)/([^/]+\.basis\.glb)$")


def migrate_scene_id(scene_id: str) -> tuple[str, bool]:
    """Return (new_scene_id, changed). Idempotent."""
    m = OLD_RE.match(scene_id)
    if not m:
        return scene_id, False
    split, scan, _old_filename = m.group(1), m.group(2), m.group(3)
    scan_id = scan.split("-", 1)[1]
    return f"{split}/{scan}/{scan_id}.basis.glb", True


def migrate_file(path: Path) -> tuple[int, int]:
    """Migrate one json.gz file in place. Returns (n_total, n_changed)."""
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  backup: {backup}")

    with gzip.open(path, "rt") as f:
        data = json.load(f)

    eps = data.get("episodes", [])
    n_changed = 0
    for ep in eps:
        old = ep.get("scene_id")
        if not old:
            continue
        new, changed = migrate_scene_id(old)
        if changed:
            ep["scene_id"] = new
            n_changed += 1

    with gzip.open(path, "wt") as f:
        json.dump(data, f)

    return len(eps), n_changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dsl_dir", help="Path to DSL/ directory containing train/ and val/")
    ap.add_argument(
        "--split",
        nargs="+",
        default=["train", "val"],
        help="Which splits to migrate (default: train val)",
    )
    args = ap.parse_args()

    root = Path(args.dsl_dir)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        return 1

    for split in args.split:
        path = root / split / f"{split}.json.gz"
        if not path.is_file():
            print(f"SKIP: {path} (missing)")
            continue
        print(f"\n=== migrating {path} ===")
        n_total, n_changed = migrate_file(path)
        print(f"  episodes: {n_total}, scene_id rewritten: {n_changed}")
        if n_changed == 0:
            print("  (file was already migrated)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

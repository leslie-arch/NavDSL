#!/usr/bin/env python3
"""Standalone smoke test for HM3DAutoVLNDatasetV1.

Run on the remote host (where habitat-lab + lmdb + networkx are installed):

  python -m navdsl.data_adapter.test_hm3d_autovln_dataset \
      --episodes ~/sshfs_root/.../DSL/train/train.json.gz \
      --scenes-dir ~/sshfs_root/.../versioned_data/hm3d-0.2/hm3d/ \
      --connectivity-dir ~/sshfs_root/.../NAV_GRAPH/connectivity \
      --view-lmdb ~/sshfs_root/.../features/view_timm_imagenet_vitb16 \
      --obj-lmdb ~/sshfs_root/.../features/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16 \
      --rel-angles ~/sshfs_root/.../annotations/scanvp_candview_relangles.json
"""
import argparse
import sys
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--scenes-dir", required=True)
    ap.add_argument("--connectivity-dir", required=True)
    ap.add_argument("--view-lmdb", required=True)
    ap.add_argument("--obj-lmdb", required=True)
    ap.add_argument("--rel-angles", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    # Build a minimal DictConfig-like object the dataset accepts.
    config = SimpleNamespace(
        data_path=args.episodes,        # already includes {split} substitution
        split=args.split,
        scenes_dir=args.scenes_dir,
        content_scenes=["*"],
        nav_graph_dir=args.connectivity_dir,
        view_feature_lmdb=args.view_lmdb,
        object_feature_lmdb=args.obj_lmdb,
        rel_angles_path=args.rel_angles,
    )

    # data_path in the yaml uses "{split}" placeholder; the test takes a concrete
    # path. If the user passed a path containing {split}, format it; else use as-is.
    if "{split}" in config.data_path:
        config.data_path = config.data_path.format(split=config.split)

    print(f"Loading dataset from {config.data_path}", file=sys.stderr)
    from navdsl.data_adapter.hm3d_autovln_dataset import HM3DAutoVLNDatasetV1

    ds = HM3DAutoVLNDatasetV1(config)
    print(f"  num_episodes = {len(ds.episodes)}", file=sys.stderr)
    assert len(ds.episodes) > 0, "no episodes loaded"

    ep = ds.episodes[0]
    print(f"  first episode: id={ep.episode_id} scan={ep.scene_scan_id}", file=sys.stderr)
    print(f"    start_vp={ep.start_viewpoint_id} goal_vp={ep.goal_viewpoint_id}", file=sys.stderr)
    print(f"    reference_viewpoints={ep.reference_viewpoints}", file=sys.stderr)
    print(f"    target_object_id={ep.target_object_id}", file=sys.stderr)
    print(f"    target_visible_viewpoints={ep.target_visible_viewpoints}", file=sys.stderr)
    print(f"    instruction_text={ep.instruction.instruction_text[:80]!r}", file=sys.stderr)

    # --- nav graph ---
    scan = ep.scene_scan_id
    G = ds._build_nav_graph(scan)
    print(f"\n  nav graph for {scan}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    start_vp = ep.start_viewpoint_id
    pos = ds.get_viewpoint_position(scan, start_vp)
    print(f"  position of {start_vp}: {pos}", file=sys.stderr)

    cands = ds.get_candidates(scan, start_vp)
    print(f"  candidates from {start_vp}: {cands}", file=sys.stderr)
    assert len(cands) > 0, "no candidates at start viewpoint (nav graph broken?)"

    # --- view features ---
    view_ft = ds.get_view_features(scan, start_vp)
    print(f"\n  view_features({scan}, {start_vp}): shape={view_ft.shape} dtype={view_ft.dtype}", file=sys.stderr)
    assert view_ft.shape == (36, 768), f"unexpected view shape {view_ft.shape}"
    print(f"    sample [0,:5] = {view_ft[0, :5]}", file=sys.stderr)
    print(f"    norm = {float((view_ft**2).sum()**0.5):.3f}", file=sys.stderr)

    # --- object features ---
    obj = ds.get_object_features(scan, start_vp)
    print(f"\n  object_features({scan}, {start_vp}):", file=sys.stderr)
    print(f"    obj_ids: {obj['obj_ids'][:5]}{'...' if len(obj['obj_ids']) > 5 else ''}", file=sys.stderr)
    print(f"    obj_names sample: {obj['obj_names'][:3]}", file=sys.stderr)
    print(f"    fts shape: {obj['fts'].shape}", file=sys.stderr)
    assert obj['fts'].shape[1] == 768, f"unexpected obj dim {obj['fts'].shape[1]}"

    # --- rel angles ---
    if cands:
        first_cand = cands[0]
        rel = ds.get_rel_angle(scan, start_vp, first_cand)
        print(f"\n  rel_angle({scan}, {start_vp} -> {first_cand}): {rel}", file=sys.stderr)

    # --- walk a reference path ---
    print(f"\n  Walking reference path of length {len(ep.reference_viewpoints)}:", file=sys.stderr)
    for i, vp in enumerate(ep.reference_viewpoints):
        cands = ds.get_candidates(scan, vp)
        print(f"    step {i}: vp={vp}  cands={cands}", file=sys.stderr)

    print("\nALL CHECKS PASSED", file=sys.stderr)


if __name__ == "__main__":
    main()

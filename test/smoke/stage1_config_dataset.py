#!/usr/bin/env python3
"""Phase 7 Stage 1: Hydra config parse + dataset instantiation.

Verifies:
  - Hydra can compose the experiment config
  - YAML overrides via CLI work
  - HM3DAutoVLNDatasetV1 loads episodes
  - Per-scan LMDB read path works

No GPU needed. ~10 seconds.

Run:
    python test/smoke/stage1_config_dataset.py
"""
import os
import sys
from pathlib import Path

# Make NavDSL importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Step 1: register all components via side-effect imports
import navdsl.config  # noqa: F401  registers HM3DAutoVLNDatasetConfig schema
import navdsl.data_adapter.hm3d_autovln_dataset  # noqa: F401
import navdsl.tasks.vln_graph_nav  # noqa: F401
import navdsl.sensor.viewpoint_feature_sensor  # noqa: F401
import navdsl.sensor.object_feature_sensor  # noqa: F401
import navdsl.sensor.candidate_viewpoints_sensor  # noqa: F401
import navdsl.sensor.graph_nodes_sensor  # noqa: F401
import navdsl.measurements.viewpoint_success  # noqa: F401

# Step 2: set up Hydra with habitat search path plugin
from hydra import initialize_config_dir, compose
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)
register_hydra_plugin(HabitatBaselinesConfigPlugin)

# Step 3: read paths from env (set by _paths.sh or directly)
BASE = os.environ.get(
    "NAVDSL_DATA_BASE", "/sata/sdb7/dataset/habitat-data"
)
SCENES = os.environ.get("NAVDSL_SCENES", f"{BASE}/versioned_data/hm3d-0.2/hm3d")
EPISODES = os.environ.get(
    "NAVDSL_EPISODES", f"{BASE}/datasets/vln/hm3d/autovln/v1.0/DSL"
)
NAV_GRAPH = os.environ.get(
    "NAVDSL_NAV_GRAPH",
    f"{BASE}/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/connectivity",
)
PER_SCAN = os.environ.get(
    "NAVDSL_PER_SCAN",
    f"{BASE}/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/features/per_scan",
)
REL_ANGLES = os.environ.get(
    "NAVDSL_REL_ANGLES",
    f"{BASE}/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/annotations/scanvp_candview_relangles.json",
)
DUET_CKPT = os.environ.get(
    "NAVDSL_DUET_CKPT",
    f"{BASE}/datasets/vln/hm3d/autovln/v1.0/REVERIE/expr_duet/pretrain_hm3d_v1/"
    "pseudo3d-depth2-cmt-timm.vitb16-mlm.sap.og-init.lxmert-bsz.64/ckpts/"
    "model_step_35000.pt",
)
SMOKE_SCENE = os.environ.get("NAVDSL_SMOKE_SCENE", "00000-kfPV7w3FaU5")


print("=" * 60)
print("STAGE 1: Config + Dataset")
print("=" * 60)

print("\n--- 1.1 Hydra config composition ---")
config_dir = str(REPO_ROOT / "config")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(
        config_name="experiments/hm3d_autovln_graph_nav",
        overrides=[
            # `{split}` is Python str.format() syntax, resolved by
            # HM3DAutoVLNDatasetV1 at load time. Quote the value so Hydra's
            # override grammar doesn't choke on the braces; DO NOT use
            # `${split}` — OmegaConf interprets that as an interpolation ref.
            f'habitat.dataset.data_path="{EPISODES}/{{split}}/{{split}}.json.gz"',
            f'habitat.dataset.scenes_dir={SCENES}/',
            f'habitat.simulator.scene_dataset={SCENES}/hm3d_basis.scene_dataset_config.json',
            f'habitat.dataset.nav_graph_dir={NAV_GRAPH}',
            f'habitat.dataset.feature_per_scan_dir={PER_SCAN}',
            f'habitat.dataset.rel_angles_path={REL_ANGLES}',
            f'habitat.dataset.content_scenes=["{SMOKE_SCENE}"]',
            'habitat.dataset.split=train',
            f'habitat_baselines.il.duet.checkpoint={DUET_CKPT}',
        ],
    )

print(f"  trainer_name: {cfg.habitat_baselines.trainer_name}")
print(f"  task type:    {cfg.habitat.task.type}")
print(f"  dataset type: {cfg.habitat.dataset.type}")
print(f"  dataset split: {cfg.habitat.dataset.split}")
print(f"  actions:      {list(cfg.habitat.task.actions.keys())}")
print(f"  lab_sensors:  {list(cfg.habitat.task.lab_sensors.keys())}")
print(f"  measurements: {list(cfg.habitat.task.measurements.keys())}")
assert cfg.habitat_baselines.trainer_name == "duet_il"
assert cfg.habitat.task.type == "VLNGraphNav-v0"
assert cfg.habitat.dataset.type == "HM3DAutoVLN-v1"

print("\n--- 1.2 Patch config (hydra → habitat DictConfig) ---")
from habitat.config.default import patch_config

cfg = patch_config(cfg)

print("\n--- 1.3 Instantiate dataset ---")
from navdsl.data_adapter.hm3d_autovln_dataset import HM3DAutoVLNDatasetV1

ds = HM3DAutoVLNDatasetV1(cfg.habitat.dataset)
n_eps = len(ds.episodes)
print(f"  num_episodes: {n_eps}")
assert n_eps > 0, f"No episodes loaded for scene {SMOKE_SCENE}"

ep = ds.episodes[0]
print(f"\n  first episode:")
print(f"    episode_id:           {ep.episode_id}")
print(f"    scene_scan_id:        {ep.scene_scan_id}")
print(f"    start_viewpoint_id:   {ep.start_viewpoint_id}")
print(f"    goal_viewpoint_id:    {ep.goal_viewpoint_id}")
print(f"    reference_viewpoints: {ep.reference_viewpoints}")
print(f"    target_object_id:     {ep.target_object_id}")
print(f"    target_visible_vps:   {ep.target_visible_viewpoints}")
print(f"    instruction[:60]:     {ep.instruction.instruction_text[:60]!r}")

print("\n--- 1.4 Per-scan LMDB reads ---")
import numpy as np

scan = ep.scene_scan_id
vp = ep.start_viewpoint_id
view = ds.get_view_features(scan, vp)
obj = ds.get_object_features(scan, vp)
cands = ds.get_candidates(scan, vp)
pos = ds.get_viewpoint_position(scan, vp)

print(f"  get_view_features({scan}, {vp}):")
print(f"    shape: {view.shape}")
print(f"    dtype: {view.dtype}")
print(f"    norm:  {float((view**2).sum() ** 0.5):.3f}")
assert view.shape == (36, 768), f"unexpected view shape {view.shape}"

print(f"  get_object_features({scan}, {vp}):")
print(f"    obj count: {len(obj['obj_ids'])}")
print(f"    fts.shape: {obj['fts'].shape}")
print(f"    obj_names[:3]: {obj['obj_names'][:3]}")

print(f"  get_candidates({scan}, {vp}):")
print(f"    neighbors: {cands}")
assert len(cands) > 0, "no candidates at start viewpoint"

print(f"  get_viewpoint_position({scan}, {vp}): {pos}")

print("\n" + "=" * 60)
print("STAGE 1 PASSED — config + dataset working")
print("=" * 60)

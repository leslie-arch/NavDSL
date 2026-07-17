#!/usr/bin/env python3
"""Phase 7 Stage 3: Single rollout + backward pass.

Verifies:
  - EpisodeBatch produces valid obs dicts (all fields populated)
  - VLNBert forward pass works in all 3 modes (language / panorama / navigation)
  - Loss is finite
  - Backward populates gradients

This is the critical integration test — if Stage 3 passes, the data pipeline
and model pipeline are correctly wired.

Requires GPU. ~1 minute.

Run:
    python test/smoke/stage3_rollout_backward.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import navdsl.data_adapter.hm3d_autovln_dataset  # noqa: F401
import navdsl.tasks.vln_graph_nav  # noqa: F401
import navdsl.sensor.viewpoint_feature_sensor  # noqa: F401
import navdsl.sensor.object_feature_sensor  # noqa: F401
import navdsl.sensor.candidate_viewpoints_sensor  # noqa: F401
import navdsl.sensor.graph_nodes_sensor  # noqa: F401
import navdsl.measurements.viewpoint_success  # noqa: F401

import torch
from omegaconf import OmegaConf
from types import SimpleNamespace

from navdsl.data_adapter.hm3d_autovln_dataset import HM3DAutoVLNDatasetV1
from navdsl.utils.duet_trainer import EpisodeBatch, _ArgsNamespace
from navdsl.policy.duet.agent_obj import GMapObjectNavAgent

BASE = os.environ.get("NAVDSL_DATA_BASE", "/sata/sdb7/dataset/habitat-data")
EPISODES = os.environ.get(
    "NAVDSL_EPISODES", f"{BASE}/datasets/vln/hm3d/autovln/v1.0/DSL"
)
SCENES = os.environ.get("NAVDSL_SCENES", f"{BASE}/versioned_data/hm3d-0.2/hm3d")
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
BATCH_SIZE = int(os.environ.get("NAVDSL_SMOKE_BATCH", "2"))

print("=" * 60)
print("STAGE 3: Rollout + Backward")
print("=" * 60)

print(f"\nCUDA: {torch.cuda.is_available()}")
print(f"smoke scene: {SMOKE_SCENE}")
print(f"batch size: {BATCH_SIZE}")

print("\n--- 3.1 Build dataset + filter to smoke scene ---")
ds_cfg = SimpleNamespace(
    data_path=f"{EPISODES}/train/train.json.gz",
    split="train",
    scenes_dir=f"{SCENES}/",
    content_scenes=["*"],
    nav_graph_dir=NAV_GRAPH,
    feature_per_scan_dir=PER_SCAN,
    rel_angles_path=REL_ANGLES,
)
ds = HM3DAutoVLNDatasetV1(ds_cfg)
print(f"  total episodes: {len(ds.episodes)}")
ds.episodes = [ep for ep in ds.episodes if ep.scene_scan_id == SMOKE_SCENE][:BATCH_SIZE]
print(f"  filtered to {len(ds.episodes)} episodes for smoke test")
assert len(ds.episodes) >= 1, f"no episodes for scene {SMOKE_SCENE}"

print("\n--- 3.2 Build EpisodeBatch + reset ---")
batch = EpisodeBatch(
    episodes=ds.episodes,
    dataset=ds,
    batch_size=len(ds.episodes),
    angle_feat_size=4,
    image_feat_size=768,
    obj_feat_size=768,
)
obs = batch.reset()
print(f"  batch size: {len(obs)}")

print(f"\n--- 3.3 Verify obs dict structure (first episode) ---")
ob0 = obs[0]
expected_keys = [
    "instr_id", "scan", "viewpoint", "heading", "elevation", "position",
    "feature", "candidate", "obj_img_fts", "obj_ang_fts", "obj_box_fts",
    "obj_ids", "instr_encoding", "gt_path", "gt_end_vps", "gt_obj_id",
]
missing = [k for k in expected_keys if k not in ob0]
if missing:
    print(f"  FAIL: obs missing keys: {missing}")
    sys.exit(1)
print(f"  all {len(expected_keys)} expected keys present")

print(f"\n  obs[0] contents:")
print(f"    instr_id:        {ob0['instr_id']}")
print(f"    scan:            {ob0['scan']}")
print(f"    viewpoint:       {ob0['viewpoint']}")
print(f"    feature.shape:   {ob0['feature'].shape}  (expected (36, 772))")
print(f"    n_candidates:    {len(ob0['candidate'])}")
print(f"    n_objects:       {len(ob0['obj_ids'])}")
print(f"    gt_path len:     {len(ob0['gt_path'])}")
print(f"    gt_end_vps:      {ob0['gt_end_vps']}")
assert ob0["feature"].shape == (36, 772), f"feature shape wrong: {ob0['feature'].shape}"
assert len(ob0["candidate"]) > 0, "no candidates"
assert len(ob0["gt_path"]) >= 2, "gt_path too short"

print("\n--- 3.4 Build agent ---")
model_cfg = OmegaConf.create({
    "checkpoint": DUET_CKPT,
    "batch_size": len(ds.episodes),
    "tokenizer": "bert",
    "image_feat_size": 768, "obj_feat_size": 768, "angle_feat_size": 4,
    "num_l_layers": 9, "num_pano_layers": 2, "num_x_layers": 4,
    "graph_sprels": True, "fusion": "dynamic",
    "fix_lang_embedding": False, "fix_pano_embedding": False, "fix_local_branch": False,
    "feat_dropout": 0.0, "dropout": 0.1,
    "optim": "adamW", "lr": 1e-5, "world_size": 1,
    "ignoreid": -1, "max_action_len": 20,
    "enc_full_graph": True, "loss_nav_3": True,
    "detailed_output": False, "expl_max_ratio": 0.0,
    "train_alg": "imitation", "ml_weight": 0.0,
    "dagger_sample": "sample", "aug": None,
})
args = _ArgsNamespace(model_cfg)
args.bert_ckpt_file = DUET_CKPT
agent = GMapObjectNavAgent(args, batch, rank=0)
print(f"  agent type: {type(agent).__name__}")

print("\n--- 3.5 Single rollout with teacher forcing ---")
agent.vln_bert_optimizer.zero_grad()
agent.critic_optimizer.zero_grad()
agent.loss = 0
agent.feedback = "teacher"
agent.rollout(train_ml=1.0, train_rl=False, reset=True)

print(f"\n  loss type:  {type(agent.loss).__name__}")
if hasattr(agent.loss, "item"):
    loss_val = agent.loss.item()
    print(f"  loss value: {loss_val:.4f}")
    assert loss_val == loss_val, "loss is NaN"  # NaN check
    assert abs(loss_val) < 1e6, f"loss explosion: {loss_val}"

il_losses = agent.logs.get("IL_loss", [])
og_losses = agent.logs.get("OG_loss", [])
print(f"  IL_loss log: {il_losses[-1] if il_losses else 'empty':.4f}")
print(f"  OG_loss log: {og_losses[-1] if og_losses else 'empty':.4f}")

print("\n--- 3.6 Backward pass ---")
agent.loss.backward()
grad_norm = sum(
    p.grad.norm().item() ** 2
    for p in agent.vln_bert.parameters()
    if p.grad is not None
) ** 0.5
n_params_with_grad = sum(1 for p in agent.vln_bert.parameters() if p.grad is not None)
n_params_total = sum(1 for p in agent.vln_bert.parameters())
print(f"  grad norm:           {grad_norm:.4f}")
print(f"  params with grad:    {n_params_with_grad}/{n_params_total}")
assert n_params_with_grad > 0, "no gradients — backward didn't flow"
assert grad_norm == grad_norm, "grad norm is NaN"

print("\n" + "=" * 60)
print("STAGE 3 PASSED — full pipeline (data + model + loss + grad) works")
print("=" * 60)
print("\nNext: Stage 4 runs the actual navdsl.run entry point.")

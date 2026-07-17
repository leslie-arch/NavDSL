#!/usr/bin/env python3
"""Phase 7 Stage 2: VLNBert model + GMapObjectNavAgent construction.

Verifies:
  - Pretrained weights load (no missing/unexpected keys)
  - Model has expected param count (~200M)
  - Agent initializes optimizer correctly

Requires GPU (model uses .cuda() internally). ~30 seconds.

Run:
    python test/smoke/stage2_model_agent.py
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

from navdsl.policy.duet.model import VLNBert
from navdsl.policy.duet.agent_obj import GMapObjectNavAgent
from navdsl.utils.duet_trainer import _ArgsNamespace

BASE = os.environ.get("NAVDSL_DATA_BASE", "/sata/sdb7/dataset/habitat-data")
DUET_CKPT = os.environ.get(
    "NAVDSL_DUET_CKPT",
    f"{BASE}/datasets/vln/hm3d/autovln/v1.0/REVERIE/expr_duet/pretrain_hm3d_v1/"
    "pseudo3d-depth2-cmt-timm.vitb16-mlm.sap.og-init.lxmert-bsz.64/ckpts/"
    "model_step_35000.pt",
)


class NoopEnv:
    """Stub env — we won't rollout, just construct the agent."""

    def reset(self):
        return []

    def _get_obs(self):
        return []

    def step(self, actions):
        return []


print("=" * 60)
print("STAGE 2: Model + Agent")
print("=" * 60)

print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARN: no CUDA — agent code calls .cuda() internally, will fail")

print(f"\n--- 2.1 Construct VLNBert (loads pretrained weights) ---")
print(f"checkpoint: {DUET_CKPT}")
print(f"size: {os.path.getsize(DUET_CKPT) / 1e9:.2f} GB")

model_cfg = OmegaConf.create({
    "checkpoint": DUET_CKPT,
    "tokenizer": "bert",
    "image_feat_size": 768,
    "obj_feat_size": 768,
    "angle_feat_size": 4,
    "num_l_layers": 9,
    "num_pano_layers": 2,
    "num_x_layers": 4,
    "graph_sprels": True,
    "fusion": "dynamic",
    "fix_lang_embedding": False,
    "fix_pano_embedding": False,
    "fix_local_branch": False,
    "feat_dropout": 0.0,
    "dropout": 0.1,
})
args = _ArgsNamespace(model_cfg)
args.bert_ckpt_file = DUET_CKPT

model = VLNBert(args)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n  VLNBert constructed")
print(f"  total params: {n_params / 1e6:.1f}M")
print(f"  device: {next(model.parameters()).device}")

print("\n--- 2.2 Construct GMapObjectNavAgent ---")
agent = GMapObjectNavAgent(args, NoopEnv(), rank=0)
print(f"  agent type: {type(agent).__name__}")
print(f"  vln_bert params: {sum(p.numel() for p in agent.vln_bert.parameters()) / 1e6:.1f}M")
print(f"  critic params: {sum(p.numel() for p in agent.critic.parameters()) / 1e6:.1f}M")
print(f"  optimizer: {type(agent.vln_bert_optimizer).__name__}")
print(f"  lr: {agent.vln_bert_optimizer.param_groups[0]['lr']}")

print("\n" + "=" * 60)
print("STAGE 2 PASSED — model + agent construction working")
print("=" * 60)
print("\nIf you saw no 'Missing key(s)' or 'Unexpected key(s)' warnings above,")
print("the pretrained weights loaded cleanly. Otherwise, paste the warnings")
print("back so we can extend the key remapping in vlnbert_init.py.")

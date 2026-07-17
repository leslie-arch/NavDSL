#!/usr/bin/env bash
# Common path config for Phase 7 smoke tests.
# Source this from other stage scripts: `source $(dirname $0)/_paths.sh`

# Default to local paths; override with NAVDSL_DATA_BASE env var if running remote.
# For remote (sshfs): export NAVDSL_DATA_BASE=/home/hjk/sshfs_root/dataset/habitat-data
export NAVDSL_DATA_BASE="${NAVDSL_DATA_BASE:-/sata/sdb7/dataset/habitat-data}"

# Derived paths
export NAVDSL_EPISODES="$NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/DSL"
export NAVDSL_SCENES="$NAVDSL_DATA_BASE/scene_datasets/hm3d_v0.2"
export NAVDSL_SCENE_DATASET_CONFIG="$NAVDSL_SCENES/hm3d_basis.scene_dataset_config.json"
export NAVDSL_NAV_GRAPH="$NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/connectivity"
export NAVDSL_PER_SCAN="$NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/features/per_scan"
export NAVDSL_REL_ANGLES="$NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/annotations/scanvp_candview_relangles.json"
export NAVDSL_DUET_CKPT="$NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/REVERIE/expr_duet/pretrain_hm3d_v1/pseudo3d-depth2-cmt-timm.vitb16-mlm.sap.og-init.lxmert-bsz.64/ckpts/model_step_35000.pt"

# Scene to use for smoke tests
export NAVDSL_SMOKE_SCENE="${NAVDSL_SMOKE_SCENE:-00000-kfPV7w3FaU5}"

# Python interpreter to use (must have habitat-lab + habitat-baselines installed).
# Override with NAVDSL_PYTHON if you want a different one.
export NAVDSL_PYTHON="${NAVDSL_PYTHON:-/sata/sda3/home/anaconda3/envs/habitat/bin/python}"

echo "[paths] NAVDSL_DATA_BASE=$NAVDSL_DATA_BASE"
echo "[paths] NAVDSL_SMOKE_SCENE=$NAVDSL_SMOKE_SCENE"
echo "[paths] NAVDSL_PYTHON=$NAVDSL_PYTHON"

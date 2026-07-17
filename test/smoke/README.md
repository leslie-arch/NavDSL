# Phase 7 Smoke Test

End-to-end verification of the NavDSL Route B pipeline (DUET BC fine-tune on
HM3D-AutoVLN data). 4 stages, each tests one layer:

| Stage | Tests | GPU | Time |
|---|---|---|---|
| 0 | Environment, paths, GPU | No | ~30s |
| 1 | Hydra config, dataset, per-scan LMDB reads | No | ~10s |
| 2 | VLNBert model, pretrained weights load | Yes | ~30s |
| 3 | EpisodeBatch obs format, full rollout, backward | Yes | ~1min |
| 4 | `navdsl.run` entry point, 1 epoch, checkpoint save | Yes | ~5min |

## Quick start

```bash
cd /path/to/NavDSL

# Local (default paths point to /sata/sdb7/...):
bash test/smoke/run_all.sh

# Remote (sshfs paths):
NAVDSL_DATA_BASE=/home/hjk/sshfs_root/dataset/habitat-data bash test/smoke/run_all.sh
```

Each stage script also runs standalone — see below.

## Stage scripts

### `stage0_preflight.sh`

Checks environment + data paths. No GPU needed.

```bash
bash test/smoke/stage0_preflight.sh
```

**On failure**: usually a missing path. Verify `NAVDSL_DATA_BASE` points to
your data root. If per-scan LMDB is missing, run:

```bash
python -m navdsl.data_adapter.split_lmdb_per_scan \
    --source-view-lmdb $NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/features/view_timm_imagenet_vitb16 \
    --source-obj-lmdb $NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/features/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16 \
    --output-dir $NAVDSL_DATA_BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/features/per_scan
```

(~2 hours, one-time.)

### `stage1_config_dataset.py`

Verifies Hydra config composition + dataset loading + per-scan LMDB reads.

```bash
python test/smoke/stage1_config_dataset.py
```

**On failure**: usually a yaml field or path override issue. Check the
override paths match your filesystem.

### `stage2_model_agent.py`

Constructs VLNBert with pretrained `model_step_35000.pt`. Needs GPU.

```bash
python test/smoke/stage2_model_agent.py
```

**On failure**: paste any `Missing key(s) in state_dict` or
`Unexpected key(s) in state_dict` warnings back. These indicate the key
remapping in `navdsl/policy/duet/vlnbert_init.py` is missing a pattern.

### `stage3_rollout_backward.py`

**Critical integration test**. Verifies EpisodeBatch produces valid obs dicts
in the format GMapObjectNavAgent expects, runs one full rollout with teacher
forcing, checks loss is finite and gradients flow.

```bash
python test/smoke/stage3_rollout_backward.py
```

**On failure**: the traceback is usually informative. Common issues:
- `KeyError: 'X'` — obs dict missing a field → fix EpisodeBatch._build_single_obs
- `RuntimeError: shape mismatch` — feature dimensions don't match agent expectations
- `loss = nan` — empty obj_ids list causing div-by-zero in obj feature assembly

### `stage4_end_to_end.sh`

Runs the actual `python -m navdsl.run` entry point with 1 epoch on 1 scene.

```bash
bash test/smoke/stage4_end_to_end.sh
```

**On failure**: check `/tmp/duet_smoke.log` for the full traceback.

## Environment variables

All scripts read these env vars (with sensible defaults for local runs):

| Var | Default | Override for |
|---|---|---|
| `NAVDSL_DATA_BASE` | `/sata/sdb7/dataset/habitat-data` | Remote: `/home/hjk/sshfs_root/dataset/habitat-data` |
| `NAVDSL_SMOKE_SCENE` | `00000-kfPV7w3FaU5` | Pick a different test scene |
| `NAVDSL_SMOKE_BATCH` | `2` | Reduce to 1 if OOM |

Derived paths (`NAVDSL_EPISODES`, `NAVDSL_SCENES`, etc.) — see
`_paths.sh`. Override individually if your layout differs.

## What success looks like

After all 4 stages pass:
- `data/duet_smoke/ckpts/epoch_1.pt` exists (~1 GB)
- `/tmp/duet_smoke.log` contains `[duet_il] training complete`
- No traceback in log
- IL_loss is a finite number (typically 1-5)

## Next steps after smoke test passes

1. Run full training (multi-scene, multi-epoch):
   ```bash
   python -m navdsl.run \
       --config-name=experiments/hm3d_autovln_graph_nav \
       habitat_baselines.il.duet.max_epochs=100 \
       habitat_baselines.il.duet.batch_size=8
   ```

2. Evaluate:
   ```bash
   python -m navdsl.run \
       --config-name=experiments/hm3d_autovln_graph_nav \
       habitat_baselines.evaluate=True \
       habitat_baselines.eval_ckpt_path_dir=data/duet_hm3d_autovln/ckpts
   ```

3. Monitor with TensorBoard:
   ```bash
   tensorboard --logdir data/duet_hm3d_autovln/tb
   ```

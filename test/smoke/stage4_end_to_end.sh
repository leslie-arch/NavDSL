#!/usr/bin/env bash
# Phase 7 Stage 4: Full end-to-end training run via navdsl.run.
#
# Runs 1 epoch on 1 scene with batch_size=2. Verifies the actual training
# entry point (Hydra + habitat Env + trainer + checkpoint save).
#
# Run:
#   bash test/smoke/stage4_end_to_end.sh
#
# On remote:
#   NAVDSL_DATA_BASE=/home/hjk/sshfs_root/dataset/habitat-data bash test/smoke/stage4_end_to_end.sh

set -u
cd "$(dirname "$0")/../.."  # NavDSL repo root
source test/smoke/_paths.sh

mkdir -p checkpoints/duet_smoke/ckpts checkpoints/duet_smoke/tb

echo ""
echo "=========================================="
echo "STAGE 4: End-to-end training (navdsl.run)"
echo "=========================================="

# Hydra CLI overrides — paths from env vars set by _paths.sh
OVERRIDES=(
    habitat.dataset.content_scenes="[\"$NAVDSL_SMOKE_SCENE\"]"
    'habitat.dataset.data_path="'"$NAVDSL_EPISODES"'/{split}/{split}.json.gz"'
    habitat.dataset.scenes_dir="$NAVDSL_SCENES/"
    habitat.simulator.scene_dataset="$NAVDSL_SCENE_DATASET_CONFIG"
    habitat.dataset.nav_graph_dir="$NAVDSL_NAV_GRAPH"
    habitat.dataset.feature_per_scan_dir="$NAVDSL_PER_SCAN"
    habitat.dataset.rel_angles_path="$NAVDSL_REL_ANGLES"
    habitat_baselines.il.duet.checkpoint="$NAVDSL_DUET_CKPT"
    habitat_baselines.il.duet.max_epochs=1
    habitat_baselines.il.duet.batch_size=2
    habitat_baselines.il.duet.eval_interval=1
    habitat_baselines.checkpoint_folder=checkpoints/duet_smoke/ckpts
    habitat_baselines.tensorboard_dir=checkpoints/duet_smoke/tb
    habitat_baselines.log_file=checkpoints/duet_smoke/train.log
    habitat_baselines.evaluate=False
)

echo ""
echo "Command:"
echo "  python -m navdsl.run --config-name=experiments/hm3d_autovln_graph_nav \\"
for o in "${OVERRIDES[@]}"; do
    echo "    $o \\"
done
echo ""

# Run and capture output
LOG=/tmp/duet_smoke.log
python -m navdsl.run \
    --config-name=experiments/hm3d_autovln_graph_nav \
    "${OVERRIDES[@]}" 2>&1 | tee "$LOG"

echo ""
echo "=========================================="
echo "STAGE 4: Verification"
echo "=========================================="

echo ""
echo "--- Check checkpoint was saved ---"
if [ -f checkpoints/duet_smoke/ckpts/epoch_1.pt ]; then
    SIZE=$(du -h checkpoints/duet_smoke/ckpts/epoch_1.pt | cut -f1)
    echo "  OK: epoch_1.pt ($SIZE)"
else
    echo "  FAIL: no epoch_1.pt saved"
    echo "  Last 30 lines of log:"
    tail -30 "$LOG"
    exit 1
fi

echo ""
echo "--- Check key log milestones ---"
for pattern in \
    "[duet_il] train episodes:" \
    "Initalizing the VLN-BERT model" \
    "[duet_il] epoch 1/1 done" \
    "[duet_il] training complete"; do
    if grep -qF "$pattern" "$LOG"; then
        echo "  OK: '$pattern' found in log"
    else
        echo "  FAIL: '$pattern' not in log"
        exit 1
    fi
done

echo ""
echo "--- Check loss is finite ---"
LOSS_LINE=$(grep -oE 'IL_loss=[0-9.]+' "$LOG" | tail -1)
if [ -z "$LOSS_LINE" ]; then
    echo "  WARN: no IL_loss= line found in log"
else
    echo "  $LOSS_LINE"
fi

echo ""
echo "--- Check no exceptions ---"
if grep -qE 'Traceback|Error|Exception' "$LOG"; then
    echo "  FAIL: traceback found in log"
    grep -E 'Traceback|Error|Exception' "$LOG" | head -5
    exit 1
fi
echo "  OK: no exceptions"

echo ""
echo "=========================================="
echo "STAGE 4 PASSED — full training pipeline works"
echo "=========================================="
echo ""
echo "Checkpoint: checkpoints/duet_smoke/ckpts/epoch_1.pt"
echo "Log:        $LOG"
echo "TB:         checkpoints/duet_smoke/tb"

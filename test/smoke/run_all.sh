#!/usr/bin/env bash
# Phase 7 smoke test runner — runs all 4 stages in sequence.
#
# Stops on first failure. Each stage script also works standalone.
#
# Run:
#   bash test/smoke/run_all.sh
#
# On remote:
#   NAVDSL_DATA_BASE=/home/hjk/sshfs_root/dataset/habitat-data bash test/smoke/run_all.sh

set -e  # exit on first failure
cd "$(dirname "$0")/../.."

DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "############################################################"
echo "#  Phase 7 Smoke Test — Running all stages sequentially    #"
echo "############################################################"

echo ""
echo "########## Stage 0: Pre-flight ##########"
bash "$DIR/stage0_preflight.sh"

echo ""
echo "########## Stage 1: Config + Dataset ##########"
python "$DIR/stage1_config_dataset.py"

echo ""
echo "########## Stage 2: Model + Agent ##########"
python "$DIR/stage2_model_agent.py"

echo ""
echo "########## Stage 3: Rollout + Backward ##########"
python "$DIR/stage3_rollout_backward.py"

echo ""
echo "########## Stage 4: End-to-end training ##########"
bash "$DIR/stage4_end_to_end.sh"

echo ""
echo "############################################################"
echo "#  ALL STAGES PASSED                                        #"
echo "############################################################"
echo ""
echo "Phase 7 smoke test complete. Pipeline is verified end-to-end."
echo "Next: full training run (100 epochs) on multiple scenes."

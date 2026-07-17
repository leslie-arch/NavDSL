#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export DSL_PYTHON=${DSL_PYTHON:-$(which python)}
export DSL_SERVER_IP=${DSL_SERVER_IP:-localhost}
export DSL_SERVER_PORT=${DSL_SERVER_PORT:-8080}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}

session_name=dsl_servers_${RANDOM}

# Create a detached tmux session with a single pane
tmux new-session -d -s ${session_name}

# Start unified VLM server (all models in one Flask process)
tmux send-keys -t ${session_name}:0 "${DSL_PYTHON} -m navdsl.vlm.server --all --ip ${DSL_SERVER_IP} --port ${DSL_SERVER_PORT}" C-m

# Attach to the tmux session to view the output
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Server will listen on ${DSL_SERVER_IP}:${DSL_SERVER_PORT}"
echo "Run the following to monitor the server:"
echo "tmux attach-session -t ${session_name}"

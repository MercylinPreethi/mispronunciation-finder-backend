#!/bin/bash
# Environment setup for MDD system

export TRANSFORMERS_CACHE="mdd_cluster_workspace/model_cache"
export HF_HOME="mdd_cluster_workspace/hf_cache"
export PYTHONPATH="mdd_cluster_workspace:$PYTHONPATH"

# Activate virtual environment
source mdd_cluster_workspace/mdd_env/bin/activate

echo "Environment activated!"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

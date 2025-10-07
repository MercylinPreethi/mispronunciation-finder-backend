#!/bin/bash
# MDD System Runner Script

cd mdd_cluster_workspace
source mdd_cluster_workspace/mdd_env/bin/activate

# Set environment variables
export TRANSFORMERS_CACHE="mdd_cluster_workspace/model_cache"
export HF_HOME="mdd_cluster_workspace/hf_cache"

# Example usage:
echo "MDD System Ready!"
echo "Usage examples:"
echo ""
echo "1. Single file with local model:"
echo "  python mdd_cluster.py --audio_file audio_files/sample.wav --reference_text 'hello world' --use_local_model"
echo ""
echo "2. With Modal endpoint (if deployed):"
echo "  python mdd_cluster.py --audio_file audio_files/sample.wav --reference_text 'hello world' --modal_url YOUR_MODAL_URL"
echo ""
echo "3. Test system:"
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""

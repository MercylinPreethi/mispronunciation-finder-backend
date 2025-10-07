
🎉 MDD System Successfully Set Up!

📁 Workspace: /DATA/mercylin/mdd_cluster_workspace

🚀 Quick Start:

1. Activate environment:
   cd mdd_cluster_workspace
   source setup_env.sh

2. Test the system:
   python mdd_cluster.py --audio_file audio_files/your_audio.wav \
                        --reference_text "your reference text" \
                        --use_local_model

3. For better performance, set up Modal Labs:
   - Sign up at https://modal.com
   - Run: modal setup
   - Deploy: modal serve modal/phoneme_modal_server.py
   - Use the provided URL with --modal_url flag

📋 File Structure:
mdd_cluster_workspace/
├── mdd_cluster.py          # Main system
├── modal/
│   └── phoneme_modal_server.py  # Modal server
├── audio_files/            # Put your audio files here
├── results/                # Output directory
├── references.json         # Reference texts
├── run_mdd.sh             # Convenient run script
└── setup_env.sh           # Environment setup

💡 Usage Examples:

# Local model (works offline)
python mdd_cluster.py --audio_file audio.wav --reference_text "hello world" --use_local_model

# With Modal (requires internet, faster)
python mdd_cluster.py --audio_file audio.wav --reference_text "hello world" --modal_url YOUR_URL

# With Groq API for advanced feedback
python mdd_cluster.py --audio_file audio.wav --reference_text "hello world" --groq_api_key YOUR_KEY

🔧 Environment Variables:
export TRANSFORMERS_CACHE="mdd_cluster_workspace/model_cache"
export HF_HOME="mdd_cluster_workspace/hf_cache"
export GROQ_API_KEY="your-key-here"  # Optional
export MODAL_URL="your-modal-url"    # Optional

📚 Next Steps:
1. Place audio files in audio_files/ directory
2. Update references.json with your text mappings
3. Run the system on your data
4. Check results/ directory for output

🆘 Troubleshooting:
- Check cluster_setup.log for detailed logs
- Verify CUDA with: python -c "import torch; print(torch.cuda.is_available())"
- Test eSpeak with: espeak --version
- For issues, re-run: python cluster_runner.py --step verify
# mispronunciation-finder-backend

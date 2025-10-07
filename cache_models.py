
import os
os.environ["TRANSFORMERS_CACHE"] = "mdd_cluster_workspace/model_cache"
os.environ["HF_HOME"] = "mdd_cluster_workspace/hf_cache"

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

models_to_cache = [
    "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    "facebook/wav2vec2-large-xlsr-53",
    "facebook/wav2vec2-xls-r-300m"
]

for model_name in models_to_cache:
    try:
        print(f"Caching {model_name}...")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        print(f"✓ Successfully cached {model_name}")
    except Exception as e:
        print(f"⚠ Could not cache {model_name}: {e}")

print("Model caching complete!")

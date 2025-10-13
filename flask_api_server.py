"""
Flask Server for Pronunciation Analysis with wav2vec2
Includes intelligent phoneme cleaning and word-level analysis
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
import time
import base64
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import ollama

import whisper

# LLM evaluation imports
from huggingface_hub import InferenceClient
from datetime import datetime


CONVERSATION_CONFIG = {
    "provider": "ollama",
    "model": "llama3.2",  # or "mistral", "phi3", etc.
}

# TTS imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("pyttsx3 not available. Install with: pip install pyttsx3")
    TTS_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    print("gTTS not available. Install with: pip install gtts")
    GTTS_AVAILABLE = False

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    print("Audio processing libraries not available.")
    AUDIO_PROCESSING_AVAILABLE = False

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not available. Install with: pip install groq")

# Import the improved MDD system
try:
    from improved_dynamic_mdd import ImprovedDynamicMDD
except ImportError:
    print("Warning: improved_dynamic_mdd.py not found.")
    ImprovedDynamicMDD = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Config
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = '/tmp/audio_uploads'
AUDIO_OUTPUT_FOLDER = '/tmp/generated_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)

improved_mdd_system = None
whisper_model = None

def get_whisper_model():
    """Initialize Whisper model for validation"""
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded for validation")
    return whisper_model

# LLM Judge Configuration
LLM_JUDGE_CONFIG = {
    "huggingface": {
        "model": "microsoft/DialoGPT-medium",
        "max_tokens": 300,
        "temperature": 0.2,
        "timeout": 30
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 400,
        "temperature": 0,
        "api_key": os.environ.get("GROQ_API_KEY")
    },
    "preferred_provider": "groq"
}

# Audio Generation Configuration
AUDIO_CONFIG = {
    "sample_rate": 22050,
    "format": "wav",
    "speed": 0.9,
    "voice_gender": "female",
    "language": "en"
}

hf_client = None
groq_client = None


def generate_conversational_response(user_message: str, pronunciation_analysis: Dict[str, Any]) -> str:
    """Generate conversational AI response using Groq"""
    
    # Extract key metrics
    accuracy = pronunciation_analysis.get('accuracy', 0) * 100
    mispronounced = pronunciation_analysis.get('mispronounced_words', [])
    partial = pronunciation_analysis.get('partial_words', [])
    correct = pronunciation_analysis.get('correct_words', [])
    fluency_score = pronunciation_analysis.get('word_accuracy', 0) * 100
    
    # Build context-aware system prompt
    system_prompt = f"""You are a friendly, encouraging AI pronunciation coach named Coach AI. You're having a natural conversation with a language learner.

The user just said: "{user_message}"

Their pronunciation analysis:
- Overall Accuracy: {accuracy:.1f}%
- Fluency Score: {fluency_score:.1f}%
- Words pronounced perfectly: {', '.join(correct[:3]) if correct else 'None yet'}
- Words that need work: {', '.join(mispronounced) if mispronounced else 'None'}
- Partially correct: {', '.join(partial) if partial else 'None'}

Your response guidelines:
1. FIRST: Respond naturally to what they said (acknowledge their message content, show you understood them)
2. Give genuine, specific praise when accuracy is good (>75%)
3. If there are pronunciation issues (accuracy <75%), mention 1-2 problem words naturally
4. Keep it conversational - sound like a supportive friend, not a formal teacher
5. Ask a follow-up question to continue the conversation naturally
6. Keep response under 50 words
7. Be encouraging and positive
8. Use casual language and contractions

Example good responses:
- High accuracy (85%+): "That's great! I love how clearly you said that. Your pronunciation of 'beautiful' was perfect! So, what's your favorite thing about learning English?"
- Medium accuracy (70-85%): "Nice! I understood you well. Just a quick tip - the word 'weather' has a tricky 'th' sound. Try putting your tongue between your teeth. Anyway, what's the weather like where you are?"
- Lower accuracy (<70%): "I appreciate you trying! Let's work on 'comfortable' - break it into parts: com-for-ta-ble. What would make you more comfortable practicing?"

Remember: You're having a CONVERSATION, not giving a lesson. Respond to what they said!"""

    try:
        client = get_groq_client()
        if not client:
            logger.warning("Groq client not available, using fallback")
            return generate_fallback_response(user_message, pronunciation_analysis)
        
        logger.info("🤖 Generating Groq response...")
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.3-70b-versatile",  # Best free model
            temperature=0.8,  # More creative and conversational
            max_tokens=200,  # Enough for 50 words with safety margin
            top_p=0.9
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        logger.info(f"✅ Groq response generated: {ai_response[:100]}...")
        
        return ai_response
        
    except Exception as e:
        logger.error(f"❌ Groq response generation failed: {e}")
        return generate_fallback_response(user_message, pronunciation_analysis)

def generate_fallback_response(user_message: str, pronunciation_analysis: Dict[str, Any]) -> str:
    """Generate rule-based response if Groq fails"""
    
    accuracy = pronunciation_analysis.get('accuracy', 0) * 100
    mispronounced = pronunciation_analysis.get('mispronounced_words', [])
    partial = pronunciation_analysis.get('partial_words', [])
    
    # Try to understand the message context
    user_lower = user_message.lower()
    
    # Respond based on accuracy + context
    if accuracy >= 85:
        # Excellent pronunciation
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! Great pronunciation! Your greeting was crystal clear. What brings you here today?"
        elif any(word in user_lower for word in ['good', 'fine', 'great', 'well']):
            return "That's wonderful to hear! Your pronunciation is excellent. What would you like to practice today?"
        elif any(word in user_lower for word in ['weather', 'day', 'today']):
            return "Perfect! I understood you clearly. You're speaking really well. Tell me more about your day!"
        else:
            return f"Excellent pronunciation! You said '{user_message}' very clearly. Keep it up! What else would you like to talk about?"
    
    elif accuracy >= 70:
        # Good with minor issues
        problem_word = mispronounced[0] if mispronounced else (partial[0] if partial else None)
        
        if problem_word:
            return f"Good effort! I understood most of what you said. Just work on the word '{problem_word}' a bit. Can you use it in another sentence?"
        else:
            return "Nice job! Your pronunciation is improving. Let's keep practicing. What topic interests you?"
    
    elif accuracy >= 50:
        # Needs work but understandable
        problem_word = mispronounced[0] if mispronounced else "some words"
        return f"I see you're working hard! Let's focus on '{problem_word}'. Try saying it slowly: {problem_word}. Can you try again?"
    
    else:
        # Significant issues
        return "Let's practice together! Try speaking a bit more slowly and clearly. Take your time - pronunciation improves with practice. What would you like to say?"

@app.route('/chat_with_coach', methods=['POST'])
def chat_with_coach():
    """Complete endpoint: Transcribe + Analyze + Generate Response with Groq"""
    start_time = time.time()
    
    print_banner("AI COACH CONVERSATION (GROQ)", "=", 80)
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "Audio file required"}), 400

        file = request.files['audio_file']
        
        # Save audio temporarily
        filename = secure_filename(file.filename)
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        file.save(temp_audio_path)
        
        print_section("STEP 1: TRANSCRIPTION", "Processing with Whisper...")
        
        # Step 1: Transcribe with Whisper
        whisper = get_whisper_model()
        if whisper is None:
            os.remove(temp_audio_path)
            return jsonify({"success": False, "error": "Whisper not available"}), 500
        
        whisper_result = whisper.transcribe(temp_audio_path)
        transcription = whisper_result["text"].strip()
        
        print_section("TRANSCRIPTION RESULT", {
            "Text": transcription,
            "Language": whisper_result.get("language", "en"),
            "Length": f"{len(transcription)} characters"
        })
        
        if not transcription or len(transcription) < 2:
            os.remove(temp_audio_path)
            return jsonify({
                "success": False,
                "error": "Could not understand audio. Please speak clearly."
            }), 400
        
        print_section("STEP 2: PRONUNCIATION ANALYSIS", "Analyzing with wav2vec2...")
        
        # Step 2: Analyze pronunciation
        phoneme_results = extract_phonemes_wav2vec2(temp_audio_path, transcription)
        
        if not phoneme_results["success"]:
            os.remove(temp_audio_path)
            return jsonify({
                "success": False,
                "error": "Pronunciation analysis failed"
            }), 500
        
        predicted_phonemes = phoneme_results["predicted_phonemes"]
        reference_phonemes = phoneme_results["reference_phonemes"]
        
        # Perform segmentation and analysis
        segmentation_result = enhanced_universal_phoneme_segmentation(predicted_phonemes, transcription)
        words_analysis = enhanced_analyze_segmented_words(segmentation_result)
        
        # Extract word status
        mispronounced_words = []
        correctly_pronounced_words = []
        partial_words = []
        
        for word_data in words_analysis:
            word = word_data["word"]
            status = word_data["status"]
            
            if status == "correct":
                correctly_pronounced_words.append(word)
            elif status == "mispronounced":
                mispronounced_words.append(word)
            elif status == "partial":
                partial_words.append(word)
        
        # Calculate metrics
        per_result = calculate_phoneme_error_rate(predicted_phonemes, reference_phonemes)
        word_accuracy = (len(correctly_pronounced_words) + len(partial_words) * 0.5) / max(1, len(words_analysis))
        accuracy = 1 - (per_result["per"] / 100)
        
        pronunciation_analysis = {
            'accuracy': accuracy,
            'word_accuracy': word_accuracy,
            'mispronounced_words': mispronounced_words,
            'partial_words': partial_words,
            'correct_words': correctly_pronounced_words,
            'per': per_result["per"],
            'fluency_score': word_accuracy * 100
        }
        
        print_section("PRONUNCIATION METRICS", {
            "Accuracy": f"{accuracy * 100:.1f}%",
            "Fluency": f"{word_accuracy * 100:.1f}%",
            "Perfect": len(correctly_pronounced_words),
            "Needs Work": len(mispronounced_words),
            "Partial": len(partial_words)
        })
        
        print_section("STEP 3: GROQ AI RESPONSE", "Generating conversational reply...")
        
        # Step 3: Generate conversational AI response with Groq
        ai_response = generate_conversational_response(transcription, pronunciation_analysis)
        
        print_section("GROQ AI RESPONSE", ai_response)
        
        # Step 4: Get detailed practice suggestions from LLM Judge
        print_section("STEP 4: DETAILED FEEDBACK", "Generating practice tips...")
        
        llm_judge = LLMPronunciationJudge()
        llm_assessment = llm_judge.evaluate_pronunciation(
            reference_text=transcription,
            reference_phonemes=reference_phonemes,
            predicted_phonemes=predicted_phonemes,
            word_analysis={"words": words_analysis},
            per=per_result["per"],
            word_accuracy=word_accuracy
        )
        
        # Extract suggestions from LLM feedback
        concise_feedback = llm_assessment.get("concise_feedback", {})
        quick_fixes = concise_feedback.get("quick_fixes", "")
        suggestions = []
        
        if quick_fixes:
            # Split by periods or newlines
            raw_suggestions = [s.strip() for s in quick_fixes.replace('\n', '.').split('.') if s.strip()]
            # Take top 3 and ensure they're useful
            suggestions = [s for s in raw_suggestions if len(s) > 10][:3]
        
        # Fallback suggestions if none generated
        if not suggestions and mispronounced_words:
            suggestions = [
                f"Practice the word '{mispronounced_words[0]}' slowly",
                "Break difficult words into syllables",
                "Listen and repeat after native speakers"
            ]
        
        # Cleanup
        os.remove(temp_audio_path)
        
        processing_time = time.time() - start_time
        
        print_section("PROCESSING COMPLETE", {
            "Total Time": f"{processing_time:.2f}s",
            "User Said": transcription,
            "AI Replied": ai_response[:80] + "..."
        })
        
        # Build response
        response = {
            "success": True,
            "transcription": transcription,
            "ai_response": ai_response,
            "pronunciation_feedback": {
                "accuracy": round(accuracy * 100, 1),
                "fluency_score": round(word_accuracy * 100, 1),
                "mispronounced_words": mispronounced_words,
                "partial_words": partial_words,
                "correct_words": correctly_pronounced_words,
                "suggestions": suggestions,
                "per": round(per_result["per"], 1)
            },
            "word_analysis": {
                "total_words": len(words_analysis),
                "correct": len(correctly_pronounced_words),
                "partial": len(partial_words),
                "mispronounced": len(mispronounced_words)
            },
            "metadata": {
                "processing_time": round(processing_time, 2),
                "language": whisper_result.get("language", "en"),
                "model": "groq/llama-3.3-70b-versatile"
            }
        }
        
        print_banner("✅ COACH CONVERSATION COMPLETE", "=", 80)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"❌ Coach conversation failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    
class AudioGenerator:
    """Handle audio generation for pronunciation examples"""
    
    def __init__(self):
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self._configure_tts_engine()
            except Exception as e:
                logger.warning(f"Could not initialize TTS engine: {e}")
                self.tts_engine = None
    
    def _configure_tts_engine(self):
        """Configure the TTS engine with optimal settings for pronunciation learning"""
        if not self.tts_engine:
            return
        
        try:
            # Set speech rate (slower for learning)
            self.tts_engine.setProperty('rate', 150)  # Default is usually 200
            
            # Set volume
            self.tts_engine.setProperty('volume', 0.9)
            
            # Try to set a clear voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice for pronunciation learning (often clearer)
                female_voices = [v for v in voices if 'female' in v.name.lower() or 'zira' in v.name.lower()]
                if female_voices:
                    self.tts_engine.setProperty('voice', female_voices[0].id)
                elif len(voices) > 1:
                    self.tts_engine.setProperty('voice', voices[1].id)
            
        except Exception as e:
            logger.warning(f"TTS configuration warning: {e}")
    
    def generate_word_audio_pyttsx3(self, word: str, output_path: str) -> bool:
        """Generate audio for a single word using pyttsx3"""
        if not self.tts_engine:
            return False
        
        try:
            # Add slight pauses around the word for clarity
            text_to_speak = f". {word} ."
            
            self.tts_engine.save_to_file(text_to_speak, output_path)
            self.tts_engine.runAndWait()
            
            return os.path.exists(output_path)
            
        except Exception as e:
            logger.error(f"pyttsx3 audio generation failed for '{word}': {e}")
            return False
    
    def generate_word_audio_gtts(self, word: str, output_path: str) -> bool:
        """Generate audio for a single word using gTTS"""
        if not GTTS_AVAILABLE:
            return False
        
        try:
            # Create TTS object with slow speech for pronunciation learning
            tts = gTTS(text=word, lang='en', slow=True)
            tts.save(output_path)
            
            return os.path.exists(output_path)
            
        except Exception as e:
            logger.error(f"gTTS audio generation failed for '{word}': {e}")
            return False
    
    def generate_sentence_audio(self, sentence: str, output_path: str, slow: bool = True) -> bool:
        """Generate audio for entire sentence"""
        if GTTS_AVAILABLE:
            try:
                tts = gTTS(text=sentence, lang='en', slow=slow)
                tts.save(output_path)
                return os.path.exists(output_path)
            except Exception as e:
                logger.error(f"Sentence audio generation failed: {e}")
        
        if self.tts_engine:
            try:
                self.tts_engine.save_to_file(sentence, output_path)
                self.tts_engine.runAndWait()
                return os.path.exists(output_path)
            except Exception as e:
                logger.error(f"TTS sentence generation failed: {e}")
        
        return False

    
    
    def generate_word_audio(self, word: str, output_path: str) -> bool:
        """Generate audio for a word using available TTS engines"""
        # Try gTTS first (usually better quality)
        if self.generate_word_audio_gtts(word, output_path):
            return True
        
        # Fallback to pyttsx3
        return self.generate_word_audio_pyttsx3(word, output_path)
    
    def audio_to_base64(self, audio_path: str) -> Optional[str]:
        """Convert audio file to base64 string for API response"""
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                return base64.b64encode(audio_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Audio to base64 conversion failed: {e}")
            return None
    
    def process_audio_quality(self, audio_path: str) -> bool:
        """Improve audio quality if audio processing libraries are available"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return True
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=AUDIO_CONFIG["sample_rate"])
            
            # Apply light noise reduction and normalization
            audio = librosa.util.normalize(audio)
            
            # Save processed audio
            sf.write(audio_path, audio, sr, format='WAV')
            
            return True
            
        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            return False

# Initialize audio generator
audio_generator = AudioGenerator()

def print_banner(text, char="=", width=80):
    """Print a formatted banner"""
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}")

def print_section(title, content, char="-", width=60):
    """Print a formatted section"""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")
    if isinstance(content, dict):
        for key, value in content.items():
            print(f"{key}: {value}")
    else:
        print(content)

def get_hf_client():
    """Initialize Hugging Face client"""
    global hf_client
    if hf_client is None:
        try:
            hf_client = InferenceClient(
                model=LLM_JUDGE_CONFIG["huggingface"]["model"],
                timeout=LLM_JUDGE_CONFIG["huggingface"]["timeout"]
            )
            logger.info(f"HF Judge initialized with model: {LLM_JUDGE_CONFIG['huggingface']['model']}")
        except Exception as e:
            logger.warning(f"Could not initialize HF client: {e}")
            hf_client = None
    return hf_client


def extract_phonemes_wav2vec2(audio_path: str, reference_text: str, is_single_phoneme: bool = False) -> Dict[str, Any]:
    """
    Extract phonemes using wav2vec2 - returns actual predictions
    
    Args:
        audio_path: Path to audio file
        reference_text: Reference text (can be a single phoneme or full word)
        is_single_phoneme: If True, only analyze for single phoneme practice
    
    Returns:
        Dictionary with predicted and reference phonemes
    """
    try:
        print_banner("WAV2VEC2 PHONEME EXTRACTION", "=", 80)
        
        system = get_improved_mdd_system()
        
        # CRITICAL: For single phoneme practice, use simplified extraction
        if is_single_phoneme:
            logger.info(f"🎯 SINGLE PHONEME MODE: Analyzing for phoneme '{reference_text}'")
            
            # Load and process audio
            audio = system.load_audio(audio_path)
            
            # Extract phonemes (should return what was actually said)
            predicted_phonemes = system.extract_phonemes_from_audio(audio, reference_text)
            
            # Reference is just the single target phoneme
            reference_phonemes = [reference_text]
            
            logger.info(f"Predicted for single phoneme: {predicted_phonemes}")
            logger.info(f"Reference (target phoneme): {reference_phonemes}")
            
            return {
                "predicted_phonemes": predicted_phonemes,
                "reference_phonemes": reference_phonemes,
                "model": "wav2vec2",
                "success": True,
                "mode": "single_phoneme",
                "target_phoneme": reference_text
            }
        
        # NORMAL MODE: Full word/sentence analysis
        else:
            logger.info(f"📝 FULL TEXT MODE: Analyzing '{reference_text}'")
            raw_results = system.process_audio(audio_path, reference_text)
            
            predicted_phonemes = raw_results["analysis"]["predicted_phonemes"]
            reference_phonemes = raw_results["analysis"]["reference_phonemes"]
            
            return {
                "predicted_phonemes": predicted_phonemes,
                "reference_phonemes": reference_phonemes,
                "model": "wav2vec2",
                "success": True,
                "mode": "full_text",
                "raw_results": raw_results
            }
            
    except Exception as e:
        logger.error(f"Wav2vec2 extraction failed: {e}")
        return {
            "predicted_phonemes": [],
            "reference_phonemes": [reference_text] if is_single_phoneme else [],
            "model": "wav2vec2",
            "success": False,
            "error": str(e),
            "mode": "single_phoneme" if is_single_phoneme else "full_text"
        }

def enhanced_universal_phoneme_segmentation(predicted_phonemes: List[str], reference_text: str) -> List[Dict]:
    """
    Enhanced phoneme segmentation that properly maps predicted phonemes to words
    """
    system = get_improved_mdd_system()
    words = reference_text.lower().split()
    
    # Get reference phonemes for each word
    word_ref_phonemes = []
    
    for word in words:
        try:
            # Clean word for phoneme conversion
            clean_word = word.replace("'", "").replace(",", "").replace(".", "")
            ref_phonemes = system._text_to_phonemes_espeak(clean_word)
            if not ref_phonemes or ref_phonemes == ['<unk>']:
                ref_phonemes = system._basic_text_to_phonemes(clean_word)
            
            word_ref_phonemes.append({
                "word": word,
                "ref_phonemes": ref_phonemes,
                "length": len(ref_phonemes)
            })
        except Exception as e:
            # Fallback: use simple phoneme mapping
            word_ref_phonemes.append({
                "word": word,
                "ref_phonemes": [word],
                "length": 1
            })
    
    # Calculate total reference phonemes
    total_ref_phonemes = sum(len(w['ref_phonemes']) for w in word_ref_phonemes)
    
    if not predicted_phonemes or total_ref_phonemes == 0:
        return enhanced_proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)
    
    # Use improved boundary detection
    boundaries = enhanced_boundary_detection(predicted_phonemes, word_ref_phonemes)
    
    return boundaries

def enhanced_boundary_detection(predicted_phonemes: List[str], word_ref_phonemes: List[Dict]) -> List[Dict]:
    """
    Enhanced boundary detection using phonetic similarity
    """
    n_pred = len(predicted_phonemes)
    n_words = len(word_ref_phonemes)
    
    # DP table for optimal alignment
    dp = [[float('inf')] * (n_words + 1) for _ in range(n_pred + 1)]
    path = [[None] * (n_words + 1) for _ in range(n_pred + 1)]
    
    dp[0][0] = 0
    
    for i in range(n_pred + 1):
        for j in range(n_words + 1):
            if dp[i][j] == float('inf'):
                continue
                
            # Try to assign phonemes to current word
            if j < n_words:
                current_word = word_ref_phonemes[j]
                ref_phonemes = current_word["ref_phonemes"]
                ref_len = len(ref_phonemes)
                
                # Try different segment lengths
                max_seg_len = min(n_pred - i, ref_len * 3)  # Allow up to 3x reference length
                for seg_len in range(1, max_seg_len + 1):
                    if i + seg_len > n_pred:
                        continue
                        
                    segment = predicted_phonemes[i:i + seg_len]
                    
                    # Calculate alignment cost
                    aligned_ref, aligned_seg = needleman_wunsch_alignment(ref_phonemes, segment)
                    cost = enhanced_calculate_alignment_cost(aligned_ref, aligned_seg)
                    
                    # Add length penalty
                    length_penalty = abs(seg_len - ref_len) * 0.1
                    total_cost = dp[i][j] + cost + length_penalty
                    
                    if total_cost < dp[i + seg_len][j + 1]:
                        dp[i + seg_len][j + 1] = total_cost
                        path[i + seg_len][j + 1] = (i, j, seg_len)
    
    # Backtrack to find optimal boundaries
    boundaries = []
    i, j = n_pred, n_words
    
    # If no path found, use fallback
    if dp[n_pred][n_words] == float('inf'):
        return enhanced_proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)
    
    while i > 0 and j > 0:
        if path[i][j] is None:
            # Fallback for this segment
            break
        
        prev_i, prev_j, seg_len = path[i][j]
        current_word = word_ref_phonemes[j - 1]
        
        boundaries.append({
            "word": current_word["word"],
            "start": prev_i,
            "end": i,
            "predicted_phonemes": predicted_phonemes[prev_i:i],
            "reference_phonemes": current_word["ref_phonemes"],
            "alignment_cost": dp[i][j] - dp[prev_i][prev_j],
            "method": "enhanced_dp"
        })
        
        i, j = prev_i, prev_j
    
    boundaries.reverse()
    
    # If we didn't get all words, use fallback for remaining
    if len(boundaries) < len(word_ref_phonemes):
        remaining_boundaries = enhanced_proportional_segmentation_fallback(
            predicted_phonemes[i:], 
            word_ref_phonemes[len(boundaries):]
        )
        # Adjust indices for remaining boundaries
        for boundary in remaining_boundaries:
            boundary["start"] += i
            boundary["end"] += i
        boundaries.extend(remaining_boundaries)
    
    return boundaries

def enhanced_calculate_alignment_cost(aligned_ref: List[str], aligned_pred: List[str]) -> float:
    """Calculate cost based on phoneme alignment with improved weights"""
    if not aligned_ref or not aligned_pred:
        return float('inf')
    
    cost = 0
    for ref_ph, pred_ph in zip(aligned_ref, aligned_pred):
        if ref_ph == "_" and pred_ph != "_":
            cost += 1.0  # Insertion
        elif pred_ph == "_" and ref_ph != "_":
            cost += 1.0  # Deletion  
        elif ref_ph != pred_ph:
            cost += 0.7  # Substitution (less than insertion/deletion)
        else:
            cost += 0.0  # Match
    
    return cost

def enhanced_proportional_segmentation_fallback(predicted_phonemes: List[str], word_ref_phonemes: List[Dict]) -> List[Dict]:
    """Enhanced proportional segmentation fallback"""
    boundaries = []
    
    if not predicted_phonemes or not word_ref_phonemes:
        return boundaries
    
    # Calculate total reference length
    total_ref_len = sum(len(w['ref_phonemes']) for w in word_ref_phonemes)
    if total_ref_len == 0:
        total_ref_len = len(word_ref_phonemes)  # Fallback to word count
    
    current_pos = 0
    
    for i, word_data in enumerate(word_ref_phonemes):
        # Get reference length
        ref_len = len(word_data['ref_phonemes'])
        if ref_len == 0:
            ref_len = 1
        
        # Calculate end position
        if i == len(word_ref_phonemes) - 1:
            # Last word gets remaining phonemes
            end_pos = len(predicted_phonemes)
        else:
            proportion = ref_len / total_ref_len
            allocated = max(1, int(len(predicted_phonemes) * proportion))
            end_pos = min(current_pos + allocated, len(predicted_phonemes))
        
        segment = predicted_phonemes[current_pos:end_pos]
        boundaries.append({
            "word": word_data["word"],
            "start": current_pos,
            "end": end_pos,
            "predicted_phonemes": segment,
            "reference_phonemes": word_data["ref_phonemes"],
            "alignment_cost": float('inf'),
            "method": "enhanced_proportional_fallback"
        })
        current_pos = end_pos
    
    return boundaries

def enhanced_analyze_segmented_words(segmentation_result: List[Dict]) -> List[Dict[str, Any]]:
    """Enhanced word analysis with better phoneme mapping"""
    words_analysis = []
    
    for seg_data in segmentation_result:
        word = seg_data["word"]
        pred_phonemes = seg_data["predicted_phonemes"]
        ref_phonemes = seg_data["reference_phonemes"]
        
        # Skip if no reference phonemes
        if not ref_phonemes:
            words_analysis.append(create_enhanced_empty_word_analysis(word))
            continue
        
        # Use Needleman-Wunsch for optimal alignment
        aligned_ref, aligned_pred = needleman_wunsch_alignment(ref_phonemes, pred_phonemes)
        
        # Calculate detailed metrics
        analysis = calculate_enhanced_word_analysis_metrics(word, ref_phonemes, pred_phonemes, aligned_ref, aligned_pred)
        words_analysis.append(analysis)
    
    return words_analysis

def calculate_enhanced_word_analysis_metrics(word: str, ref_phonemes: List[str], pred_phonemes: List[str], 
                                          aligned_ref: List[str], aligned_pred: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive word analysis metrics"""
    
    # Count matches and errors
    correct_matches = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    errors = []
    
    for j, (ref_p, pred_p) in enumerate(zip(aligned_ref, aligned_pred)):
        if ref_p == "_" and pred_p != "_":
            insertions += 1
            errors.append({
                "position": j,
                "type": "insertion",
                "predicted": pred_p,
                "expected": None
            })
        elif pred_p == "_" and ref_p != "_":
            deletions += 1
            errors.append({
                "position": j,
                "type": "deletion",
                "predicted": None,
                "expected": ref_p
            })
        elif ref_p != pred_p and ref_p != "_" and pred_p != "_":
            substitutions += 1
            errors.append({
                "position": j,
                "type": "substitution",
                "predicted": pred_p,
                "expected": ref_p
            })
        elif ref_p == pred_p:
            correct_matches += 1
    
    # Calculate accuracy metrics
    total_ref = len([r for r in aligned_ref if r != "_"])
    if total_ref == 0:
        word_per = 100.0
        word_accuracy = 0.0
    else:
        word_per = ((substitutions + deletions + insertions) / total_ref) * 100
        word_accuracy = (correct_matches / total_ref) * 100
    
    # Determine word status with improved thresholds
    if word_accuracy >= 85:
        status = "correct"
    elif word_accuracy >= 50:
        status = "partial"
    else:
        status = "mispronounced"
    
    return {
        "word": word,
        "reference_phonemes": ref_phonemes,
        "predicted_phonemes": pred_phonemes,
        "aligned_reference": aligned_ref,
        "aligned_predicted": aligned_pred,
        "per": {"per": round(word_per, 2), "method": "enhanced_alignment"},
        "accuracy": round(word_accuracy, 2),
        "status": status,
        "phoneme_errors": errors,
        "error_count": len(errors),
        "error_counts": {
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "correct": correct_matches
        },
        "match_quality": round(word_accuracy / 100, 3)
    }

def create_enhanced_empty_word_analysis(word: str) -> Dict[str, Any]:
    """Create analysis for words with no reference phonemes"""
    return {
        "word": word,
        "reference_phonemes": [],
        "predicted_phonemes": [],
        "aligned_reference": [],
        "aligned_predicted": [],
        "per": {"per": 100.0, "method": "no_reference"},
        "accuracy": 0.0,
        "status": "mispronounced",
        "phoneme_errors": [],
        "error_count": 0,
        "error_counts": {
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "correct": 0
        },
        "match_quality": 0.0
    }

def debug_phoneme_mapping(reference_text: str, predicted_phonemes: List[str], segmentation_result: List[Dict]):
    """Debug function to log phoneme mapping details"""
    print("\n" + "="*80)
    print("DEBUG PHONEME MAPPING")
    print("="*80)
    print(f"Reference: '{reference_text}'")
    print(f"Predicted: {predicted_phonemes}")
    print(f"Total predicted phonemes: {len(predicted_phonemes)}")
    
    for i, boundary in enumerate(segmentation_result):
        print(f"\nWord {i+1}: '{boundary['word']}'")
        print(f"  Reference phonemes: {boundary['reference_phonemes']}")
        print(f"  Predicted segment: {boundary['predicted_phonemes']}")
        print(f"  Boundary: {boundary['start']}-{boundary['end']}")
        print(f"  Method: {boundary.get('method', 'unknown')}")
        print(f"  Segment length: {len(boundary['predicted_phonemes'])}")
    
    print("="*80 + "\n")

def clean_predicted_phonemes(predicted_phonemes: List[str], 
                             reference_phonemes: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Intelligent phoneme cleaning: removes repetitions, noise, and finds best match
    """
    stats = {
        'original_count': len(predicted_phonemes),
        'method': 'none'
    }
    
    # Step 1: Remove noise markers
    NOISE_PHONEMES = {'[noise]', '[breath]', '[laugh]', '[cough]', 'brth', 'ns', 'lg', '<unk>', 'sil', 'sp'}
    cleaned = [p for p in predicted_phonemes if p not in NOISE_PHONEMES]
    
    # Step 2: Remove consecutive duplicates
    deduplicated = []
    prev = None
    for p in cleaned:
        if p != prev:
            deduplicated.append(p)
        prev = p
    
    # Step 3: If too many phonemes, find best matching window
    if len(deduplicated) > len(reference_phonemes) * 1.5:
        ref_len = len(reference_phonemes)
        ref_set = set(reference_phonemes)
        best_score = -1
        best_window = deduplicated[:ref_len]
        
        # Sliding window to find best match
        for start in range(len(deduplicated) - ref_len + 1):
            window = deduplicated[start:start + ref_len]
            matches = sum(1 for p in window if p in ref_set)
            score = matches / ref_len
            
            if score > best_score:
                best_score = score
                best_window = window
        
        stats['filtered_count'] = len(best_window)
        stats['removed_count'] = stats['original_count'] - len(best_window)
        stats['method'] = 'windowing'
        stats['match_score'] = round(best_score, 2)
        
        return best_window, stats
    
    # If reasonable length, just return deduplicated
    stats['filtered_count'] = len(deduplicated)
    stats['removed_count'] = stats['original_count'] - len(deduplicated)
    stats['method'] = 'deduplication'
    
    return deduplicated, stats


def filter_predicted_phonemes_before_analysis(predicted_phonemes: List[str], 
                                            reference_phonemes: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Aggressive filtering of predicted phonemes BEFORE any analysis
    Removes repetitions, filler sounds, and extraneous speech artifacts
    """
    
    # Comprehensive noise and filler phonemes
    NOISE_PHONEMES = {
        '[noise]', '[breath]', '[laugh]', '[cough]', '[silence]', '[unk]',
        'brth', 'ns', 'lg', 'cg', 'sp', 'sil', '<unk>', '##', '++'
    }
    
    # Common filler sounds and repetitions
    FILLER_PHONEMES = {
        'ə', 'ʌ', 'ɚ', 'ɝ', 'ː', 'ʔ', 'ʰ'  # Schwa sounds, glottal stops, aspirations
    }
    
    # Vowel variations that might be filler
    VOWEL_VARIANTS = {
        'ə', 'ɚ', 'ɝ', 'ʌ', 'ɪ', 'ʊ', 'ɐ', 'ɑ', 'ɔ', 'ɛ', 'æ'
    }
    
    filtered = []
    removed = []
    stats = {
        'original_count': len(predicted_phonemes),
        'noise_removed': 0,
        'repetitions_removed': 0,
        'filler_removed': 0,
        'excess_vowels_removed': 0
    }
    
    # Step 1: Remove obvious noise markers
    cleaned_phonemes = []
    for phoneme in predicted_phonemes:
        if phoneme.lower() in NOISE_PHONEMES:
            stats['noise_removed'] += 1
            removed.append({'phoneme': phoneme, 'reason': 'noise_marker'})
        else:
            cleaned_phonemes.append(phoneme)
    
    # Step 2: Remove consecutive duplicates (stuttering/repetitions)
    deduplicated = []
    prev_phoneme = None
    for phoneme in cleaned_phonemes:
        if phoneme == prev_phoneme:
            stats['repetitions_removed'] += 1
            removed.append({'phoneme': phoneme, 'reason': 'repetition'})
        else:
            deduplicated.append(phoneme)
            prev_phoneme = phoneme
    
    # Step 3: Find the best matching subsequence using reference as guide
    if len(deduplicated) > len(reference_phonemes) * 1.2:  # If we have too many phonemes
        best_match = find_best_matching_subsequence(deduplicated, reference_phonemes)
        filtered = best_match
        stats['excess_removed'] = len(deduplicated) - len(best_match)
    else:
        filtered = deduplicated
    
    # Step 4: Remove excessive filler sounds if still too long
    if len(filtered) > len(reference_phonemes) * 1.5:
        filtered = remove_excessive_fillers(filtered, reference_phonemes)
        stats['filler_removed'] = len(deduplicated) - len(filtered)
    
    stats['filtered_count'] = len(filtered)
    stats['total_removed'] = stats['original_count'] - len(filtered)
    
    return filtered, stats

def find_best_matching_subsequence(predicted: List[str], reference: List[str]) -> List[str]:
    """Find the best matching subsequence using sliding window"""
    if len(predicted) <= len(reference):
        return predicted
    
    ref_set = set(reference)
    best_score = -1
    best_subsequence = predicted
    
    # Try different window sizes
    for window_size in range(len(reference), min(len(predicted), len(reference) * 2)):
        for start_idx in range(len(predicted) - window_size + 1):
            window = predicted[start_idx:start_idx + window_size]
            
            # Score based on reference phoneme matches
            matches = sum(1 for p in window if p in ref_set)
            score = matches / window_size
            
            # Prefer sequences that match reference length
            length_penalty = abs(len(window) - len(reference)) * 0.1
            final_score = score - length_penalty
            
            if final_score > best_score:
                best_score = final_score
                best_subsequence = window
    
    return best_subsequence

def remove_excessive_fillers(predicted: List[str], reference: List[str]) -> List[str]:
    """Remove excessive filler sounds while preserving content"""
    # Common content phonemes (consonants and stressed vowels)
    CONTENT_PHONEMES = {
        'b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 
        't', 'v', 'w', 'z', 'ʃ', 'ʒ', 'ʧ', 'ʤ', 'ŋ', 'θ', 'ð',
        'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ'
    }
    
    filler_count = 0
    content_phonemes = []
    
    for phoneme in predicted:
        if phoneme in CONTENT_PHONEMES:
            content_phonemes.append(phoneme)
            filler_count = 0
        else:
            filler_count += 1
            # Allow only limited consecutive fillers
            if filler_count <= 2:  # Allow up to 2 filler sounds in a row
                content_phonemes.append(phoneme)
            # Otherwise skip this filler
    
    return content_phonemes

def get_groq_client():
    """Initialize Groq client"""
    global groq_client
    if groq_client is None and GROQ_AVAILABLE:
        try:
            api_key = LLM_JUDGE_CONFIG["groq"]["api_key"]
            if api_key:
                groq_client = Groq(api_key=api_key)
                logger.info("Groq client initialized successfully")
            else:
                logger.warning("GROQ_API_KEY not found in environment")
        except Exception as e:
            logger.warning(f"Could not initialize Groq client: {e}")
            groq_client = None
    return groq_client

def get_improved_mdd_system():
    global improved_mdd_system
    if improved_mdd_system is None:
        if ImprovedDynamicMDD is None:
            raise Exception("Improved MDD system not available.")
        improved_mdd_system = ImprovedDynamicMDD(
            model_name="facebook/wav2vec2-base-960h",
            device="auto"
        )
    return improved_mdd_system

# Keep existing functions (needleman_wunsch_alignment, calculate_phoneme_error_rate, etc.)
def needleman_wunsch_alignment(seq_a: List[str], seq_b: List[str], 
                              match_score: float = 1.0, 
                              mismatch_score: float = -1.0, 
                              indel_score: float = -1.0, 
                              gap: str = "_") -> Tuple[List[str], List[str]]:
    """Implement Needleman-Wunsch algorithm for optimal sequence alignment"""
    n, m = len(seq_a), len(seq_b)
    
    # Initialize scoring matrix
    score_matrix = np.zeros((n + 1, m + 1))
    
    # Initialize first row and column with gap penalties
    for i in range(1, n + 1):
        score_matrix[i][0] = i * indel_score
    for j in range(1, m + 1):
        score_matrix[0][j] = j * indel_score
    
    # Fill the scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i-1][j-1] + (match_score if seq_a[i-1] == seq_b[j-1] else mismatch_score)
            delete = score_matrix[i-1][j] + indel_score
            insert = score_matrix[i][j-1] + indel_score
            score_matrix[i][j] = max(match, delete, insert)
    
    # Traceback to get alignment
    aligned_a, aligned_b = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        current_score = score_matrix[i][j]
        
        if i > 0 and j > 0:
            diagonal_score = score_matrix[i-1][j-1] + (match_score if seq_a[i-1] == seq_b[j-1] else mismatch_score)
            if current_score == diagonal_score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                continue
        
        if i > 0:
            up_score = score_matrix[i-1][j] + indel_score
            if current_score == up_score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(gap)
                i -= 1
                continue
        
        if j > 0:
            left_score = score_matrix[i][j-1] + indel_score
            if current_score == left_score:
                aligned_a.append(gap)
                aligned_b.append(seq_b[j-1])
                j -= 1
                continue
        
        # Fallback
        if i > 0 and j > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(gap)
            i -= 1
        else:
            aligned_a.append(gap)
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the sequences (since we built them backwards)
    aligned_a.reverse()
    aligned_b.reverse()
    
    return aligned_a, aligned_b

def calculate_phoneme_error_rate(predicted: List[str], reference: List[str]) -> Dict[str, Any]:
    """Calculate Phoneme Error Rate (PER)"""
    if not reference:
        return {
            "per": 100.0, 
            "method": "phoneme_error_rate", 
            "substitutions": 0, 
            "deletions": 0, 
            "insertions": 0,
            "correct": 0,  # ADD THIS
            "total_reference": 0,
            "total_predicted": 0
        }
    
    # Use dynamic programming to compute edit distance and operations
    n, m = len(reference), len(predicted)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    operations = [[""] * (m + 1) for _ in range(n + 1)]
    
    # Initialize DP table
    for i in range(n + 1):
        dp[i][0] = i
        operations[i][0] = 'D' * i if i > 0 else ''
    for j in range(m + 1):
        dp[0][j] = j
        operations[0][j] = 'I' * j if j > 0 else ''
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == predicted[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                operations[i][j] = operations[i - 1][j - 1] + 'C'  # Correct
            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                
                min_val = min(substitution, deletion, insertion)
                dp[i][j] = min_val
                
                if min_val == substitution:
                    operations[i][j] = operations[i - 1][j - 1] + 'S'  # Substitution
                elif min_val == deletion:
                    operations[i][j] = operations[i - 1][j] + 'D'  # Deletion
                else:
                    operations[i][j] = operations[i][j - 1] + 'I'  # Insertion
    
    # Count operations from the operations string
    op_string = operations[n][m]
    substitutions = op_string.count('S')
    deletions = op_string.count('D')
    insertions = op_string.count('I')
    correct = op_string.count('C')
    
    per = (substitutions + deletions + insertions) / len(reference) * 100 if reference else 100
    
    return {
        "per": round(per, 2),
        "method": "phoneme_error_rate",
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "correct": correct,
        "total_reference": len(reference),
        "total_predicted": len(predicted)
    }

# Enhanced word analysis with audio generation
def generate_word_audio_data(words_analysis: List[Dict[str, Any]], reference_text: str) -> Dict[str, Any]:
    """Generate audio for each word and the complete sentence"""
    
    print_banner("GENERATING PRONUNCIATION AUDIO", "=", 80)
    
    audio_data = {
        "word_audio": {},
        "sentence_audio": None,
        "audio_generation_success": False,
        "generated_count": 0,
        "failed_count": 0
    }
    
    timestamp = str(int(time.time()))
    
    # Generate audio for individual words
    for word_data in words_analysis:
        word = word_data["word"]
        status = word_data["status"]
        
        try:
            # Generate audio file path
            audio_filename = f"word_{word}_{timestamp}.wav"
            audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, audio_filename)
            
            # Generate audio for the word
            if audio_generator.generate_word_audio(word, audio_path):
                # Process audio quality if available
                audio_generator.process_audio_quality(audio_path)
                
                # Convert to base64 for API response
                audio_base64 = audio_generator.audio_to_base64(audio_path)
                
                if audio_base64:
                    audio_data["word_audio"][word] = {
                        "audio_base64": audio_base64,
                        "status": status,
                        "filename": audio_filename,
                        "pronunciation_status": status,
                        "needs_practice": status in ["mispronounced", "partial"]
                    }
                    audio_data["generated_count"] += 1
                    
                    print(f"✓ Generated audio for '{word}' ({status})")
                else:
                    audio_data["failed_count"] += 1
                    print(f"✗ Failed to encode audio for '{word}'")
                
                # Clean up temporary file
                try:
                    os.remove(audio_path)
                except:
                    pass
            else:
                audio_data["failed_count"] += 1
                print(f"✗ Failed to generate audio for '{word}'")
                
        except Exception as e:
            logger.error(f"Audio generation error for word '{word}': {e}")
            audio_data["failed_count"] += 1
    
    # Generate audio for complete sentence (reference pronunciation)
    try:
        sentence_filename = f"sentence_{timestamp}.wav"
        sentence_path = os.path.join(AUDIO_OUTPUT_FOLDER, sentence_filename)
        
        if audio_generator.generate_sentence_audio(reference_text, sentence_path, slow=True):
            audio_generator.process_audio_quality(sentence_path)
            sentence_audio_base64 = audio_generator.audio_to_base64(sentence_path)
            
            if sentence_audio_base64:
                audio_data["sentence_audio"] = {
                    "audio_base64": sentence_audio_base64,
                    "text": reference_text,
                    "filename": sentence_filename,
                    "type": "reference_pronunciation"
                }
                print(f"✓ Generated sentence audio: '{reference_text}'")
            
            # Clean up
            try:
                os.remove(sentence_path)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Sentence audio generation error: {e}")
    
    # Set success flag
    audio_data["audio_generation_success"] = audio_data["generated_count"] > 0
    
    print_section("AUDIO GENERATION SUMMARY", {
        "Words processed": len(words_analysis),
        "Audio files generated": audio_data["generated_count"],
        "Generation failures": audio_data["failed_count"],
        "Sentence audio": "Generated" if audio_data["sentence_audio"] else "Failed",
        "Overall success": audio_data["audio_generation_success"]
    })
    
    return audio_data

# Keep existing LLM classes and analysis functions...
# (LLMPronunciationJudge class remains the same)

class LLMPronunciationJudge:
    """Concise LLM-as-a-Judge system for short, focused feedback"""
    
    def __init__(self, preferred_provider="groq"):
        self.preferred_provider = preferred_provider
        self.hf_client = get_hf_client()
        self.groq_client = get_groq_client()
        
    def _call_groq_llm(self, system_message: str, user_message: str) -> Optional[str]:
        """Call Groq LLM for concise feedback"""
        if not self.groq_client:
            logger.warning("Groq client not available")
            return None
            
        try:
            print_section("GROQ API REQUEST (CONCISE)", {
                "Model": LLM_JUDGE_CONFIG["groq"]["model"],
                "Max Tokens": LLM_JUDGE_CONFIG["groq"]["max_tokens"],
                "System Message Length": f"{len(system_message)} characters"
            })
            
            logger.info("Sending concise request to Groq API...")
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=LLM_JUDGE_CONFIG["groq"]["model"],
                temperature=LLM_JUDGE_CONFIG["groq"]["temperature"],
                max_tokens=LLM_JUDGE_CONFIG["groq"]["max_tokens"]
            )
            
            if response.choices and len(response.choices) > 0:
                feedback = response.choices[0].message.content.strip()
                
                print_banner("CONCISE GROQ FEEDBACK", "=", 80)
                print(feedback)
                print_banner("END OF CONCISE FEEDBACK", "=", 80)
                
                logger.info(f"Concise Groq feedback received ({len(feedback)} characters)")
                return feedback
            
            return None
                
        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            return None
    
    def _create_concise_phoneme_comparison(self, reference_phonemes: List[str], 
                                         predicted_phonemes: List[str]) -> str:
        """Create a concise phoneme comparison display"""
        ref_str = " ".join(reference_phonemes) if reference_phonemes else "No reference"
        pred_str = " ".join(predicted_phonemes) if predicted_phonemes else "No prediction"
        
        return f"Expected: {ref_str}\nYour pronunciation: {pred_str}"
    
    def _parse_concise_feedback(self, feedback: str) -> Dict[str, Any]:
        """Parse concise feedback into structured components"""
        result = {
            "assessment": "",
            "key_issues": "",
            "quick_fixes": "",
            "raw_feedback": feedback
        }
        
        try:
            # Extract concise sections
            lines = feedback.strip().split('\n')
            current_section = "assessment"
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if "assessment:" in line.lower():
                    current_section = "assessment"
                    result[current_section] = line.split(":", 1)[1].strip() if ":" in line else ""
                elif "key issues:" in line.lower() or "issues:" in line.lower():
                    current_section = "key_issues"
                    result[current_section] = line.split(":", 1)[1].strip() if ":" in line else ""
                elif "quick fixes:" in line.lower() or "fixes:" in line.lower() or "practice:" in line.lower():
                    current_section = "quick_fixes"
                    result[current_section] = line.split(":", 1)[1].strip() if ":" in line else ""
                else:
                    # Append to current section
                    if result[current_section]:
                        result[current_section] += " " + line
                    else:
                        result[current_section] = line
            
            # If no structured sections found, put everything in assessment
            if not any(result[key] for key in ["assessment", "key_issues", "quick_fixes"]):
                result["assessment"] = feedback.strip()
                
        except Exception as e:
            logger.error(f"Error parsing concise feedback: {e}")
            result["assessment"] = feedback.strip()
        
        return result
    
    def evaluate_pronunciation_with_groq_concise(self, reference_text: str, 
                                               reference_phonemes: List[str],
                                               predicted_phonemes: List[str],
                                               word_analysis: Dict[str, Any], 
                                               per: float, word_accuracy: float) -> Dict[str, Any]:
        """Generate concise pronunciation feedback using Groq"""
        
        print_banner("CONCISE GROQ EVALUATION", "=", 80)
        
        # Create concise system message for short feedback
        system_message = """You are an American English pronunciation coach. Provide concise, actionable feedback in under 150 words.

Format your response exactly like this:
Assessment: [One sentence about overall quality]
Key Issues: [Specific pronunciation problems found]
Quick Fixes: [Brief practice suggestions]

Be encouraging but direct. Recognize when formal speech (like "do not" instead of "don't") is correct. Focus on the most important issues."""
        
        phoneme_comparison = self._create_concise_phoneme_comparison(reference_phonemes, predicted_phonemes)
        
        # Find most problematic words for focused feedback
        problem_words = []
        for word_data in word_analysis.get("words", []):
            if word_data.get("status") in ["mispronounced", "partial"] and len(problem_words) < 3:
                problem_words.append({
                    "word": word_data["word"],
                    "expected": " ".join(word_data.get("reference_phonemes", [])),
                    "actual": " ".join(word_data.get("predicted_phonemes", []))
                })
        
        user_message = f"""Text: "{reference_text}"

{phoneme_comparison}

Metrics: PER={per:.1f}%, Word Accuracy={word_accuracy*100:.1f}%

Problem words: {problem_words if problem_words else "None identified"}

Provide concise feedback in the exact format requested."""
        
        print_section("CONCISE INPUT", {
            "Reference Text": reference_text,
            "PER": f"{per:.1f}%",
            "Problem Words": len(problem_words)
        })
        
        llm_response = self._call_groq_llm(system_message, user_message)
        
        if llm_response:
            parsed_feedback = self._parse_concise_feedback(llm_response)
            
            # Simple scoring based on metrics
            accuracy_score = max(1, min(5, int(5 - (per / 20))))
            fluency_score = max(1, min(5, int(word_accuracy * 5)))
            overall_score = round((accuracy_score + fluency_score) / 2)
            
            result = {
                "provider": "groq",
                "model": LLM_JUDGE_CONFIG["groq"]["model"],
                "overall_score": overall_score,
                "accuracy_score": accuracy_score,
                "fluency_score": fluency_score,
                "concise_feedback": parsed_feedback,
                "raw_response": llm_response,
                "phoneme_display": phoneme_comparison,
                "timestamp": datetime.now().isoformat()
            }
            
            print_banner("CONCISE EVALUATION COMPLETED", "=", 80)
            return result
        
        return self._fallback_assessment(per, word_accuracy, reference_phonemes, predicted_phonemes)
    
    def _fallback_assessment(self, per: float, word_accuracy: float, 
                           reference_phonemes: List[str] = None, 
                           predicted_phonemes: List[str] = None) -> Dict[str, Any]:
        """Concise fallback assessment"""
        
        accuracy_score = max(1, min(5, int(5 - (per / 20))))
        fluency_score = max(1, min(5, int(word_accuracy * 5)))
        overall_score = round((accuracy_score + fluency_score) / 2)
        
        # Generate concise feedback based on scores
        if per <= 15:
            assessment = "Strong pronunciation with minimal errors."
            issues = "Minor phoneme variations detected."
            fixes = "Continue practicing to maintain accuracy."
        elif per <= 30:
            assessment = "Good foundation with some pronunciation issues."
            issues = f"Approximately {int(per)}% of phonemes need attention."
            fixes = "Focus on problem sounds and practice word boundaries."
        else:
            assessment = "Pronunciation needs focused improvement."
            issues = "Multiple phoneme errors affecting clarity."
            fixes = "Break down practice into individual sounds first."
        
        phoneme_display = ""
        if reference_phonemes and predicted_phonemes:
            phoneme_display = self._create_concise_phoneme_comparison(reference_phonemes, predicted_phonemes)
        
        return {
            'provider': 'fallback',
            'model': 'traditional_metrics',
            'overall_score': overall_score,
            'accuracy_score': accuracy_score,
            'fluency_score': fluency_score,
            'concise_feedback': {
                "assessment": assessment,
                "key_issues": issues,
                "quick_fixes": fixes
            },
            'phoneme_display': phoneme_display,
            'method': 'concise_fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_pronunciation(self, reference_text: str, reference_phonemes: List[str],
                             predicted_phonemes: List[str], word_analysis: Dict[str, Any], 
                             per: float, word_accuracy: float) -> Dict[str, Any]:
        """Main concise evaluation method"""
        
        print_banner("CONCISE LLM EVALUATION START", "=", 80)
        
        if self.preferred_provider == "groq" and self.groq_client:
            return self.evaluate_pronunciation_with_groq_concise(
                reference_text, reference_phonemes, predicted_phonemes, 
                word_analysis, per, word_accuracy
            )
        else:
            logger.warning("No LLM providers available, using concise fallback")
            return self._fallback_assessment(per, word_accuracy, reference_phonemes, predicted_phonemes)

# Enhanced word analysis functions
def find_optimal_word_boundaries(predicted_phonemes: List[str], word_phoneme_mapping: List[Dict]) -> List[Tuple[int, int]]:
    """Find optimal word boundaries using dynamic programming"""
    n_pred = len(predicted_phonemes)
    n_words = len(word_phoneme_mapping)
    
    # dp[i][j] = minimum cost to align first i predicted phonemes with first j words
    dp = [[float('inf')] * (n_words + 1) for _ in range(n_pred + 1)]
    parent = [[(-1, -1)] * (n_words + 1) for _ in range(n_pred + 1)]
    
    # Base case: no phonemes, no words
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(n_pred + 1):
        for j in range(n_words + 1):
            if dp[i][j] == float('inf'):
                continue
                
            if j < n_words:  # Can add another word
                ref_phonemes = word_phoneme_mapping[j]['phonemes']
                min_len = max(1, len(ref_phonemes) // 2)
                max_len = min(n_pred - i, len(ref_phonemes) * 3)
                
                # Try different lengths for current word
                for word_len in range(min_len, max_len + 1):
                    if i + word_len <= n_pred:
                        # Calculate alignment cost for this word
                        word_pred = predicted_phonemes[i:i + word_len]
                        aligned_ref, aligned_pred = needleman_wunsch_alignment(ref_phonemes, word_pred)
                        
                        # Cost = number of mismatches
                        cost = sum(1 for r, p in zip(aligned_ref, aligned_pred) if r != p)
                        
                        if dp[i][j] + cost < dp[i + word_len][j + 1]:
                            dp[i + word_len][j + 1] = dp[i][j] + cost
                            parent[i + word_len][j + 1] = (i, j)
    
    # Backtrack to find optimal boundaries
    boundaries = []
    i, j = n_pred, n_words
    
    while i > 0 and j > 0:
        prev_i, prev_j = parent[i][j]
        if prev_j == j - 1:  # This word consumed phonemes from prev_i to i
            boundaries.append((prev_i, i))
        i, j = prev_i, prev_j
    
    boundaries.reverse()
    return boundaries

def generate_word_level_analysis_with_audio(reference_text: str, predicted_phonemes: List[str]) -> Dict[str, Any]:
    """Generate word-level analysis and audio data"""
    try:
        system = get_improved_mdd_system()
        words = reference_text.lower().split()
        word_phoneme_mapping = []
        
        # Build word-phoneme mapping
        for word in words:
            try:
                word_phonemes = system._text_to_phonemes_espeak(word)
                if not word_phonemes or word_phonemes == ['<unk>']:
                    word_phonemes = system._basic_text_to_phonemes(word)
                
                word_phoneme_mapping.append({
                    "word": word,
                    "phonemes": word_phonemes
                })
            except Exception as e:
                word_phoneme_mapping.append({
                    "word": word,
                    "phonemes": [word]
                })
        
        # Find optimal word boundaries
        try:
            boundaries = find_optimal_word_boundaries(predicted_phonemes, word_phoneme_mapping)
        except Exception as e:
            logger.warning(f"Optimal boundary finding failed, using proportional method: {e}")
            # Fallback to proportional method
            boundaries = []
            total_ref_phonemes = sum(len(wd['phonemes']) for wd in word_phoneme_mapping)
            current_pos = 0
            
            for i, word_data in enumerate(word_phoneme_mapping):
                ref_len = len(word_data['phonemes'])
                if i == len(word_phoneme_mapping) - 1:  # Last word
                    end_pos = len(predicted_phonemes)
                else:
                    proportion = ref_len / total_ref_phonemes
                    allocated = max(1, int(len(predicted_phonemes) * proportion))
                    end_pos = min(current_pos + allocated, len(predicted_phonemes))
                
                boundaries.append((current_pos, end_pos))
                current_pos = end_pos
        
        # Analyze each word using optimal boundaries
        words_analysis = []
        global_substitutions = 0
        global_insertions = 0
        global_deletions = 0
        global_correct = 0
        
        for i, (word_data, (start_idx, end_idx)) in enumerate(zip(word_phoneme_mapping, boundaries)):
            word = word_data['word']
            ref_phonemes = word_data['phonemes']
            pred_phonemes = predicted_phonemes[start_idx:end_idx]
            
            # Align this specific word
            if pred_phonemes:
                aligned_ref, aligned_pred = needleman_wunsch_alignment(ref_phonemes, pred_phonemes)
            else:
                aligned_ref = ref_phonemes
                aligned_pred = ["_"] * len(ref_phonemes)
            
            # Calculate errors for this word alignment
            word_substitutions = 0
            word_insertions = 0
            word_deletions = 0
            word_correct = 0
            errors = []
            
            for j, (ref_p, pred_p) in enumerate(zip(aligned_ref, aligned_pred)):
                if ref_p == "_":
                    word_insertions += 1
                    global_insertions += 1
                    errors.append({
                        "position": j,
                        "type": "insertion",
                        "predicted": pred_p,
                        "expected": None
                    })
                elif pred_p == "_":
                    word_deletions += 1
                    global_deletions += 1
                    errors.append({
                        "position": j,
                        "type": "deletion",
                        "predicted": None,
                        "expected": ref_p
                    })
                elif ref_p != pred_p:
                    word_substitutions += 1
                    global_substitutions += 1
                    errors.append({
                        "position": j,
                        "type": "substitution",
                        "predicted": pred_p,
                        "expected": ref_p
                    })
                else:
                    word_correct += 1
                    global_correct += 1
            
            # Calculate word-level PER
            total_ref_phonemes = len([p for p in aligned_ref if p != "_"])
            word_per = ((word_substitutions + word_deletions + word_insertions) / max(1, total_ref_phonemes)) * 100
            
            # Determine word status
            if word_per == 0:
                status = "correct"
            elif word_per <= 25:
                status = "partial"
            else:
                status = "mispronounced"
            
            words_analysis.append({
                "word": word,
                "reference_phonemes": ref_phonemes,
                "predicted_phonemes": pred_phonemes,
                "aligned_reference": aligned_ref,
                "aligned_predicted": aligned_pred,
                "per": {"per": round(word_per, 2), "method": "optimal_boundary_alignment"},
                "status": status,
                "phoneme_errors": errors,
                "error_count": len(errors),
                "error_counts": {
                    "substitutions": word_substitutions,
                    "insertions": word_insertions,
                    "deletions": word_deletions,
                    "correct": word_correct
                }
            })
        
        # Generate audio data for all words
        audio_data = generate_word_audio_data(words_analysis, reference_text)
        
        # Calculate overall statistics
        correct_words = sum(1 for w in words_analysis if w["status"] == "correct")
        partial_words = sum(1 for w in words_analysis if w["status"] == "partial")
        mispronounced_words = sum(1 for w in words_analysis if w["status"] == "mispronounced")
        
        total_per = sum(w["per"]["per"] for w in words_analysis)
        avg_per = total_per / max(1, len(words_analysis))
        
        word_accuracy = (correct_words + (partial_words * 0.5)) / max(1, len(words_analysis))
        
        # Calculate global PER
        total_reference_phonemes = sum(len(word_data['phonemes']) for word_data in word_phoneme_mapping)
        global_per = ((global_substitutions + global_deletions + global_insertions) / max(1, total_reference_phonemes)) * 100
        
        return {
            "words": words_analysis,
            "audio_data": audio_data,
            "summary": {
                "total_words": len(words_analysis),
                "correct_words": correct_words,
                "partial_words": partial_words,
                "mispronounced_words": mispronounced_words,
                "error_words": 0,
                "word_accuracy_percent": round(word_accuracy * 100, 2),
                "partial_credit_percent": round(word_accuracy * 100, 2),
                "average_per": round(avg_per, 2),
                "global_per": round(global_per, 2),
                "overall_status": "good" if word_accuracy >= 0.8 else "needs_improvement" if word_accuracy >= 0.6 else "poor"
            }
        }
        
    except Exception as e:
        logger.error(f"Word analysis with audio failed: {e}")
        return {"error": str(e)}


def create_concise_analysis_with_llm(reference_text: str, raw_results: Dict[str, Any], 
                                    word_level: Dict[str, Any], 
                                    filter_extraneous: bool = True) -> Dict[str, Any]:
    """Create concise analysis combining traditional metrics with LLM evaluation"""
    
    print_banner("CREATING CONCISE ANALYSIS WITH AUDIO", "=", 80)
    
    # Extract basic data
    predicted_phonemes_raw = raw_results.get("analysis", {}).get("predicted_phonemes", [])
    reference_phonemes = raw_results.get("analysis", {}).get("reference_phonemes", [])
    
    # Apply phoneme filtering if enabled
    if filter_extraneous:
        predicted_phonemes, filtering_stats = improved_intelligent_phoneme_cleaning(
            predicted_phonemes_raw, 
            reference_phonemes
        )
    else:
        predicted_phonemes = predicted_phonemes_raw
        filtering_stats = {
            'original_count': len(predicted_phonemes_raw),
            'filtered_count': len(predicted_phonemes_raw),
            'removed_count': 0,
            'filtering_applied': False
        }
    
    # Calculate traditional metrics with filtered phonemes
    per_result = calculate_phoneme_error_rate(predicted_phonemes, reference_phonemes)
    
    # Get word-level statistics and audio data
    word_summary = word_level.get("summary", {})
    word_accuracy = word_summary.get("word_accuracy_percent", 0) / 100
    audio_data = word_level.get("audio_data", {})
    
    print_section("METRICS FOR CONCISE FEEDBACK", {
        "Original Phonemes": filtering_stats['original_count'],
        "Filtered Phonemes": filtering_stats['filtered_count'],
        "Removed Phonemes": filtering_stats['removed_count'],
        "PER (filtered)": f"{per_result['per']:.1f}%",
        "Word Accuracy": f"{word_accuracy * 100:.1f}%",
        "Audio Files Generated": audio_data.get("generated_count", 0)
    })
    
    # Initialize LLM judge
    llm_judge = LLMPronunciationJudge()
    
    # Get LLM assessment with filtered phoneme comparison
    llm_assessment = llm_judge.evaluate_pronunciation(
        reference_text=reference_text,
        reference_phonemes=reference_phonemes,
        predicted_phonemes=predicted_phonemes,
        word_analysis=word_level,
        per=per_result["per"],
        word_accuracy=word_accuracy
    )
    
    # Create mispronunciations array
    mispronunciations = []
    phoneme_position = 0
    
    for word_data in word_level.get("words", []):
        word_phonemes = word_data.get("reference_phonemes", [])
        
        for error in word_data.get("phoneme_errors", []):
            error_with_position = error.copy()
            error_with_position["absolute_position"] = phoneme_position + error["position"]
            error_with_position["word"] = word_data["word"]
            mispronunciations.append(error_with_position)
        
        phoneme_position += len(word_phonemes)
    
    # USE WORD-LEVEL ACCURACY instead of phoneme-level PER
    # Word accuracy is more meaningful for user feedback
    accuracy = word_accuracy  # This is already 0-1 scale (0.70 for 70%)
    
    final_analysis = {
        "accuracy": accuracy,  # Now correctly shows 0.70 (70%)
        "per": per_result["per"],
        "word_accuracy": word_accuracy,
        "correct_phonemes": per_result["correct"],
        "total_phonemes": len(reference_phonemes),
        "predicted_phonemes": predicted_phonemes,
        "predicted_phonemes_raw": predicted_phonemes_raw,
        "reference_phonemes": reference_phonemes,
        "mispronunciations": mispronunciations,
        "word_summary": word_summary,
        "llm_assessment": llm_assessment,
        "audio_data": audio_data,
        "filtering_stats": filtering_stats,
        "evaluation_method": "concise_traditional_plus_llm_with_audio_filtered"
    }
    
    print_section("CONCISE ANALYSIS WITH FILTERING COMPLETED", {
        "Traditional Accuracy (filtered)": f"{accuracy * 100:.1f}%",  # Now shows 70%
        "LLM Overall Score": f"{llm_assessment.get('overall_score', 'N/A')}/5",
        "LLM Provider": llm_assessment.get('provider', 'fallback'),
        "Total Mispronunciations": len(mispronunciations),
        "Extraneous Sounds Removed": filtering_stats['removed_count'],
        "Word Audio Files": len(audio_data.get("word_audio", {}))
    })
    
    return final_analysis

def generate_concise_feedback_with_audio(analysis_result: Dict[str, Any], word_level: Dict[str, Any]) -> str:
    """Generate concise feedback that includes phoneme display and audio availability info"""
    
    print_banner("GENERATING CONCISE FEEDBACK WITH AUDIO INFO", "=", 80)
    
    # Get LLM assessment and audio data
    llm_assessment = analysis_result.get("llm_assessment", {})
    concise_feedback = llm_assessment.get("concise_feedback", {})
    phoneme_display = llm_assessment.get("phoneme_display", "")
    audio_data = analysis_result.get("audio_data", {})
    
    # Traditional metrics for fallback
    per = analysis_result["per"]
    word_accuracy = analysis_result["word_accuracy"] * 100
    
    # Build the concise feedback
    feedback_parts = []
    
    # Reference text and phoneme comparison
    reference_text = analysis_result.get("metadata", {}).get("reference_text", "")
    if reference_text:
        feedback_parts.append(f'Text: "{reference_text}"')
    
    # Phoneme display
    if phoneme_display:
        feedback_parts.append(f"\n{phoneme_display}")
    
    # Audio availability info
    word_audio_count = len(audio_data.get("word_audio", {}))
    if word_audio_count > 0:
        sentence_audio = "Available" if audio_data.get("sentence_audio") else "Not available"
        feedback_parts.append(f"\nAudio: {word_audio_count} word pronunciations + sentence audio {sentence_audio.lower()}")
    
    # LLM Assessment sections
    if concise_feedback.get("assessment"):
        feedback_parts.append(f"\nAssessment: {concise_feedback['assessment']}")
    
    if concise_feedback.get("key_issues"):
        feedback_parts.append(f"Key Issues: {concise_feedback['key_issues']}")
    
    if concise_feedback.get("quick_fixes"):
        feedback_parts.append(f"Quick Fixes: {concise_feedback['quick_fixes']}")
    
    # Always include metrics at the end
    feedback_parts.append(f"Accuracy: {100 - per:.1f}% | Word Score: {word_accuracy:.1f}%")
    
    # Fallback if no LLM feedback
    if not any(concise_feedback.values()):
        if per <= 20:
            assessment = "Good pronunciation with minor issues."
        elif per <= 40:
            assessment = "Fair pronunciation needing focused practice."
        else:
            assessment = "Significant pronunciation improvements needed."
        
        audio_info = f"Audio: {word_audio_count} word pronunciations available" if word_audio_count > 0 else ""
        
        feedback_parts = [
            f'Text: "{reference_text}"',
            phoneme_display if phoneme_display else "",
            audio_info,
            f"Assessment: {assessment}",
            f"Accuracy: {100 - per:.1f}% | Word Score: {word_accuracy:.1f}%"
        ]
    
    final_feedback = "\n".join(filter(None, feedback_parts))
    
    print_section("CONCISE FEEDBACK WITH AUDIO GENERATED", {
        "Total Length": f"{len(final_feedback)} characters",
        "Lines": len(final_feedback.split('\n')),
        "LLM Provider": llm_assessment.get('provider', 'fallback'),
        "Audio Files Referenced": word_audio_count
    })
    
    return final_feedback

import whisper

# Global Whisper model
whisper_model = None

def get_whisper_model():
    """Initialize Whisper model for validation"""
    global whisper_model
    if whisper_model is None:
        try:
            whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded for validation")
        except Exception as e:
            logger.warning(f"Could not load Whisper model: {e}")
            whisper_model = None
    return whisper_model

def validate_phoneme_extraction(audio_path: str, reference_text: str, predicted_phonemes: List[str], reference_phonemes: List[str]) -> Dict[str, Any]:
    """
    Compare wav2vec2 phoneme predictions with Whisper transcription
    Prints comparison on backend console
    """
    print("\n" + "="*80)
    print("PHONEME EXTRACTION VALIDATION - WAV2VEC2 vs WHISPER (ALM)")
    print("="*80)
    
    try:
        # Display wav2vec2 results
        print("\n[1] WAV2VEC2 PHONEME EXTRACTION RESULTS:")
        print(f"    Predicted phonemes: {predicted_phonemes}")
        print(f"    Reference phonemes: {reference_phonemes}")
        print(f"    Phonemes match: {predicted_phonemes == reference_phonemes}")
        
        # Get Whisper transcription (ground truth)
        print("\n[2] WHISPER (ALM) TRANSCRIPTION:")
        whisper = get_whisper_model()
        
        if whisper is None:
            print("    ⚠️  Whisper model not available - skipping validation")
            return {
                "validation_skipped": True,
                "reason": "Whisper model not loaded"
            }
        
        whisper_result = whisper.transcribe(audio_path)
        whisper_text = whisper_result["text"].strip()
        
        print(f"    Whisper heard: '{whisper_text}'")
        print(f"    Reference text: '{reference_text}'")
        
        # Compare results
        phonemes_identical = predicted_phonemes == reference_phonemes
        text_matches = whisper_text.lower().strip() == reference_text.lower().strip()
        
        print("\n" + "-"*80)
        print("VALIDATION COMPARISON:")
        print("-"*80)
        print(f"✓ wav2vec2 predicted == reference: {phonemes_identical}")
        print(f"✓ Whisper heard == reference: {text_matches}")
        
        # Detect the bug
        bug_detected = False
        if phonemes_identical and not text_matches:
            print("\n" + "!"*80)
            print("⚠️  CRITICAL BUG DETECTED IN WAV2VEC2 SYSTEM!")
            print("!"*80)
            print(f"Expected:      '{reference_text}'")
            print(f"Whisper heard: '{whisper_text}'")
            print(f"wav2vec2 returned REFERENCE phonemes: {predicted_phonemes}")
            print("\nThis indicates wav2vec2 is NOT analyzing the actual audio.")
            print("It's copying reference phonemes instead of extracting from audio signal.")
            print("!"*80)
            bug_detected = True
            bug_status = "CRITICAL_BUG"
        elif phonemes_identical and text_matches:
            print("\n✓ PERFECT PRONUNCIATION")
            print("  Both models confirm audio matches reference perfectly.")
            bug_status = "OK"
        else:
            print("\n✓ SYSTEM WORKING CORRECTLY")
            print("  wav2vec2 detected pronunciation differences.")
            bug_status = "OK"
        
        print("="*80 + "\n")
        
        return {
            "wav2vec2_predicted": predicted_phonemes,
            "wav2vec2_reference": reference_phonemes,
            "whisper_transcription": whisper_text,
            "reference_text": reference_text,
            "phonemes_identical": phonemes_identical,
            "transcription_matches": text_matches,
            "bug_detected": bug_detected,
            "bug_status": bug_status
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ Validation error: {e}")
        print("="*80 + "\n")
        return {
            "validation_error": str(e),
            "bug_status": "UNKNOWN"
        }

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Whisper"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "Audio file required"}), 400

        file = request.files['audio_file']
        
        # Save temporarily
        filename = secure_filename(file.filename)
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        file.save(temp_audio_path)

        # Transcribe with Whisper
        whisper = get_whisper_model()
        if whisper is None:
            return jsonify({"success": False, "error": "Whisper model not available"}), 500

        result = whisper.transcribe(temp_audio_path)
        transcription = result["text"].strip()

        # Cleanup
        os.remove(temp_audio_path)

        return jsonify({
            "success": True,
            "transcription": transcription
        })

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
# In your Flask app, update the analysis endpoint to include word-level phoneme data
@app.route('/analyze_with_llm_judge', methods=['POST'])
def analyze_pronunciation_with_llm():
    """Pronunciation analysis with Whisper validation"""
    start_time = time.time()
    
    print_banner("PRONUNCIATION ANALYSIS - WAV2VEC2", "=", 80)
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "Audio file required"}), 400

        file = request.files['audio_file']
        reference_text = request.form.get('reference_text', '').strip()
        use_llm_judge = request.form.get('use_llm_judge', 'true').lower() == 'true'
        generate_audio = request.form.get('generate_audio', 'true').lower() == 'true'
        intelligent_cleaning = request.form.get('intelligent_cleaning', 'true').lower() == 'true'
        skip_validation = request.form.get('skip_validation', 'false').lower() == 'true'
        
        print_section("REQUEST PARAMETERS", {
            "Reference Text": reference_text,
            "Audio File": file.filename,
            "Use LLM Judge": use_llm_judge,
            "Generate Audio": generate_audio,
            "Intelligent Cleaning": intelligent_cleaning,
            "Skip Validation": skip_validation,
            "File Size": f"{len(file.read())} bytes"
        })
        file.seek(0)
        
        if not reference_text:
            return jsonify({"success": False, "error": "Reference text required"}), 400

        filename = secure_filename(file.filename)
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        file.save(temp_audio_path)

        # Extract phonemes
        phoneme_results = extract_phonemes_wav2vec2(temp_audio_path, reference_text)
        
        if not phoneme_results["success"]:
            raise Exception(f"Phoneme extraction failed: {phoneme_results.get('error', 'Unknown error')}")
        
        predicted_phonemes_raw = phoneme_results["predicted_phonemes"]
        reference_phonemes = phoneme_results["reference_phonemes"]
        
        # *** VALIDATION STEP - Compare with Whisper (ALM) ***
        validation_result = None
        if not skip_validation:
            validation_result = validate_phoneme_extraction(
                temp_audio_path, 
                reference_text,
                predicted_phonemes_raw,
                reference_phonemes
            )
            
            # If critical bug detected, stop processing
            if validation_result.get("bug_status") == "CRITICAL_BUG":
                os.remove(temp_audio_path)
                return jsonify({
                    "success": False,
                    "error": "Critical system malfunction detected",
                    "validation": validation_result,
                    "details": {
                        "issue": "wav2vec2 returned reference phonemes instead of analyzing audio",
                        "expected": reference_text,
                        "whisper_heard": validation_result.get("whisper_transcription"),
                        "wav2vec2_output": predicted_phonemes_raw,
                        "recommendation": "Fix improved_dynamic_mdd.py - it's not analyzing audio correctly"
                    }
                }), 500
        
        # Apply ENHANCED intelligent cleaning
        print_section("ENHANCED INTELLIGENT PHONEME CLEANING", "Processing...")
        if intelligent_cleaning:
            predicted_phonemes_clean, cleaning_stats = enhanced_clean_predicted_phonemes(
                predicted_phonemes_raw, 
                reference_phonemes,
                reference_text
            )
            
            print_section("ENHANCED CLEANING RESULTS", {
                "Original": len(predicted_phonemes_raw),
                "Cleaned": len(predicted_phonemes_clean),
                "Words Preserved": cleaning_stats.get('words_preserved', 0),
                "Insertions Removed": cleaning_stats.get('insertions_removed', 0),
                "Method": cleaning_stats.get('method', 'enhanced'),
                "Match Score": f"{cleaning_stats.get('match_score', 0):.1%}" if cleaning_stats.get('match_score') else "N/A"
            })
        else:
            predicted_phonemes_clean = predicted_phonemes_raw
            cleaning_stats = {'method': 'disabled'}

        # Perform segmentation and analysis - THIS IS THE KEY CHANGE
        print_banner("ENHANCED PHONEME SEGMENTATION", "=", 80)
        segmentation_result = enhanced_universal_phoneme_segmentation(predicted_phonemes_clean, reference_text)
        words_analysis = enhanced_analyze_segmented_words(segmentation_result)

        # Add debug logging
        debug_phoneme_mapping(reference_text, predicted_phonemes_clean, segmentation_result)

        # Extract word lists based on status
        mispronounced_words = []
        correctly_pronounced_words = []
        partial_words = []

        print("\n=== WORD STATUS DEBUG ===")
        for word_data in words_analysis:
            word = word_data["word"]
            status = word_data["status"]
            
            # Debug logging
            print(f"{word}: {status} (PER: {word_data['per']['per']:.1f}%, Match: {word_data.get('match_quality', 0):.1%})")
            
            if status == "correct":
                correctly_pronounced_words.append(word)
            elif status == "mispronounced":
                mispronounced_words.append(word)
            elif status == "partial":
                partial_words.append(word)
        
        print(f"\nCorrect (GREEN): {correctly_pronounced_words}")
        print(f"Partial (YELLOW): {partial_words}")
        print(f"Mispronounced (RED): {mispronounced_words}")
        print("========================\n")
        
        # Generate audio data
        audio_data = generate_word_audio_data(words_analysis, reference_text)

        # Create word level results - ENHANCED WITH WORD-LEVEL PHONEMES
        word_level_results = {
            "words": words_analysis,
            "audio_data": audio_data,
            "summary": {
                "total_words": len(words_analysis),
                "correct_words": len(correctly_pronounced_words),
                "partial_words": len(partial_words),
                "mispronounced_words": len(mispronounced_words),
                "word_accuracy_percent": round(
                    (len(correctly_pronounced_words) + len(partial_words) * 0.5) / 
                    max(1, len(words_analysis)) * 100, 2
                )
            },
            # ADD THIS: Word-level phoneme mapping for frontend table
            "word_phoneme_mapping": [
                {
                    "word": word_data["word"],
                    "reference_phonemes": word_data["reference_phonemes"],
                    "predicted_phonemes": word_data["predicted_phonemes"],
                    "aligned_reference": word_data["aligned_reference"],
                    "aligned_predicted": word_data["aligned_predicted"],
                    "status": word_data["status"]
                }
                for word_data in words_analysis
            ]
        }
        
        # Create raw results
        raw_results = {
            "analysis": {
                "predicted_phonemes": predicted_phonemes_clean,
                "reference_phonemes": reference_phonemes
            }
        }
        
        # Generate analysis and feedback
        if use_llm_judge:
            frontend_analysis = create_concise_analysis_with_llm(
                reference_text, raw_results, word_level_results, filter_extraneous=False
            )
            feedback = generate_concise_feedback_with_audio(frontend_analysis, word_level_results)
        else:
            frontend_analysis = create_traditional_analysis_with_audio(
                reference_text, raw_results, word_level_results
            )
            feedback = generate_traditional_feedback_with_audio(frontend_analysis, word_level_results)

        processing_time = time.time() - start_time
        
        # Clean up temporary file
        try:
            os.remove(temp_audio_path)
        except:
            pass

        # Build response - ENHANCED WITH WORD-LEVEL PHONEMES
        response = {
            "success": True,
            "analysis": frontend_analysis,
            "word_level_analysis": word_level_results,
            "feedback": feedback,
            "cleaning_info": {
                "applied": intelligent_cleaning,
                "stats": cleaning_stats,
                "original_phonemes": predicted_phonemes_raw,
                "cleaned_phonemes": predicted_phonemes_clean
            },
            "validation": validation_result,  # Include validation results
            "metadata": {
                "reference_text": reference_text,
                "processing_time": round(processing_time, 3),
                "phoneme_model": "wav2vec2",
                "llm_judge_enabled": use_llm_judge,
                "audio_generation_enabled": generate_audio,
                "intelligent_cleaning_enabled": intelligent_cleaning,
                "validation_enabled": not skip_validation,
                "word_status_summary": {
                    "correct": correctly_pronounced_words,
                    "partial": partial_words,
                    "mispronounced": mispronounced_words
                }
            },
            # ADD THIS: Enhanced word phoneme data for table display
            "word_phonemes": {
                "reference_text": reference_text,
                "words": [
                    {
                        "word": word_data["word"],
                        "reference_phonemes": word_data["reference_phonemes"],
                        "predicted_phonemes": word_data["predicted_phonemes"],
                        "aligned_reference": word_data["aligned_reference"],
                        "aligned_predicted": word_data["aligned_predicted"],
                        "status": word_data["status"],
                        "phoneme_errors": word_data["phoneme_errors"],
                        "per": word_data["per"]["per"]
                    }
                    for word_data in words_analysis
                ]
            }
        }
        
        print_banner("ANALYSIS COMPLETE", "=", 80)
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    
def detect_and_handle_repetitions(predicted_phonemes: List[str], reference_phonemes: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Detect and handle repeated phrases by extracting the best single instance
    """
    stats = {
        'repetition_detected': False,
        'original_length': len(predicted_phonemes),
        'repetitions_found': 0,
        'method_used': 'none'
    }
    
    # If predicted is much longer than reference, look for repetitions
    if len(predicted_phonemes) <= len(reference_phonemes) * 1.5:
        return predicted_phonemes, stats
    
    # Method: Look for the best matching segment
    ref_len = len(reference_phonemes)
    best_score = -1
    best_instance = predicted_phonemes[:ref_len]  # Default to first segment
    
    # Try different starting positions
    for start in range(0, min(len(predicted_phonemes) - ref_len + 1, 10)):  # Limit search for efficiency
        candidate = predicted_phonemes[start:start + ref_len]
        
        # Calculate alignment score
        aligned_ref, aligned_cand = needleman_wunsch_alignment(reference_phonemes, candidate)
        matches = sum(1 for r, c in zip(aligned_ref, aligned_cand) if r == c)
        score = matches / len(reference_phonemes)
        
        if score > best_score:
            best_score = score
            best_instance = candidate
            stats['best_start'] = start
    
    if best_score > 0.3:  # Reasonable match
        stats['repetition_detected'] = True
        stats['repetitions_found'] = len(predicted_phonemes) // ref_len
        stats['best_instance_score'] = round(best_score, 2)
        stats['method_used'] = 'best_segment_match'
        
        print_section("REPETITION DETECTION", {
            "Original Length": len(predicted_phonemes),
            "Reference Length": len(reference_phonemes),
            "Repetitions Found": stats['repetitions_found'],
            "Best Instance Score": f"{best_score:.1%}",
            "Selected Instance": f"positions {stats.get('best_start', 0)}-{stats.get('best_start', 0) + ref_len}"
        })
        
        return best_instance, stats
    
    return predicted_phonemes[:ref_len], stats  # Fallback: take first ref_len phonemes


def universal_phoneme_segmentation(predicted_phonemes: List[str], reference_text: str) -> List[Dict]:
    """
    Improved phoneme segmentation that properly maps predicted phonemes to words
    """
    system = get_improved_mdd_system()
    words = reference_text.lower().split()
    
    # Get reference phonemes for each word
    word_ref_phonemes = []
    
    for word in words:
        try:
            # Clean word for phoneme conversion
            clean_word = word.replace("'", "").replace(",", "").replace(".", "")
            ref_phonemes = system._text_to_phonemes_espeak(clean_word)
            if not ref_phonemes or ref_phonemes == ['<unk>']:
                ref_phonemes = system._basic_text_to_phonemes(clean_word)
            
            word_ref_phonemes.append({
                "word": word,
                "ref_phonemes": ref_phonemes,
                "length": len(ref_phonemes)
            })
        except Exception as e:
            # Fallback: use simple phoneme mapping
            word_ref_phonemes.append({
                "word": word,
                "ref_phonemes": [word],
                "length": 1
            })
    
    # Calculate total reference phonemes
    total_ref_phonemes = sum(len(w['ref_phonemes']) for w in word_ref_phonemes)
    
    if not predicted_phonemes or total_ref_phonemes == 0:
        return proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)
    
    # Use improved boundary detection
    boundaries = improved_boundary_detection(predicted_phonemes, word_ref_phonemes)
    
    return boundaries

def improved_boundary_detection(predicted_phonemes: List[str], word_ref_phonemes: List[Dict]) -> List[Dict]:
    """
    Improved boundary detection using phonetic similarity
    """
    n_pred = len(predicted_phonemes)
    n_words = len(word_ref_phonemes)
    
    # DP table for optimal alignment
    dp = [[float('inf')] * (n_words + 1) for _ in range(n_pred + 1)]
    path = [[None] * (n_words + 1) for _ in range(n_pred + 1)]
    
    dp[0][0] = 0
    
    for i in range(n_pred + 1):
        for j in range(n_words + 1):
            if dp[i][j] == float('inf'):
                continue
                
            # Try to assign phonemes to current word
            if j < n_words:
                current_word = word_ref_phonemes[j]
                ref_phonemes = current_word["ref_phonemes"]
                ref_len = len(ref_phonemes)
                
                # Try different segment lengths
                for seg_len in range(1, min(n_pred - i, ref_len * 2) + 1):
                    if i + seg_len > n_pred:
                        continue
                        
                    segment = predicted_phonemes[i:i + seg_len]
                    
                    # Calculate alignment cost
                    aligned_ref, aligned_seg = needleman_wunsch_alignment(ref_phonemes, segment)
                    cost = calculate_alignment_cost(aligned_ref, aligned_seg)
                    
                    # Add length penalty
                    length_penalty = abs(seg_len - ref_len) * 0.1
                    total_cost = dp[i][j] + cost + length_penalty
                    
                    if total_cost < dp[i + seg_len][j + 1]:
                        dp[i + seg_len][j + 1] = total_cost
                        path[i + seg_len][j + 1] = (i, j, seg_len)
    
    # Backtrack to find optimal boundaries
    boundaries = []
    i, j = n_pred, n_words
    
    while i > 0 and j > 0:
        if path[i][j] is None:
            # Fallback: use proportional segmentation
            return proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)
        
        prev_i, prev_j, seg_len = path[i][j]
        current_word = word_ref_phonemes[j - 1]
        
        boundaries.append({
            "word": current_word["word"],
            "start": prev_i,
            "end": i,
            "predicted_phonemes": predicted_phonemes[prev_i:i],
            "reference_phonemes": current_word["ref_phonemes"],
            "alignment_cost": dp[i][j] - dp[prev_i][prev_j],
            "method": "improved_dp"
        })
        
        i, j = prev_i, prev_j
    
    boundaries.reverse()
    return boundaries

def calculate_alignment_cost(aligned_ref: List[str], aligned_pred: List[str]) -> float:
    """Calculate cost based on phoneme alignment"""
    if not aligned_ref or not aligned_pred:
        return float('inf')
    
    cost = 0
    for ref_ph, pred_ph in zip(aligned_ref, aligned_pred):
        if ref_ph == "_" and pred_ph != "_":
            cost += 1.0  # Insertion
        elif pred_ph == "_" and ref_ph != "_":
            cost += 1.0  # Deletion  
        elif ref_ph != pred_ph:
            cost += 0.7  # Substitution (less than insertion/deletion)
        else:
            cost += 0.0  # Match
    
    return cost

def proportional_segmentation_fallback(predicted_phonemes: List[str], word_ref_phonemes: List[Dict]) -> List[Dict]:
    """Fallback segmentation using proportional allocation - returns list of dicts"""
    boundaries = []
    
    # Calculate total reference length from available keys
    total_ref_len = 0
    for w in word_ref_phonemes:
        if "length" in w:
            total_ref_len += w["length"]
        elif "phonemes" in w:
            total_ref_len += len(w["phonemes"])
        elif "ref_phonemes" in w:
            total_ref_len += len(w["ref_phonemes"])
        else:
            total_ref_len += 1  # Default
    
    if total_ref_len == 0:
        total_ref_len = 1
    
    current_pos = 0
    
    for i, word_data in enumerate(word_ref_phonemes):
        # Get reference length
        if "length" in word_data:
            ref_len = word_data["length"]
        elif "phonemes" in word_data:
            ref_len = len(word_data["phonemes"])
        elif "ref_phonemes" in word_data:
            ref_len = len(word_data["ref_phonemes"])
        else:
            ref_len = 1
        
        # Get reference phonemes
        if "phonemes" in word_data:
            ref_phonemes = word_data["phonemes"]
        elif "ref_phonemes" in word_data:
            ref_phonemes = word_data["ref_phonemes"]
        else:
            ref_phonemes = [word_data.get("word", "?")]
        
        # Calculate end position
        if i == len(word_ref_phonemes) - 1:
            # Last word gets remaining phonemes
            end_pos = len(predicted_phonemes)
        else:
            proportion = ref_len / total_ref_len
            allocated = max(1, int(len(predicted_phonemes) * proportion))
            end_pos = min(current_pos + allocated, len(predicted_phonemes))
        
        segment = predicted_phonemes[current_pos:end_pos]
        boundaries.append({
            "word": word_data.get("word", "unknown"),
            "start": current_pos,
            "end": end_pos,
            "predicted_phonemes": segment,
            "reference_phonemes": ref_phonemes,
            "alignment_cost": float('inf'),
            "method": "proportional_fallback"
        })
        current_pos = end_pos
    
    return boundaries

def analyze_segmented_words(segmentation_result: List[Dict]) -> List[Dict[str, Any]]:
    """Enhanced word analysis with better phoneme mapping"""
    words_analysis = []
    
    for seg_data in segmentation_result:
        word = seg_data["word"]
        pred_phonemes = seg_data["predicted_phonemes"]
        ref_phonemes = seg_data["reference_phonemes"]
        
        # Skip if no reference phonemes
        if not ref_phonemes:
            words_analysis.append(create_empty_word_analysis(word))
            continue
        
        # Use Needleman-Wunsch for optimal alignment
        aligned_ref, aligned_pred = needleman_wunsch_alignment(ref_phonemes, pred_phonemes)
        
        # Calculate detailed metrics
        analysis = calculate_word_analysis_metrics(word, ref_phonemes, pred_phonemes, aligned_ref, aligned_pred)
        words_analysis.append(analysis)
    
    return words_analysis

def calculate_word_analysis_metrics(word: str, ref_phonemes: List[str], pred_phonemes: List[str], 
                                  aligned_ref: List[str], aligned_pred: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive word analysis metrics"""
    
    # Count matches and errors
    correct_matches = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    errors = []
    
    for j, (ref_p, pred_p) in enumerate(zip(aligned_ref, aligned_pred)):
        if ref_p == "_" and pred_p != "_":
            insertions += 1
            errors.append({
                "position": j,
                "type": "insertion",
                "predicted": pred_p,
                "expected": None
            })
        elif pred_p == "_" and ref_p != "_":
            deletions += 1
            errors.append({
                "position": j,
                "type": "deletion",
                "predicted": None,
                "expected": ref_p
            })
        elif ref_p != pred_p and ref_p != "_" and pred_p != "_":
            substitutions += 1
            errors.append({
                "position": j,
                "type": "substitution",
                "predicted": pred_p,
                "expected": ref_p
            })
        elif ref_p == pred_p:
            correct_matches += 1
    
    # Calculate accuracy metrics
    total_ref = len([r for r in aligned_ref if r != "_"])
    if total_ref == 0:
        word_per = 100.0
        word_accuracy = 0.0
    else:
        word_per = ((substitutions + deletions + insertions) / total_ref) * 100
        word_accuracy = (correct_matches / total_ref) * 100
    
    # Determine word status with improved thresholds
    if word_accuracy >= 85:
        status = "correct"
    elif word_accuracy >= 50:
        status = "partial"
    else:
        status = "mispronounced"
    
    return {
        "word": word,
        "reference_phonemes": ref_phonemes,
        "predicted_phonemes": pred_phonemes,
        "aligned_reference": aligned_ref,
        "aligned_predicted": aligned_pred,
        "per": {"per": round(word_per, 2), "method": "enhanced_alignment"},
        "accuracy": round(word_accuracy, 2),
        "status": status,
        "phoneme_errors": errors,
        "error_count": len(errors),
        "error_counts": {
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "correct": correct_matches
        },
        "match_quality": round(word_accuracy / 100, 3)
    }

def create_empty_word_analysis(word: str) -> Dict[str, Any]:
    """Create analysis for words with no reference phonemes"""
    return {
        "word": word,
        "reference_phonemes": [],
        "predicted_phonemes": [],
        "aligned_reference": [],
        "aligned_predicted": [],
        "per": {"per": 100.0, "method": "no_reference"},
        "accuracy": 0.0,
        "status": "mispronounced",
        "phoneme_errors": [],
        "error_count": 0,
        "error_counts": {
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "correct": 0
        },
        "match_quality": 0.0
    }

def improved_intelligent_phoneme_cleaning(predicted_phonemes: List[str], 
                                        reference_phonemes: List[str],
                                        reference_text: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Enhanced phoneme cleaning that preserves correct word phonemes
    while removing only unwanted insertions between words
    """
    stats = {
        'original_count': len(predicted_phonemes),
        'method': 'enhanced_word_preservation',
        'words_preserved': 0,
        'insertions_removed': 0,
        'preserved_phonemes': []
    }
    
    # Step 1: Get word-level reference phonemes
    system = get_improved_mdd_system()
    words = reference_text.lower().split()
    word_ref_phonemes = []
    
    for word in words:
        try:
            word_phonemes = system._text_to_phonemes_espeak(word)
            if not word_phonemes or word_phonemes == ['<unk>']:
                word_phonemes = system._basic_text_to_phonemes(word)
            word_ref_phonemes.append({
                "word": word,
                "phonemes": word_phonemes,
                "length": len(word_phonemes)  # ✅ ADD THIS LINE
            })
        except Exception as e:
            word_ref_phonemes.append({
                "word": word,
                "phonemes": [word],
                "length": 1  # ✅ ADD THIS LINE
            })
    
    # Step 2: Find word boundaries in predicted phonemes
    word_boundaries = find_word_boundaries_with_insertion_handling(
        predicted_phonemes, word_ref_phonemes
    )
    
    
    # Step 3: Extract phonemes from word boundaries, skipping insertions between words
    cleaned_phonemes = []
    current_pos = 0
    
    for i, boundary in enumerate(word_boundaries):
        word_data = word_ref_phonemes[i]
        start_idx, end_idx = boundary
        
        # Add phonemes from the current word
        word_phonemes = predicted_phonemes[start_idx:end_idx]
        cleaned_phonemes.extend(word_phonemes)
        stats['preserved_phonemes'].extend(word_phonemes)
        stats['words_preserved'] += 1
        
        # If there's a next word, check for insertions between current and next word
        if i < len(word_boundaries) - 1:
            next_start = word_boundaries[i + 1][0]
            # Remove any phonemes between current word end and next word start
            if end_idx < next_start:
                stats['insertions_removed'] += (next_start - end_idx)
    
    # If we found word boundaries, use the cleaned version
    if cleaned_phonemes:
        stats['filtered_count'] = len(cleaned_phonemes)
        stats['removed_count'] = stats['original_count'] - len(cleaned_phonemes)
        return cleaned_phonemes, stats
    
    # Fallback to original cleaning if word boundary detection fails
    return clean_predicted_phonemes(predicted_phonemes, reference_phonemes)

def validate_phoneme_extraction(audio_path: str, reference_text: str, predicted_phonemes: List[str], reference_phonemes: List[str]) -> Dict[str, Any]:
    """
    Compare wav2vec2 phoneme predictions with Whisper transcription
    Prints comparison on backend console
    """
    print("\n" + "="*80)
    print("PHONEME EXTRACTION VALIDATION - WAV2VEC2 vs WHISPER (ALM)")
    print("="*80)
    
    try:
        # Display wav2vec2 results
        print("\n[1] WAV2VEC2 PHONEME EXTRACTION RESULTS:")
        print(f"    Predicted phonemes: {predicted_phonemes}")
        print(f"    Reference phonemes: {reference_phonemes}")
        print(f"    Phonemes match: {predicted_phonemes == reference_phonemes}")
        
        # Get Whisper transcription (ground truth)
        print("\n[2] WHISPER (ALM) TRANSCRIPTION:")
        whisper = get_whisper_model()
        
        if whisper is None:
            print("    ⚠️  Whisper model not available - skipping validation")
            return {
                "validation_skipped": True,
                "reason": "Whisper model not loaded"
            }
        
        whisper_result = whisper.transcribe(audio_path)
        whisper_text = whisper_result["text"].strip()
        
        print(f"    Whisper heard: '{whisper_text}'")
        print(f"    Reference text: '{reference_text}'")
        
        # Compare results
        phonemes_identical = predicted_phonemes == reference_phonemes
        text_matches = whisper_text.lower().strip() == reference_text.lower().strip()
        
        print("\n" + "-"*80)
        print("VALIDATION COMPARISON:")
        print("-"*80)
        print(f"✓ wav2vec2 predicted == reference: {phonemes_identical}")
        print(f"✓ Whisper heard == reference: {text_matches}")
        
        # Detect the bug
        bug_detected = False
        if phonemes_identical and not text_matches:
            print("\n" + "!"*80)
            print("⚠️  CRITICAL BUG DETECTED IN WAV2VEC2 SYSTEM!")
            print("!"*80)
            print(f"Expected:      '{reference_text}'")
            print(f"Whisper heard: '{whisper_text}'")
            print(f"wav2vec2 returned REFERENCE phonemes: {predicted_phonemes}")
            print("\nThis indicates wav2vec2 is NOT analyzing the actual audio.")
            print("It's copying reference phonemes instead of extracting from audio signal.")
            print("!"*80)
            bug_detected = True
            bug_status = "CRITICAL_BUG"
        elif phonemes_identical and text_matches:
            print("\n✓ PERFECT PRONUNCIATION")
            print("  Both models confirm audio matches reference perfectly.")
            bug_status = "OK"
        else:
            print("\n✓ SYSTEM WORKING CORRECTLY")
            print("  wav2vec2 detected pronunciation differences.")
            bug_status = "OK"
        
        print("="*80 + "\n")
        
        return {
            "wav2vec2_predicted": predicted_phonemes,
            "wav2vec2_reference": reference_phonemes,
            "whisper_transcription": whisper_text,
            "reference_text": reference_text,
            "phonemes_identical": phonemes_identical,
            "transcription_matches": text_matches,
            "bug_detected": bug_detected,
            "bug_status": bug_status
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ Validation error: {e}")
        print("="*80 + "\n")
        return {
            "validation_error": str(e),
            "bug_status": "UNKNOWN"
        }
def find_word_boundaries_with_insertion_handling(predicted_phonemes: List[str], 
                                               word_ref_phonemes: List[Dict]) -> List[Tuple[int, int]]:
    """
    Find word boundaries while handling insertions between words
    Uses dynamic programming to find optimal word segments
    """
    n_pred = len(predicted_phonemes)
    n_words = len(word_ref_phonemes)
    
    # DP table: dp[i][j] = min cost to align first i phonemes with first j words
    dp = [[float('inf')] * (n_words + 1) for _ in range(n_pred + 1)]
    boundaries = [[[] for _ in range(n_words + 1)] for _ in range(n_pred + 1)]
    
    dp[0][0] = 0
    boundaries[0][0] = []
    
    for i in range(n_pred + 1):
        for j in range(n_words + 1):
            if dp[i][j] == float('inf'):
                continue
                
            # Option 1: Skip predicted phoneme (handle insertion)
            if i < n_pred:
                skip_cost = 0.3  # Low cost for skipping insertions
                if dp[i][j] + skip_cost < dp[i + 1][j]:
                    dp[i + 1][j] = dp[i][j] + skip_cost
                    boundaries[i + 1][j] = boundaries[i][j]  # No new boundary
            
            # Option 2: Assign phonemes to current word
            if j < n_words:
                # ✅ FIX: Handle both "phonemes" and "ref_phonemes" keys
                ref_phonemes = word_ref_phonemes[j].get('phonemes', 
                              word_ref_phonemes[j].get('ref_phonemes', []))
                ref_len = len(ref_phonemes)
                
                # Try different segment lengths for current word
                min_len = max(1, ref_len - 1)  # Allow 1 deletion
                max_len = min(n_pred - i, ref_len + 3)  # Allow 3 insertions max
                
                for seg_len in range(min_len, max_len + 1):
                    if i + seg_len <= n_pred:
                        segment = predicted_phonemes[i:i + seg_len]
                        
                        # Calculate alignment cost
                        aligned_ref, aligned_seg = needleman_wunsch_alignment(ref_phonemes, segment)
                        matches = sum(1 for r, s in zip(aligned_ref, aligned_seg) if r == s)
                        mismatches = len([r for r in aligned_ref if r != "_"]) - matches
                        
                        # Length penalty to prefer segments close to reference length
                        length_penalty = abs(seg_len - ref_len) * 0.2
                        total_cost = mismatches + length_penalty
                        
                        if dp[i][j] + total_cost < dp[i + seg_len][j + 1]:
                            dp[i + seg_len][j + 1] = dp[i][j] + total_cost
                            new_boundaries = boundaries[i][j] + [(i, i + seg_len)]
                            boundaries[i + seg_len][j + 1] = new_boundaries
    
    # Return best boundaries found
    if dp[n_pred][n_words] != float('inf'):
        return boundaries[n_pred][n_words]
    
    # Fallback: proportional segmentation
    return [(b['start'], b['end']) for b in proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)]


def enhanced_clean_predicted_phonemes(predicted_phonemes: List[str], 
                                    reference_phonemes: List[str],
                                    reference_text: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Main enhanced cleaning function that preserves word integrity
    """
    # First try the word-preserving method
    cleaned, stats = improved_intelligent_phoneme_cleaning(
        predicted_phonemes, reference_phonemes, reference_text
    )
    
    # If the word-preserving method didn't work well, fall back to original
    if len(cleaned) < len(reference_phonemes) * 0.5:  # If we lost too many phonemes
        logger.warning("Word-preserving cleaning removed too many phonemes, using fallback")
        return clean_predicted_phonemes(predicted_phonemes, reference_phonemes)
    
    return cleaned, stats

def universal_phoneme_segmentation(predicted_phonemes: List[str], reference_text: str) -> List[Dict]:
    """
    Universal phoneme segmentation with insertion handling
    """
    system = get_improved_mdd_system()
    words = reference_text.lower().split()
    
    # Get reference phonemes for each word
    word_ref_phonemes = []
    total_ref_phonemes = 0
    
    for word in words:
        try:
            ref_phonemes = system._text_to_phonemes_espeak(word)
            if not ref_phonemes or ref_phonemes == ['<unk>']:
                ref_phonemes = system._basic_text_to_phonemes(word)
            word_ref_phonemes.append({
                "word": word,
                "ref_phonemes": ref_phonemes,
                "length": len(ref_phonemes)
            })
            total_ref_phonemes += len(ref_phonemes)
        except Exception as e:
            word_ref_phonemes.append({
                "word": word,
                "ref_phonemes": [word],
                "length": 1
            })
            total_ref_phonemes += 1
    
    n_pred = len(predicted_phonemes)
    n_words = len(words)
    
    # DP table
    dp = [[float('inf')] * (n_words + 1) for _ in range(n_pred + 1)]
    word_boundaries = [[[] for _ in range(n_words + 1)] for _ in range(n_pred + 1)]
    
    dp[0][0] = 0
    word_boundaries[0][0] = []
    
    # Fill DP table
    for i in range(n_pred + 1):
        for j in range(n_words + 1):
            if dp[i][j] == float('inf'):
                continue
            
            # **NEW: Allow skipping predicted phonemes (insertions between words)**
            if i < n_pred:
                skip_cost = 0.3  # Low cost for skipping insertion phonemes
                if dp[i][j] + skip_cost < dp[i + 1][j]:
                    dp[i + 1][j] = dp[i][j] + skip_cost
                    word_boundaries[i + 1][j] = word_boundaries[i][j]  # Don't add to any word
            
            if j < n_words:
                current_word_data = word_ref_phonemes[j]
                ref_phonemes = current_word_data["ref_phonemes"]
                ref_len = len(ref_phonemes)
                
                # Try different segment lengths
                min_segment = max(1, ref_len - 1)  # Allow 1 deletion
                max_segment = min(n_pred - i, ref_len + 2)  # Allow 2 insertions
                
                for seg_len in range(min_segment, max_segment + 1):
                    if i + seg_len <= n_pred:
                        segment = predicted_phonemes[i:i + seg_len]
                        
                        # Calculate alignment cost
                        aligned_ref, aligned_seg = needleman_wunsch_alignment(ref_phonemes, segment)
                        
                        # Count exact matches only (stricter)
                        matches = sum(1 for r, s in zip(aligned_ref, aligned_seg) 
                                    if r == s and r != "_" and s != "_")
                        mismatches = len(ref_phonemes) - matches
                        
                        # Length penalty
                        length_penalty = abs(seg_len - ref_len) * 0.2
                        total_cost = mismatches + length_penalty
                        
                        if dp[i][j] + total_cost < dp[i + seg_len][j + 1]:
                            dp[i + seg_len][j + 1] = dp[i][j] + total_cost
                            new_boundaries = word_boundaries[i][j] + [{
                                "word": current_word_data["word"],
                                "start": i,
                                "end": i + seg_len,
                                "predicted_phonemes": segment,
                                "reference_phonemes": ref_phonemes,
                                "alignment_cost": total_cost
                            }]
                            word_boundaries[i + seg_len][j + 1] = new_boundaries
    
    # Find best alignment
    if dp[n_pred][n_words] != float('inf'):
        best_boundaries = word_boundaries[n_pred][n_words]
    else:
        # Fallback to proportional
        best_boundaries = proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)
    
    return best_boundaries
    

def create_corrected_llm_input(word_analysis: Dict[str, Any]) -> str:
    """
    Create corrected LLM input with accurate word information
    """
    problem_words = []
    
    for word_data in word_analysis.get("words", []):
        if word_data.get("status") in ["mispronounced", "partial"]:
            problem_words.append({
                "word": word_data["word"],
                "expected": " ".join(word_data.get("reference_phonemes", [])),
                "actual": " ".join(word_data.get("predicted_phonemes", [])),
                "issue": word_data.get("notes", "")
            })
    
    # Build accurate description
    accurate_description = "Actual pronunciation issues:\n"
    for pw in problem_words:
        accurate_description += f"- '{pw['word']}': Said '{pw['actual']}' instead of '{pw['expected']}' ({pw['issue']})\n"
    
    return accurate_description


def improved_filter_predicted_phonemes(predicted_phonemes: List[str], 
                                     reference_phonemes: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Improved filtering with repetition detection
    """
    stats = {
        'original_count': len(predicted_phonemes),
        'repetition_handling': {},
        'final_filtering': {}
    }
    
    # Step 1: Handle phrase repetitions FIRST
    filtered, repetition_stats = detect_and_handle_repetitions(predicted_phonemes, reference_phonemes)
    stats['repetition_handling'] = repetition_stats
    
    # Step 2: Apply additional cleaning if needed
    if len(filtered) > len(reference_phonemes) * 1.2:
        # Use your existing filtering logic
        final_filtered, filtering_stats = filter_predicted_phonemes_before_analysis(filtered, reference_phonemes)
        stats['final_filtering'] = filtering_stats
    else:
        final_filtered = filtered
        stats['final_filtering'] = {'additional_filtering': False}
    
    stats['final_count'] = len(final_filtered)
    stats['total_removed'] = stats['original_count'] - len(final_filtered)
    
    return final_filtered, stats

def segment_phonemes_into_words(predicted_phonemes: List[str], reference_text: str) -> Dict[str, List[str]]:
    """
    Explicitly segment phonemes into words based on reference text structure
    """
    system = get_improved_mdd_system()
    words = reference_text.lower().split()
    
    # Get reference phonemes for each word
    word_phoneme_map = {}
    total_ref_phonemes = 0
    
    for word in words:
        try:
            word_phonemes = system._text_to_phonemes_espeak(word)
            if not word_phonemes or word_phonemes == ['<unk>']:
                word_phonemes = system._basic_text_to_phonemes(word)
            word_phoneme_map[word] = word_phonemes
            total_ref_phonemes += len(word_phonemes)
        except Exception as e:
            word_phoneme_map[word] = [word]
            total_ref_phonemes += 1
    
    # Calculate expected proportions
    word_proportions = {}
    for word, phonemes in word_phoneme_map.items():
        word_proportions[word] = len(phonemes) / total_ref_phonemes
    
    # Segment predicted phonemes based on proportions
    segmented_words = {}
    current_pos = 0
    
    for i, word in enumerate(words):
        if i == len(words) - 1:  # Last word gets remaining phonemes
            end_pos = len(predicted_phonemes)
        else:
            proportion = word_proportions[word]
            allocated = max(1, int(len(predicted_phonemes) * proportion))
            end_pos = min(current_pos + allocated, len(predicted_phonemes))
        
        segmented_words[word] = predicted_phonemes[current_pos:end_pos]
        current_pos = end_pos
    
    return segmented_words

def generate_word_level_analysis_explicit_segmentation(reference_text: str, predicted_phonemes: List[str]) -> Dict[str, Any]:
    """
    Generate word-level analysis with explicit word segmentation
    """
    try:
        # Segment phonemes into words explicitly
        segmented_words = segment_phonemes_into_words(predicted_phonemes, reference_text)
        
        words_analysis = []
        
        for word, pred_phonemes in segmented_words.items():
            # Get reference phonemes for this word
            system = get_improved_mdd_system()
            try:
                ref_phonemes = system._text_to_phonemes_espeak(word)
                if not ref_phonemes or ref_phonemes == ['<unk>']:
                    ref_phonemes = system._basic_text_to_phonemes(word)
            except:
                ref_phonemes = [word]
            
            # Analyze this word
            word_analysis = analyze_word_robust(
                {"word": word, "phonemes": ref_phonemes},
                pred_phonemes,
                0,  # word_index
                len(segmented_words)  # total_words
            )
            words_analysis.append(word_analysis)
        
        # Generate audio data
        audio_data = generate_word_audio_data(words_analysis, reference_text)
        
        # Calculate overall statistics
        correct_words = sum(1 for w in words_analysis if w["status"] == "correct")
        partial_words = sum(1 for w in words_analysis if w["status"] == "partial")
        mispronounced_words = sum(1 for w in words_analysis if w["status"] == "mispronounced")
        
        total_per = sum(w["per"]["per"] for w in words_analysis)
        avg_per = total_per / max(1, len(words_analysis))
        
        word_accuracy = (correct_words + (partial_words * 0.5)) / max(1, len(words_analysis))
        
        return {
            "words": words_analysis,
            "audio_data": audio_data,
            "summary": {
                "total_words": len(words_analysis),
                "correct_words": correct_words,
                "partial_words": partial_words,
                "mispronounced_words": mispronounced_words,
                "word_accuracy_percent": round(word_accuracy * 100, 2),
                "average_per": round(avg_per, 2),
                "overall_status": "good" if word_accuracy >= 0.8 else "needs_improvement" if word_accuracy >= 0.6 else "poor",
                "analysis_method": "explicit_word_segmentation"
            }
        }
        
    except Exception as e:
        logger.error(f"Explicit word segmentation analysis failed: {e}")
        return {"error": str(e)}
    
def find_optimal_word_boundaries_robust(predicted_phonemes: List[str], word_phoneme_mapping: List[Dict]) -> List[Tuple[int, int]]:
    """Find word boundaries that can handle completely mispronounced words"""
    n_pred = len(predicted_phonemes)
    n_words = len(word_phoneme_mapping)
    
    # dp[i][j] = minimum cost to align first i predicted phonemes with first j words
    dp = [[float('inf')] * (n_words + 1) for _ in range(n_pred + 1)]
    parent = [[(-1, -1)] * (n_words + 1) for _ in range(n_pred + 1)]
    
    # Base case: no phonemes, no words
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(n_pred + 1):
        for j in range(n_words + 1):
            if dp[i][j] == float('inf'):
                continue
                
            if j < n_words:  # Can add another word
                ref_phonemes = word_phoneme_mapping[j]['phonemes']
                ref_len = len(ref_phonemes)
                
                # Try different lengths for current word (with flexibility)
                min_len = max(1, ref_len // 3)  # Allow very short matches for mispronounced words
                max_len = min(n_pred - i, ref_len * 3)  # Allow longer matches
                
                for word_len in range(min_len, max_len + 1):
                    if i + word_len <= n_pred:
                        word_pred = predicted_phonemes[i:i + word_len]
                        
                        # Calculate alignment cost for this word
                        aligned_ref, aligned_pred = needleman_wunsch_alignment(ref_phonemes, word_pred)
                        
                        # Cost = number of mismatches + length penalty
                        mismatches = sum(1 for r, p in zip(aligned_ref, aligned_pred) if r != p and r != "_" and p != "_")
                        length_penalty = abs(len(word_pred) - ref_len) * 0.1
                        cost = mismatches + length_penalty
                        
                        # Allow skipping words with very high cost (complete mispronunciation)
                        if cost > ref_len * 0.8:  # If more than 80% wrong, consider skipping
                            # Try to skip this word by assigning minimal phonemes
                            skip_cost = ref_len * 0.9  # High cost for skipping
                            if dp[i][j] + skip_cost < dp[i + 1][j + 1]:
                                dp[i + 1][j + 1] = dp[i][j] + skip_cost
                                parent[i + 1][j + 1] = (i, j)
                        else:
                            if dp[i][j] + cost < dp[i + word_len][j + 1]:
                                dp[i + word_len][j + 1] = dp[i][j] + cost
                                parent[i + word_len][j + 1] = (i, j)
            
            # Allow skipping predicted phonemes (handle extra sounds)
            if i < n_pred:
                skip_cost = 0.5  # Low cost for skipping extra phonemes
                if dp[i][j] + skip_cost < dp[i + 1][j]:
                    dp[i + 1][j] = dp[i][j] + skip_cost
                    parent[i + 1][j] = (i, j)
    
    # Backtrack to find optimal boundaries
    boundaries = []
    i, j = n_pred, n_words
    
    while i > 0 and j > 0:
        prev_i, prev_j = parent[i][j]
        if prev_j == j - 1:  # This word consumed phonemes from prev_i to i
            boundaries.append((prev_i, i))
        elif prev_j == j:  # Skipped phonemes
            # No boundary added, just move back
            pass
        i, j = prev_i, prev_j
    
    boundaries.reverse()
    return boundaries

def analyze_word_robust(word_data: Dict, pred_phonemes: List[str], word_index: int, total_words: int) -> Dict[str, Any]:
    """Analyze a single word with robust handling of complete mispronunciations"""
    word = word_data['word']
    ref_phonemes = word_data['phonemes']
    
    # If no predicted phonemes for this word, mark as completely mispronounced
    if not pred_phonemes:
        return {
            "word": word,
            "reference_phonemes": ref_phonemes,
            "predicted_phonemes": [],
            "aligned_reference": ref_phonemes,
            "aligned_predicted": ["_"] * len(ref_phonemes),
            "per": {"per": 100.0, "method": "complete_mispronunciation"},
            "status": "mispronounced",
            "phoneme_errors": [{
                "position": i,
                "type": "deletion", 
                "predicted": None,
                "expected": ref_p
            } for i, ref_p in enumerate(ref_phonemes)],
            "error_count": len(ref_phonemes),
            "error_counts": {
                "substitutions": 0,
                "insertions": 0,
                "deletions": len(ref_phonemes),
                "correct": 0
            },
            "confidence": "low",
            "notes": "Word completely missing or unrecognizable"
        }
    
    print({"word": word,
            "reference_phonemes": ref_phonemes,
            "predicted_phonemes": [],
            "aligned_reference": ref_phonemes,
            "aligned_predicted": ["_"] * len(ref_phonemes),
            "per": {"per": 100.0, "method": "complete_mispronunciation"},
            "status": "mispronounced",
            "phoneme_errors": [{
                "position": i,
                "type": "deletion", 
                "predicted": None,
                "expected": ref_p
            } for i, ref_p in enumerate(ref_phonemes)],
            "error_count": len(ref_phonemes),
            "error_counts": {
                "substitutions": 0,
                "insertions": 0,
                "deletions": len(ref_phonemes),
                "correct": 0
            },
            "confidence": "low",
            "notes": "Word completely missing or unrecognizable"
        })
    # Align this specific word
    aligned_ref, aligned_pred = needleman_wunsch_alignment(ref_phonemes, pred_phonemes)
    
    # Calculate errors
    word_substitutions = 0
    word_insertions = 0
    word_deletions = 0
    word_correct = 0
    errors = []
    
    for j, (ref_p, pred_p) in enumerate(zip(aligned_ref, aligned_pred)):
        if ref_p == "_":
            word_insertions += 1
            errors.append({
                "position": j,
                "type": "insertion",
                "predicted": pred_p,
                "expected": None
            })
        elif pred_p == "_":
            word_deletions += 1
            errors.append({
                "position": j,
                "type": "deletion", 
                "predicted": None,
                "expected": ref_p
            })
        elif ref_p != pred_p:
            word_substitutions += 1
            errors.append({
                "position": j,
                "type": "substitution",
                "predicted": pred_p,
                "expected": ref_p
            })
        else:
            word_correct += 1
    
    # Calculate word-level PER
    total_ref_phonemes = len([p for p in aligned_ref if p != "_"])
    if total_ref_phonemes == 0:
        word_per = 100.0
    else:
        word_per = ((word_substitutions + word_deletions + word_insertions) / total_ref_phonemes) * 100
    
    # Determine word status with more nuanced thresholds
    if word_per == 0:
        status = "correct"
        confidence = "high"
    elif word_per <= 20 and word_correct >= len(ref_phonemes) * 0.5:
        status = "partial"
        confidence = "medium"
    elif word_per <= 50 and word_correct >= len(ref_phonemes) * 0.3:
        status = "partial" 
        confidence = "low"
    else:
        status = "mispronounced"
        confidence = "low"
    
    # If alignment is very poor, mark as completely mispronounced
    if word_correct == 0 and len(pred_phonemes) > 0:
        status = "mispronounced"
        confidence = "very_low"
    
    return {
        "word": word,
        "reference_phonemes": ref_phonemes,
        "predicted_phonemes": pred_phonemes,
        "aligned_reference": aligned_ref,
        "aligned_predicted": aligned_pred,
        "per": {"per": round(word_per, 2), "method": "robust_alignment"},
        "status": status,
        "phoneme_errors": errors,
        "error_count": len(errors),
        "error_counts": {
            "substitutions": word_substitutions,
            "insertions": word_insertions,
            "deletions": word_deletions,
            "correct": word_correct
        },
        "confidence": confidence,
        "alignment_quality": round(word_correct / max(1, len(ref_phonemes)), 2)
    }

def generate_word_level_analysis_robust(reference_text: str, predicted_phonemes: List[str]) -> Dict[str, Any]:
    """Generate word-level analysis with robust handling of mispronunciations"""
    try:
        system = get_improved_mdd_system()
        words = reference_text.lower().split()
        word_phoneme_mapping = []
        
        # Build word-phoneme mapping
        for word in words:
            try:
                word_phonemes = system._text_to_phonemes_espeak(word)
                if not word_phonemes or word_phonemes == ['<unk>']:
                    word_phonemes = system._basic_text_to_phonemes(word)
                
                word_phoneme_mapping.append({
                    "word": word,
                    "phonemes": word_phonemes
                })
            except Exception as e:
                word_phoneme_mapping.append({
                    "word": word,
                    "phonemes": [word]
                })
        
        # Use robust boundary finding
        try:
            boundaries = find_optimal_word_boundaries_robust(predicted_phonemes, word_phoneme_mapping)
        except Exception as e:
            logger.warning(f"Robust boundary finding failed, using fallback: {e}")
            # Fallback: proportional allocation with overlap handling
            boundaries = []
            total_ref_phonemes = sum(len(wd['phonemes']) for wd in word_phoneme_mapping)
            current_pos = 0
            
            for i, word_data in enumerate(word_phoneme_mapping):
                ref_len = len(word_data['phonemes'])
                if i == len(word_phoneme_mapping) - 1:  # Last word gets remaining phonemes
                    end_pos = len(predicted_phonemes)
                else:
                    proportion = ref_len / total_ref_phonemes
                    allocated = max(1, int(len(predicted_phonemes) * proportion))
                    end_pos = min(current_pos + allocated, len(predicted_phonemes))
                
                boundaries.append((current_pos, end_pos))
                current_pos = end_pos
        
        # Analyze each word robustly
        words_analysis = []
        used_phonemes = set()
        
        for i, (word_data, (start_idx, end_idx)) in enumerate(zip(word_phoneme_mapping, boundaries)):
            # Ensure we don't reuse phonemes (handle overlap)
            start_idx = max(start_idx, max(used_phonemes) if used_phonemes else 0)
            if start_idx >= end_idx:
                # No phonemes left for this word - mark as completely mispronounced
                pred_phonemes_for_word = []
            else:
                pred_phonemes_for_word = predicted_phonemes[start_idx:end_idx]
                used_phonemes.update(range(start_idx, end_idx))
            
            word_analysis = analyze_word_robust(word_data, pred_phonemes_for_word, i, len(words))
            words_analysis.append(word_analysis)
        
        # Generate audio data
        audio_data = generate_word_audio_data(words_analysis, reference_text)
        
        # Calculate overall statistics
        correct_words = sum(1 for w in words_analysis if w["status"] == "correct")
        partial_words = sum(1 for w in words_analysis if w["status"] == "partial")
        mispronounced_words = sum(1 for w in words_analysis if w["status"] == "mispronounced")
        
        total_per = sum(w["per"]["per"] for w in words_analysis)
        avg_per = total_per / max(1, len(words_analysis))
        
        word_accuracy = (correct_words + (partial_words * 0.5)) / max(1, len(words_analysis))
        
        return {
            "words": words_analysis,
            "audio_data": audio_data,
            "summary": {
                "total_words": len(words_analysis),
                "correct_words": correct_words,
                "partial_words": partial_words,
                "mispronounced_words": mispronounced_words,
                "word_accuracy_percent": round(word_accuracy * 100, 2),
                "average_per": round(avg_per, 2),
                "overall_status": "good" if word_accuracy >= 0.8 else "needs_improvement" if word_accuracy >= 0.6 else "poor",
                "analysis_method": "robust_word_alignment"
            }
        }
        
    except Exception as e:
        logger.error(f"Robust word analysis failed: {e}")
        return {"error": str(e)}

# Traditional analysis functions with audio support
def create_traditional_analysis_with_audio(reference_text: str, raw_results: Dict[str, Any], word_level: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback to traditional analysis when LLM is not used, but with audio support"""
    predicted_phonemes = raw_results.get("analysis", {}).get("predicted_phonemes", [])
    reference_phonemes = raw_results.get("analysis", {}).get("reference_phonemes", [])
    
    per_result = calculate_phoneme_error_rate(predicted_phonemes, reference_phonemes)
    word_summary = word_level.get("summary", {})
    word_accuracy = word_summary.get("word_accuracy_percent", 0) / 100
    audio_data = word_level.get("audio_data", {})
    
    # Create mispronunciations array
    mispronunciations = []
    phoneme_position = 0
    
    for word_data in word_level.get("words", []):
        word_phonemes = word_data.get("reference_phonemes", [])
        for error in word_data.get("phoneme_errors", []):
            error_with_position = error.copy()
            error_with_position["absolute_position"] = phoneme_position + error["position"]
            error_with_position["word"] = word_data["word"]
            mispronunciations.append(error_with_position)
        phoneme_position += len(word_phonemes)
    
    accuracy = max(0, 100 - per_result["per"]) / 100
    
    return {
        "accuracy": accuracy,
        "per": per_result["per"],
        "word_accuracy": word_accuracy,
        "correct_phonemes": per_result["correct"],
        "total_phonemes": len(reference_phonemes),
        "predicted_phonemes": predicted_phonemes,
        "reference_phonemes": reference_phonemes,
        "mispronunciations": mispronunciations,
        "word_summary": word_summary,
        "audio_data": audio_data,
        "evaluation_method": "traditional_only_with_audio"
    }

def generate_traditional_feedback_with_audio(analysis_result: Dict[str, Any], word_level: Dict[str, Any]) -> str:
    """Generate traditional feedback without LLM assessment but with audio info"""
    per = analysis_result["per"]
    word_accuracy = analysis_result["word_accuracy"] * 100
    audio_data = analysis_result.get("audio_data", {})
    
    # Create phoneme display
    ref_phonemes = analysis_result.get("reference_phonemes", [])
    pred_phonemes = analysis_result.get("predicted_phonemes", [])
    phoneme_display = f"Expected: {' '.join(ref_phonemes)}\nYour pronunciation: {' '.join(pred_phonemes)}"
    
    feedback_parts = []
    
    # Add phoneme comparison
    feedback_parts.append(phoneme_display)
    
    # Add audio info
    word_audio_count = len(audio_data.get("word_audio", {}))
    if word_audio_count > 0:
        feedback_parts.append(f"Audio: {word_audio_count} word pronunciations available for practice")
    
    if per <= 10:
        feedback_parts.append("Assessment: Excellent pronunciation! Very clear and accurate.")
    elif per <= 25:
        feedback_parts.append("Assessment: Good pronunciation overall with minor errors.")
    elif per <= 40:
        feedback_parts.append("Assessment: Fair pronunciation. Several areas need improvement.")
    else:
        feedback_parts.append("Assessment: Significant pronunciation issues detected.")
    
    feedback_parts.append(f"Accuracy: {100 - per:.1f}% | Word Score: {word_accuracy:.1f}%")
    
    return "\n".join(feedback_parts)

# Keep existing generate_word_level_analysis for compatibility
def generate_word_level_analysis(reference_text: str, predicted_phonemes: List[str]) -> Dict[str, Any]:
    """Original word-level analysis without audio generation"""
    result = generate_word_level_analysis_with_audio(reference_text, predicted_phonemes)
    # Remove audio data for backward compatibility
    result.pop("audio_data", None)
    return result

# Original routes preserved for compatibility
@app.route('/analyze', methods=['POST'])
def analyze_pronunciation():
    """Original pronunciation analysis endpoint - now uses audio-enhanced format"""
    return analyze_pronunciation_with_llm()

@app.route('/analyze_word_practice', methods=['POST'])
def analyze_word_practice():
    """Enhanced word practice with context-aware phoneme audio generation"""
    start_time = time.time()
    
    print_banner("WORD PRACTICE ANALYSIS", "=", 80)
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "Audio file required"}), 400

        file = request.files['audio_file']
        word = request.form.get('word', '').strip()
        
        if not word:
            return jsonify({"success": False, "error": "Word required"}), 400

        print_section("REQUEST", {
            "Word": word,
            "File": file.filename
        })
        
        # Save audio temporarily
        filename = secure_filename(file.filename)
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        file.save(temp_audio_path)

        # Extract phonemes from user's pronunciation
        phoneme_results = extract_phonemes_wav2vec2(temp_audio_path, word)
        
        if not phoneme_results["success"]:
            os.remove(temp_audio_path)
            return jsonify({
                "success": False, 
                "error": "Phoneme extraction failed"
            }), 500
        
        predicted_phonemes = phoneme_results["predicted_phonemes"]
        reference_phonemes = phoneme_results["reference_phonemes"]
        
        print_section("EXTRACTION", {
            "Word": word,
            "Predicted": ' '.join(predicted_phonemes),
            "Reference": ' '.join(reference_phonemes)
        })
        
        # Calculate word-level accuracy
        accuracy, status, feedback = calculate_word_accuracy(
            word=word,
            predicted_phonemes=predicted_phonemes,
            reference_phonemes=reference_phonemes
        )
        
        # **CRITICAL**: Generate phoneme audio from the actual word context
        phoneme_audio_data = generate_word_phoneme_audio(word, reference_phonemes)
        
        # Build detailed analysis
        phoneme_breakdown = []
        for i, ref_ph in enumerate(reference_phonemes):
            pred_ph = predicted_phonemes[i] if i < len(predicted_phonemes) else "_"
            
            if ref_ph == pred_ph:
                ph_status = "correct"
            elif pred_ph == "_":
                ph_status = "mispronounced"
            else:
                # Calculate similarity
                similarity = phoneme_similarity_score(ref_ph, pred_ph)
                ph_status = "partial" if similarity >= 0.6 else "mispronounced"
            
            phoneme_breakdown.append({
                "phoneme": ref_ph,
                "predicted": pred_ph,
                "status": ph_status,
                "position": i
            })
        
        # Build response
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "word": word,
            "accuracy": round(accuracy, 1),
            "status": status,
            "feedback": feedback,
            "analysis": {
                "predicted_phonemes": predicted_phonemes,
                "reference_phonemes": reference_phonemes,
                "correct_phonemes": sum(1 for p, r in zip(predicted_phonemes, reference_phonemes) if p == r),
                "total_phonemes": len(reference_phonemes),
                "phoneme_breakdown": phoneme_breakdown
            },
            "phoneme_audio": phoneme_audio_data,  # Word-context phoneme audio
            "metadata": {
                "processing_time": round(processing_time, 3),
                "timestamp": int(time.time()),
                "word_context": word
            }
        }
        
        # Cleanup
        os.remove(temp_audio_path)
        
        print_section("RESULT", {
            "Word": word,
            "Accuracy": f"{accuracy:.1f}%",
            "Status": status.upper(),
            "Phonemes Generated": len(phoneme_audio_data) if phoneme_audio_data else 0,
            "Time": f"{processing_time:.3f}s"
        })
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Word practice failed: {e}", exc_info=True)
        try:
            if 'temp_audio_path' in locals():
                os.remove(temp_audio_path)
        except:
            pass
        return jsonify({"success": False, "error": str(e)}), 500


def calculate_word_accuracy(word, predicted_phonemes, reference_phonemes):
    """Calculate word accuracy with detailed feedback"""
    
    if not reference_phonemes:
        return 0.0, "mispronounced", "No reference phonemes available"
    
    # Align phonemes
    aligned_ref, aligned_pred = needleman_wunsch_alignment(reference_phonemes, predicted_phonemes)
    
    # Count matches
    correct = sum(1 for r, p in zip(aligned_ref, aligned_pred) if r == p and r != "_")
    total = len([r for r in aligned_ref if r != "_"])
    
    if total == 0:
        return 0.0, "mispronounced", "Unable to analyze"
    
    accuracy = (correct / total) * 100
    
    # Determine status
    if accuracy >= 90:
        status = "correct"
        feedback = f"Excellent! '{word}' pronounced correctly."
    elif accuracy >= 70:
        status = "partial"
        feedback = f"Good attempt at '{word}'. Some sounds need adjustment."
    else:
        status = "mispronounced"
        feedback = f"'{word}' needs more practice. Focus on the individual sounds."
    
    return accuracy, status, feedback

def phoneme_similarity_score(ph1, ph2):
    """
    Calculate similarity between two phonemes
    Returns score between 0.0 and 1.0
    """
    if not ph1 or not ph2:
        return 0.0
    
    # Exact match
    if ph1 == ph2:
        return 1.0
    
    # Case-insensitive match
    if ph1.lower() == ph2.lower():
        return 0.95
    
    # Character overlap
    max_len = max(len(ph1), len(ph2))
    if max_len == 0:
        return 0.0
    
    matches = sum(c1 == c2 for c1, c2 in zip(ph1, ph2))
    similarity = matches / max_len
    
    # Phonetic class bonus
    if same_phonetic_class(ph1, ph2):
        similarity = min(1.0, similarity + 0.2)
    
    return similarity

def generate_word_phoneme_audio(word, phonemes):
    """
    Generate audio for each phoneme FROM the actual word pronunciation
    This ensures phonemes sound like they do in the word context
    
    Returns: Dictionary mapping phoneme -> audio_base64
    """
    
    try:
        print_section("GENERATING PHONEME AUDIO FROM WORD", {
            "Word": word,
            "Phonemes": ' '.join(phonemes)
        })
        
        # Step 1: Generate the full word audio
        timestamp = int(time.time())
        word_audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"word_{word}_{timestamp}.wav")
        
        if not audio_generator.generate_word_audio(word, word_audio_path):
            logger.warning(f"Could not generate audio for word '{word}'")
            return None
        
        # Process quality
        audio_generator.process_audio_quality(word_audio_path)
        
        # Step 2: Load the word audio
        import librosa
        import soundfile as sf
        import numpy as np
        
        audio_data, sr = librosa.load(word_audio_path, sr=22050)  # Use 22050 for better quality
        duration = len(audio_data) / sr
        
        print_section("WORD AUDIO LOADED", {
            "Duration": f"{duration:.2f}s",
            "Sample Rate": f"{sr}Hz",
            "Samples": len(audio_data),
            "Phonemes": len(phonemes)
        })
        
        # Step 3: Segment audio into phoneme chunks
        phoneme_audio_map = {}
        
        if len(phonemes) > 0:
            # Calculate time per phoneme with padding
            segment_duration = duration / len(phonemes)
            overlap = 0.02  # 20ms overlap for smoother transitions
            
            for i, phoneme in enumerate(phonemes):
                # Calculate start and end samples with overlap
                start_time = max(0, i * segment_duration - overlap)
                end_time = min(duration, (i + 1) * segment_duration + overlap)
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Extract phoneme audio segment
                phoneme_audio = audio_data[start_sample:end_sample]
                
                # Apply envelope for smooth edges
                fade_samples = int(0.015 * sr)  # 15ms fade
                if len(phoneme_audio) > 2 * fade_samples:
                    # Smooth fade in
                    fade_in = np.linspace(0, 1, fade_samples) ** 2
                    phoneme_audio[:fade_samples] *= fade_in
                    # Smooth fade out
                    fade_out = np.linspace(1, 0, fade_samples) ** 2
                    phoneme_audio[-fade_samples:] *= fade_out
                
                # Normalize audio
                phoneme_audio = librosa.util.normalize(phoneme_audio)
                
                # Save phoneme audio to temp file
                phoneme_audio_path = os.path.join(
                    AUDIO_OUTPUT_FOLDER, 
                    f"phoneme_{word}_{phoneme}_{i}_{timestamp}.wav"
                )
                sf.write(phoneme_audio_path, phoneme_audio, sr, format='WAV')
                
                # Convert to base64
                phoneme_audio_base64 = audio_generator.audio_to_base64(phoneme_audio_path)
                
                if phoneme_audio_base64:
                    phoneme_audio_map[phoneme] = {
                        "audio_base64": phoneme_audio_base64,
                        "phoneme": phoneme,
                        "position": i,
                        "duration": round(len(phoneme_audio) / sr, 3),
                        "word_context": word,
                        "sample_rate": sr
                    }
                    
                    logger.info(f"✓ Generated audio for phoneme '{phoneme}' at position {i} from word '{word}'")
                
                # Cleanup temp phoneme file
                try:
                    os.remove(phoneme_audio_path)
                except:
                    pass
        
        # Cleanup word audio file
        try:
            os.remove(word_audio_path)
        except:
            pass
        
        print_section("PHONEME AUDIO GENERATED", {
            "Total Phonemes": len(phonemes),
            "Successfully Generated": len(phoneme_audio_map),
            "Word Context": word
        })
        
        return phoneme_audio_map if phoneme_audio_map else None
        
    except Exception as e:
        logger.error(f"Failed to generate phoneme audio from word '{word}': {e}")
        import traceback
        traceback.print_exc()
        return None
    
@app.route('/get_word_phoneme_audio/<word>/<phoneme>', methods=['GET'])
def get_word_phoneme_audio_endpoint(word, phoneme):
    """
    Get audio for a specific phoneme in the context of a word
    This generates the phoneme from the word pronunciation
    """
    
    try:
        print_section("WORD PHONEME AUDIO REQUEST", {
            "Word": word,
            "Phoneme": phoneme
        })
        
        if not word.strip() or not phoneme.strip():
            return jsonify({
                "success": False, 
                "error": "Word and phoneme required"
            }), 400
        
        # Get reference phonemes for the word
        system = get_improved_mdd_system()
        try:
            phonemes = system._text_to_phonemes_espeak(word.lower())
            if not phonemes or phonemes == ['<unk>']:
                phonemes = system._basic_text_to_phonemes(word.lower())
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Could not get phonemes for word '{word}'"
            }), 400
        
        # Clean phonemes
        phonemes = [p for p in phonemes if p not in [' ', '']]
        
        if phoneme not in phonemes:
            return jsonify({
                "success": False,
                "error": f"Phoneme '{phoneme}' not found in word '{word}'. Word phonemes: {phonemes}"
            }), 400
        
        # Get phoneme position
        phoneme_position = phonemes.index(phoneme)
        
        # Generate word audio
        timestamp = int(time.time())
        word_audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"word_{word}_{timestamp}.wav")
        
        if not audio_generator.generate_word_audio(word, word_audio_path):
            return jsonify({
                "success": False,
                "error": "Audio generation failed"
            }), 500
        
        # Process quality
        audio_generator.process_audio_quality(word_audio_path)
        
        # Load and segment
        import librosa
        import soundfile as sf
        
        audio_data, sr = librosa.load(word_audio_path, sr=22050)
        duration = len(audio_data) / sr
        
        # Calculate phoneme segment with overlap
        segment_duration = duration / len(phonemes)
        overlap = 0.02  # 20ms overlap
        
        start_time = max(0, phoneme_position * segment_duration - overlap)
        end_time = min(duration, (phoneme_position + 1) * segment_duration + overlap)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract phoneme audio
        phoneme_audio = audio_data[start_sample:end_sample]
        
        # Apply smooth fades
        fade_samples = int(0.015 * sr)
        if len(phoneme_audio) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples) ** 2
            phoneme_audio[:fade_samples] *= fade_in
            fade_out = np.linspace(1, 0, fade_samples) ** 2
            phoneme_audio[-fade_samples:] *= fade_out
        
        # Normalize
        phoneme_audio = librosa.util.normalize(phoneme_audio)
        
        # Save phoneme audio
        phoneme_audio_path = os.path.join(
            AUDIO_OUTPUT_FOLDER,
            f"phoneme_{word}_{phoneme}_{timestamp}.wav"
        )
        sf.write(phoneme_audio_path, phoneme_audio, sr, format='WAV')
        
        # Return audio file
        response = send_file(
            phoneme_audio_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"phoneme_{phoneme}_from_{word}.wav"
        )
        
        # Schedule cleanup (use background task in production)
        try:
            os.remove(word_audio_path)
            # Don't remove phoneme_audio_path yet - Flask needs it for response
        except:
            pass
        
        return response
            
    except Exception as e:
        logger.error(f"Get word phoneme audio failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
   
@app.route('/analyze_with_alignment', methods=['POST'])
def analyze_pronunciation_with_alignment():
    """Original enhanced analysis endpoint - now uses audio-enhanced format"""
    return analyze_pronunciation_with_llm()

@app.route('/get_word_audio/<word>', methods=['GET'])
def get_word_audio(word):
    """Endpoint to generate and return audio for a specific word"""
    try:
        print_section("WORD AUDIO REQUEST", {"Word": word})
        
        if not word.strip():
            return jsonify({"success": False, "error": "Word parameter required"}), 400
        
        # Clean the word
        clean_word = word.strip().lower()
        timestamp = str(int(time.time()))
        
        # Generate audio file
        audio_filename = f"word_{clean_word}_{timestamp}.wav"
        audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, audio_filename)
        
        if audio_generator.generate_word_audio(clean_word, audio_path):
            # Process audio quality
            audio_generator.process_audio_quality(audio_path)
            
            # Return the audio file directly
            return send_file(
                audio_path,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f"pronunciation_{clean_word}.wav"
            )
        else:
            return jsonify({
                "success": False, 
                "error": f"Could not generate audio for word '{clean_word}'"
            }), 500
            
    except Exception as e:
        logger.error(f"Word audio generation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_sentence_audio', methods=['POST'])
def get_sentence_audio():
    """Endpoint to generate and return audio for a complete sentence"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "Text required"}), 400
        
        sentence = data['text'].strip()
        slow = data.get('slow', True)
        
        print_section("SENTENCE AUDIO REQUEST", {
            "Text": sentence,
            "Slow Speech": slow
        })
        
        if not sentence:
            return jsonify({"success": False, "error": "Text cannot be empty"}), 400
        
        timestamp = str(int(time.time()))
        audio_filename = f"sentence_{timestamp}.wav"
        audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, audio_filename)
        
        if audio_generator.generate_sentence_audio(sentence, audio_path, slow=slow):
            # Process audio quality
            audio_generator.process_audio_quality(audio_path)
            
            # Return the audio file directly
            return send_file(
                audio_path,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f"pronunciation_sentence_{timestamp}.wav"
            )
        else:
            return jsonify({
                "success": False,
                "error": "Could not generate sentence audio"
            }), 500
            
    except Exception as e:
        logger.error(f"Sentence audio generation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """API information"""
    return jsonify({
        "service": "Dual-Model Pronunciation Analysis API", 
        "status": "active",
        "version": "9.0-DUAL-MODEL",
        "phoneme_models": {
            "wav2vec2": {
                "available": True,
                "trigger": "source=recording",
                "benefits": ["Faster processing", "Optimized for real-time", "Consistent with training"]
            }
        },
        "usage": {
            "upload_file": "POST /analyze_with_llm_judge with source=upload",
            "recording": "POST /analyze_with_llm_judge with source=recording"
        }
    })

@app.route('/get_reference_phonemes', methods=['POST'])
def get_reference_phonemes():
    """Endpoint to get reference phonemes for text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "Text required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"success": False, "error": "Text cannot be empty"}), 400
        
        system = get_improved_mdd_system()
        words = text.lower().split()
        word_phoneme_mapping = []
        phoneme_index = 0
        
        for word in words:
            try:
                word_phonemes = system._text_to_phonemes_espeak(word)
                if not word_phonemes or word_phonemes == ['<unk>']:
                    word_phonemes = system._basic_text_to_phonemes(word)
                
                word_phoneme_mapping.append({
                    "word": word,
                    "phonemes": word_phonemes,
                    "startIndex": phoneme_index,
                    "endIndex": phoneme_index + len(word_phonemes) - 1
                })
                
                phoneme_index += len(word_phonemes)
                
            except Exception as e:
                logger.warning(f"Could not get phonemes for word '{word}': {e}")
                word_phoneme_mapping.append({
                    "word": word,
                    "phonemes": [word],
                    "startIndex": phoneme_index,
                    "endIndex": phoneme_index
                })
                phoneme_index += 1
        
        return jsonify({
            "success": True,
            "word_phoneme_mapping": word_phoneme_mapping,
            "total_phonemes": phoneme_index
        })
        
    except Exception as e:
        logger.error(f"Phoneme mapping failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "mdd_system": ImprovedDynamicMDD is not None,
            "audio_generation": TTS_AVAILABLE or GTTS_AVAILABLE,
            "audio_processing": AUDIO_PROCESSING_AVAILABLE,
            "llm_groq": GROQ_AVAILABLE and LLM_JUDGE_CONFIG["groq"]["api_key"] is not None,
            "llm_huggingface": get_hf_client() is not None
        },
        "phoneme_models": {
            "wav2vec2": {
                "available": ImprovedDynamicMDD is not None,
                "use_case": "Real-time recordings (faster processing)",
                "format": "ARPA-like phonemes"
            }
        },
        "audio_engines": {
            "gtts": GTTS_AVAILABLE,
            "pyttsx3": TTS_AVAILABLE,
            "librosa": AUDIO_PROCESSING_AVAILABLE
        }
    })

@app.route('/analyze_phoneme_practice', methods=['POST'])
def analyze_phoneme_practice():
    """Analyze a SINGLE phoneme pronunciation practice"""
    try:
        # Get the audio file and target phoneme
        audio_file = request.files.get('audio_file')
        target_phoneme = request.form.get('phoneme')
        word_context = request.form.get('word_context', '')
        
        if not audio_file or not target_phoneme:
            return jsonify({
                'success': False,
                'error': 'Missing audio file or phoneme'
            }), 400
        
        # Save audio temporarily
        timestamp = int(time.time())
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_phoneme_{target_phoneme}.wav")
        audio_file.save(temp_audio_path)
        
        logger.info("="*60)
        logger.info(f"🎯 SINGLE PHONEME PRACTICE")
        logger.info(f"Target Phoneme: '{target_phoneme}'")
        logger.info(f"Word Context: '{word_context}'")
        logger.info("="*60)
        
        # CRITICAL: Extract phonemes with single-phoneme mode
        extraction_result = extract_phonemes_wav2vec2(
            temp_audio_path, 
            target_phoneme,
            is_single_phoneme=True  # THIS IS THE KEY FLAG
        )
        
        if not extraction_result['success']:
            return jsonify({
                'success': False,
                'error': 'Phoneme extraction failed'
            }), 500
        
        predicted_phonemes = extraction_result['predicted_phonemes']
        reference_phonemes = extraction_result['reference_phonemes']
        
        logger.info(f"📊 ANALYSIS:")
        logger.info(f"  Target: {target_phoneme}")
        logger.info(f"  Predicted: {predicted_phonemes}")
        logger.info(f"  Reference: {reference_phonemes}")
        
        # Calculate accuracy for single phoneme
        accuracy, status, feedback = calculate_single_phoneme_accuracy(
            target_phoneme,
            predicted_phonemes,
            word_context
        )
        
        logger.info(f"✅ RESULT:")
        logger.info(f"  Accuracy: {accuracy:.1f}%")
        logger.info(f"  Status: {status}")
        logger.info("="*60)
        
        # Build response
        result = {
            'success': True,
            'phoneme': target_phoneme,
            'word_context': word_context,
            'accuracy': accuracy,
            'status': status,
            'feedback': feedback,
            'analysis': {
                'predicted_phonemes': predicted_phonemes,
                'reference_phonemes': reference_phonemes,
                'target_phoneme': target_phoneme,
                'extraction_mode': extraction_result.get('mode', 'single_phoneme')
            }
        }
        
        # Clean up temp file
        try:
            os.remove(temp_audio_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Phoneme practice analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def calculate_single_phoneme_accuracy(target_phoneme: str, 
                                      predicted_phonemes: List[str],
                                      word_context: str = "") -> Tuple[float, str, str]:
    """
    Calculate accuracy for single phoneme practice
    
    Returns:
        (accuracy, status, feedback)
    """
    if not predicted_phonemes:
        return (0.0, 'mispronounced', 
                f"No clear sound detected. Please try saying '/{target_phoneme}/' more clearly.")
    
    # Check if target phoneme is in predicted phonemes
    if target_phoneme in predicted_phonemes:
        accuracy = 100.0
        status = 'correct'
        context = f" in '{word_context}'" if word_context else ""
        feedback = f"Perfect! You pronounced '/{target_phoneme}/'{context} correctly."
        return (accuracy, status, feedback)
    
    # Calculate similarity with phoneme groups
    similarity = calculate_phoneme_similarity_score(target_phoneme, predicted_phonemes)
    accuracy = similarity * 100
    
    # Determine status and feedback
    if accuracy >= 70:
        status = 'partial'
        predicted_str = predicted_phonemes[0] if predicted_phonemes else '?'
        feedback = (f"Close! You said something like '/{predicted_str}/' instead of '/{target_phoneme}/'. "
                   f"Try again and focus on the '/{target_phoneme}/' sound.")
    else:
        status = 'mispronounced'
        predicted_str = '/'.join(predicted_phonemes) if predicted_phonemes else '?'
        feedback = (f"Not quite. You said '/{predicted_str}/' but we're looking for '/{target_phoneme}/'. "
                   f"Listen to the correct pronunciation and try again.")
    
    return (accuracy, status, feedback)


def calculate_phoneme_similarity_score(target: str, predicted: List[str]) -> float:
    """
    Calculate similarity between target phoneme and predicted phonemes
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if target in predicted:
        return 1.0
    
    # Phoneme similarity groups
    similar_groups = {
        'vowels': ['a', 'e', 'i', 'o', 'u', 'æ', 'ɛ', 'ɪ', 'ɔ', 'ʊ', 'ə', 'ʌ', 'ɑ'],
        'front_vowels': ['i', 'ɪ', 'e', 'ɛ', 'æ'],
        'back_vowels': ['u', 'ʊ', 'o', 'ɔ', 'ɑ'],
        'plosives': ['p', 'b', 't', 'd', 'k', 'g'],
        'fricatives': ['f', 'v', 's', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'h'],
        'nasals': ['m', 'n', 'ŋ'],
        'liquids': ['l', 'r', 'ɹ'],
        'glides': ['w', 'j']
    }
    
    # Find which group the target belongs to
    target_groups = [group_name for group_name, phonemes in similar_groups.items() 
                     if target in phonemes]
    
    if not target_groups:
        return 0.0
    
    # Check if any predicted phoneme is in the same group
    max_similarity = 0.0
    for pred in predicted:
        for group_name in target_groups:
            if pred in similar_groups[group_name]:
                # More specific groups get higher similarity
                if group_name in ['front_vowels', 'back_vowels']:
                    max_similarity = max(max_similarity, 0.75)
                else:
                    max_similarity = max(max_similarity, 0.65)
    
    return max_similarity

def calculate_phoneme_similarity(predicted, reference):
    """Calculate similarity between predicted and reference phonemes"""
    if not predicted or not reference:
        return 0.0
    
    # For single phoneme, check if it matches
    if len(reference) == 1:
        target = reference[0]
        if target in predicted:
            return 1.0
        
        # Check for similar phonemes (vowel groups, consonant groups, etc.)
        similar_groups = {
            'vowels': ['a', 'e', 'i', 'o', 'u', 'æ', 'ɛ', 'ɪ', 'ɔ', 'ʊ', 'ə'],
            'plosives': ['p', 'b', 't', 'd', 'k', 'g'],
            'fricatives': ['f', 'v', 's', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'h'],
            'nasals': ['m', 'n', 'ŋ'],
            'liquids': ['l', 'r'],
            'glides': ['w', 'j']
        }
        
        # Find which group the target belongs to
        target_group = None
        for group_name, phonemes in similar_groups.items():
            if target in phonemes:
                target_group = phonemes
                break
        
        # Check if any predicted phoneme is in the same group
        if target_group:
            for pred in predicted:
                if pred in target_group:
                    return 0.7  # Partial credit for same category
        
        return 0.0
    
    # For multiple phonemes (shouldn't happen in phoneme practice)
    matches = sum(1 for p, r in zip(predicted, reference) if p == r)
    return matches / max(len(predicted), len(reference))

def calculate_phoneme_accuracy_realtime(target_phoneme, predicted_phonemes, reference_phonemes, word_context=None):
    """
    Real-time phoneme accuracy calculation
    Returns: (accuracy, status, feedback)
    """
    
    # EXACT MATCH - 100%
    if target_phoneme in predicted_phonemes:
        return 100.0, 'correct', f"Perfect! '{target_phoneme}' pronounced correctly."
    
    # CLOSE MATCH - Calculate similarity
    best_similarity = 0.0
    best_match = None
    
    for pred_ph in predicted_phonemes:
        if not pred_ph.strip():
            continue
        
        # Fast similarity calculation
        similarity = phoneme_similarity_score(target_phoneme, pred_ph)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = pred_ph
    
    # Determine status based on similarity
    if best_similarity >= 0.7:
        # PARTIAL - Close but not perfect
        accuracy = best_similarity * 100
        status = 'partial'
        feedback = f"Close! You said '{best_match}'. Focus on '{target_phoneme}' sound."
        
    elif best_similarity >= 0.4:
        # MISPRONOUNCED - Recognizable but wrong
        accuracy = best_similarity * 100
        status = 'mispronounced'
        feedback = f"You said '{best_match}' instead of '{target_phoneme}'. Try again."
        
    else:
        # MISPRONOUNCED - Not detected
        accuracy = 0.0
        status = 'mispronounced'
        if predicted_phonemes:
            feedback = f"'{target_phoneme}' not detected. You said: {' '.join(predicted_phonemes[:3])}"
        else:
            feedback = f"No clear sound detected. Speak louder and try again."
    
    return accuracy, status, feedback


def phoneme_similarity_score(ph1, ph2):
    """
    Fast phoneme similarity calculation
    Returns score between 0.0 and 1.0
    """
    if not ph1 or not ph2:
        return 0.0
    
    # Exact match
    if ph1 == ph2:
        return 1.0
    
    # Case-insensitive match
    if ph1.lower() == ph2.lower():
        return 0.95
    
    # Character overlap scoring
    max_len = max(len(ph1), len(ph2))
    min_len = min(len(ph1), len(ph2))
    
    if max_len == 0:
        return 0.0
    
    # Count matching positions
    matches = sum(c1 == c2 for c1, c2 in zip(ph1, ph2))
    
    # Base similarity
    similarity = (matches / max_len) * (min_len / max_len)
    
    # Phonetic class bonus
    if same_phonetic_class(ph1, ph2):
        similarity = min(1.0, similarity + 0.2)
    
    return similarity


def same_phonetic_class(ph1, ph2):
    """Quick check if phonemes are in same class"""
    
    # Vowels
    vowels = ['i', 'ɪ', 'e', 'ɛ', 'æ', 'ə', 'ʌ', 'ɜ', 'u', 'ʊ', 'o', 'ɔ', 'ɑ']
    
    # Stops
    stops = ['p', 'b', 't', 'd', 'k', 'g']
    
    # Fricatives
    fricatives = ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h']
    
    # Nasals
    nasals = ['m', 'n', 'ŋ']
    
    # Approximants
    approximants = ['l', 'r', 'w', 'j']
    
    # Check each class
    for phoneme_class in [vowels, stops, fricatives, nasals, approximants]:
        if ph1 in phoneme_class and ph2 in phoneme_class:
            return True
    
    return False


def generate_phoneme_audio_realtime(phoneme):
    """Generate phoneme audio in real-time - NO FALLBACKS"""
    
    try:
        timestamp = int(time.time())
        audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"phoneme_{phoneme}_{timestamp}.wav")
        
        # Generate audio
        if not audio_generator.generate_word_audio(phoneme, audio_path):
            return None
        
        # Process quality
        audio_generator.process_audio_quality(audio_path)
        
        # Convert to base64
        audio_base64 = audio_generator.audio_to_base64(audio_path)
        
        # Cleanup
        try:
            os.remove(audio_path)
        except:
            pass
        
        if audio_base64:
            return {
                "audio_base64": audio_base64,
                "phoneme": phoneme
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Audio generation failed for '{phoneme}': {e}")
        return None


@app.route('/get_phoneme_audio/<phoneme>', methods=['GET'])
def get_phoneme_audio(phoneme):
    """Real-time phoneme audio generation - NO FALLBACKS"""
    
    try:
        print_section("PHONEME AUDIO", {"Phoneme": phoneme})
        
        if not phoneme.strip():
            return jsonify({"success": False, "error": "Phoneme required"}), 400
        
        clean_phoneme = phoneme.strip()
        timestamp = int(time.time())
        
        # Generate audio
        audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"phoneme_{clean_phoneme}_{timestamp}.wav")
        
        if not audio_generator.generate_word_audio(clean_phoneme, audio_path):
            return jsonify({
                "success": False, 
                "error": "Audio generation failed"
            }), 500
        
        # Process quality
        audio_generator.process_audio_quality(audio_path)
        
        # Verify file
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return jsonify({
                "success": False, 
                "error": "Audio file invalid"
            }), 500
        
        # Return audio
        return send_file(
            audio_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"phoneme_{clean_phoneme}.wav"
        )
            
    except Exception as e:
        logger.error(f"Get phoneme audio failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# OPTIONAL: Batch phoneme analysis for efficiency
@app.route('/analyze_phonemes_batch', methods=['POST'])
def analyze_phonemes_batch():
    """Analyze multiple phonemes at once for better performance"""
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "Audio required"}), 400
        
        file = request.files['audio_file']
        phonemes = request.form.get('phonemes', '').strip().split(',')
        word_context = request.form.get('word_context', '').strip()
        
        if not phonemes:
            return jsonify({"success": False, "error": "Phonemes required"}), 400
        
        # Save audio
        filename = secure_filename(file.filename)
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        file.save(temp_audio_path)
        
        # Extract once for all phonemes
        reference_text = word_context if word_context else ' '.join(phonemes)
        phoneme_results = extract_phonemes_wav2vec2(temp_audio_path, reference_text)
        
        if not phoneme_results["success"]:
            os.remove(temp_audio_path)
            return jsonify({"success": False, "error": "Extraction failed"}), 500
        
        predicted_phonemes = phoneme_results["predicted_phonemes"]
        
        # Analyze each phoneme
        results = []
        for phoneme in phonemes:
            phoneme = phoneme.strip()
            if not phoneme:
                continue
            
            accuracy, status, feedback = calculate_phoneme_accuracy_realtime(
                target_phoneme=phoneme,
                predicted_phonemes=predicted_phonemes,
                reference_phonemes=phoneme_results["reference_phonemes"],
                word_context=word_context
            )
            
            results.append({
                "phoneme": phoneme,
                "accuracy": round(accuracy, 1),
                "status": status,
                "feedback": feedback
            })
        
        # Cleanup
        os.remove(temp_audio_path)
        
        return jsonify({
            "success": True,
            "results": results,
            "predicted_phonemes": predicted_phonemes,
            "word_context": word_context
        })
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        try:
            if 'temp_audio_path' in locals():
                os.remove(temp_audio_path)
        except:
            pass
        return jsonify({"success": False, "error": str(e)}), 500
    
if __name__ == "__main__":
    print_banner("STARTING DUAL-MODEL PRONUNCIATION SERVER", "=", 80)
    print_section("SERVER CONFIGURATION", {
        "Host": "0.0.0.0",
        "Port": "5000", 
        "wav2vec2": "Yes",
        "Audio Generation": "Enabled" if (TTS_AVAILABLE or GTTS_AVAILABLE) else "Disabled",
        "Groq LLM": "Yes" if GROQ_AVAILABLE and LLM_JUDGE_CONFIG["groq"]["api_key"] else "No"
    })
   
    
    print_banner("DUAL-MODEL SERVER READY", "=", 80)
    
    app.run(host="0.0.0.0", port=5050, debug=True, threaded=True)
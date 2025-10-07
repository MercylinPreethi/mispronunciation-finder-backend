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

import whisper

# LLM evaluation imports
from huggingface_hub import InferenceClient
from datetime import datetime

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


def extract_phonemes_wav2vec2(audio_path: str, reference_text: str) -> Dict[str, Any]:
    """
    Extract phonemes using wav2vec2 - returns actual predictions
    """
    try:
        print_banner("WAV2VEC2 PHONEME EXTRACTION", "=", 80)
        
        system = get_improved_mdd_system()
        raw_results = system.process_audio(audio_path, reference_text)
        
        predicted_phonemes = raw_results["analysis"]["predicted_phonemes"]
        reference_phonemes = raw_results["analysis"]["reference_phonemes"]
        
        
        return {
            "predicted_phonemes": predicted_phonemes,  # Use actual predictions
            "reference_phonemes": reference_phonemes,
            "model": "wav2vec2",
            "success": True,
            "raw_results": raw_results
        }
    except Exception as e:
        logger.error(f"Wav2vec2 extraction failed: {e}")
        return {
            "predicted_phonemes": [],
            "reference_phonemes": [],
            "model": "wav2vec2",
            "success": False,
            "error": str(e)
        }
    
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

        # Perform segmentation and analysis
        segmentation_result = universal_phoneme_segmentation(predicted_phonemes_clean, reference_text)
        words_analysis = analyze_segmented_words(segmentation_result)

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

        # Create word level results
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
            }
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

        # Build response
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


def universal_phoneme_segmentation(predicted_phonemes: List[str], reference_text: str) -> Dict[str, Any]:
    """
    Universal phoneme segmentation that works for any sentence
    Uses dynamic programming to find optimal word boundaries
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
    
    # DP table: dp[i][j] = min cost to align first i phonemes with first j words
    dp = [[float('inf')] * (n_words + 1) for _ in range(n_pred + 1)]
    word_boundaries = [[[] for _ in range(n_words + 1)] for _ in range(n_pred + 1)]
    
    # Base case
    dp[0][0] = 0
    word_boundaries[0][0] = []
    
    # Fill DP table
    for i in range(n_pred + 1):
        for j in range(n_words + 1):
            if dp[i][j] == float('inf'):
                continue
                
            if j < n_words:
                # Try different segment lengths for current word
                current_word_data = word_ref_phonemes[j]
                ref_phonemes = current_word_data["ref_phonemes"]
                ref_len = len(ref_phonemes)
                
                # Allow flexible segment lengths (0.5x to 2x reference length)
                min_segment = max(1, ref_len // 2)
                max_segment = min(n_pred - i, ref_len * 2)
                
                for seg_len in range(min_segment, max_segment + 1):
                    if i + seg_len <= n_pred:
                        segment = predicted_phonemes[i:i + seg_len]
                        
                        # Calculate alignment cost
                        aligned_ref, aligned_seg = needleman_wunsch_alignment(ref_phonemes, segment)
                        cost = sum(1 for r, s in zip(aligned_ref, aligned_seg) if r != s)
                        
                        # Add length penalty to prefer segments close to reference length
                        length_penalty = abs(seg_len - ref_len) * 0.1
                        total_cost = dp[i][j] + cost + length_penalty
                        
                        if total_cost < dp[i + seg_len][j + 1]:
                            dp[i + seg_len][j + 1] = total_cost
                            # Store the boundary and segment info
                            new_boundaries = word_boundaries[i][j] + [{
                                "word": current_word_data["word"],
                                "start": i,
                                "end": i + seg_len,
                                "predicted_phonemes": segment,
                                "reference_phonemes": ref_phonemes,
                                "alignment_cost": cost
                            }]
                            word_boundaries[i + seg_len][j + 1] = new_boundaries
    
    # Find the best alignment that uses all words
    if dp[n_pred][n_words] != float('inf'):
        best_boundaries = word_boundaries[n_pred][n_words]
    else:
        # Fallback: use proportional segmentation
        best_boundaries = proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)
    
    return best_boundaries


def proportional_segmentation_fallback(predicted_phonemes: List[str], word_ref_phonemes: List[Dict]) -> List[Dict]:
    """Fallback segmentation using proportional allocation"""
    boundaries = []
    total_ref_len = sum(w["length"] for w in word_ref_phonemes)
    current_pos = 0
    
    for i, word_data in enumerate(word_ref_phonemes):
        ref_len = word_data["length"]
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
            "alignment_cost": float('inf'),  # Unknown for fallback
            "method": "proportional_fallback"
        })
        current_pos = end_pos
    
    return boundaries

def analyze_segmented_words(segmentation_result: List[Dict]) -> List[Dict[str, Any]]:
    """Analyze each word segment with Needleman-Wunsch-based accuracy"""
    words_analysis = []
    
    for seg_data in segmentation_result:
        word = seg_data["word"]
        pred_phonemes = seg_data["predicted_phonemes"]
        ref_phonemes = seg_data["reference_phonemes"]
        
        # Use Needleman-Wunsch for optimal alignment
        aligned_ref, aligned_pred = needleman_wunsch_alignment(ref_phonemes, pred_phonemes)
        
        # Calculate match metrics based on ALIGNMENT
        total_positions = len(aligned_ref)
        correct_matches = sum(1 for r, p in zip(aligned_ref, aligned_pred) 
                             if r == p and r != "_" and p != "_")
        
        # Count error types from alignment
        substitutions = sum(1 for r, p in zip(aligned_ref, aligned_pred) 
                           if r != p and r != "_" and p != "_")
        deletions = sum(1 for r, p in zip(aligned_ref, aligned_pred) 
                       if r != "_" and p == "_")
        insertions = sum(1 for r, p in zip(aligned_ref, aligned_pred) 
                        if r == "_" and p != "_")
        
        # Reference phoneme count (excluding gaps)
        total_ref = len([r for r in aligned_ref if r != "_"])
        
        # Calculate accuracy based on Needleman-Wunsch alignment
        if total_ref == 0:
            word_accuracy = 0.0
            word_per = 100.0
        else:
            # Accuracy = correct matches / reference phonemes
            word_accuracy = (correct_matches / total_ref) * 100
            # PER = errors / reference phonemes
            word_per = ((substitutions + deletions + insertions) / total_ref) * 100
        
        # Build error list
        errors = []
        for j, (ref_p, pred_p) in enumerate(zip(aligned_ref, aligned_pred)):
            if ref_p == "_":
                errors.append({
                    "position": j,
                    "type": "insertion",
                    "predicted": pred_p,
                    "expected": None
                })
            elif pred_p == "_":
                errors.append({
                    "position": j,
                    "type": "deletion",
                    "predicted": None,
                    "expected": ref_p
                })
            elif ref_p != pred_p:
                errors.append({
                    "position": j,
                    "type": "substitution",
                    "predicted": pred_p,
                    "expected": ref_p
                })
        
        # STATUS DETERMINATION - Based on accuracy percentage
        if word_accuracy >= 90:
            status = "correct"  # 90%+ is correct
        elif word_accuracy >= 60:
            status = "partial"  # 60-89% is partial
        else:
            status = "mispronounced"  # <60% is mispronounced
        
        words_analysis.append({
            "word": word,
            "reference_phonemes": ref_phonemes,
            "predicted_phonemes": pred_phonemes,
            "aligned_reference": aligned_ref,
            "aligned_predicted": aligned_pred,
            "per": {"per": round(word_per, 2), "method": "needleman_wunsch"},
            "accuracy": round(word_accuracy, 2),  # ADD THIS
            "status": status,
            "phoneme_errors": errors,
            "error_count": len(errors),
            "error_counts": {
                "substitutions": substitutions,
                "insertions": insertions,
                "deletions": deletions,
                "correct": correct_matches
            },
            "match_quality": round(word_accuracy / 100, 2)
        })
    
    return words_analysis

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
                "phonemes": word_phonemes
            })
        except Exception as e:
            word_ref_phonemes.append({
                "word": word,
                "phonemes": [word]
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
                ref_phonemes = word_ref_phonemes[j]['phonemes']
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
    return proportional_segmentation_fallback(predicted_phonemes, word_ref_phonemes)

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
    """Dedicated endpoint for word-level pronunciation practice"""
    start_time = time.time()
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "Audio file required"}), 400

        file = request.files['audio_file']
        word = request.form.get('word', '').strip()
        
        if not word:
            return jsonify({"success": False, "error": "Word required"}), 400

        filename = secure_filename(file.filename)
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        file.save(temp_audio_path)

        # Extract phonemes for single word
        phoneme_results = extract_phonemes_wav2vec2(temp_audio_path, word)
        
        if not phoneme_results["success"]:
            raise Exception(f"Phoneme extraction failed: {phoneme_results.get('error', 'Unknown error')}")
        
        predicted_phonemes = phoneme_results["predicted_phonemes"]
        reference_phonemes = phoneme_results["reference_phonemes"]
        
        # Calculate PER for the word
        per_result = calculate_phoneme_error_rate(predicted_phonemes, reference_phonemes)
        accuracy = max(0, 100 - per_result["per"])
        
        # Determine status based on accuracy
        if accuracy >= 85:
            status = 'correct'
            feedback = f"Excellent! Your pronunciation of '{word}' is accurate ({accuracy:.1f}%)."
        elif accuracy >= 60:
            status = 'partial'
            feedback = f"Good attempt! Your pronunciation of '{word}' is {accuracy:.1f}% accurate. "
            # Add specific phoneme feedback
            if per_result['substitutions'] > 0:
                feedback += f"Watch out for {per_result['substitutions']} substituted sound(s). "
            if per_result['deletions'] > 0:
                feedback += f"You're missing {per_result['deletions']} sound(s). "
            if per_result['insertions'] > 0:
                feedback += f"You're adding {per_result['insertions']} extra sound(s). "
        else:
            status = 'mispronounced'
            feedback = f"Keep practicing '{word}'. Current accuracy: {accuracy:.1f}%. "
            # More detailed feedback for low scores
            feedback += f"\n\nExpected phonemes: {' '.join(reference_phonemes)}\n"
            feedback += f"Your phonemes: {' '.join(predicted_phonemes)}\n\n"
            if per_result['substitutions'] > 0:
                feedback += f"• {per_result['substitutions']} sounds need correction\n"
            if per_result['deletions'] > 0:
                feedback += f"• {per_result['deletions']} sounds are missing\n"
            if per_result['insertions'] > 0:
                feedback += f"• {per_result['insertions']} extra sounds to remove\n"
        
        # Create detailed analysis
        analysis_result = {
            "accuracy": accuracy / 100,
            "per": per_result["per"],
            "correct_phonemes": per_result["correct"],
            "total_phonemes": len(reference_phonemes),
            "predicted_phonemes": predicted_phonemes,
            "reference_phonemes": reference_phonemes,
            "error_breakdown": {
                "substitutions": per_result["substitutions"],
                "deletions": per_result["deletions"],
                "insertions": per_result["insertions"]
            }
        }
        
        processing_time = time.time() - start_time
        
        # Clean up
        try:
            os.remove(temp_audio_path)
        except:
            pass

        return jsonify({
            "success": True,
            "word": word,
            "status": status,
            "accuracy": accuracy,
            "feedback": feedback,
            "analysis": analysis_result,
            "metadata": {
                "processing_time": round(processing_time, 3),
                "phoneme_model": "wav2vec2"
            }
        })

    except Exception as e:
        logger.error(f"Word practice analysis failed: {e}", exc_info=True)
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
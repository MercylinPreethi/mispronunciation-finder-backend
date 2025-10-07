#!/usr/bin/env python3
"""
Improved Dynamic Mispronunciation Detection System
Better audio processing, phoneme extraction, and alignment
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import json
import time
import subprocess
import re
import scipy.signal
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC, 
    Wav2Vec2FeatureExtractor, Wav2Vec2Model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_asr_output(asr_text: str, reference_text: str) -> str:
    """Log ASR quality but NEVER return reference text"""
    ref_words = set(reference_text.lower().split())
    asr_words = asr_text.lower().split()
    
    # Check word overlap
    overlap = sum(1 for word in asr_words if word in ref_words)
    overlap_ratio = overlap / len(ref_words) if ref_words else 0
    
    if overlap_ratio < 0.5:
        logger.info(f"⚠ ASR output very different from reference ({overlap_ratio:.1%} match)")
        logger.info(f"ASR heard: '{asr_text}'")
        logger.info(f"Reference: '{reference_text}'")
        # CRITICAL: Still use ASR output, not reference!
    
    # ALWAYS return what ASR heard, never the reference
    return asr_text

class ImprovedDynamicMDD:
    def __init__(self, 
                 model_name: str = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme",
                 device: str = "auto"):
        
        self.device = self._setup_device(device)
        self.model_name = model_name
        
        logger.info("Initializing improved dynamic MDD system")
        self._load_models()
        self._init_phoneme_analysis()
        
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        return torch.device(device)
    
    def _load_models(self):
        """Load models with better error handling"""
        try:
            logger.info(f"Loading models: {self.model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.feature_model = Wav2Vec2Model.from_pretrained(self.model_name)
            
            self.asr_model.to(self.device)
            self.feature_model.to(self.device)
            self.asr_model.eval()
            self.feature_model.eval()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise Exception(f"Cannot initialize models: {str(e)}")
    
    def _init_phoneme_analysis(self):
        """Initialize phoneme analysis tools"""
        # IPA phoneme categories for better classification
        self.vowels = {
            'i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ʌ', 'ə', 'ɚ', 'ɝ',
            'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ'  # diphthongs
        }
        
        self.consonants = {
            'p', 'b', 't', 'd', 'k', 'g', 'ʔ',  # stops
            'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h',  # fricatives
            'tʃ', 'dʒ',  # affricates
            'm', 'n', 'ŋ',  # nasals
            'l', 'r', 'ɹ',  # liquids
            'w', 'j'  # glides
        }
        
        # Phoneme similarity matrix for better alignment
        self.phoneme_similarity = self._create_similarity_matrix()
    
    def _create_similarity_matrix(self):
        """Create phoneme similarity matrix"""
        # Simplified similarity - same category = higher similarity
        similarity = {}
        
        # High similarity within same category
        for p1 in self.vowels:
            for p2 in self.vowels:
                similarity[(p1, p2)] = 0.8 if p1 != p2 else 1.0
        
        for p1 in self.consonants:
            for p2 in self.consonants:
                similarity[(p1, p2)] = 0.6 if p1 != p2 else 1.0
        
        # Lower similarity across categories
        for v in self.vowels:
            for c in self.consonants:
                similarity[(v, c)] = similarity[(c, v)] = 0.2
        
        return similarity
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """Improved audio loading with format conversion"""
        logger.info(f"Loading audio: {audio_path}")
        
        # Try multiple loading approaches
        audio = None
        actual_sr = None
        
        # Method 1: torchaudio (often handles more formats)
        try:
            audio_tensor, actual_sr = torchaudio.load(audio_path)
            audio = audio_tensor.numpy().flatten()
            logger.info(f"Loaded with torchaudio: {len(audio)} samples at {actual_sr}Hz")
        except Exception as e:
            logger.warning(f"torchaudio failed: {e}")
        
        # Method 2: soundfile
        if audio is None:
            try:
                audio, actual_sr = sf.read(audio_path)
                logger.info(f"Loaded with soundfile: {len(audio)} samples at {actual_sr}Hz")
            except Exception as e:
                logger.warning(f"soundfile failed: {e}")
        
        # Method 3: librosa
        if audio is None:
            try:
                audio, actual_sr = librosa.load(audio_path, sr=None)
                logger.info(f"Loaded with librosa: {len(audio)} samples at {actual_sr}Hz")
            except Exception as e:
                logger.error(f"All audio loading methods failed: {e}")
                raise Exception(f"Cannot load audio file: {audio_path}")
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to target sample rate
        if actual_sr != target_sr:
            logger.info(f"Resampling from {actual_sr}Hz to {target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=actual_sr, target_sr=target_sr)
        
        # Normalize and clean
        if len(audio) > 0:
            # Remove DC offset
            audio = audio - np.mean(audio)
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
        
        # Ensure minimum length
        min_length = target_sr // 4  # 0.25 seconds minimum
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), 'constant')
        
        logger.info(f"Final audio: {len(audio)} samples, duration: {len(audio)/target_sr:.2f}s")
        return audio.astype(np.float32)
    
    def extract_phonemes_from_audio(self, audio: np.ndarray, reference_text: str = None) -> List[str]:
        """Improved phoneme extraction combining multiple approaches"""
        logger.info("Extracting phonemes using improved method")
        
        try:
            # Method 1: ASR transcription + phoneme conversion (with validation)
            asr_phonemes = self._asr_to_phonemes(audio, reference_text)
            
            # Method 2: Acoustic feature analysis
            acoustic_phonemes = self._acoustic_to_phonemes(audio)
            
            # Method 3: Hybrid approach - combine both
            final_phonemes = self._combine_phoneme_extractions(asr_phonemes, acoustic_phonemes)
            
            logger.info(f"Final phonemes: {final_phonemes}")
            return final_phonemes
            
        except Exception as e:
            logger.error(f"Phoneme extraction failed: {e}")
            return ['<unk>']

    def _asr_to_phonemes(self, audio: np.ndarray, reference_text: str = None) -> List[str]:
        """Extract phonemes via ASR transcription with validation"""
        try:
            with torch.no_grad():
                inputs = self.processor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                logits = self.asr_model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0].lower().strip()
                self._last_asr_transcription = transcription
                
                logger.info(f"ASR transcription: '{transcription}'")
                
                # VALIDATE ASR OUTPUT if reference provided
                if reference_text:
                    transcription = validate_asr_output(transcription, reference_text)
                
                # REMOVE DUPLICATES
                cleaned_transcription = self._remove_transcription_duplicates(transcription)
                if cleaned_transcription != transcription:
                    logger.info(f"Removed duplicates: '{cleaned_transcription}'")
                
                # Convert to phonemes
                if cleaned_transcription and not cleaned_transcription.isspace():
                    phonemes = self._text_to_phonemes_espeak(cleaned_transcription)
                    return phonemes
                    
        except Exception as e:
            logger.warning(f"ASR phoneme extraction failed: {e}")
        
        return ['<unk>']

    def process_audio(self, audio_path: str, reference_text: str) -> Dict[str, Any]:
        """Process audio with improved pipeline"""
        logger.info(f"Processing with improved system: '{reference_text}'")
        
        try:
            start_time = time.time()
            
            # Load audio
            audio = self.load_audio(audio_path)
            
            # Extract phonemes - BUT IGNORE THEM if ASR was bad
            predicted_phonemes = self.extract_phonemes_from_audio(audio, reference_text)  # Must pass it
            
            
            reference_phonemes = self.extract_reference_phonemes(reference_text)
            
    
            alignment_result = self.align_phonemes(predicted_phonemes, reference_phonemes)
            
            # Generate feedback
            feedback = self.generate_feedback(alignment_result, reference_text)
            
            processing_time = time.time() - start_time
            
            return {
                'audio_path': audio_path,
                'reference_text': reference_text,
                'analysis': alignment_result,
                'feedback': feedback,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'method': 'improved_dynamic'
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'audio_path': audio_path,
                'reference_text': reference_text,
                'analysis': {
                    'accuracy': 0.0,
                    'correct_phonemes': 0,
                    'total_phonemes': 0,
                    'predicted_phonemes': [],
                    'reference_phonemes': [],
                    'mispronunciations': []
                },
                'feedback': f"Analysis failed: {str(e)}",
                'timestamp': time.time(),
                'error': str(e),
                'method': 'improved_dynamic'
            }

    def _remove_transcription_duplicates(self, transcription: str) -> str:
        """Remove duplicate phrases from transcription"""
        words = transcription.split()
        if len(words) < 6:
            return transcription
        
        # Check if first half matches second half (indicating duplication)
        mid = len(words) // 2
        first_half = ' '.join(words[:mid])
        second_half = ' '.join(words[mid:])
        
        # If halves are very similar, use only first half
        similarity = self._calculate_text_similarity(first_half, second_half)
        if similarity > 0.7:  # 70% similarity threshold
            logger.info(f"Detected transcription duplication: {similarity:.1%} similarity")
            return first_half
        
        return transcription

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def _acoustic_to_phonemes(self, audio: np.ndarray) -> List[str]:
        """Extract phonemes using acoustic analysis"""
        try:
            # Extract acoustic features
            features = self._extract_acoustic_features(audio)
            
            # Segment audio into phoneme-like units
            segments = self._segment_audio(audio, features)
            
            # Classify each segment
            phonemes = []
            for segment in segments:
                phoneme = self._classify_segment(segment)
                phonemes.append(phoneme)
            
            return phonemes
            
        except Exception as e:
            logger.warning(f"Acoustic phoneme extraction failed: {e}")
            return ['<unk>']
    
    def _extract_acoustic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive acoustic features with fixed broadcasting"""
        features = {}
        
        # 1. Energy features
        frame_length = 512
        hop_length = 256
        
        # Ensure audio length is compatible with framing
        if len(audio) < frame_length:
            audio = np.pad(audio, (0, frame_length - len(audio)), 'constant')
        
        # Calculate number of frames to ensure consistent shapes
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        
        try:
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            
            # Ensure all features have the same number of frames
            features['energy'] = np.sum(frames**2, axis=0)[:n_frames]
            features['rms'] = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0][:n_frames]
            
            # 2. Spectral features with length control
            features['zcr'] = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0][:n_frames]
            features['spectral_centroids'] = librosa.feature.spectral_centroid(y=audio, sr=16000, hop_length=hop_length)[0][:n_frames]
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=16000, hop_length=hop_length)[0][:n_frames]
            
            # 3. MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, hop_length=hop_length)
            features['mfccs'] = mfccs[:, :n_frames]
            
            # 4. Formant-like features (simplified)
            stft = librosa.stft(audio, hop_length=hop_length)
            magnitude = np.abs(stft)
            features['formant_peaks'] = self._estimate_formants(magnitude[:, :n_frames])
            
            # Verify all features have consistent frame counts
            frame_counts = {}
            for key, value in features.items():
                if key != 'formant_peaks':  # Skip object array
                    if len(value.shape) == 1:
                        frame_counts[key] = len(value)
                    else:
                        frame_counts[key] = value.shape[1] if value.shape[1] < value.shape[0] else value.shape[0]
            
            logger.info(f"Feature frame counts: {frame_counts}")
            
            return features
            
        except Exception as e:
            logger.error(f"Acoustic feature extraction failed: {e}")
            # Return dummy features with consistent shape
            dummy_frames = max(1, len(audio) // hop_length)
            return {
                'energy': np.ones(dummy_frames) * 0.1,
                'rms': np.ones(dummy_frames) * 0.1,
                'zcr': np.ones(dummy_frames) * 0.05,
                'spectral_centroids': np.ones(dummy_frames) * 1000,
                'spectral_rolloff': np.ones(dummy_frames) * 2000,
                'mfccs': np.ones((13, dummy_frames)) * 0.1,
                'formant_peaks': np.array([np.array([800, 1200, 1600])] * dummy_frames, dtype=object)
            }

    def _segment_audio(self, audio: np.ndarray, features: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Segment audio into phoneme-like units with fixed shape handling"""
        try:
            # Use energy and spectral changes to find boundaries
            energy = features['energy']
            zcr = features['zcr']
            
            # Ensure both arrays have the same length
            min_length = min(len(energy), len(zcr))
            energy = energy[:min_length]
            zcr = zcr[:min_length]
            
            # Smooth features
            from scipy.ndimage import gaussian_filter1d
            energy_smooth = gaussian_filter1d(energy, sigma=2)
            zcr_smooth = gaussian_filter1d(zcr, sigma=2)
            
            # Find change points - ensure arrays are same length
            if len(energy_smooth) > 1 and len(zcr_smooth) > 1:
                energy_diff = np.abs(np.diff(energy_smooth))
                zcr_diff = np.abs(np.diff(zcr_smooth))
                
                # Make sure both diff arrays have same length
                min_diff_length = min(len(energy_diff), len(zcr_diff))
                energy_diff = energy_diff[:min_diff_length]
                zcr_diff = zcr_diff[:min_diff_length]
                
                # Combine change indicators
                change_score = energy_diff + zcr_diff * 10
            else:
                # Fallback for very short audio
                change_score = np.array([0.1])
            
            # Find peaks (potential boundaries)
            from scipy.signal import find_peaks
            if len(change_score) > 3:
                boundary_indices, _ = find_peaks(
                    change_score, 
                    height=np.mean(change_score) + np.std(change_score),
                    distance=max(1, int(0.05 * 16000 / 256))  # Minimum 50ms between boundaries
                )
            else:
                boundary_indices = []
            
            # Convert to audio sample indices
            hop_length = 256
            boundary_samples = boundary_indices * hop_length
            
            # Add start and end
            boundary_samples = np.concatenate([[0], boundary_samples, [len(audio)]])
            boundary_samples = np.unique(boundary_samples)
            boundary_samples = boundary_samples[boundary_samples < len(audio)]
            
            # Extract segments
            segments = []
            for i in range(len(boundary_samples) - 1):
                start = int(boundary_samples[i])
                end = int(boundary_samples[i + 1])
                
                if end > start and end <= len(audio):
                    segment = audio[start:end]
                    if len(segment) > 100:  # Minimum segment length
                        segments.append(segment)
            
            # Ensure we have at least one segment
            if not segments:
                segments = [audio]
            
            logger.info(f"Segmented audio into {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Audio segmentation failed: {e}")
            # Return single segment as fallback
            return [audio]
    
    def _estimate_formants(self, magnitude: np.ndarray) -> np.ndarray:
        """Simple formant estimation"""
        # Find spectral peaks that could represent formants
        formants = []
        for frame in magnitude.T:  # Iterate over time frames
            # Smooth the spectrum
            smoothed = gaussian_filter1d(frame, sigma=2)
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.1)
            
            # Take first few peaks as formant candidates
            formant_freqs = peaks[:3] if len(peaks) >= 3 else peaks
            formants.append(formant_freqs)
        
        return np.array(formants, dtype=object)
    
    
    def _classify_segment(self, segment: np.ndarray) -> str:
        """Classify audio segment as phoneme"""
        if len(segment) < 50:
            return '<sil>'
        
        # Calculate acoustic properties
        energy = np.mean(segment**2)
        zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
        
        # Simple classification based on acoustic properties
        if energy < 0.001:
            return '<sil>'
        elif zcr > 0.15:  # High ZCR -> fricative-like
            if energy > 0.05:
                return 's'  # Strong fricative
            else:
                return 'f'  # Weak fricative
        elif energy > 0.1:  # High energy -> vowel-like
            # Try to distinguish vowels by spectral properties
            try:
                mfccs = librosa.feature.mfcc(y=segment, sr=16000, n_mfcc=5)
                first_mfcc = np.mean(mfccs[1])  # Skip energy (mfcc[0])
                
                if first_mfcc > 0:
                    return 'i'  # Front vowel
                else:
                    return 'a'  # Back vowel
            except:
                return 'ə'  # Schwa default
        else:
            # Medium energy -> consonant
            return 't'  # Default consonant
    
    def _combine_phoneme_extractions(self, asr_phonemes: List[str], 
                                   acoustic_phonemes: List[str]) -> List[str]:
        """Combine phonemes from different extraction methods"""
        # If ASR worked well, prefer it
        if asr_phonemes and asr_phonemes != ['<unk>'] and len(asr_phonemes) > 1:
            return asr_phonemes
        
        # Otherwise use acoustic
        if acoustic_phonemes and len(acoustic_phonemes) > 1:
            # Clean up repetitive phonemes
            cleaned = []
            prev = None
            for phoneme in acoustic_phonemes:
                if phoneme != prev:
                    cleaned.append(phoneme)
                    prev = phoneme
            return cleaned
        
        # Fallback
        return ['<unk>']
    
    def _text_to_phonemes_espeak(self, text: str) -> List[str]:
        """Convert text to phonemes using eSpeak"""
        try:
            result = subprocess.run(
                ['espeak', '-q', '--ipa', text], 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=10
            )
            ipa_text = result.stdout.strip()
            
            if ipa_text:
                phonemes = self._parse_ipa(ipa_text)
                return phonemes if phonemes else ['<unk>']
                
        except Exception as e:
            logger.warning(f"eSpeak failed: {e}")
        
        # Fallback to basic conversion
        return self._basic_text_to_phonemes(text)
    
    def _parse_ipa(self, ipa_text: str) -> List[str]:
        """Parse IPA text to phoneme list"""
        # Remove stress markers and spaces
        cleaned = re.sub(r'[ˈˌ\s]', '', ipa_text)
        
        phonemes = []
        i = 0
        
        while i < len(cleaned):
            # Check for multi-character symbols first
            found = False
            for length in [3, 2, 1]:
                if i + length <= len(cleaned):
                    symbol = cleaned[i:i+length]
                    if symbol in (self.vowels | self.consonants):
                        phonemes.append(symbol)
                        i += length
                        found = True
                        break
            
            if not found:
                # Map unknown characters to closest known phoneme
                char = cleaned[i]
                mapped = self._map_unknown_char(char)
                if mapped:
                    phonemes.append(mapped)
                i += 1
        
        return phonemes
    
    def _map_unknown_char(self, char: str) -> Optional[str]:
        """Map unknown IPA characters to known phonemes"""
        mapping = {
            'ɪ': 'ɪ', 'i': 'i', 'e': 'e', 'ɛ': 'ɛ', 'æ': 'æ',
            'a': 'a', 'ɑ': 'ɑ', 'ɔ': 'ɔ', 'o': 'o', 'ʊ': 'ʊ', 'u': 'u',
            'ʌ': 'ʌ', 'ə': 'ə', 'ɚ': 'ɚ', 'ɝ': 'ɝ',
            'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
            'f': 'f', 'v': 'v', 's': 's', 'z': 'z', 'h': 'h',
            'm': 'm', 'n': 'n', 'l': 'l', 'r': 'r', 'w': 'w', 'j': 'j'
        }
        return mapping.get(char)
    
    def _basic_text_to_phonemes(self, text: str) -> List[str]:
        """Basic text to phoneme conversion"""
        phonemes = []
        for char in text.lower():
            if char in 'aeiou':
                phonemes.append('ə')  # Schwa for vowels
            elif char.isalpha():
                phonemes.append('t')  # Generic consonant
        
        return phonemes if phonemes else ['<unk>']
    
    def extract_reference_phonemes(self, text: str) -> List[str]:
        """Extract reference phonemes from text"""
        logger.info(f"Extracting reference phonemes for: '{text}'")
        return self._text_to_phonemes_espeak(text)
    
    def align_phonemes(self, predicted: List[str], reference: List[str]) -> Dict[str, Any]:
        """Align phoneme sequences with improved similarity scoring"""
        logger.info("Performing improved phoneme alignment")
        logger.info(f"Predicted: {predicted}")
        logger.info(f"Reference: {reference}")
        
        if not predicted:
            predicted = ['<sil>']
        if not reference:
            reference = ['<unk>']
        
        # Use dynamic programming with phoneme similarity
        aligned_pred, aligned_ref = self._align_with_similarity(predicted, reference)
        
        # Calculate accuracy with similarity-based scoring
        matches = 0
        total_score = 0
        mispronunciations = []
        
        for i, (pred, ref) in enumerate(zip(aligned_pred, aligned_ref)):
            similarity = self._get_phoneme_similarity(pred, ref)
            total_score += similarity
            
            if similarity >= 0.8:
                matches += 1
            elif pred != ref:
                error_type = 'deletion' if pred == '-' else 'insertion' if ref == '-' else 'substitution'
                severity = 'low' if similarity > 0.6 else 'medium' if similarity > 0.3 else 'high'
                
                mispronunciations.append({
                    'position': i,
                    'predicted': pred,
                    'reference': ref,
                    'type': error_type,
                    'severity': severity,
                    'similarity': similarity
                })
        
        accuracy = total_score / len(reference) if reference else 0
        
        logger.info(f"Improved alignment: {matches}/{len(reference)} exact, {accuracy:.2%} similarity-weighted")
        
        return {
            'predicted_phonemes': predicted,
            'reference_phonemes': reference,
            'aligned_predicted': aligned_pred,
            'aligned_reference': aligned_ref,
            'accuracy': accuracy,
            'total_phonemes': len(reference),
            'correct_phonemes': matches,
            'mispronunciations': mispronunciations
        }
    
    def _align_with_similarity(self, pred: List[str], ref: List[str]) -> Tuple[List[str], List[str]]:
        """Alignment using phoneme similarity scores"""
        m, n = len(pred), len(ref)
        
        # DP matrix for similarity-based alignment
        dp = np.zeros((m + 1, n + 1))
        
        # Initialize with gap penalties
        for i in range(1, m + 1):
            dp[i, 0] = dp[i-1, 0] - 0.5  # Insertion penalty
        for j in range(1, n + 1):
            dp[0, j] = dp[0, j-1] - 0.5  # Deletion penalty
        
        # Fill DP matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                similarity = self._get_phoneme_similarity(pred[i-1], ref[j-1])
                
                dp[i, j] = max(
                    dp[i-1, j-1] + similarity,  # Match/mismatch
                    dp[i-1, j] - 0.5,          # Insertion
                    dp[i, j-1] - 0.5           # Deletion
                )
        
        # Backtrack
        aligned_pred, aligned_ref = [], []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                similarity = self._get_phoneme_similarity(pred[i-1], ref[j-1])
                if dp[i, j] == dp[i-1, j-1] + similarity:
                    aligned_pred.append(pred[i-1])
                    aligned_ref.append(ref[j-1])
                    i -= 1
                    j -= 1
                elif dp[i, j] == dp[i-1, j] - 0.5:
                    aligned_pred.append(pred[i-1])
                    aligned_ref.append('-')
                    i -= 1
                else:
                    aligned_pred.append('-')
                    aligned_ref.append(ref[j-1])
                    j -= 1
            elif i > 0:
                aligned_pred.append(pred[i-1])
                aligned_ref.append('-')
                i -= 1
            else:
                aligned_pred.append('-')
                aligned_ref.append(ref[j-1])
                j -= 1
        
        aligned_pred.reverse()
        aligned_ref.reverse()
        
        return aligned_pred, aligned_ref
    
    def _get_phoneme_similarity(self, p1: str, p2: str) -> float:
        """Get similarity score between two phonemes"""
        if p1 == p2:
            return 1.0
        
        if p1 == '-' or p2 == '-':
            return 0.0
        
        # Use similarity matrix
        return self.phoneme_similarity.get((p1, p2), 0.1)
    
    def generate_feedback(self, alignment_result: Dict[str, Any], reference_text: str) -> str:
        """Generate improved feedback"""
        accuracy = alignment_result['accuracy']
        mispronunciations = alignment_result['mispronunciations']
        
        feedback = f"Improved Dynamic Analysis\n"
        feedback += f"Text: '{reference_text}'\n"
        feedback += f"Similarity Score: {accuracy:.1%}\n"
        feedback += f"Phonemes: {alignment_result['correct_phonemes']}/{alignment_result['total_phonemes']}\n\n"
        
        if accuracy > 0.9:
            feedback += "Excellent pronunciation!\n"
        elif accuracy > 0.7:
            feedback += "Good pronunciation with minor variations.\n"
        elif accuracy > 0.5:
            feedback += "Fair pronunciation with some issues to address.\n"
        else:
            feedback += "Significant pronunciation differences detected.\n"
        
        if mispronunciations:
            feedback += f"\nDetailed Analysis ({len(mispronunciations)} variations):\n"
            
            high_priority = [e for e in mispronunciations if e['severity'] == 'high']
            if high_priority:
                feedback += f"\nHigh Priority ({len(high_priority)}):\n"
                for i, error in enumerate(high_priority[:3], 1):
                    feedback += f"  {i}. {self._format_error_detailed(error)}\n"
        
        return feedback
    
    def _format_error_detailed(self, error: Dict[str, Any]) -> str:
        """Format error with similarity information"""
        similarity = error.get('similarity', 0)
        
        if error['type'] == 'substitution':
            return f"'{error['predicted']}' → '{error['reference']}' (similarity: {similarity:.1%})"
        elif error['type'] == 'deletion':
            return f"Missing: '{error['reference']}'"
        else:
            return f"Extra: '{error['predicted']}'"
    
    
def main():
    parser = argparse.ArgumentParser(description='Improved Dynamic MDD System')
    parser.add_argument('--audio_file', required=True)
    parser.add_argument('--reference_text', required=True)
    parser.add_argument('--model_name', default='mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output_file')
    
    args = parser.parse_args()
    
    try:
        system = ImprovedDynamicMDD(
            model_name=args.model_name,
            device=args.device
        )
        
        results = system.process_audio(args.audio_file, args.reference_text)
        
        print("\n" + "="*60)
        print("IMPROVED DYNAMIC PRONUNCIATION ANALYSIS")
        print("="*60)
        print(results['feedback'])
        print(f"Processing time: {results.get('processing_time', 0):.2f}s")
        print("="*60)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Mispronunciation Detection and Diagnosis System - Real-time Version
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import json
import requests
import time
import subprocess
import re

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MispronunciationDetectionSystem:
    def __init__(self, 
                 model_name: str = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme",
                 device: str = "auto",
                 groq_api_key: Optional[str] = None,
                 modal_url: Optional[str] = None,
                 use_local_model: bool = False):
        
        self.device = self._setup_device(device)
        self.groq_api_key = groq_api_key
        self.modal_url = modal_url
        self.use_local_model = use_local_model
        self.model_name = model_name
        
        if modal_url and not use_local_model:
            logger.info(f"Using Modal endpoint: {modal_url}")
            self.processor = None
            self.model = None
            self._test_modal_connection()
        else:
            logger.info(f"Using local model: {model_name}")
            self._load_local_model()
        
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        return torch.device(device)
    
    def _load_local_model(self):
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}, using fallback")
            self.model_name = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _test_modal_connection(self):
        if not self.modal_url:
            return
        try:
            health_url = self.modal_url.replace('/phoneme_recognition_endpoint', '/health')
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                logger.info("Modal connection successful")
        except Exception as e:
            logger.warning(f"Modal connection failed: {e}")
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        logger.info(f"Loading audio: {audio_path}")
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
        except Exception as e:
            logger.warning(f"Librosa failed: {e}, trying alternative loading")
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if sr != target_sr:
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio
    
    def extract_phonemes_from_audio(self, audio: np.ndarray) -> List[str]:
        if self.modal_url and not self.use_local_model:
            return self._extract_phonemes_modal(audio)
        else:
            return self._extract_phonemes_local(audio)
    
    def _extract_phonemes_modal(self, audio: np.ndarray) -> List[str]:
        try:
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            files = {'audio_file': ('audio.wav', audio_bytes, 'audio/wav')}
            data = {'sample_rate': 16000}
            
            response = requests.post(
                self.modal_url + '/phoneme_recognition_endpoint',
                files=files, data=data, timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                return result.get('phonemes', [])
            else:
                raise Exception(f"Modal failed: {result.get('error')}")
        except Exception as e:
            logger.warning(f"Modal failed: {e}, using local fallback")
            if hasattr(self, 'processor') and self.processor:
                return self._extract_phonemes_local(audio)
            raise e
    
    def _extract_phonemes_local(self, audio: np.ndarray) -> List[str]:
        if not hasattr(self, 'model') or self.model is None:
            raise Exception("Local model not available")
        
        with torch.no_grad():
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            phonemes = self._parse_phonemes(transcription)
        
        return phonemes
    
    def _parse_phonemes(self, transcription: str) -> List[str]:
        """Convert wav2vec2 word output to phonemes"""
        logger.info(f"Raw transcription: '{transcription}'")
        cleaned = re.sub(r'[<>|]', '', transcription)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        logger.info(f"Cleaned transcription: '{cleaned}'")
        
        # wav2vec2-base-960h outputs words, convert to phonemes
        if cleaned:
            words = cleaned.lower().split()
            phonemes = []
            for word in words:
                word_phonemes = self._word_to_phonemes(word)
                phonemes.extend(word_phonemes)
        else:
            phonemes = ['<UNK>']
        
        logger.info(f"Final phonemes from audio: {phonemes}")
        return phonemes if phonemes else ['<UNK>']
    
    def _word_to_phonemes(self, word: str) -> List[str]:
        """Convert English words to basic phonemes"""
        word_mappings = {
            'hello': ['h', 'ɛ', 'l', 'oʊ'],
            'world': ['w', 'ɝ', 'l', 'd'],
            'hi': ['h', 'aɪ'],
            'hey': ['h', 'eɪ'],
            'the': ['ð', 'ə'],
            'a': ['ə'],
            'and': ['æ', 'n', 'd'],
            'is': ['ɪ', 'z'],
            'are': ['ɑr'],
            'you': ['j', 'u'],
            'i': ['aɪ'],
            'we': ['w', 'i'],
            'they': ['ð', 'eɪ'],
            'good': ['g', 'ʊ', 'd'],
            'morning': ['m', 'ɔr', 'n', 'ɪ', 'ŋ'],
            'afternoon': ['æ', 'f', 't', 'ər', 'n', 'u', 'n'],
            'evening': ['i', 'v', 'n', 'ɪ', 'ŋ'],
            'night': ['n', 'aɪ', 't'],
            'how': ['h', 'aʊ'],
            'what': ['w', 'ʌ', 't'],
            'when': ['w', 'ɛ', 'n'],
            'where': ['w', 'ɛr'],
            'why': ['w', 'aɪ'],
            'who': ['h', 'u'],
            'yes': ['j', 'ɛ', 's'],
            'no': ['n', 'oʊ'],
            'ok': ['oʊ', 'k', 'eɪ'],
            'okay': ['oʊ', 'k', 'eɪ'],
            'please': ['p', 'l', 'i', 'z'],
            'thank': ['θ', 'æ', 'ŋ', 'k'],
            'thanks': ['θ', 'æ', 'ŋ', 'k', 's'],
            'sorry': ['s', 'ɔr', 'i'],
            'excuse': ['ɪ', 'k', 's', 'k', 'j', 'u', 'z'],
            'me': ['m', 'i']
        }
        
        if word in word_mappings:
            return word_mappings[word]
        else:
            return self._fallback_text_to_phonemes(word)
    
    def extract_phonemes_from_text(self, text: str) -> List[str]:
        try:
            result = subprocess.run(['espeak', '-q', '--ipa', text], 
                                  capture_output=True, text=True, check=True)
            ipa_transcription = result.stdout.strip()
            phonemes = self._parse_ipa_phonemes(ipa_transcription)
        except Exception as e:
            logger.warning(f"eSpeak failed: {e}, using fallback")
            phonemes = self._fallback_text_to_phonemes(text)
        
        logger.info(f"Reference phonemes: {phonemes}")
        return phonemes if phonemes else ['<UNK>']
    
    def _parse_ipa_phonemes(self, ipa_text: str) -> List[str]:
        ipa_clean = re.sub(r'[ˈˌ]', '', ipa_text)
        phonemes = []
        i = 0
        while i < len(ipa_clean):
            if ipa_clean[i].isspace():
                i += 1
                continue
            if i + 1 < len(ipa_clean):
                two_char = ipa_clean[i:i+2]
                if two_char in ['tʃ', 'dʒ', 'θ', 'ð', 'ʃ', 'ʒ', 'ŋ', 'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ']:
                    phonemes.append(two_char)
                    i += 2
                    continue
            phonemes.append(ipa_clean[i])
            i += 1
        return [p for p in phonemes if p and not p.isspace()]
    
    def _fallback_text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to basic phonemes using letter-to-sound rules"""
        basic_mapping = {
            'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɔ', 'u': 'ʌ',
            'th': 'θ', 'ch': 'tʃ', 'sh': 'ʃ', 'ng': 'ŋ',
            'h': 'h', 'w': 'w', 'r': 'r', 'l': 'l', 'd': 'd',
            'b': 'b', 'c': 'k', 'f': 'f', 'g': 'g', 'j': 'dʒ',
            'k': 'k', 'm': 'm', 'n': 'n', 'p': 'p', 'q': 'k',
            's': 's', 't': 't', 'v': 'v', 'x': 'ks', 'y': 'j', 'z': 'z'
        }
        
        text = text.lower()
        phonemes = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            if i + 1 < len(text):
                digraph = text[i:i+2]
                if digraph in basic_mapping:
                    phonemes.append(basic_mapping[digraph])
                    i += 2
                    continue
            char = text[i]
            if char.isalpha():
                phonemes.append(basic_mapping.get(char, char))
            i += 1
        return phonemes
    
    def align_phoneme_sequences(self, predicted_phonemes: List[str], 
                              reference_phonemes: List[str]) -> Dict[str, Any]:
        """Align phoneme sequences using dynamic programming"""
        logger.info("Aligning phoneme sequences")
        logger.info(f"Predicted: {predicted_phonemes}")
        logger.info(f"Reference: {reference_phonemes}")
        
        if not predicted_phonemes:
            predicted_phonemes = ['<silence>']
        if not reference_phonemes:
            reference_phonemes = ['<empty>']
        
        # Use dynamic programming for better alignment
        pred_len = len(predicted_phonemes)
        ref_len = len(reference_phonemes)
        
        # Create DP matrix
        dp = [[0] * (ref_len + 1) for _ in range(pred_len + 1)]
        
        # Initialize base cases
        for i in range(pred_len + 1):
            dp[i][0] = i
        for j in range(ref_len + 1):
            dp[0][j] = j
        
        # Fill DP matrix
        for i in range(1, pred_len + 1):
            for j in range(1, ref_len + 1):
                if predicted_phonemes[i-1] == reference_phonemes[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Match
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Insertion
                        dp[i][j-1],      # Deletion
                        dp[i-1][j-1]     # Substitution
                    )
        
        # Backtrack to find alignment
        aligned_pred = []
        aligned_ref = []
        i, j = pred_len, ref_len
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if predicted_phonemes[i-1] == reference_phonemes[j-1]:
                    aligned_pred.append(predicted_phonemes[i-1])
                    aligned_ref.append(reference_phonemes[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j-1] + 1:  # Substitution
                    aligned_pred.append(predicted_phonemes[i-1])
                    aligned_ref.append(reference_phonemes[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j] + 1:  # Insertion
                    aligned_pred.append(predicted_phonemes[i-1])
                    aligned_ref.append('-')
                    i -= 1
                else:  # Deletion
                    aligned_pred.append('-')
                    aligned_ref.append(reference_phonemes[j-1])
                    j -= 1
            elif i > 0:
                aligned_pred.append(predicted_phonemes[i-1])
                aligned_ref.append('-')
                i -= 1
            else:
                aligned_pred.append('-')
                aligned_ref.append(reference_phonemes[j-1])
                j -= 1
        
        # Reverse (backtracking gives reverse order)
        aligned_pred.reverse()
        aligned_ref.reverse()
        
        # Calculate accuracy and find errors
        matches = sum(1 for p, r in zip(aligned_pred, aligned_ref) 
                     if p == r and p != '-')
        
        mispronunciations = []
        for i, (pred, ref) in enumerate(zip(aligned_pred, aligned_ref)):
            if pred != ref:
                error_type = 'deletion' if pred == '-' else 'insertion' if ref == '-' else 'substitution'
                mispronunciations.append({
                    'position': i,
                    'predicted': pred,
                    'reference': ref,
                    'type': error_type
                })
        
        accuracy = matches / len(reference_phonemes) if reference_phonemes else 0
        
        logger.info(f"Alignment complete: {matches}/{len(reference_phonemes)} = {accuracy:.2%}")
        
        return {
            'predicted_phonemes': predicted_phonemes,
            'reference_phonemes': reference_phonemes,
            'aligned_predicted': aligned_pred,
            'aligned_reference': aligned_ref,
            'alignment_score': matches,
            'accuracy': accuracy,
            'total_phonemes': len(reference_phonemes),
            'correct_phonemes': matches,
            'mispronunciations': mispronunciations
        }
    
    def generate_feedback(self, alignment_result: Dict[str, Any], reference_text: str) -> str:
        accuracy = alignment_result['accuracy']
        mispronunciations = alignment_result['mispronunciations']
        
        feedback = f"Real-time Pronunciation Analysis\n"
        feedback += f"Text: '{reference_text}'\n"
        feedback += f"Accuracy: {accuracy:.1%}\n"
        feedback += f"Phonemes: {alignment_result['correct_phonemes']}/{alignment_result['total_phonemes']}\n\n"
        
        if len(mispronunciations) == 0:
            feedback += "Perfect pronunciation! Well done.\n"
        else:
            feedback += f"Areas to improve ({len(mispronunciations)} issues):\n"
            for i, error in enumerate(mispronunciations[:5], 1):
                if error['type'] == 'substitution':
                    feedback += f"{i}. Replace '{error['predicted']}' with '{error['reference']}'\n"
                elif error['type'] == 'deletion':
                    feedback += f"{i}. Add missing sound '{error['reference']}'\n"
                elif error['type'] == 'insertion':
                    feedback += f"{i}. Remove extra sound '{error['predicted']}'\n"
            
            if len(mispronunciations) > 5:
                feedback += f"... and {len(mispronunciations) - 5} more\n"
        
        return feedback
    
    def process_audio(self, audio_path: str, reference_text: str) -> Dict[str, Any]:
        logger.info(f"Real-time processing: {reference_text}")
        
        try:
            start_time = time.time()
            audio = self.load_audio(audio_path)
            predicted_phonemes = self.extract_phonemes_from_audio(audio)
            reference_phonemes = self.extract_phonemes_from_text(reference_text)
            alignment_result = self.align_phoneme_sequences(predicted_phonemes, reference_phonemes)
            feedback = self.generate_feedback(alignment_result, reference_text)
            processing_time = time.time() - start_time
            
            return {
                'audio_path': audio_path,
                'reference_text': reference_text,
                'analysis': alignment_result,
                'feedback': feedback,
                'processing_time': processing_time,
                'timestamp': time.time()
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
                'error': str(e)
            }

def main():
    parser = argparse.ArgumentParser(description='Real-time MDD System')
    parser.add_argument('--audio_file', required=True)
    parser.add_argument('--reference_text', required=True)
    parser.add_argument('--model_name', default='mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--groq_api_key')
    parser.add_argument('--modal_url')
    parser.add_argument('--use_local_model', action='store_true')
    parser.add_argument('--output_file')
    
    args = parser.parse_args()
    
    try:
        mdd_system = MispronunciationDetectionSystem(
            model_name=args.model_name,
            device=args.device,
            groq_api_key=args.groq_api_key,
            modal_url=args.modal_url,
            use_local_model=args.use_local_model
        )
        
        results = mdd_system.process_audio(args.audio_file, args.reference_text)
        
        print("\n" + "="*50)
        print("REAL-TIME PRONUNCIATION ANALYSIS")
        print("="*50)
        print(results['feedback'])
        print(f"Processing time: {results.get('processing_time', 0):.2f}s")
        print("="*50)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import sys
import argparse
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional
import tempfile

try:
    import sounddevice as sd
    import soundfile as sf
    import queue
except ImportError:
    print("Installing audio dependencies...")
    os.system("pip install sounddevice soundfile")
    import sounddevice as sd
    import soundfile as sf
    import queue

from mdd_cluster import MispronunciationDetectionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeMicrophoneMDD:
    def __init__(self,
                 sample_rate: int = 16000,
                 recording_timeout: float = 10.0,
                 silence_threshold: float = 0.01,
                 model_name: str = "facebook/wav2vec2-base-960h",
                 device: str = "auto"):
        
        self.sample_rate = sample_rate
        self.recording_timeout = recording_timeout
        self.silence_threshold = silence_threshold
        
        # Initialize MDD system
        logger.info("Initializing real-time MDD system...")
        self.mdd_system = MispronunciationDetectionSystem(
            model_name=model_name,
            device=device,
            use_local_model=True
        )
        
        self.audio_queue = queue.Queue()
        self.recording = False
        self.recorded_audio = []
        
        # Check audio devices
        self._check_audio_devices()
        logger.info("Real-time microphone system ready")
    
    def _check_audio_devices(self):
        """Check available audio devices"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            logger.info(f"Found {len(input_devices)} audio input devices")
            
            default_device = sd.query_devices(kind='input')
            logger.info(f"Using: {default_device['name']}")
        except Exception as e:
            logger.warning(f"Audio device check failed: {e}")
    
    def audio_callback(self, indata, frames, time, status):
        """Audio recording callback"""
        if status:
            logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Start recording from microphone"""
        self.recording = True
        self.recorded_audio = []
        
        print("🎤 Recording... Speak now!")
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self.audio_callback
            ):
                start_time = time.time()
                silent_chunks = 0
                min_recording_time = 0.5
                
                while self.recording:
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        self.recorded_audio.append(audio_chunk.flatten())
                        
                        # Check for silence
                        rms = np.sqrt(np.mean(audio_chunk**2))
                        if rms < self.silence_threshold:
                            silent_chunks += 1
                        else:
                            silent_chunks = 0
                        
                        elapsed_time = time.time() - start_time
                        
                        # Auto-stop conditions
                        if elapsed_time > self.recording_timeout:
                            print("⏰ Recording timeout")
                            break
                        
                        if elapsed_time > min_recording_time and silent_chunks > 15:
                            print("🔇 Silence detected")
                            break
                            
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        print("🛑 Recording stopped")
                        break
        
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
        
        self.recording = False
        
        if self.recorded_audio:
            full_audio = np.concatenate(self.recorded_audio)
            duration = len(full_audio) / self.sample_rate
            print(f"✅ Recorded {duration:.2f} seconds")
            return full_audio
        else:
            print("⚠️ No audio recorded")
            return None
    
    def analyze_recording(self, audio_data, reference_text):
        """Analyze recorded audio"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, self.sample_rate)
            temp_path = temp_file.name
        
        try:
            print("🔍 Analyzing pronunciation...")
            results = self.mdd_system.process_audio(temp_path, reference_text)
            os.unlink(temp_path)
            return results
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return {"error": str(e)}
    
    def display_results(self, results):
        """Display analysis results"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        analysis = results['analysis']
        accuracy = analysis['accuracy']
        
        print("\n" + "="*60)
        print("🎯 PRONUNCIATION ANALYSIS RESULTS")
        print("="*60)
        print(f"📊 Accuracy: {accuracy:.1%}")
        print(f"✅ Correct: {analysis['correct_phonemes']}/{analysis['total_phonemes']} phonemes")
        
        if analysis['mispronunciations']:
            print(f"\n❌ Issues found ({len(analysis['mispronunciations'])}):")
            for i, error in enumerate(analysis['mispronunciations'][:5], 1):
                if error['type'] == 'substitution':
                    print(f"   {i}. Replace '{error['predicted']}' with '{error['reference']}'")
                elif error['type'] == 'deletion':
                    print(f"   {i}. Add missing sound '{error['reference']}'")
                elif error['type'] == 'insertion':
                    print(f"   {i}. Remove extra sound '{error['predicted']}'")
        else:
            print("🎉 Perfect pronunciation!")
        
        print("\n💬 Feedback:")
        print(f"   {results['feedback']}")
        print("="*60)
    
    def practice_session(self, reference_text):
        """Single practice session"""
        print(f"\n📝 Practice text: \"{reference_text}\"")
        print("Press Enter when ready to speak...")
        input()
        
        audio_data = self.start_recording()
        if audio_data is not None:
            results = self.analyze_recording(audio_data, reference_text)
            self.display_results(results)
            return results
        return None
    
    def interactive_mode(self):
        """Interactive practice mode"""
        print("\n🎯 Interactive Pronunciation Practice")
        print("="*50)
        
        preset_texts = [
            "hello world",
            "good morning",
            "thank you very much",
            "how are you today",
            "nice to meet you",
            "have a good day"
        ]
        
        while True:
            try:
                print("\nOptions:")
                print("1. Enter custom text")
                print("2. Choose preset text")
                print("3. Quit")
                
                choice = input("Choose (1-3): ").strip()
                
                if choice == '1':
                    text = input("Enter text to practice: ").strip()
                    if not text:
                        print("Please enter some text.")
                        continue
                
                elif choice == '2':
                    print("\nPreset texts:")
                    for i, text in enumerate(preset_texts, 1):
                        print(f"  {i}. {text}")
                    
                    try:
                        idx = int(input("Choose (1-6): ")) - 1
                        if 0 <= idx < len(preset_texts):
                            text = preset_texts[idx]
                        else:
                            print("Invalid choice.")
                            continue
                    except ValueError:
                        print("Please enter a number.")
                        continue
                
                elif choice == '3':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("Invalid choice.")
                    continue
                
                # Practice the text
                results = self.practice_session(text)
                
                if results and "error" not in results:
                    again = input("\n🔄 Practice again? (y/n): ").strip().lower()
                    if again not in ['y', 'yes']:
                        continue
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

def main():
    parser = argparse.ArgumentParser(description='Real-time Microphone Pronunciation Analysis')
    parser.add_argument('--reference_text', help='Text to practice')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model_name', default='facebook/wav2vec2-base-960h', help='Model name')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device')
    parser.add_argument('--timeout', type=float, default=10.0, help='Recording timeout')
    
    args = parser.parse_args()
    
    if not args.interactive and not args.reference_text:
        parser.error("Either --reference_text or --interactive is required")
    
    try:
        # Initialize system
        rt_mdd = RealTimeMicrophoneMDD(
            recording_timeout=args.timeout,
            model_name=args.model_name,
            device=args.device
        )
        
        if args.interactive:
            rt_mdd.interactive_mode()
        else:
            rt_mdd.practice_session(args.reference_text)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
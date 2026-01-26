import torch
import whisper
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Union
import os
from configs.config import MODELS


class AudioTranscriber:
    """
    Whisper-based audio transcription for Indonesian language
    """
    
    def __init__(self, model_size: str = "small"):
        """
        Initialize Whisper model
        
        Args:
            model_size: One of ['tiny', 'base', 'small', 'medium', 'large']
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size, device=self.device)
        
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.language = MODELS["audio"]["language"]  # "id" for Indonesian
    
    def load_audio(self, audio_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load and preprocess audio
        
        Args:
            audio_input: Path to audio file or numpy array
            
        Returns:
            Audio data as numpy array
        """
        if isinstance(audio_input, str):
            # Load from file
            audio, sr = librosa.load(audio_input, sr=self.sample_rate, mono=True)
        else:
            audio = audio_input
            
        # Normalize
        audio = audio.astype(np.float32)
        
        # Ensure correct sample rate
        if hasattr(audio_input, 'sample_rate'):
            if audio_input.sample_rate != self.sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=audio_input.sample_rate, 
                    target_sr=self.sample_rate
                )
        
        return audio
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Simple noise reduction using spectral gating"""
        # Compute short-time Fourier transform
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Estimate noise profile from first 0.5 seconds
        noise_duration = int(0.5 * self.sample_rate / 512)
        noise_profile = np.mean(magnitude[:, :noise_duration], axis=1, keepdims=True)
        
        # Apply spectral gating
        mask = magnitude > (noise_profile * 1.5)
        stft_cleaned = stft * mask
        
        # Inverse STFT
        audio_cleaned = librosa.istft(stft_cleaned)
        
        return audio_cleaned
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio quality for better transcription"""
        # Apply noise reduction
        audio = self.apply_noise_reduction(audio)
        
        # Normalize volume
        audio = librosa.util.normalize(audio)
        
        # Apply pre-emphasis filter
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        return audio
    
    def transcribe(
        self, 
        audio_input: Union[str, np.ndarray],
        enhance: bool = True
    ) -> Dict:
        """
        Transcribe audio to text
        
        Args:
            audio_input: Path to audio file or numpy array
            enhance: Whether to apply audio enhancement
            
        Returns:
            Dictionary with transcription and metadata
        """
        # Load audio
        audio = self.load_audio(audio_input)
        
        # Enhance if requested
        if enhance:
            audio = self.enhance_audio(audio)
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            verbose=False
        )
        
        # Extract segments for detailed analysis
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "confidence": segment.get("no_speech_prob", 0.0)
            })
        
        return {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": segments,
            "duration": len(audio) / self.sample_rate
        }
    
    def transcribe_with_timestamps(
        self, 
        audio_input: Union[str, np.ndarray]
    ) -> Dict:
        """
        Transcribe with word-level timestamps
        
        Args:
            audio_input: Path to audio file or numpy array
            
        Returns:
            Dictionary with word-level timestamps
        """
        audio = self.load_audio(audio_input)
        
        result = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            word_timestamps=True,
            fp16=torch.cuda.is_available()
        )
        
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info["word"].strip(),
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "confidence": word_info.get("probability", 1.0)
                })
        
        return {
            "text": result["text"].strip(),
            "words": words,
            "language": result["language"]
        }


class AudioToExpense:
    """
    Convert audio transcription to expense information
    Uses the text extractor after transcription
    """
    
    def __init__(self, text_extractor=None):
        self.transcriber = AudioTranscriber()
        self.text_extractor = text_extractor
    
    def process(self, audio_input: Union[str, np.ndarray]) -> Dict:
        """
        Process audio to extract expense information
        
        Args:
            audio_input: Path to audio file or numpy array
            
        Returns:
            Dictionary with expense information
        """
        # Transcribe audio
        transcription = self.transcriber.transcribe(audio_input)
        
        result = {
            "transcription": transcription["text"],
            "language": transcription["language"],
            "duration": transcription["duration"],
            "segments": transcription["segments"]
        }
        
        # Extract expense info if text extractor available
        if self.text_extractor:
            expense_info = self.text_extractor.predict(transcription["text"])
            result.update({
                "category": expense_info["category"],
                "category_confidence": expense_info["category_confidence"],
                "amount": expense_info["amount"],
                "entities": expense_info["entities"],
                "embedding": expense_info["embedding"]
            })
        
        return result


if __name__ == "__main__":
    # Test transcription
    transcriber = AudioTranscriber()
    
    # For testing, you would need an actual audio file
    # This is a demonstration of the interface
    
    print("Audio Transcriber initialized successfully")
    print(f"Model loaded on: {transcriber.device}")
    print(f"Language: {transcriber.language}")
    
    # Example usage (uncomment when you have audio file):
    # result = transcriber.transcribe("path/to/audio.wav")
    # print(f"Transcription: {result['text']}")
    # print(f"Duration: {result['duration']:.2f}s")
    # print(f"Segments: {len(result['segments'])}")
    
    # Example with AudioToExpense:
    # from models.text_extractor import TextExtractor
    # text_extractor = TextExtractor()
    # audio_processor = AudioToExpense(text_extractor)
    # expense_result = audio_processor.process("path/to/audio.wav")
    # print(f"Category: {expense_result['category']}")
    # print(f"Amount: Rp {expense_result['amount']:,.0f}")

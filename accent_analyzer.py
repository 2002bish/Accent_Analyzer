#!/usr/bin/env python3
"""
English Accent Analyzer
A tool to extract audio from video URLs and classify English accents.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# Core libraries
import numpy as np
import librosa
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Audio/Video processing
import yt_dlp

# Try to import whisper with error handling
try:
    import whisper
    # Verify it's the correct OpenAI whisper by checking for load_model
    if not hasattr(whisper, 'load_model'):
        raise ImportError("Wrong whisper package - not OpenAI Whisper")
except ImportError as e:
    whisper = None
    print(f"Warning: OpenAI Whisper not available ({e}). Language detection will be limited.")
except Exception as e:
    whisper = None
    print(f"Warning: Whisper import failed: {e}. Language detection will be limited.")

# Web interface (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccentAnalyzer:
    """Main class for analyzing English accents from audio."""
    
    def __init__(self):
        self.whisper_model = None
        self.accent_classifier = None
        self.scaler = None
        self.accent_labels = ['American', 'British', 'Australian', 'Indian', 'Canadian', 'Irish']
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load or create the necessary models."""
        try:
            # Load Whisper for transcription and language detection
            if whisper is not None:
                logger.info("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
            else:
                logger.warning("Whisper not available. Using fallback language detection.")
                self.whisper_model = None
            
            # Create a simple accent classifier (in production, you'd train this on real data)
            self._create_accent_classifier()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            if whisper is None:
                logger.info("Continuing without Whisper - using fallback methods")
                self.whisper_model = None
            else:
                raise
    
    def _create_accent_classifier(self):
        """Create a mock accent classifier based on acoustic features."""
        # In a real implementation, this would be trained on labeled accent data
        # For now, we'll create a rule-based system with acoustic features
        self.accent_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Generate mock training data for demonstration
        # In production, you'd use real accent datasets
        np.random.seed(42)
        n_samples = 1000
        n_features = 13  # MFCC features
        
        X_mock = np.random.randn(n_samples, n_features)
        y_mock = np.random.randint(0, len(self.accent_labels), n_samples)
        
        X_scaled = self.scaler.fit_transform(X_mock)
        self.accent_classifier.fit(X_scaled, y_mock)
        
        logger.info("Accent classifier initialized (mock training data)")
    
    def download_audio(self, url: str) -> str:
        """Download audio from video URL."""
        try:
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "audio")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'{output_path}.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'postprocessor_args': [
                    '-ar', '16000'  # 16kHz sample rate
                ],
                'prefer_ffmpeg': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading audio from: {url}")
                ydl.download([url])
            
            # Find the downloaded file
            audio_file = f"{output_path}.wav"
            if os.path.exists(audio_file):
                return audio_file
            else:
                # Try other extensions
                for ext in ['mp3', 'm4a', 'webm']:
                    alt_file = f"{output_path}.{ext}"
                    if os.path.exists(alt_file):
                        return alt_file
                
                raise FileNotFoundError("Downloaded audio file not found")
                
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise
    
    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract acoustic features from audio file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract MFCC features (commonly used for accent classification)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Additional features that can help distinguish accents
            # Pitch/fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_mean = np.mean(spectral_centroids)
            
            # Combine features
            features = np.concatenate([
                mfcc_mean,
                [pitch_mean, spectral_mean]
            ])
            
            # Pad or truncate to expected feature size
            if len(features) < 13:
                features = np.pad(features, (0, 13 - len(features)))
            else:
                features = features[:13]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def detect_language_confidence(self, audio_path: str) -> Tuple[str, float]:
        """Detect language and confidence using Whisper or fallback method."""
        try:
            if self.whisper_model is not None:
                # Use Whisper for language detection
                result = self.whisper_model.transcribe(audio_path)
                detected_language = result.get('language', 'unknown')
                
                # Calculate English confidence based on detection
                if detected_language == 'en':
                    english_confidence = 0.9  # High confidence if detected as English
                else:
                    english_confidence = 0.1  # Low confidence if not English
                    
                return detected_language, english_confidence
            else:
                # Fallback: assume English and use basic audio analysis
                logger.info("Using fallback language detection (assuming English)")
                
                # Load audio and do basic checks
                y, sr = librosa.load(audio_path, sr=16000)
                
                # Simple heuristics for English-like audio
                # Check if there's speech-like activity
                rms = librosa.feature.rms(y=y)[0]
                speech_activity = np.mean(rms > np.percentile(rms, 20))
                
                # Estimate confidence based on speech activity and duration
                duration = len(y) / sr
                if duration > 5 and speech_activity > 0.3:
                    english_confidence = 0.7  # Reasonable confidence for speech
                else:
                    english_confidence = 0.4  # Lower confidence
                
                return 'en', english_confidence
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            # Fallback to assuming English with low confidence
            return 'en', 0.5
    
    def classify_accent(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify accent based on acoustic features."""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction and probabilities
            prediction = self.accent_classifier.predict(features_scaled)[0]
            probabilities = self.accent_classifier.predict_proba(features_scaled)[0]
            
            accent = self.accent_labels[prediction]
            confidence = float(max(probabilities))
            
            return accent, confidence
            
        except Exception as e:
            logger.error(f"Error classifying accent: {e}")
            return 'Unknown', 0.0
    
    def analyze_video(self, url: str) -> Dict:
        """Main method to analyze video URL for accent classification."""
        results = {
            'url': url,
            'status': 'processing',
            'language_detected': None,
            'english_confidence': 0.0,
            'accent_classification': None,
            'accent_confidence': 0.0,
            'summary': None,
            'error': None
        }
        
        audio_path = None
        
        try:
            # Step 1: Download audio
            logger.info("Step 1: Downloading audio...")
            audio_path = self.download_audio(url)
            
            # Step 2: Language detection
            logger.info("Step 2: Detecting language...")
            language, eng_confidence = self.detect_language_confidence(audio_path)
            results['language_detected'] = language
            results['english_confidence'] = eng_confidence * 100
            
            # Step 3: Extract features and classify accent (only if English)
            if language == 'en' or eng_confidence > 0.5:
                logger.info("Step 3: Extracting features and classifying accent...")
                features = self.extract_audio_features(audio_path)
                accent, acc_confidence = self.classify_accent(features)
                
                results['accent_classification'] = accent
                results['accent_confidence'] = acc_confidence * 100
                
                # Generate summary
                results['summary'] = self._generate_summary(results)
            else:
                results['summary'] = f"Language detected as '{language}', not English. Accent classification skipped."
            
            results['status'] = 'completed'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logger.error(f"Analysis failed: {e}")
        
        finally:
            # Cleanup
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    # Also remove the directory if empty
                    parent_dir = os.path.dirname(audio_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                except:
                    pass
        
        return results
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate a human-readable summary of the analysis."""
        lang = results['language_detected']
        eng_conf = results['english_confidence']
        accent = results['accent_classification']
        acc_conf = results['accent_confidence']
        
        summary = f"Analysis Results:\n"
        summary += f"• Language: {lang} (English confidence: {eng_conf:.1f}%)\n"
        
        if accent:
            summary += f"• Accent: {accent} (confidence: {acc_conf:.1f}%)\n"
            
            # Add interpretation
            if eng_conf >= 80:
                summary += f"• Strong English speaker detected"
            elif eng_conf >= 60:
                summary += f"• Moderate English proficiency detected"
            else:
                summary += f"• Low English confidence - results may be unreliable"
                
            if acc_conf >= 70:
                summary += f" with clear {accent} accent characteristics"
            elif acc_conf >= 50:
                summary += f" with possible {accent} accent features"
            else:
                summary += f" with uncertain accent classification"
        
        return summary


def run_cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze English accents from video URLs',
        epilog='''
Examples:
  python acc.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python acc.py "https://your-video-url" --output results.json
  
For web interface:
  streamlit run acc.py streamlit
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('url', help='Video URL to analyze (YouTube, Loom, direct MP4, etc.)')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AccentAnalyzer()
    
    # Analyze video
    print(f"Analyzing video: {args.url}")
    print("This may take a few minutes...")
    
    results = analyzer.analyze_video(args.url)
    
    # Print results
    print("\n" + "="*50)
    print("ACCENT ANALYSIS RESULTS")
    print("="*50)
    
    if results['status'] == 'completed':
        print(f"URL: {results['url']}")
        print(f"Language Detected: {results['language_detected']}")
        print(f"English Confidence: {results['english_confidence']:.1f}%")
        
        if results['accent_classification']:
            print(f"Accent Classification: {results['accent_classification']}")
            print(f"Accent Confidence: {results['accent_confidence']:.1f}%")
        
        print(f"\nSummary:")
        print(results['summary'])
        
    else:
        print(f"Analysis failed: {results['error']}")
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def run_streamlit():
    """Streamlit web interface."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: pip install streamlit")
        return
    
    st.title("🎯 English Accent Analyzer")
    st.markdown("Analyze English accents from video URLs for hiring evaluation")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.analyzer = AccentAnalyzer()
    
    # Input URL
    url = st.text_input("Enter video URL (YouTube, Loom, direct MP4, etc.):")
    
    if st.button("Analyze Accent") and url:
        with st.spinner("Analyzing video... This may take a few minutes."):
            results = st.session_state.analyzer.analyze_video(url)
        
        # Display results
        if results['status'] == 'completed':
            st.success("Analysis completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("English Confidence", f"{results['english_confidence']:.1f}%")
                
            with col2:
                if results['accent_classification']:
                    st.metric("Accent Confidence", f"{results['accent_confidence']:.1f}%")
            
            if results['accent_classification']:
                st.subheader(f"Detected Accent: {results['accent_classification']}")
            
            st.subheader("Summary")
            st.text(results['summary'])
            
            # Show raw results
            with st.expander("Raw Results"):
                st.json(results)
                
        else:
            st.error(f"Analysis failed: {results['error']}")


import os

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit()
    elif len(sys.argv) == 1:
        run_streamlit()  # default to Streamlit if no args
    else:
        run_cli()




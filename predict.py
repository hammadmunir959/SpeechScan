#!/usr/bin/env python3
"""
Dysarthria Detection and Gender Classification - Standalone Prediction Script
Command-line tool for predicting dysarthria and gender in audio files
Based on the implementation from dyarthria_model.py
"""

import argparse
import os
import sys
import time
import gc
import librosa
import numpy as np
import tensorflow as tf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import json
import pandas as pd

class DysarthriaGenderPredictor:
    def __init__(self, model_dir="ProcessedData/models"):
        """
        Initialize the dysarthria and gender predictor
        
        Args:
            model_dir: Directory containing the model files
        """
        self.model_dir = Path(model_dir)
        self.tf_model = None
        self.processor = None
        self.wav2vec_model = None
        
        # Audio processing parameters
        self.target_sr = 16000
        self.max_duration = 10
        self.min_duration = 2
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.load_models()
    
    def load_models(self):
        """Load the trained models"""
        try:
            print("🔄 Starting model loading process...")
            print(f"📁 Model directory: {self.model_dir}")
            print(f"🖥️ Device: {self.device}")
            
            # Load TensorFlow model
            print("📦 Loading TensorFlow model...")
            model_path = self.model_dir / "best_wav2vec2_fnn_model.h5"
            print(f"📄 Model file path: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"✅ Model file found, size: {model_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Load the complete model (architecture + weights)
            print("⏳ Loading TensorFlow model architecture and weights...")
            self.tf_model = tf.keras.models.load_model(model_path)
            print("✅ TensorFlow model loaded successfully")
            print(f"📊 TensorFlow model summary: {self.tf_model.count_params()} parameters")
            
            # Load Wav2Vec2 models
            print("🔄 Loading Wav2Vec2 processor...")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            print("✅ Wav2Vec2 processor loaded successfully")
            
            print("🔄 Loading Wav2Vec2 model...")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            self.wav2vec_model.eval()
            print("✅ Wav2Vec2 model loaded successfully")
            print(f"📊 Wav2Vec2 model parameters: {sum(p.numel() for p in self.wav2vec_model.parameters())}")
            
            print("🎉 All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            sys.exit(1)
    
    def preprocess_audio_for_prediction(self, audio_path, target_sr=16000, max_duration=10, min_duration=2):
        """Loads and cleans audio for feature extraction."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=target_sr)
            
            # Check if librosa successfully loaded the file
            if y is None or len(y) == 0:
                raise ValueError(f"Could not load or found empty audio data in {audio_path}.")
            
            # Remove silence from beginning and end
            y, _ = librosa.effects.trim(y, top_db=20)
            duration = len(y) / sr
            
            # Handle duration constraints
            if duration < min_duration:
                # Pad short audio
                required_length = int(min_duration * sr)
                if len(y) < required_length:
                    y = np.pad(y, (0, required_length - len(y)), mode='constant')
            elif duration > max_duration:
                # Truncate long audio
                y = y[:int(max_duration * sr)]
            
            # Normalize audio
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            return y
            
        except Exception as e:
            raise Exception(f"Audio preprocessing failed: {e}")
    
    def extract_features_for_prediction(self, audio_array, processor, model):
        """Extracts Wav2Vec2 global average features."""
        try:
            # Process audio
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # Move input to the same device as the model (GPU/CPU)
            input_values = inputs.input_values.to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = model(input_values)
            
            # Get hidden states
            hidden_states = outputs.last_hidden_state
            
            # Global average pooling and move to CPU before converting to numpy
            features = torch.mean(hidden_states, dim=1).cpu().squeeze().numpy()
            return features
            
        except Exception as e:
            raise Exception(f"Feature extraction failed: {e}")
    
    def predict_voice_health(self, audio_path):
        """Predicts voice health (Normal/Abnormal)."""
        try:
            # 1. Preprocess audio
            audio = self.preprocess_audio_for_prediction(audio_path)
            
            # 2. Extract features
            features = self.extract_features_for_prediction(audio, self.processor, self.wav2vec_model)
            
            # 3. Prepare features for Keras model
            features = features.reshape(1, -1)
            
            # 4. Predict
            prediction_prob = self.tf_model.predict(features, verbose=0)[0][0]
            prediction_label = "Abnormal (Unhealthy)" if prediction_prob > 0.5 else "Normal (Healthy)"
            
            return prediction_label, prediction_prob
            
        except Exception as e:
            print(f"An error occurred during health prediction for {Path(audio_path).name}: {e}")
            return "ERROR", np.nan
    
    def predict_gender_by_f0(self, audio, sr=16000):
        """Predicts gender using fundamental frequency (F0) estimation."""
        try:
            # F0 estimation
            f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            f0_mean = np.mean(f0_clean) if len(f0_clean) > 0 else 0
            
            # Standard thresholds
            if f0_mean < 100:
                return "Male", f0_mean
            elif f0_mean < 165:
                return "Male/Low-Pitched", f0_mean
            else:
                return "Female/High-Pitched", f0_mean
                
        except Exception as e:
            print(f"Error in F0 estimation: {e}")
            return "ERROR", np.nan
    
    def predict_gender(self, audio_path):
        """Drives the gender prediction process."""
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Predict based on F0
            gender, f0_mean = self.predict_gender_by_f0(audio, sr)
            return gender, f0_mean
            
        except Exception as e:
            print(f"An error occurred during gender prediction for {Path(audio_path).name}: {e}")
            return "ERROR", np.nan
    
    def predict(self, audio_path):
        """
        Predict both dysarthria and gender from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Prediction results
        """
        start_time = time.time()
        
        try:
            # Validate file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Health Prediction
            health_class, health_prob = self.predict_voice_health(audio_path)
            
            # Gender Prediction
            gender_class, f0_mean = self.predict_gender(audio_path)
            
            # Calculate confidence for health prediction
            if not np.isnan(health_prob):
                confidence_score = max(health_prob, 1 - health_prob)
                if confidence_score > 0.8:
                    confidence = "High"
                elif confidence_score > 0.6:
                    confidence = "Medium"
                else:
                    confidence = "Low"
            else:
                confidence = "Unknown"
                confidence_score = 0.0
            
            processing_time = time.time() - start_time
            
            return {
                "file_path": audio_path,
                "health_prediction": health_class,
                "health_probability": float(health_prob) if not np.isnan(health_prob) else None,
                "health_confidence": confidence,
                "health_confidence_score": float(confidence_score),
                "gender_prediction": gender_class,
                "f0_mean_hz": float(f0_mean) if not np.isnan(f0_mean) else None,
                "processing_time": round(processing_time, 3)
            }
            
        except Exception as e:
            return {
                "file_path": audio_path,
                "error": str(e),
                "health_prediction": None,
                "gender_prediction": None
            }
    
    def predict_batch(self, audio_dir, output_file=None):
        """
        Predict dysarthria and gender for all audio files in a directory
        
        Args:
            audio_dir: Directory containing audio files
            output_file: Optional output file for results (CSV format)
            
        Returns:
            list: List of prediction results
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {audio_dir}")
        
        # Supported audio extensions
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a')
        
        # Get all audio files
        audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print("⚠️ WARNING: No audio files found in the directory.")
            return []
        
        print(f"Found {len(audio_files)} audio files to process...")
        
        results = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"\n--- PROCESSING FILE {i+1}/{len(audio_files)}: {audio_file.name} ---")
            
            # Make prediction
            result = self.predict(audio_file)
            results.append(result)
            
            # Cleanup
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Save results if requested
        if output_file:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Dysarthria Detection and Gender Classification")
    parser.add_argument("input", help="Path to audio file or directory")
    parser.add_argument("--model-dir", default="ProcessedData/models", 
                       help="Directory containing model files (default: ProcessedData/models)")
    parser.add_argument("--output", "-o", help="Output file for results (CSV format for batch, JSON for single file)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all audio files in directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DysarthriaGenderPredictor(model_dir=args.model_dir)
    
    if args.batch or os.path.isdir(args.input):
        # Batch processing
        results = predictor.predict_batch(args.input, args.output)
        
        if results:
            print("\n" + "="*80)
            print("FINAL BATCH PREDICTION RESULTS")
            print("="*80)
            
            # Display results in a table
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            
            # Summary statistics
            if len(results) > 0:
                health_predictions = [r['health_prediction'] for r in results if r['health_prediction']]
                gender_predictions = [r['gender_prediction'] for r in results if r['gender_prediction']]
                
                print(f"\n📊 SUMMARY:")
                print(f"   Total files processed: {len(results)}")
                print(f"   Health predictions: {len(health_predictions)}")
                print(f"   Gender predictions: {len(gender_predictions)}")
                
                if health_predictions:
                    normal_count = sum(1 for p in health_predictions if "Normal" in p)
                    abnormal_count = sum(1 for p in health_predictions if "Abnormal" in p)
                    print(f"   Normal voices: {normal_count}")
                    print(f"   Abnormal voices: {abnormal_count}")
    
    else:
        # Single file processing
        result = predictor.predict(args.input)
        
        # Handle errors
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        
        # Display results
        print(f"\n🎤 Dysarthria Detection and Gender Classification Results")
        print(f"{'='*60}")
        print(f"File: {result['file_path']}")
        print(f"\n🏥 Health Analysis:")
        print(f"   Prediction: {result['health_prediction']}")
        print(f"   Probability (Abnormal): {result['health_probability']:.3f}" if result['health_probability'] else "   Probability: N/A")
        print(f"   Confidence: {result['health_confidence']} ({result['health_confidence_score']:.3f})")
        
        print(f"\n👤 Gender Analysis:")
        print(f"   Prediction: {result['gender_prediction']}")
        print(f"   F0 Mean: {result['f0_mean_hz']:.2f} Hz" if result['f0_mean_hz'] else "   F0 Mean: N/A")
        
        print(f"\n⏱️  Processing Time: {result['processing_time']}s")
        
        # Additional interpretation
        if result['health_prediction'] == "Abnormal (Unhealthy)":
            print(f"\n⚠️  The audio shows signs of dysarthria (abnormal speech patterns)")
            print(f"   This could indicate speech disorders, neurological conditions,")
            print(f"   or other factors affecting speech clarity.")
        elif result['health_prediction'] == "Normal (Healthy)":
            print(f"\n✅ The audio appears to have normal speech patterns")
            print(f"   No significant signs of dysarthria detected.")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
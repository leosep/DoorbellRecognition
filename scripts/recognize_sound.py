import librosa
import numpy as np
import sounddevice as sd
from scipy.spatial.distance import cosine
import os
import sys
import alert
import time
import scipy.io.wavfile as wav
from datetime import datetime

# Set the sampling rate for recording
SAMPLING_RATE = 16000  # Adjust as needed for your microphone
DURATION = 15  # Duration of each recording in seconds (adjust as needed)
SIMILARITY_THRESHOLD = 0.92  # Lower threshold for testing detection
SIMILARITY_WINDOW_SIZE = 5  # Number of past similarity scores to average
similarity_window = []  # Store past similarity scores for smoothing

def normalize_audio(audio_data):
    """Normalize audio to the range [-1, 1]."""
    audio_max = np.max(np.abs(audio_data))
    if audio_max > 0:
        return audio_data / audio_max
    return audio_data

def calculate_rms(audio_data):
    """Calculate the RMS (Root Mean Square) of the audio data."""
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms

def record_audio(duration, sampling_rate):
    """Capture audio from the microphone and normalize it."""
    print("Recording...")
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    
    # Normalize the audio volume to be between -1 and 1
    audio_data = normalize_audio(audio_data.flatten())
    
    rms = calculate_rms(audio_data)
    print(f"RMS of recorded audio: {rms}")
    
    if rms < 0.01:  # If the RMS is too low, consider it too quiet
        print("Warning: Audio is too quiet.")
        return None  # Optionally return None for very low volume
    return audio_data

def save_audio_to_wav(audio_data, filename, sampling_rate):
    """Save the recorded audio to a .wav file."""
    # Convert audio data to 16-bit PCM format for .wav
    audio_data_int16 = np.int16(audio_data * 32767)  # Scale to 16-bit PCM
    wav.write(filename, sampling_rate, audio_data_int16)
    print(f"Saved audio to {filename}")

def extract_features(y, sr):
    """Extract MFCC and Chroma features from audio"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Mean of MFCC coefficients
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # Mean of Chroma features
    
    features = np.concatenate((mfcc_mean, chroma_mean))  # Combine MFCC and Chroma
    print(f"Extracted combined features (MFCC + Chroma): {features}")
    return features

def compare_audio(audio_data, reference_path, sampling_rate=SAMPLING_RATE):
    """Compare the audio input with a reference audio file using cosine similarity"""
    # Convert the recorded audio into features
    features_audio = extract_features(audio_data, sampling_rate)

    # Load reference audio and extract features
    y_ref, sr_ref = librosa.load(reference_path, sr=sampling_rate)
    features_ref = extract_features(y_ref, sr_ref)

    # Compare audio features using cosine similarity
    similarity = 1 - cosine(features_audio, features_ref)
    print(f"Similarity between recorded and reference audio: {similarity}")  # Log similarity
    return similarity

def is_doorbell_sound(similarity, threshold=SIMILARITY_THRESHOLD):
    """Smoothing and final decision based on rolling average of similarities"""
    similarity_window.append(similarity)
    
    if len(similarity_window) > SIMILARITY_WINDOW_SIZE:
        similarity_window.pop(0)

    # Calculate the weighted average of similarities, giving more weight to recent recordings
    weighted_similarity = np.mean(similarity_window[-SIMILARITY_WINDOW_SIZE:])

    print(f"Weighted average similarity over {SIMILARITY_WINDOW_SIZE} recordings: {weighted_similarity}")

    # Only detect if the weighted similarity exceeds the threshold
    if weighted_similarity > threshold:
        return True
    return False

def recognize_doorbell_sound(audio_data, reference_path):
    """Recognize if the doorbell sound is detected from live audio"""
    similarity = compare_audio(audio_data, reference_path)
    print(f"Audio similarity: {similarity}")  # Log raw similarity

    # Trigger detection if similarity exceeds threshold in one recording
    if similarity > SIMILARITY_THRESHOLD:
        return True
    return False

if __name__ == '__main__':
    reference_file = '../audio/doorbell_sound.wav'  # Doorbell sound file

    while True:
        # Record audio from the microphone (small chunk)
        audio_data = record_audio(DURATION, SAMPLING_RATE)

        if audio_data is None:
            print("Skipping recording due to low volume.")
            continue  # Skip the current iteration if audio is too quiet

        # Optionally, you can save the recorded audio to a .wav file for inspection
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"recording_{timestamp}.wav"
        # save_audio_to_wav(audio_data, filename, SAMPLING_RATE)

        # Check if the doorbell sound is recognized
        if recognize_doorbell_sound(audio_data, reference_file):
            print("Doorbell sound detected!")
            # Send Pushbullet notification via the alert.py script
            alert.send_push_notification("Someone is at the door!")

        # Optionally, you can add a delay before the next recording
        print(f"Waiting for seconds before next recording...")
        time.sleep(3)  # Delay to avoid overloading

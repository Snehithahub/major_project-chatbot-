import speech_recognition as sr
from pydub import AudioSegment
import os
import warnings

AudioSegment.converter = None
warnings.simplefilter("ignore", RuntimeWarning)

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            print("🎧 Processing Audio File...")
            recognizer.adjust_for_ambient_noise(source) 
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        print(f"🗣 Recognized Text: {text}")
        return text

    except sr.UnknownValueError:
        print("⚠️ Could not understand the audio!")
        return None
    except sr.RequestError:
        print("⚠️ Could not request results from Google Speech Recognition service!")
        return None
    except FileNotFoundError:
        print(f"❌ File not found: {audio_path}")
        return None

# Your MP3 file path
mp3_file_path = r"C:\Users\SHENATEJA\Downloads\harvard.wav"
audio_text = audio_to_text(mp3_file_path)


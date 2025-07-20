from flask import Flask, request, jsonify
from flask_cors import CORS
import pyttsx3
from personality_engine import Personality
import threading
import json
import os
import uuid
from typing import Dict, List, Optional
import time # NEW: Import the time module for delays

# --- Import shared resources ---
from sentence_transformers import SentenceTransformer
from mood_engine.offline_mood_classifier import OfflineMoodClassifier
import ollama

# --- One-time loading of heavy, shared resources ---
print("[Startup] Initializing shared resources... This may take a moment.")

# --- NEW: Retry loop for Ollama connection ---
ollama_connected = False
for i in range(3): # Try to connect 3 times
    try:
        print(f"[Startup] Checking for Ollama connection (Attempt {i+1}/3)...")
        ollama.list()
        print("[Startup] Ollama connection successful.")
        ollama_connected = True
        break # Exit the loop if successful
    except Exception as e:
        print(f"[Startup] Ollama connection failed. Retrying in 5 seconds...")
        time.sleep(5) # Wait for 5 seconds before trying again

if not ollama_connected:
    print(f"[Startup] FATAL ERROR: Could not connect to Ollama after several attempts. Is it running?")
    exit() # Exit if the connection fails after all retries
# --- End of new retry loop ---

# Load the single, shared embedding model instance
try:
    print("[Startup] Loading embedding model (all-MiniLM-L6-v2)...")
    SHARED_EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    print("[Startup] Embedding model loaded successfully.")
except Exception as e:
    print(f"[Startup] FATAL ERROR: Could not load embedding model. {e}")
    exit()

# Load the mood mappings data once
MOOD_MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), 'mood_engine', 'mood_mappings.json')
try:
    with open(MOOD_MAPPINGS_PATH, 'r', encoding="utf-8") as f:
        mood_mappings_data = json.load(f)
    semantic_mood_data = mood_mappings_data.get('semantic_mood_mapping', [])
except Exception as e:
    print(f"[Startup] FATAL ERROR: Could not load mood mappings from {MOOD_MAPPINGS_PATH}. {e}")
    exit()

# Create the single, shared mood classifier instance
print("[Startup] Training shared mood classifier...")
SHARED_MOOD_CLASSIFIER = OfflineMoodClassifier(
    semantic_mood_data=semantic_mood_data,
    embedder=SHARED_EMBEDDER
)
print("[Startup] Shared resources initialized successfully.")
# --- End of one-time loading ---


app = Flask(__name__)
CORS(app)

active_sessions: Dict[str, Personality] = {}
session_lock = threading.Lock()

# --- TTS Engine Initialization and Configuration ---
tts_engine = pyttsx3.init()
tts_lock = threading.Lock()

MOOD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mood_engine', 'mood_config.json')
tts_settings = {}
try:
    with open(MOOD_CONFIG_PATH, 'r', encoding="utf-8") as f:
        config_data = json.load(f)
        tts_settings = config_data.get('tts_settings', {})
    print(f"[TTS] Loaded TTS settings from {MOOD_CONFIG_PATH}")
except Exception as e:
    print(f"[TTS] Warning: Could not load TTS settings. {e}")

DEFAULT_VOICE_ID = tts_settings.get('default_voice_id', None)
DEFAULT_RATE = tts_settings.get('default_rate', 175)
DEFAULT_VOLUME = tts_settings.get('default_volume', 0.9)
MOOD_VOICE_MODIFIERS = tts_settings.get('mood_voice_modifiers', [])

if DEFAULT_VOICE_ID:
    voices = tts_engine.getProperty('voices')
    found_voice = False
    for voice in voices:
        if voice.id == DEFAULT_VOICE_ID:
            tts_engine.setProperty('voice', voice.id)
            print(f"[TTS] Set default voice to: {voice.name} (ID: {voice.id})")
            found_voice = True
            break
    if not found_voice:
        print(f"[TTS] Warning: Configured default voice ID '{DEFAULT_VOICE_ID}' not found. Using system default.")
else:
    print("[TTS] No default voice ID configured. Using system default.")

tts_engine.setProperty('rate', DEFAULT_RATE)
tts_engine.setProperty('volume', DEFAULT_VOLUME)


def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("\n--- Available TTS Voices ---")
    for i, voice in enumerate(voices):
        print(f"  {i+1}. Name: {voice.name} | ID: {voice.id}")
    print("----------------------------")
    engine.stop()

list_voices()

def speak(text: str, current_mood_vector: Dict[str, float]):
    global tts_engine
    with tts_lock:
        try:
            if text and text.strip() != "...":
                tts_engine.setProperty('rate', DEFAULT_RATE)
                tts_engine.setProperty('volume', DEFAULT_VOLUME)
                effective_rate_delta = 0
                effective_volume_delta = 0
                for modifier in MOOD_VOICE_MODIFIERS:
                    mood_band = modifier.get('mood')
                    threshold = modifier.get('threshold')
                    if mood_band and mood_band in current_mood_vector and current_mood_vector[mood_band] >= threshold:
                        effective_rate_delta += modifier.get('rate_delta', 0)
                        effective_volume_delta += modifier.get('volume_delta', 0)
                final_rate = max(100, DEFAULT_RATE + effective_rate_delta)
                final_volume = max(0.0, min(1.0, DEFAULT_VOLUME + effective_volume_delta))
                tts_engine.setProperty('rate', final_rate)
                tts_engine.setProperty('volume', final_volume)
                tts_engine.say(text)
                tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Engine Error: {e}. Re-initializing.")
            tts_engine = pyttsx3.init()


@app.route('/chat', methods=['POST'])
def handle_chat():
    user_input = request.json['message']
    session_id = request.json.get('session_id')

    with session_lock:
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"[Session] New session_id generated: {session_id}")
            active_sessions[session_id] = Personality(
                session_id=session_id,
                shared_embedder=SHARED_EMBEDDER,
                shared_mood_classifier=SHARED_MOOD_CLASSIFIER
            )
        elif session_id not in active_sessions:
            print(f"[Session] Existing session_id '{session_id}' re-connected. Initializing Personality.")
            active_sessions[session_id] = Personality(
                session_id=session_id,
                shared_embedder=SHARED_EMBEDDER,
                shared_mood_classifier=SHARED_MOOD_CLASSIFIER
            )
    

    current_personality = active_sessions[session_id]
    print(f"USER ({session_id}): {user_input}")
    response_text = current_personality.process_input(user_input)
    print(f"SILVER ({session_id}): {response_text}")
    current_mood_vector = current_personality.mood_eq.get_current_vector()
    speech_thread = threading.Thread(target=speak, args=(response_text, current_mood_vector))
    speech_thread.start()

     # --- FINAL CHANGE ---
    # Now, we include the mood_vector in the response to the UI.
    return jsonify({
        "reply": response_text, 
        "session_id": session_id,
        "mood_vector": current_mood_vector 
    })

    

if __name__ == '__main__':
    print("Silver is online. Silently.")
    app.run(port=5000, debug=False)
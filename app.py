import streamlit as st
import cv2
from streamlit_webrtc import VideoProcessorBase, AudioProcessorBase, webrtc_streamer, WebRtcMode
from deepface import DeepFace
import av
import logging
import numpy as np
import time
import threading
import queue
# Legacy PyAudio and webrtcvad removed; we now use browser-captured audio via streamlit-webrtc
import io
import librosa
import tensorflow as tf
import os
import sys
import random
import pandas as pd
from datetime import datetime
import wave
import mediapipe as mp
from scipy.spatial import distance as dist
import joblib 
import collections
from streamlit_autorefresh import st_autorefresh
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import warnings
import soundfile as sf # For more robust audio loading from buffer

# --- Setup ---
# Suppress noisy logs
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("deepface")
logger.setLevel(logging.ERROR)
logger_webrtc = logging.getLogger("streamlit_webrtc")
logger_webrtc.setLevel(logging.ERROR)
# legacy pyaudio logger removed

st.set_page_config(page_title="Postpartum Wellness Monitor", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Minimal UI mode: hide non-essential informational messages and logs from the frontend
# Set to True for a clean prototype UI. Set to False to restore verbose user messages.
MINIMAL_UI = True
# --- Session State Initialization (MOVED TO TOP TO FIX ERROR) ---
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = queue.Queue()

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        'time', 'final_state', 'mock_heart_rate', 'face_emotion', 'voice_emotion', 'is_sleeping'
    ])
    
if 'current_state' not in st.session_state:
    st.session_state.current_state = {
        "face_emotion": "N/A",
        "voice_emotion": "N/A",
        "is_sleeping": False,
        "final_state": "Calibrating",
        "mock_heart_rate": 70,
        "last_alert": None,
        # frame_count is handled per-stream in EmotionTransformer to avoid cross-session drift
    }
    
if 'audio_webrtc_active' not in st.session_state:
    # Flag for browser-captured audio (streamlit-webrtc)
    st.session_state.audio_webrtc_active = False
    
if 'fl_status_message' not in st.session_state:
    st.session_state.fl_status_message = None


def ensure_data_queue():
    """Ensure a usable queue exists in session_state (idempotent)."""
    if 'data_queue' not in st.session_state or not isinstance(st.session_state.data_queue, queue.Queue):
        st.session_state.data_queue = queue.Queue()
    return st.session_state.data_queue

# --- Constants for Audio Processing ---
# Use config if available, otherwise defaults
try:
    from config import VAD_AGGRESSIVENESS, VAD_FRAME_MS, VAD_RATE, VAD_SILENCE_FRAMES
    from config import VOICE_MODEL_PATH, VOICE_ENCODER_PATH
except ImportError:
    VAD_AGGRESSIVENESS = 3
    VAD_FRAME_MS = 30
    VAD_RATE = 16000
    VAD_SILENCE_FRAMES = 30
    VOICE_MODEL_PATH = 'voice_model.h5'
    VOICE_ENCODER_PATH = 'voice_emotion_encoder.joblib'

VAD_CHUNK = (VAD_RATE * VAD_FRAME_MS) // 1000
# AUDIO_FORMAT and AUDIO_CHANNELS (PyAudio) removed; browser audio is used instead
MODEL_SAMPLE_RATE = 22050
VOICE_EMOTIONS = ['anxious', 'calm', 'happy', 'neutral', 'sad', 'angry'] 

# --- Import Configuration ---
try:
    import config
    EAR_THRESHOLD = config.EAR_THRESHOLD
    EAR_CONSEC_FRAMES = config.EAR_CONSEC_FRAMES
    LEFT_EYE_INDICES = config.LEFT_EYE_INDICES
    RIGHT_EYE_INDICES = config.RIGHT_EYE_INDICES
    DEEPFACE_FRAME_SKIP = config.DEEPFACE_FRAME_SKIP
    DATA_LOG_INTERVAL_SECONDS = config.DATA_LOG_INTERVAL_SECONDS
    MAX_CHART_DATA_POINTS = config.MAX_CHART_DATA_POINTS
    HEART_RATE_RANGES = config.HEART_RATE_RANGES
except ImportError:
    # Fallback to defaults if config.py not found
    EAR_THRESHOLD = 0.23
    EAR_CONSEC_FRAMES = 45
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    DEEPFACE_FRAME_SKIP = 5
    DATA_LOG_INTERVAL_SECONDS = 3.0
    MAX_CHART_DATA_POINTS = 500
    HEART_RATE_RANGES = {
        'Anxiety': (100, 125),
        'Agitated': (95, 115),
        'Stress': (85, 100),
        'Sadness': (80, 95),
        'Sleeping': (50, 65),
        'Calmness': (60, 75),
        'Calibrating': (70, 80)
    }

# --- Global Models (Loaded Once) ---
# Voice model and encoder will be stored here after preload_models
global_voice_model = None
global_voice_encoder = None

# --- Model Pre-loading Function ---
@st.cache_resource
def preload_models():
    """Loads all AI models once and caches them."""
    
    with st.spinner("Loading MediaPipe FaceMesh model..."):
        face_mesh_model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, # Increased confidence for better stability
            min_tracking_confidence=0.5  # Increased confidence
        )
    if not MINIMAL_UI:
        st.success("MediaPipe FaceMesh model loaded.")

    with st.spinner("Loading DeepFace emotion model (this may take a moment)..."):
        try:
            # Create a dummy image to force DeepFace to load its models
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_result = DeepFace.analyze(
                dummy_img, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='opencv',  # Use opencv for initial detection
                silent=True  # Suppress verbose output
            )
            
            # Verify DeepFace is working
            if test_result:
                # Model ready; avoid verbose UI messages to keep prototype clean
                logging.getLogger(__name__).debug("DeepFace initialized and ready for emotion detection")
            else:
                # Keep a minimal warning for users if DeepFace reports no result during preflight
                if not MINIMAL_UI:
                    st.warning("⚠️ DeepFace loaded but returned no result - may work during actual detection")
        except Exception as e:
            if not MINIMAL_UI:
                st.error("❌ CRITICAL: Failed to pre-load DeepFace model. Face emotion detection will be unavailable.")
            logging.getLogger(__name__).exception("DeepFace pre-load failed")
            return None, None, None # Return Nones if critical model fails
    
    with st.spinner("Loading Voice Emotion model and encoder..."):
        try:
            # Check if model file exists
            if not os.path.exists(VOICE_MODEL_PATH):
                if not MINIMAL_UI:
                    st.warning(f"Voice model file not found: {VOICE_MODEL_PATH}")
                    st.info("💡 **Solution**: Run `python train_voice_model.py` to train a new model.")
                voice_model = None
                voice_encoder = None
            else:
                # Check for deprecated model format error
                voice_model = None
                load_errors = []
                error_messages = []
                
                # Strategy 1: Try loading with compile=False (most compatible)
                try:
                    voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH, compile=False)
                    # Recompile the model with the same settings used during training
                    voice_model.compile(
                        loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                        metrics=['accuracy']
                    )
                except Exception as e1:
                    error_str = str(e1)
                    error_messages.append(f"Method 1 failed: {error_str}")
                    load_errors.append(error_str)
                    
                    # Check if this is the batch_shape error
                    if 'batch_shape' in error_str.lower() or 'unrecognized keyword arguments' in error_str.lower():
                        if not MINIMAL_UI:
                            st.error("❌ **Model Compatibility Issue Detected**")
                            st.warning("The existing model file was created with an older TensorFlow version and is incompatible.")
                            # Provide auto-fix option
                            st.markdown("""
                            **🔧 Quick Fix (Automatic):**
                            ```bash
                            python fix_model_compatibility.py
                            ```
                            This will automatically delete the incompatible files.
                            
                            **Or Manual Fix:**
                            1. Delete the old model file: `voice_model.h5`
                            2. Delete the encoder: `voice_emotion_encoder.joblib`
                            3. Retrain: `python train_voice_model.py`
                            """)
                            # Try to auto-fix if fix script exists
                            if os.path.exists('fix_model_compatibility.py'):
                                if st.button("🔧 Auto-Fix Now", type="primary"):
                                    try:
                                        import subprocess
                                        result = subprocess.run(
                                            [sys.executable, 'fix_model_compatibility.py'],
                                            capture_output=True,
                                            text=True,
                                            timeout=10
                                        )
                                        st.code(result.stdout, language=None)
                                        if result.returncode == 0:
                                            st.success("✅ Incompatible files removed! Please refresh the page.")
                                            st.info("💡 Next: Run `python train_voice_model.py` to train a new compatible model.")
                                        else:
                                            st.error("Auto-fix failed. Please run manually.")
                                    except Exception as auto_err:
                                        st.error(f"Auto-fix error: {auto_err}")
                                        st.info("Please run `python fix_model_compatibility.py` manually.")
                        
                        voice_model = None
                        voice_encoder = None
                    else:
                        # Try other loading strategies for different errors
                        # Strategy 2: Try loading normally (for older TF versions)
                        try:
                            voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH)
                        except Exception as e2:
                            error_messages.append(f"Method 2 failed: {str(e2)}")
                            load_errors.append(str(e2))
                            # Strategy 3: Try loading SavedModel if it exists
                            try:
                                saved_model_path = VOICE_MODEL_PATH.replace('.h5', '_savedmodel')
                                if os.path.exists(saved_model_path):
                                    voice_model = tf.keras.models.load_model(saved_model_path, compile=False)
                                    voice_model.compile(
                                        loss='categorical_crossentropy',
                                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                                        metrics=['accuracy']
                                    )
                                else:
                                    raise Exception("SavedModel format not found")
                            except Exception as e3:
                                error_messages.append(f"Method 3 failed: {str(e3)}")
                                load_errors.append(str(e3))
                                
                                # All methods failed - show concise message and log details
                                if not MINIMAL_UI:
                                    st.error("❌ All attempts to load the voice model failed. Please retrain using `python train_voice_model.py`.")
                                logging.getLogger(__name__).debug("Voice model load errors:\n" + "\n".join(error_messages))
                                voice_model = None
                                voice_encoder = None
                
                # If model loaded successfully, load encoder
                if voice_model is not None:
                    try:
                        if not os.path.exists(VOICE_ENCODER_PATH):
                            if not MINIMAL_UI:
                                st.warning(f"Encoder file not found: {VOICE_ENCODER_PATH}")
                            voice_encoder = None
                        else:
                            voice_encoder = joblib.load(VOICE_ENCODER_PATH)
                            if not MINIMAL_UI:
                                st.success("✅ Voice Emotion model loaded successfully.")
                    except Exception as enc_error:
                        if not MINIMAL_UI:
                            st.error(f"Error loading encoder: {enc_error}")
                        voice_encoder = None
                else:
                    voice_encoder = None
                    
        except FileNotFoundError as e:
            if not MINIMAL_UI:
                st.warning("Voice model files not found. Voice analysis will be unavailable until a model is trained.")
            logging.getLogger(__name__).debug(f"FileNotFoundError loading voice model: {e}")
            voice_model = None
            voice_encoder = None
        except Exception as e:
            error_msg = str(e)
            if not MINIMAL_UI:
                st.error("❌ Error loading voice model. See logs for details.")
            logging.getLogger(__name__).exception("Error loading voice model")

            # Check for batch_shape error specifically
            if 'batch_shape' in error_msg.lower():
                if not MINIMAL_UI:
                    st.warning("**Old model format detected** - Please retrain using updated script.")
                    st.info("💡 **Fix**: Delete `voice_model.h5` and run `python train_voice_model.py`")
            
            voice_model = None
            voice_encoder = None

    return face_mesh_model, voice_model, voice_encoder

def calculate_ear(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR) from eye landmarks.
    Uses 6-point EAR formula: EAR = (vertical1 + vertical2) / (2 * horizontal)
    """
    try:
        if len(eye_landmarks) < 6:
            return None
        
        # Ensure points are numpy arrays for euclidean distance
        # Extract 6 points for EAR calculation
        p1 = np.array(eye_landmarks[1])  # Top vertical point
        p2 = np.array(eye_landmarks[5])  # Bottom vertical point
        p3 = np.array(eye_landmarks[2])  # Top vertical point (alternate)
        p4 = np.array(eye_landmarks[4])  # Bottom vertical point (alternate)
        p5 = np.array(eye_landmarks[0])  # Left corner (horizontal)
        p6 = np.array(eye_landmarks[3])  # Right corner (horizontal)

        # Calculate vertical distances
        A = dist.euclidean(p1, p2)
        B = dist.euclidean(p3, p4)
        # Calculate horizontal distance
        C = dist.euclidean(p5, p6)
        
        # Avoid division by zero
        if C == 0:
            return None
        
        ear = (A + B) / (2.0 * C)
        return ear
    except (IndexError, ValueError, TypeError) as e:
        return None  # Indicate calculation failure, let caller handle

# --- Audio Processing Functions ---
from audio_features import extract_features, FEATURE_VECTOR_LENGTH, MIN_SAMPLES

def analyze_voice_emotion(audio_frames_list, voice_model, voice_encoder):
    """
    Analyzes audio using the REAL trained model.
    """
    if voice_model is None or voice_encoder is None:
        return random.choice(VOICE_EMOTIONS).capitalize() # Fallback

    if not audio_frames_list or len(audio_frames_list) == 0:
        return None

    try:
        # Convert list of raw frames to single bytes object
        raw_audio_bytes = b''.join(audio_frames_list)
        if len(raw_audio_bytes) < 1600:  # Minimum audio length (100ms at 16kHz)
            return None

        # Convert raw int16 PCM bytes to numpy array and normalize
        audio_array = np.frombuffer(raw_audio_bytes, dtype=np.int16)
        audio_data = audio_array.astype(np.float32) / 32768.0

        # Resample if needed
        if VAD_RATE != MODEL_SAMPLE_RATE:
            audio_data = librosa.resample(y=audio_data, orig_sr=VAD_RATE, target_sr=MODEL_SAMPLE_RATE)

        # Ensure minimum length for feature extraction
        if len(audio_data) < 2048:
            return None

        # Extract features and validate
        features = extract_features(audio_data, sample_rate=MODEL_SAMPLE_RATE)
        if features is None or features.shape[0] != FEATURE_VECTOR_LENGTH:
            return None

        # Reshape features for model input and predict
        features = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
        prediction = voice_model.predict(features, verbose=0)
        if prediction.shape[1] != len(voice_encoder.classes_):
            return None

        predicted_index = int(np.argmax(prediction, axis=1)[0])
        if predicted_index >= len(voice_encoder.classes_):
            return None

        predicted_emotion = voice_encoder.classes_[predicted_index]
        return predicted_emotion.capitalize()
    except Exception:
        logging.getLogger(__name__).exception("Error analyzing audio frame")
        return None

# Legacy PyAudio-based audio_processor_thread removed. Browser-based capture is the supported path.

# --- Video Transformer Class (MODIFIED) ---
class EmotionTransformer(VideoProcessorBase):
    def __init__(self, data_queue, face_mesh_model):
        super().__init__()
        self.data_queue = data_queue
        self.face_mesh = face_mesh_model
        
        self.sleep_counter = 0
        self.is_sleeping_or_fatigued = False
        self.frame_count = 0 # For DeepFace frame skipping
        # Counters for why DeepFace/face emotion returned N/A for debugging
        self.na_reasons = collections.Counter()
        # Lightweight frame instrumentation to detect frozen frames
        self.recv_count = 0
        self.last_recv_time = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False # Optimize
        # Instrumentation: increment recv counter and timestamp
        try:
            self.recv_count += 1
            self.last_recv_time = time.time()
        except Exception:
            pass
        
        mesh_results = self.face_mesh.process(img_rgb)
        
        face_emotion = "N/A"
        ear = None # Initialize as None to distinguish no detection from 0.0
        
        # Check if face is detected by MediaPipe
        if mesh_results.multi_face_landmarks and len(mesh_results.multi_face_landmarks) > 0:
            face_landmarks = mesh_results.multi_face_landmarks[0]
            
            # --- 1. Fatigue/Sleep Detection (EAR) ---
            landmarks_np = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
            landmarks_scaled = landmarks_np * [img.shape[1], img.shape[0]]
            
            # Extract eye landmarks using correct MediaPipe indices
            try:
                left_eye_landmarks = landmarks_scaled[LEFT_EYE_INDICES]
                right_eye_landmarks = landmarks_scaled[RIGHT_EYE_INDICES]
            except IndexError:
                # Fallback: use first 6 and next 6 if indices are out of range
                num_landmarks = len(landmarks_scaled)
                if num_landmarks >= 12:
                    left_eye_landmarks = landmarks_scaled[:6]
                    right_eye_landmarks = landmarks_scaled[6:12]
                else:
                    left_eye_landmarks = None
                    right_eye_landmarks = None
            
            if left_eye_landmarks is not None and right_eye_landmarks is not None:
                left_ear = calculate_ear(left_eye_landmarks)
                right_ear = calculate_ear(right_eye_landmarks)
                
                if left_ear is not None and right_ear is not None:
                    ear = (left_ear + right_ear) / 2.0
                
                    if ear < EAR_THRESHOLD:
                        self.sleep_counter += 1
                    else:
                        self.sleep_counter = 0
                        
                    self.is_sleeping_or_fatigued = self.sleep_counter > EAR_CONSEC_FRAMES
                else:
                    # Partial EAR calculation - use available data
                    if left_ear is not None:
                        ear = left_ear
                    elif right_ear is not None:
                        ear = right_ear
                    else:
                        ear = None
                        self.sleep_counter = 0
                        self.is_sleeping_or_fatigued = False
            else:
                self.sleep_counter = 0  # Reset if landmark extraction fails
                self.is_sleeping_or_fatigued = False
                ear = None


            # --- 2. Emotion Detection (DeepFace - with frame skipping) ---
            self.frame_count += 1
            should_analyze_emotion = (self.frame_count % DEEPFACE_FRAME_SKIP == 0)
            
            if should_analyze_emotion:
                try:
                    # Find bounding box for the face from MediaPipe landmarks
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    
                    if x_coords and y_coords:
                        x_min, x_max = int(min(x_coords) * img.shape[1]), int(max(x_coords) * img.shape[1])
                        y_min, y_max = int(min(y_coords) * img.shape[0]), int(max(y_coords) * img.shape[0])
                        
                        # Ensure valid bounding box
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img.shape[1], x_max)
                        y_max = min(img.shape[0], y_max)
                        
                        if x_max > x_min and y_max > y_min:
                            padding = 20
                            face_roi = img[
                                max(0, y_min - padding):min(img.shape[0], y_max + padding),
                                max(0, x_min - padding):min(img.shape[1], x_max + padding)
                            ]

                            # Validate ROI before analysis
                            if face_roi.size > 0 and face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
                                try:
                                    # Use opencv backend for ROI analysis
                                    results = DeepFace.analyze(
                                        face_roi, 
                                        actions=['emotion'], 
                                        enforce_detection=False,
                                        detector_backend='opencv',
                                        silent=True  # Suppress DeepFace verbose output
                                    )
                                    
                                    # Handle different return formats from DeepFace
                                    if isinstance(results, list) and len(results) > 0:
                                        if isinstance(results[0], dict) and 'dominant_emotion' in results[0]:
                                            face_emotion = results[0]['dominant_emotion'].capitalize()
                                        else:
                                            face_emotion = "N/A"
                                            try:
                                                self.na_reasons['deepface_no_dominant'] += 1
                                            except Exception:
                                                pass
                                    elif isinstance(results, dict) and 'dominant_emotion' in results:
                                        face_emotion = results['dominant_emotion'].capitalize()
                                    else:
                                        face_emotion = "N/A"
                                        try:
                                            self.na_reasons['deepface_unexpected_format'] += 1
                                        except Exception:
                                            pass
                                    
                                    # Validate emotion is in expected list
                                    valid_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                                    if face_emotion not in valid_emotions:
                                        # Possibly DeepFace returned an emotion outside our whitelist (e.g., 'contempt')
                                        try:
                                            self.na_reasons['not_in_whitelist'] += 1
                                        except Exception:
                                            pass
                                        face_emotion = "N/A"
                                        
                                except Exception as deepface_error:
                                    # Log the error for debugging but don't crash
                                    logging.getLogger(__name__).exception(f"[Video Thread] DeepFace analysis error: {deepface_error}")
                                    try:
                                        self.na_reasons['deepface_exception'] += 1
                                    except Exception:
                                        pass
                                    face_emotion = "N/A"
                            else:
                                face_emotion = "N/A"  # ROI too small
                                try:
                                    self.na_reasons['roi_too_small'] += 1
                                except Exception:
                                    pass
                        else:
                            face_emotion = "N/A"  # Invalid bounding box
                            try:
                                self.na_reasons['invalid_bbox'] += 1
                            except Exception:
                                pass
                    else:
                        face_emotion = "N/A"  # No coordinates
                        try:
                            self.na_reasons['no_coords'] += 1
                        except Exception:
                            pass
                except Exception as e:
                    logging.getLogger(__name__).exception(f"[Video Thread] Error in emotion detection pipeline: {e}")
                    try:
                        self.na_reasons['pipeline_exception'] += 1
                    except Exception:
                        pass
                    face_emotion = "N/A"
            else:
                # Use last known emotion if skipping frame
                # Try to get from session state, but default to N/A if not available
                last_emotion = st.session_state.current_state.get("face_emotion", "N/A")
                face_emotion = last_emotion if last_emotion != "N/A" else "N/A"
                
                # If we don't have a last emotion yet, still try to analyze this frame (fault tolerance)
                if face_emotion == "N/A":
                    try:
                        self.na_reasons['skipped_frame_no_history'] += 1
                    except Exception:
                        pass
                if face_emotion == "N/A" and self.frame_count > DEEPFACE_FRAME_SKIP:
                    # We've skipped enough frames, try analyzing now to get initial emotion
                    try:
                        x_coords = [lm.x for lm in face_landmarks.landmark]
                        y_coords = [lm.y for lm in face_landmarks.landmark]
                        if x_coords and y_coords:
                            x_min, x_max = int(min(x_coords) * img.shape[1]), int(max(x_coords) * img.shape[1])
                            y_min, y_max = int(min(y_coords) * img.shape[0]), int(max(y_coords) * img.shape[0])
                            padding = 20
                            face_roi = img[
                                max(0, y_min - padding):min(img.shape[0], y_max + padding),
                                max(0, x_min - padding):min(img.shape[1], x_max + padding)
                            ]
                            if face_roi.size > 0 and face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
                                results = DeepFace.analyze(
                                    face_roi, 
                                    actions=['emotion'], 
                                    enforce_detection=False,
                                    detector_backend='opencv',
                                    silent=True
                                )
                                if isinstance(results, list) and len(results) > 0 and 'dominant_emotion' in results[0]:
                                    face_emotion = results[0]['dominant_emotion'].capitalize()
                    except:
                        pass  # Silent fail - will try next frame
                
            # --- 3. Send Data to Queue ---
            # Always send data if face is detected, even if emotion is N/A (indicates processing issue)
            # Include a snapshot of NA reasons for diagnostic analysis (keeps data non-blocking)
            data_point = {
                "type": "video",
                "time": datetime.now(),
                "face_emotion": face_emotion,
                "is_sleeping": self.is_sleeping_or_fatigued,
                "current_ear": ear,  # Send EAR for potential display/debugging
                "face_detected": True,  # Flag to indicate face was detected
                "na_reasons": dict(self.na_reasons)
            }
            try:
                self.data_queue.put(data_point, block=False, timeout=0.1)
            except queue.Full:
                # Queue full - skip this frame but don't crash
                pass

            # --- 4. Draw Visuals ---
            # Display face emotion on video feed
            emotion_color = (0, 255, 0) if face_emotion != "N/A" else (0, 0, 255)
            cv2.putText(img, f"Face: {face_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
            if ear is not None:
                cv2.putText(img, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.is_sleeping_or_fatigued:
                cv2.putText(img, "SLEEPING / RESTING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No face detected - show warning on video
            cv2.putText(img, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Overlay basic instrumentation so user can tell if frames are updating
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(img, f"Frame#: {self.recv_count}", (10, img.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
            cv2.putText(img, f"TS: {ts}", (10, img.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
        except Exception:
            pass
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Browser Audio Processor (uses streamlit-webrtc) ---
class BrowserAudioProcessor(AudioProcessorBase):
    def __init__(self, data_queue, voice_model, voice_encoder):
        self.data_queue = data_queue
        self.voice_model = voice_model
        self.voice_encoder = voice_encoder

        self.frames_float = []
        self.silence_counter = 0
        self.max_buffer_frames = 300
        # Energy threshold for simple VAD (adjustable)
        self.energy_threshold = 0.01

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            # Convert to numpy array (shape: channels x samples)
            arr = frame.to_ndarray()

            # If multi-channel, average to mono
            if arr.ndim == 2:
                audio_float = np.mean(arr, axis=0).astype(np.float32)
            else:
                audio_float = arr.astype(np.float32)

            # Normalize if values are outside [-1,1]
            if np.max(np.abs(audio_float)) > 1.5:
                # Assume int16 PCM
                audio_float = (audio_float / 32768.0).astype(np.float32)

            # Simple energy-based VAD
            energy = np.sqrt(np.mean(audio_float ** 2))
            is_speech = energy > self.energy_threshold

            if is_speech:
                self.frames_float.append(audio_float)
                if len(self.frames_float) > self.max_buffer_frames:
                    self.frames_float.pop(0)
                self.silence_counter = 0
            else:
                if len(self.frames_float) > 0:
                    self.silence_counter += 1
                    if self.silence_counter > VAD_SILENCE_FRAMES:
                        # Concatenate buffered floats
                        try:
                            concat = np.concatenate(self.frames_float)
                            # Resample to MODEL_SAMPLE_RATE if needed
                            src_rate = getattr(frame, 'sample_rate', MODEL_SAMPLE_RATE)
                            if src_rate != MODEL_SAMPLE_RATE:
                                concat = librosa.resample(concat, orig_sr=src_rate, target_sr=MODEL_SAMPLE_RATE)

                            # Ensure minimum length
                            if len(concat) >= MIN_SAMPLES:
                                features = extract_features(concat, sample_rate=MODEL_SAMPLE_RATE)
                                if features is not None and features.shape[0] == FEATURE_VECTOR_LENGTH:
                                    # Prepare for model
                                    feats = np.expand_dims(features, axis=0)
                                    feats = np.expand_dims(feats, axis=2)
                                    try:
                                        pred = self.voice_model.predict(feats, verbose=0)
                                        pred_idx = np.argmax(pred, axis=1)[0]
                                        if hasattr(self.voice_encoder, 'classes_') and pred_idx < len(self.voice_encoder.classes_):
                                            predicted = self.voice_encoder.classes_[pred_idx]
                                            data_point = {"type": "audio", "time": datetime.now(), "voice_emotion": predicted}
                                            try:
                                                self.data_queue.put(data_point, block=False, timeout=0.1)
                                            except queue.Full:
                                                logging.getLogger(__name__).warning("[BrowserAudio] Queue full, skipping audio datapoint")
                                    except Exception:
                                        logging.getLogger(__name__).exception("[BrowserAudio] Error predicting voice emotion")

                        except Exception:
                            logging.getLogger(__name__).exception("[BrowserAudio] Error processing buffered audio")
                        finally:
                            self.frames_float = []
                            self.silence_counter = 0

            return frame
        except Exception:
            logging.getLogger(__name__).exception("[BrowserAudio] Unexpected error in recv")
            return frame


# --- Multi-Modal Fusion & Logic Modules ---
def get_final_state(face, voice, is_sleeping):
    """
    This is the "brain" of the app. It fuses all sensor data
    into a single, actionable state. Works with partial data.
    """
    # Priority 1: Sleeping
    if is_sleeping:
        return "Sleeping"
    
    # Normalize emotion strings for comparison (case-insensitive, handle variations)
    face_lower = str(face).lower() if face else 'n/a'
    voice_lower = str(voice).lower() if voice else 'n/a'
    
    # Priority 2: Anxiety (High Confidence - both sensors agree, OR single strong signal)
    if (face_lower in ['anxious', 'fear'] and voice_lower == 'anxious') or \
       (face_lower in ['anxious', 'fear'] and voice_lower == 'n/a') or \
       (voice_lower == 'anxious' and face_lower not in ['angry', 'sad']):
        return "Anxiety"
        
    # Priority 3: Agitated (High Confidence - single strong signal enough)
    if face_lower == 'angry' or voice_lower == 'angry':
        return "Agitated"
        
    # Priority 4: Sadness (High Confidence - both agree OR single strong signal)
    if (face_lower == 'sad' and voice_lower == 'sad') or \
       (face_lower == 'sad' and voice_lower == 'n/a') or \
       (voice_lower == 'sad' and face_lower not in ['angry', 'anxious', 'fear']):
        return "Sadness"

    # Priority 5: Stress (Medium Confidence - mixed signals)
    if (face_lower in ['sad', 'anxious', 'fear'] and voice_lower in ['neutral', 'calm', 'n/a', 'happy']) or \
       (voice_lower in ['sad', 'anxious'] and face_lower in ['neutral', 'calm', 'n/a', 'happy']):
        return "Stress"
        
    # Priority 6: Calmness (Positive or neutral states)
    if (face_lower in ['neutral', 'happy', 'calm', 'surprise'] and voice_lower in ['neutral', 'happy', 'calm', 'n/a']) or \
       (face_lower in ['neutral', 'happy', 'calm', 'surprise'] and voice_lower == 'n/a') or \
       (voice_lower in ['neutral', 'happy', 'calm'] and face_lower in ['neutral', 'happy', 'calm', 'n/a']):
        return "Calmness"
        
    # FAULT-TOLERANT: Work with just face data (voice model may be unavailable)
    # Priority: Use face data even if voice is N/A
    if face_lower != 'n/a':
        # If face shows positive/neutral emotions
        if face_lower in ['neutral', 'happy', 'calm', 'surprise']:
            return "Calmness"
        # If face shows negative emotions (already handled by priorities above, but ensure detection)
        elif face_lower in ['sad', 'anxious', 'fear', 'angry']:
            # If we had matched with voice, it would be Anxiety/Agitated/Sadness
            # With just face, make reasonable inference
            if face_lower == 'angry':
                return "Agitated"  # Single strong signal enough
            elif face_lower in ['anxious', 'fear']:
                return "Anxiety"  # Single strong signal enough
            elif face_lower == 'sad':
                return "Sadness"  # Single strong signal enough
            else:
                return "Stress"  # Generic negative state
        else:
            # Any other face emotion detected
            return "Calmness"  # Default to positive if unknown but detected
    
    # If we have voice data but no face data (unlikely but handle it)
    if voice_lower != 'n/a' and voice_lower not in ['anxious', 'sad', 'angry']:
        return "Calmness"  # Default positive state if voice shows neutral/positive
    if voice_lower != 'n/a' and voice_lower in ['anxious', 'sad', 'angry']:
        return "Stress"  # Negative voice but no face to confirm

    # Only return Calibrating if truly no data from either sensor
    # This should rarely happen if webcam is working
    return "Calibrating"

def get_mock_heart_rate(final_state):
    """
    Simulates a realistic heart rate based on the final emotional state.
    Uses configurable ranges from config.py or defaults.
    """
    hr_range = HEART_RATE_RANGES.get(final_state, (70, 80))
    return random.randint(int(hr_range[0]), int(hr_range[1]))

def get_personalized_feedback(final_state):
    """
    Provides actionable, personalized feedback for each state.
    --- UPDATED: Tailored for Postpartum Patients ---
    """
    if final_state == "Anxiety":
        return ("It looks like you're feeling anxious. That's *so* common for new moms; your mind is on high alert. "
                "Let's try a simple 'grounding' exercise you can do right now: Name 3 things you can see, 2 things you can hear, "
                "and 1 thing you can feel (like your feet on the floor).")
        
    elif final_state == "Agitated":
        return ("Signs of agitation or anger were detected. Feeling irritable, especially when you're exhausted, is completely normal. "
                "If you can, place the baby safely in their crib for 2 minutes and step into another room. "
                "Just 2 minutes of quiet can help reset your nervous system.")
        
    elif final_state == "Sadness":
        return ("The system is detecting strong signs of sadness. Please know this is *not* your fault, and you are not alone. "
                "These 'baby blues' are real. Please consider sending a quick text to a friend, partner, or family member, "
                "just to let them know you're having a tough moment.")
        
    elif final_state == "Stress":
        return ("Signs of stress were detected. You're juggling a lot. A quick 'reset' can help: Stand up, stretch your arms overhead, "
                "and take three deep, slow breaths.")
        
    elif final_state == "Sleeping":
        return ("You appear to be sleeping or resting. This is the most important thing you can do right now. "
                "Every minute of rest helps your body and mind heal. We'll be quiet. You're doing a great job.")
        
    elif final_state == "Calmness":
        return ("You're in a calm state. This is wonderful. Take a deep breath and just soak in this moment. You are doing great.")
        
    return "The monitor is active and checking in." # Default fallback

# --- Federated Learning Client Code ---

# Global model definition
# Use a multinomial-capable solver and higher max_iter for stability in multiclass settings.
# We avoid a global scaler to prevent client-to-client leakage; each client will create its own scaler.
fl_model = LogisticRegression(max_iter=1000, warm_start=True, solver='saga', multi_class='multinomial')

# We need a fixed set of classes for the encoder
STATE_CLASSES = ['Sleeping', 'Anxiety', 'Agitated', 'Sadness', 'Stress', 'Calmness', 'Calibrating']
state_encoder = LabelEncoder().fit(STATE_CLASSES)

# We need a fixed set of features for the model
SENSOR_FEATURES = ['mock_heart_rate', 'is_sleeping_int']
EMOTION_CLASSES = ['Happy', 'Sad', 'Neutral', 'Anxious', 'Fear', 'Angry', 'N/A']
VOICE_CLASSES = ['Happy', 'Sad', 'Neutral', 'Anxious', 'Angry', 'Calm', 'N/A']

# Create all possible feature columns
ALL_FL_FEATURES = SENSOR_FEATURES + \
                  [f"face_{e}" for e in EMOTION_CLASSES] + \
                  [f"voice_{e}" for e in VOICE_CLASSES]

# Preprocessing for FL
def preprocess_fl_data(df):
    """
    Prepares the session dataframe for the FL model.
    """
    if df.empty or len(df) < 10:
        return None, None
    df_feat = df.copy()
    df_feat['is_sleeping_int'] = df_feat['is_sleeping'].astype(bool).astype(int)

    df_feat['face_emotion'] = pd.Categorical(df_feat['face_emotion'], categories=EMOTION_CLASSES)
    df_feat['voice_emotion'] = pd.Categorical(df_feat['voice_emotion'], categories=VOICE_CLASSES)

    df_feat = pd.get_dummies(df_feat, columns=['face_emotion', 'voice_emotion'], prefix=['face', 'voice'])

    # Create an empty DataFrame with all expected feature columns
    X_aligned = pd.DataFrame(0, index=df_feat.index, columns=ALL_FL_FEATURES)
    # Copy matching columns from df_feat
    for col in X_aligned.columns:
        if col in df_feat.columns:
            X_aligned[col] = df_feat[col]

    X_raw = X_aligned[ALL_FL_FEATURES].values

    # Fit a local scaler to avoid global state leakage
    local_scaler = StandardScaler()
    try:
        local_scaler.fit(X_raw)
        X = local_scaler.transform(X_raw)
    except Exception as e:
        logging.getLogger(__name__).exception("Error scaling FL features")
        return None, None

    # Create target 'y'
    valid_states_mask = df_feat['final_state'].isin(state_encoder.classes_)
    if not valid_states_mask.any():
        return None, None

    y = state_encoder.transform(df_feat.loc[valid_states_mask, 'final_state'])
    X = X[valid_states_mask]

    if X.shape[0] != y.shape[0]:
        logging.getLogger(__name__).warning(f"Mismatch in X and y shapes after filtering: X={X.shape}, y={y.shape}")
        return None, None

    X_shuf, y_shuf = shuffle(X, y, random_state=42)
    return X_shuf, y_shuf


# Flower client class
class AppClient(fl.client.NumPyClient):
    def __init__(self, data_df):
        self.X, self.y = preprocess_fl_data(data_df)

        # Initialize fl_model with classes if it hasn't been globally set
        if self.X is not None and self.y is not None and not hasattr(fl_model, "classes_"):
            logging.getLogger(__name__).debug("Initializing FL LogisticRegression classes.")
            # Create dummy data with all possible classes to fit for class initialization
            dummy_X = np.zeros((len(STATE_CLASSES), len(ALL_FL_FEATURES)))
            dummy_y = state_encoder.transform(STATE_CLASSES)
            try:
                fl_model.fit(dummy_X, dummy_y)  # This just sets the classes_ attribute
                logging.getLogger(__name__).debug(f"Initialized FL model classes: {fl_model.classes_}")
            except Exception as e:
                logging.getLogger(__name__).exception("Error initializing FL model classes")

    def get_parameters(self, config):
        if not hasattr(fl_model, "coef_") or fl_model.coef_ is None:
             # Model hasn't been fit yet, return zero-filled arrays with correct shape
             # Based on ALL_FL_FEATURES and STATE_CLASSES
             n_features = len(ALL_FL_FEATURES)
             n_classes = len(STATE_CLASSES)
             # Handle multiclass LogisticRegression shape (liblinear solver creates coef_ with shape (n_classes, n_features))
             coef_shape = (n_classes, n_features)
             intercept_shape = (n_classes,)

             logging.getLogger(__name__).debug(f"FL model not yet fit, returning zero parameters (features={n_features}, classes={n_classes}).")
             return [np.zeros(coef_shape, dtype=np.float64), np.zeros(intercept_shape, dtype=np.float64)]
        
        # Ensure coef_ and intercept_ have correct shapes
        coef = fl_model.coef_ if hasattr(fl_model, 'coef_') else np.zeros((len(STATE_CLASSES), len(ALL_FL_FEATURES)), dtype=np.float64)
        intercept = fl_model.intercept_ if hasattr(fl_model, 'intercept_') else np.zeros(len(STATE_CLASSES), dtype=np.float64)
        return [coef, intercept]

    def fit(self, parameters, config):
        if self.X is None or self.y is None or len(self.X) == 0:
            logging.getLogger(__name__).debug("No local data to train or data too small for FL client.")
            return self.get_parameters(config), 0, {} # Return current global parameters, 0 samples
        
        try:
            # Validate parameters shape
            if parameters is None or len(parameters) < 2:
                logging.getLogger(__name__).warning("Invalid parameters received from FL server.")
                return self.get_parameters(config), 0, {}
            
            # Set parameters with validation
            if parameters[0] is not None and parameters[1] is not None:
                fl_model.coef_ = np.array(parameters[0], dtype=np.float64)
                fl_model.intercept_ = np.array(parameters[1], dtype=np.float64)
            
            # Train with error handling
            fl_model.fit(self.X, self.y)
            
            # Ensure model has correct attributes after training
            if not hasattr(fl_model, 'coef_') or fl_model.coef_ is None:
                logging.getLogger(__name__).warning("FL model fit failed - coef_ not set.")
                return self.get_parameters(config), 0, {}
            
            return [fl_model.coef_, fl_model.intercept_], len(self.X), {}
        except Exception as e:
            logging.getLogger(__name__).exception("Error during FL fit")
            return self.get_parameters(config), 0, {}

    def evaluate(self, parameters, config):
        if self.X is None or self.y is None or len(self.X) == 0:
            logging.getLogger(__name__).debug("No local data to evaluate or data too small for FL client.")
            return 0.0, 0, {"accuracy": 0.0}
        
        try:
            # Validate and set parameters
            if parameters is None or len(parameters) < 2:
                logging.getLogger(__name__).warning("Invalid parameters received for FL evaluation.")
                return 0.0, 0, {"accuracy": 0.0}
            
            if parameters[0] is not None and parameters[1] is not None:
                fl_model.coef_ = np.array(parameters[0], dtype=np.float64)
                fl_model.intercept_ = np.array(parameters[1], dtype=np.float64)
            
            # Calculate loss using log_loss for consistency
            from sklearn.metrics import log_loss
            y_pred_proba = fl_model.predict_proba(self.X)
            loss = log_loss(self.y, y_pred_proba, labels=np.arange(len(STATE_CLASSES)))
            accuracy = fl_model.score(self.X, self.y)
            
            return float(loss), len(self.X), {"accuracy": float(accuracy)}
        except Exception as e:
            logging.getLogger(__name__).exception("Error during FL evaluate")
            return 0.0, 0, {"accuracy": 0.0}

# Thread target for FL
def start_fl_client(q, data_df):
    try:
        client = AppClient(data_df)
        if client.X is None:
            q.put({"type": "fl_status", "message": "🚫 Not enough data (min 10 valid entries) to contribute. Use app more."})
            return

        logging.getLogger(__name__).info("Attempting to connect to Flower server")
        q.put({"type": "fl_status", "message": "🔄 Connecting to FL server and training..."})
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
        q.put({"type": "fl_status", "message": "✅ Successfully contributed to global model!"})
    except Exception as e:
        logging.getLogger(__name__).exception("FL client error")
        q.put({"type": "fl_status", "message": f"❌ Error connecting to FL server (Is server running on 127.0.0.1:8080?): {e}"})

# --- Streamlit App UI (Main Function) ---
def main():
    st.title("Postpartum Wellness Monitor")

    # --- Load all models on startup ---
    face_mesh_model, voice_model_global, voice_encoder_global = preload_models()
    
    # Store in session state to pass to thread
    st.session_state.voice_model = voice_model_global
    st.session_state.voice_encoder = voice_encoder_global

    if face_mesh_model is None: # If DeepFace critical model failed
        st.stop() # Stop the app
    
    # --- Sidebar for FL ---
    with st.sidebar:
        st.header("Privacy-Preserving AI")
        st.write("Contribute to improving the global model without sharing any of your private data.")
        st.caption("A separate Flower server application (`server.py`) must be running in a terminal for this feature.")
        if st.button("Start Contributing (1 Round)"):
            if len(st.session_state.df) < 10:
                st.sidebar.warning("Not enough data to contribute (min 10 valid entries required). Please use the app for a few more minutes.")
            else:
                st.sidebar.info("Initiating Federated Learning client...")
                
                # Ensure data_queue exists before starting FL client
                if 'data_queue' not in st.session_state:
                    st.session_state.data_queue = queue.Queue()
                
                # Run FL client in a thread so it doesn't block the app
                fl_thread = threading.Thread(
                    target=start_fl_client, 
                    args=(st.session_state.data_queue, st.session_state.df.copy()), # Pass a copy to avoid modification issues
                    daemon=True
                )
                fl_thread.start()

        # --- Audio controls: start/stop background audio processing ---
        st.markdown("---")
        st.subheader("Audio Monitoring")
        st.write("The app can listen to microphone audio for voice emotion. Start only if you consent and have a microphone.")

        # Ensure queue exists before starting audio
        ensure_data_queue()

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Start Audio Monitor"):
                if st.session_state.get('audio_webrtc_active'):
                    st.sidebar.info("Audio monitor already running (browser audio).")
                else:
                    # Activate browser-based audio capture
                    st.session_state.audio_webrtc_active = True
                    st.sidebar.success("Audio monitor started (browser microphone). Please allow microphone access in your browser.")

        with col_b:
            if st.button("Stop Audio Monitor"):
                if st.session_state.get('audio_webrtc_active'):
                    st.session_state.audio_webrtc_active = False
                    st.sidebar.info("Stopped browser audio monitor.")
                else:
                    st.sidebar.info("Audio monitor is not running.")

        # If browser audio is active, render the webrtc component (must be called every run)
        if st.session_state.get('audio_webrtc_active'):
            # Capture the queue and models in a closure to avoid accessing session_state from the webrtc worker thread
            data_queue_for_audio = ensure_data_queue()
            captured_voice_model = st.session_state.get('voice_model')
            captured_voice_encoder = st.session_state.get('voice_encoder')

            def create_browser_audio_processor():
                # Use the captured values (safe to reference from worker thread)
                return BrowserAudioProcessor(data_queue_for_audio, captured_voice_model, captured_voice_encoder)

            webrtc_streamer(
                key="browser_audio",
                mode=WebRtcMode.SENDRECV,
                audio_processor_factory=create_browser_audio_processor,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        
        # Display FL status messages from the queue
        if st.session_state.fl_status_message:
            if "✅" in st.session_state.fl_status_message:
                st.sidebar.success(st.session_state.fl_status_message)
            elif "❌" in st.session_state.fl_status_message:
                st.sidebar.error(st.session_state.fl_status_message)
            else:
                st.sidebar.info(st.session_state.fl_status_message)

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Live Monitor")
        
        # Get the queue safely - ensure it exists before passing to factory
        data_queue = ensure_data_queue()
        
        # Capture the queue and model in closure to avoid session_state access in factory
        captured_face_mesh_model = face_mesh_model
        
        # Create factory function that uses captured values (not session_state)
        def create_video_processor():
            # Use captured queue instead of accessing session_state
            # This avoids KeyError when factory is called in different context
            return EmotionTransformer(data_queue, captured_face_mesh_model)
        
        webrtc_streamer(
            key="emotion_detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_video_processor, 
            media_stream_constraints={"video": True, "audio": False}, # Audio is handled by separate thread
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        st.info("Click 'START' to activate your webcam. The monitor will also passively listen to voice tone (requires a microphone).")

    with col2:
        st.header("Real-Time Dashboard")
        
        # --- Autorefresh for Dashboard Updates ---
        try:
            from config import AUTOREFRESH_INTERVAL_MS
        except ImportError:
            AUTOREFRESH_INTERVAL_MS = 2000
        st_autorefresh(interval=AUTOREFRESH_INTERVAL_MS, limit=None, key="dashboard_refresher")
        
        # --- Process Queue: Get latest of each type ---
        # Thread-safe queue processing with max iterations to prevent blocking
        latest_video_data = None
        latest_audio_data = None
        
        # Ensure data_queue exists
        ensure_data_queue()
        
        max_iterations = 100  # Limit iterations to prevent blocking
        iteration_count = 0
        while iteration_count < max_iterations and not st.session_state.data_queue.empty():
            try:
                data_point = st.session_state.data_queue.get(block=False, timeout=0.1)
                
                if data_point.get("type") == "video":
                    latest_video_data = data_point
                    # Keep minimal debug behavior: record in log rather than printing
                    face_emotion_in_data = data_point.get('face_emotion', 'N/A')
                    face_detected = data_point.get('face_detected', False)
                    if face_detected and face_emotion_in_data == "N/A":
                        logging.getLogger(__name__).debug("Face detected but emotion is N/A - possible DeepFace issue")
                    # Preserve diagnostic counters in session state for UI display
                    try:
                        st.session_state.current_state['na_reasons'] = data_point.get('na_reasons', {})
                    except Exception:
                        pass
                elif data_point.get("type") == "audio":
                    latest_audio_data = data_point
                    emotion = data_point.get('voice_emotion', 'Unknown')
                    # Keep UI minimal: avoid transient toasts in prototype
                    logging.getLogger(__name__).debug(f"Voice emotion datapoint received: {emotion}")
                elif data_point.get("type") == "fl_status":
                    st.session_state.fl_status_message = data_point.get("message")
                
                iteration_count += 1
            except queue.Empty:
                break
            except Exception:
                # Log error but don't break - continue processing queue
                logging.getLogger(__name__).exception("Error processing data queue")
                iteration_count += 1
                if iteration_count >= max_iterations:
                    break

        # --- Update current_state with latest data ---
        # Face data is primary - always update if available
        if latest_video_data:
            face_emotion_from_video = latest_video_data.get("face_emotion", "N/A")
            face_detected_flag = latest_video_data.get("face_detected", False)
            
            # Update face emotion
            st.session_state.current_state["face_emotion"] = face_emotion_from_video
            st.session_state.current_state["is_sleeping"] = latest_video_data.get("is_sleeping", False)
            
            # Debug info: If face is detected but emotion is N/A, there's a DeepFace issue
            if face_detected_flag and face_emotion_from_video == "N/A":
                # Face detected but emotion analysis failed - show diagnostic
                st.session_state.current_state["face_detection_debug"] = "Face detected, analyzing emotion..."
            else:
                st.session_state.current_state.pop("face_detection_debug", None)
        
        # Voice data is optional - only update if available
        # If voice model is unavailable, voice_emotion stays "N/A" but doesn't block detection
        if latest_audio_data:
            st.session_state.current_state["voice_emotion"] = latest_audio_data.get("voice_emotion", "N/A")
        
        # Ensure voice_emotion is set (even if "N/A") to avoid blocking state detection
        if "voice_emotion" not in st.session_state.current_state:
            st.session_state.current_state["voice_emotion"] = "N/A"

        # --- FUSION & LOGGING ---
        # This block now runs ONCE per refresh with the latest observed sensor data
        # Get face and voice emotions (voice may be N/A if model unavailable - that's OK)
        face_emotion = st.session_state.current_state.get("face_emotion", "N/A")
        voice_emotion = st.session_state.current_state.get("voice_emotion", "N/A")
        is_sleeping = st.session_state.current_state.get("is_sleeping", False)
        
        # Ensure we have at least face data (webcam should provide this)
        # If face is N/A, we can't determine state
        if face_emotion == "N/A" and voice_emotion == "N/A":
            # No sensor data available yet
            new_final_state = "Calibrating"
        else:
            # We have some sensor data - determine state (works with just face or just voice)
            new_final_state = get_final_state(
                face_emotion,
                voice_emotion,
                is_sleeping
            )
        
        new_heart_rate = get_mock_heart_rate(new_final_state)
        
        # Always update current state for display
        st.session_state.current_state["final_state"] = new_final_state
        st.session_state.current_state["mock_heart_rate"] = new_heart_rate

        # Log data periodically for continuous charting (every 2-3 seconds or on state change)
        # Initialize last_log_time if not exists
        if 'last_log_time' not in st.session_state:
            st.session_state.last_log_time = datetime.now()
        
        current_time = datetime.now()
        time_since_last_log = (current_time - st.session_state.last_log_time).total_seconds()
        
        # Log if: state changed OR 3 seconds have passed (for continuous charting)
        should_log = False
        if st.session_state.df.empty or len(st.session_state.df) == 0:
            should_log = True  # Always log first entry
        else:
            try:
                last_state = st.session_state.df.iloc[-1]['final_state']
                state_changed = new_final_state != last_state
                time_elapsed = time_since_last_log >= DATA_LOG_INTERVAL_SECONDS  # Log every N seconds
                should_log = state_changed or time_elapsed
            except (IndexError, KeyError):
                should_log = True  # Log if we can't check previous state
        
        # Thread-safe DataFrame update
        if should_log:
            try:
                # Limit DataFrame size to prevent memory issues
                try:
                    from config import MAX_HISTORY_ROWS
                except ImportError:
                    MAX_HISTORY_ROWS = 1000
                
                new_row = {
                    "time": current_time,
                    "final_state": new_final_state,
                    "mock_heart_rate": new_heart_rate,
                    "face_emotion": st.session_state.current_state["face_emotion"],
                    "voice_emotion": st.session_state.current_state["voice_emotion"],
                    "is_sleeping": st.session_state.current_state["is_sleeping"]
                }
                # Use pd.concat for efficiency with error handling
                try:
                    st.session_state.df = pd.concat([
                        st.session_state.df,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    # Keep only last N rows to prevent memory bloat
                    if len(st.session_state.df) > MAX_HISTORY_ROWS:
                        st.session_state.df = st.session_state.df.tail(MAX_HISTORY_ROWS).reset_index(drop=True)
                    st.session_state.last_log_time = current_time
                except Exception:
                    # If DataFrame operations fail, fallback to creating a new DataFrame
                    logging.getLogger(__name__).exception("Error updating session DataFrame; resetting to new DataFrame")
                    st.session_state.df = pd.DataFrame([new_row])
                    st.session_state.last_log_time = current_time
            except (IndexError, KeyError):
                # Handle case where df is empty or malformed: initialize with new row
                logging.getLogger(__name__).debug("Session DataFrame malformed or empty; initializing new DataFrame")
                st.session_state.df = pd.DataFrame([{
                    "time": current_time,
                    "final_state": new_final_state,
                    "mock_heart_rate": new_heart_rate,
                    "face_emotion": st.session_state.current_state["face_emotion"],
                    "voice_emotion": st.session_state.current_state["voice_emotion"],
                    "is_sleeping": st.session_state.current_state["is_sleeping"]
                }])
                st.session_state.last_log_time = current_time
        

        # --- Build Dashboard ---
        current_display_state = st.session_state.current_state["final_state"]
        current_display_hr = st.session_state.current_state["mock_heart_rate"]
        current_face = st.session_state.current_state.get("face_emotion", "N/A")
        current_voice = st.session_state.current_state.get("voice_emotion", "N/A")
        
        # Display current state prominently
        st.metric("Current Emotional State", current_display_state)
        st.metric("Simulated Heart Rate", f"{int(current_display_hr)} BPM")
        
        # Show sensor status for debugging
        with st.expander("🔍 Sensor Status & Diagnostics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                face_status = "✅ Detecting" if current_face != "N/A" else "⏳ Waiting..."
                st.write(f"**Face Emotion:** {current_face} ({face_status})")
                
                # Show debug info for face detection
                face_debug = st.session_state.current_state.get("face_detection_debug", "")
                if face_debug:
                    if not MINIMAL_UI:
                        st.warning(f"⚠️ {face_debug}")
                        st.caption("💡 **Troubleshooting**: If face is detected but emotion is N/A:")
                        st.caption("   1. Ensure good lighting and face visibility")
                        st.caption("   2. Face should be clearly visible in webcam")
                        st.caption("   3. DeepFace may need more time to analyze")
                
                # Check if webcam is active
                if current_face == "N/A" and not face_debug:
                    st.info("💡 **Tip**: Make sure webcam is started and your face is visible")
                # Show NA reason counters if available (diagnostics)
                na_reasons_snapshot = st.session_state.current_state.get('na_reasons', {})
                if not MINIMAL_UI and na_reasons_snapshot and isinstance(na_reasons_snapshot, dict):
                    st.subheader("Diagnostics: N/A Reasons")
                    try:
                        # Show sorted counts for readability
                        sorted_items = sorted(na_reasons_snapshot.items(), key=lambda x: -x[1])
                        for k, v in sorted_items:
                            st.write(f"• {k}: {v}")
                    except Exception:
                        st.write(na_reasons_snapshot)
                    
            with col2:
                if st.session_state.voice_model is None or st.session_state.voice_encoder is None:
                    voice_status = "❌ Model Unavailable"
                    st.write(f"**Voice Emotion:** {current_voice} ({voice_status})")
                    st.caption("Voice analysis requires trained model. Face-only detection is active.")
                else:
                    voice_status = "✅ Detecting" if current_voice != "N/A" else "⏳ Listening..."
                    st.write(f"**Voice Emotion:** {current_voice} ({voice_status})")
                    
            # Additional diagnostics
            if current_face == "N/A" and current_display_state == "Calibrating":
                st.info("📋 **System Status**: Waiting for face detection. Please ensure:")
                st.caption("   • Webcam is active (click START button)")
                st.caption("   • Face is clearly visible in camera")
                st.caption("   • Good lighting conditions")
                st.caption("   • DeepFace models are loaded (check startup messages)")
        
        if current_display_state in ["Anxiety", "Agitated", "Sadness", "Stress", "Sleeping"]:
            # Only show alert if it's a *new* state or different from the last alert
            if current_display_state != st.session_state.current_state.get("last_alert"):
                feedback = get_personalized_feedback(current_display_state)
                
                if current_display_state == "Sleeping":
                    st.info(f"**Personalized Feedback:**\n{feedback}", icon="😴")
                elif current_display_state == "Stress" or current_display_state == "Sadness":
                    st.warning(f"**Early Alert:** A level of **{current_display_state}** has been detected.", icon="⚠️")
                    st.info(f"**Personalized Feedback:**\n{feedback}")
                else: # Anxiety, Agitated
                    st.error(f"**Early Alert:** A high level of **{current_display_state}** has been detected.", icon="🚨")
                    st.info(f"**Personalized Feedback:**\n{feedback}")
                    
                st.caption("This is a supportive tool, not a medical diagnosis. If you are feeling overwhelmed, please speak to a healthcare professional.")
                st.session_state.current_state["last_alert"] = current_display_state
        else:
             st.session_state.current_state["last_alert"] = None
        
        st.caption("Disclaimer: This tool is a prototype and not a medical device. Always consult with a healthcare professional for medical advice.")

        st.subheader("Heart Rate Trend")
        chart_placeholder = st.empty()
        # Continuous time series chart - accumulate all historical data
        try:
            if st.session_state.df.empty or len(st.session_state.df) == 0:
                # Show initial point if no data yet
                initial_chart_data = pd.DataFrame({
                    'time': [datetime.now()], 
                    'mock_heart_rate': [current_display_hr]
                })
                if len(initial_chart_data) > 0:
                    chart_data = initial_chart_data.set_index('time')['mock_heart_rate']
                    chart_placeholder.line_chart(chart_data, use_container_width=True)
                else:
                    chart_placeholder.info("Collecting data...")
            else:
                # Get all historical data (or last N points for performance)
                chart_df = st.session_state.df.tail(MAX_CHART_DATA_POINTS).copy()
                
                # Ensure 'time' is datetime and 'mock_heart_rate' exists
                if 'time' in chart_df.columns:
                    chart_df['time'] = pd.to_datetime(chart_df['time'])
                else:
                    chart_df['time'] = pd.to_datetime([datetime.now()] * len(chart_df))
                    
                if 'mock_heart_rate' not in chart_df.columns:
                    chart_df['mock_heart_rate'] = current_display_hr
                
                # Sort by time to ensure proper time series line plotting
                chart_df = chart_df.sort_values('time')
                
                # Prepare chart data - use time as index for proper time series
                chart_data = chart_df.set_index('time')['mock_heart_rate']
                
                # Create continuous line chart with all historical data
                if len(chart_data) > 0:
                    chart_placeholder.line_chart(chart_data, use_container_width=True)
                else:
                    chart_placeholder.info("Collecting data...")
        except Exception:
            logging.getLogger(__name__).exception("Error creating chart")
            chart_placeholder.warning("Unable to display chart at this time.")

        st.subheader("Recent Detections Log")
        try:
            if not st.session_state.df.empty and len(st.session_state.df) > 0:
                try:
                    from config import RECENT_DETECTIONS_COUNT
                except ImportError:
                    RECENT_DETECTIONS_COUNT = 5
                display_df = st.session_state.df.tail(RECENT_DETECTIONS_COUNT).copy()
                if 'time' in display_df.columns:
                    try:
                        styled_df = display_df.style.format({'time': lambda t: t.strftime("%H:%M:%S") if isinstance(t, datetime) else str(t)})
                        st.dataframe(styled_df, width='stretch', use_container_width=True)
                    except Exception:
                        # Fallback: display without styling
                        st.dataframe(display_df, width='stretch', use_container_width=True)
                else:
                    st.dataframe(display_df, width='stretch', use_container_width=True)
            else:
                st.info("No detection data logged yet.")
        except Exception:
            logging.getLogger(__name__).exception("Error displaying detection log")
            if not MINIMAL_UI:
                st.warning("Unable to display detection log at this time.")
            
    # Legacy automatic PyAudio background thread removed. Browser audio via the UI is the supported path.

if __name__ == "__main__":
    main()


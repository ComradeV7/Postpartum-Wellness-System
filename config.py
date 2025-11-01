"""
Configuration file for Postpartum Wellness Monitoring System.
All thresholds and parameters can be customized here or via environment variables.
"""
import os

# --- Sensor Configuration ---
# Fatigue/Sleep Detection
EAR_THRESHOLD = float(os.getenv('EAR_THRESHOLD', 0.23))  # Eye Aspect Ratio threshold for blink detection
EAR_CONSEC_FRAMES = int(os.getenv('EAR_CONSEC_FRAMES', 45))  # Frames for sleep detection (~2-3 seconds)

# DeepFace Emotion Detection
DEEPFACE_FRAME_SKIP = int(os.getenv('DEEPFACE_FRAME_SKIP', 5))  # Analyze emotion every N frames
DEEPFACE_BACKEND = os.getenv('DEEPFACE_BACKEND', 'opencv')  # Backend: opencv, ssd, mtcnn

# Audio Processing
VAD_AGGRESSIVENESS = int(os.getenv('VAD_AGGRESSIVENESS', 3))  # Voice Activity Detection: 0-3 (3 = most aggressive)
VAD_FRAME_MS = int(os.getenv('VAD_FRAME_MS', 30))  # Frame size in milliseconds
VAD_RATE = int(os.getenv('VAD_RATE', 16000))  # Sample rate for VAD
VAD_SILENCE_FRAMES = int(os.getenv('VAD_SILENCE_FRAMES', 30))  # Frames of silence to detect speech end

# Model Paths
VOICE_MODEL_PATH = os.getenv('VOICE_MODEL_PATH', 'voice_model.h5')
VOICE_ENCODER_PATH = os.getenv('VOICE_ENCODER_PATH', 'voice_emotion_encoder.joblib')

# --- Data Logging Configuration ---
DATA_LOG_INTERVAL_SECONDS = float(os.getenv('DATA_LOG_INTERVAL_SECONDS', 3.0))  # Log data every N seconds
MAX_CHART_DATA_POINTS = int(os.getenv('MAX_CHART_DATA_POINTS', 500))  # Maximum points in chart
MAX_HISTORY_ROWS = int(os.getenv('MAX_HISTORY_ROWS', 1000))  # Maximum rows to keep in memory

# --- State Detection Configuration ---
# Emotion state priorities and thresholds
STRESS_INDICATORS = ['Sad', 'Anxious', 'Fear']
AGITATION_INDICATORS = ['Angry']
CALM_INDICATORS = ['Neutral', 'Happy', 'Calm', 'Surprise']

# Heart Rate Ranges (BPM)
HEART_RATE_RANGES = {
    'Anxiety': (100, 125),
    'Agitated': (95, 115),
    'Stress': (85, 100),
    'Sadness': (80, 95),
    'Sleeping': (50, 65),
    'Calmness': (60, 75),
    'Calibrating': (70, 80)
}

# --- MediaPipe Configuration ---
# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# --- Dashboard Configuration ---
AUTOREFRESH_INTERVAL_MS = int(os.getenv('AUTOREFRESH_INTERVAL_MS', 2000))  # Dashboard refresh interval
RECENT_DETECTIONS_COUNT = int(os.getenv('RECENT_DETECTIONS_COUNT', 5))  # Number of recent detections to show


# System Architecture Documentation

## Overview

The Postpartum Wellness Monitoring System is a real-time, multimodal AI application that monitors emotional and physical states through video and audio analysis. The system uses federated learning for privacy-preserving model improvement.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Video Stream │  │ Audio Thread│  │ FL Client    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │               │
└─────────┼─────────────────┼──────────────────┼───────────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Video Processor │ │Audio Processor│ │ FL Client Logic  │
├─────────────────┤ ├──────────────┤ ├──────────────────┤
│ MediaPipe       │ │ Browser Audio │ │ Data Preprocessing│
│ DeepFace        │ │ streamlit-webrtc VAD │ │ Model Training    │
│ EAR Calculation │ │ Voice Model   │ │ Parameter Sync    │
└────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Data Queue      │
                │  (Thread-Safe)   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ State Fusion     │
                │ & Dashboard      │
                └──────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐    ┌──────────┐    ┌─────────┐
    │ Alerts │    │ Feedback │    │ Logging │
    └────────┘    └──────────┘    └─────────┘
```

## Component Details

### 1. Video Processing Pipeline

**Components:**

- **MediaPipe FaceMesh**: Extracts 468 facial landmarks
- **Eye Aspect Ratio (EAR)**: Calculates eye openness for sleep detection
- **DeepFace**: Analyzes facial emotion (every 5th frame for performance)

**Flow:**

1. Video frame received from WebRTC stream
2. Convert to RGB format
3. Extract face landmarks using MediaPipe
4. Calculate EAR from eye landmarks
5. Detect emotion using DeepFace (with frame skipping)
6. Send results to data queue

**Landmarks Used:**

- Left Eye: Indices [33, 160, 158, 133, 153, 144]
- Right Eye: Indices [362, 385, 387, 263, 373, 380]

### 2. Audio Processing Pipeline

**Components:**

- **Browser-captured audio (streamlit-webrtc)**: Captures microphone audio from the user's browser session
- **Simple VAD / energy-based detection**: Detects speech segments client-side or server-side depending on configuration
- **Voice Emotion Model**: CNN-based emotion classifier (if available)

**Flow:**

1. Browser captures microphone audio via WebRTC and forwards audio frames to the Streamlit app
2. Simple VAD (energy threshold) identifies speech segments
3. Buffer speech frames until silence detected
4. Convert audio to features (MFCC, Chroma, Mel-Spectrogram)
5. Classify emotion using trained model
6. Send results to data queue

**Optimizations:**

- Buffer size limits to prevent memory issues
- Non-blocking queue operations
- Browser-based capture eliminates native audio dependency issues (PyAudio)

### 3. State Fusion Engine

**Input:**

- Face emotion (from video)
- Voice emotion (from audio)
- Sleep status (from EAR)

**Processing:**

- Priority-based state determination
- Confidence-based fusion
- Temporal consistency checks

**Output States:**

- Sleeping (Priority 1)
- Anxiety (Priority 2)
- Agitated (Priority 3)
- Sadness (Priority 4)
- Stress (Priority 5)
- Calmness (Priority 6)
- Calibrating (Default)

### 4. Federated Learning Client

**Flow:**

1. Collect local session data
2. Preprocess into feature vectors
3. Connect to FL server
4. Receive global model parameters
5. Train locally on user data
6. Send updated parameters back
7. Server aggregates from all clients

**Features:**

- Privacy-preserving (no raw data leaves device)
- Only model parameters are shared
- Supports multiple clients contributing to global model

## Data Flow

### Real-Time Processing

```
Video Frame → MediaPipe → EAR Calc → DeepFace → Queue
                                      ↓
                                   Dashboard ← Queue ← Audio Thread
                                                           ↑
                                    Microphone → VAD → Voice Model
```

### Federated Learning Flow

```
Local Data → Preprocessing → Local Training → Parameters
                                              ↓
                            FL Server ← Parameters
                                              ↓
                            Aggregation → Global Model
                                              ↓
                            Parameters → All Clients
```

## Thread Safety

### Critical Sections

1. **Data Queue**: Thread-safe queue for inter-thread communication

   - Video thread → Queue (non-blocking)
   - Audio thread → Queue (non-blocking)
   - Dashboard thread ← Queue (with iteration limits)

2. **Session State**: Streamlit handles session state thread safety

   - Access controlled by Streamlit's session mechanism

3. **DataFrame Updates**: Protected with try-except blocks
   - Fallback mechanisms for edge cases

## Error Handling Strategy

### Levels of Error Handling

1. **Model Loading**: Graceful degradation if models unavailable
2. **Audio Device**: Retry with backoff if device unavailable
3. **Queue Operations**: Non-blocking with timeout protection
4. **DataFrame Operations**: Multiple fallback strategies
5. **FL Client**: Error messages returned via queue

## Performance Optimizations

1. **Frame Skipping**: DeepFace analyzes every 5th frame
2. **Buffer Limits**: Audio buffer size capped
3. **Queue Limits**: Max iterations in queue processing
4. **Model Caching**: Models loaded once and reused
5. **Lazy Evaluation**: Features computed only when needed

## Scalability Considerations

### Current Limitations

- Single-user session (Streamlit session state)
- Synchronous model loading
- Limited concurrent FL clients

### Future Improvements

- Database backend for persistent storage
- Async model loading
- Horizontal scaling for FL server
- Multi-user support with authentication

## Security Considerations

1. **Local Processing**: All AI processing happens locally
2. **No Cloud Storage**: Data not sent to external servers
3. **FL Privacy**: Only model parameters shared, not raw data
4. **Session Isolation**: Streamlit provides session-level isolation

## Monitoring & Logging

- Console logging for debugging
- Error messages displayed in UI
- Queue status monitoring
- Model loading status indicators

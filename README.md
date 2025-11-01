# Postpartum Wellness Monitoring System

A comprehensive real-time monitoring system for postpartum wellness that uses multimodal AI (facial expression analysis, voice emotion detection, and sleep/fatigue detection) to provide personalized feedback and early alerts.

## Architecture Overview

### System Components

1. **Main Application (`app.py`)**: Streamlit-based real-time monitoring dashboard

   - Video processing: MediaPipe FaceMesh for facial landmarks, DeepFace for emotion detection
   - Audio processing: Browser-captured audio via streamlit-webrtc for voice activity detection and emotion analysis
   - Real-time dashboard with alerts and personalized feedback
   - Federated Learning client integration for privacy-preserving model improvement

2. **Federated Learning Server (`server.py`)**: Flower server for federated learning

   - Aggregates model updates from multiple clients without accessing raw data
   - Uses Federated Averaging (FedAvg) strategy

3. **Dummy Client (`dummy_client.py`)**: Test client for federated learning system

### Key Features

- **Multimodal Emotion Detection**:

  - Facial emotion recognition using DeepFace
  - Voice emotion analysis using trained CNN model
  - Sleep/fatigue detection using Eye Aspect Ratio (EAR)

- **Real-Time Monitoring**:

  - Live video feed with overlay annotations
  - Continuous audio processing
  - Real-time dashboard updates

- **Intelligent State Fusion**:

  - Combines multiple sensor inputs to determine wellness state
  - Provides personalized feedback based on detected states

- **Privacy-Preserving Learning**:
  - Federated Learning integration for model improvement without data sharing
  - Client-side training on local data

## Installation

1. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Install system dependencies** (if needed):

   - FFmpeg (for audio/video processing): Install from https://ffmpeg.org/

3. **Voice Model Files** (Optional):
   - Place `voice_model.h5` and `voice_emotion_encoder.joblib` in the project root
   - If missing, voice emotion analysis will be unavailable but other features will work

## Usage

### Running the Main Application

```bash
streamlit run app.py
```

The application will:

- Load AI models (MediaPipe, DeepFace, and optionally voice model)
- Start video processing when you click "START"
- Begin passive audio listening in the background
- Display real-time dashboard with wellness metrics

### Running Federated Learning Server

In a separate terminal:

```bash
python server.py
```

The server will:

- Wait for at least 2 clients to connect
- Run 3 rounds of federated training
- Aggregate model updates from all connected clients

### Running Test Client (Optional)

In another terminal:

```bash
python dummy_client.py
```

This will connect as a test client to the FL server.

## System Requirements

- Python 3.8+
- Webcam for video input
- Microphone for audio input (optional)
- Modern web browser
- Minimum 4GB RAM recommended
- GPU optional but recommended for faster processing

## Troubleshooting

### Common Issues

1. **Audio not working**:

   - Check microphone permissions
   - Verify audio device is available
   - Ensure PortAudio is installed

2. **Video not showing**:

   - Grant camera permissions
   - Check browser compatibility
   - Try different browser

3. **Model loading errors**:

   - Ensure all dependencies are installed
   - Check available RAM/disk space
   - DeepFace may download models on first run

4. **FL Server connection errors**:
   - Ensure server is running before starting client
   - Check firewall settings for port 8080
   - Verify server address matches client configuration

## Technical Details

### Eye Aspect Ratio (EAR) Calculation

- Uses standard MediaPipe FaceMesh landmarks
- 6-point EAR formula for each eye
- Threshold: 0.23 (adjustable)
- Sleep detection: 45 consecutive frames below threshold

### Emotion States

- **Anxiety**: High confidence when face and voice both indicate anxious state
- **Agitated**: Detected from angry face or voice
- **Sadness**: High confidence with both face and voice
- **Stress**: Medium confidence from mixed signals
- **Calmness**: Neutral/happy states from sensors
- **Sleeping**: Detected from sustained closed eyes

### Performance Optimizations

- DeepFace frame skipping (every 5th frame)
- Audio buffer size limits
- Queue processing with iteration limits
- Thread-safe data handling

## Security & Privacy

- All processing happens locally
- No data is sent to external servers (except when explicitly using FL)
- Federated Learning preserves privacy by only sharing model updates
- All audio/video processing is done in-memory

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

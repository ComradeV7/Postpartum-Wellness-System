# Deployment Guide - Postpartum Wellness Monitoring System

## Overview

This is a fully functional, deployable wellness monitoring system with no hardcoded values. All parameters are configurable via `config.py` or environment variables.

## Key Features

### ✅ Continuous Time Series Charting

- Heart rate chart accumulates data over time (like SPA)
- Logs data every 3 seconds (configurable)
- Shows complete historical trend, not just single points
- Automatic memory management (keeps last 500-1000 points)

### ✅ Intelligent State Detection

- **Not stuck on "Calibrating"**: Works with partial sensor data
- Progressive state inference from available data
- Handles cases where only face OR voice data is available
- Smart fusion logic that makes educated guesses rather than requiring all sensors

### ✅ Fully Configurable

- All thresholds in `config.py`
- No hardcoded values in main code
- Environment variable support
- Easy to customize for different deployment scenarios

## Configuration

### Using config.py (Recommended)

Edit `config.py` to customize:

```python
# Adjust detection sensitivity
EAR_THRESHOLD = 0.23  # Lower = more sensitive to closed eyes
EAR_CONSEC_FRAMES = 45  # Frames before detecting sleep

# Data logging
DATA_LOG_INTERVAL_SECONDS = 3.0  # How often to log data
MAX_CHART_DATA_POINTS = 500  # Max points in chart

# Heart rate ranges
HEART_RATE_RANGES = {
    'Anxiety': (100, 125),
    'Calmness': (60, 75),
    # ... customize all ranges
}
```

### Using Environment Variables

```bash
export EAR_THRESHOLD=0.25
export DATA_LOG_INTERVAL_SECONDS=2.0
export MAX_CHART_DATA_POINTS=1000
streamlit run app.py
```

## How It Works

### State Detection Logic

1. **Sleeping** (Priority 1): Detected via EAR (Eye Aspect Ratio)
2. **Anxiety** (Priority 2): Face shows anxious/fear OR voice is anxious
3. **Agitated** (Priority 3): Face OR voice shows anger
4. **Sadness** (Priority 4): Both agree OR single strong signal
5. **Stress** (Priority 5): Mixed signals from sensors
6. **Calmness** (Priority 6): Neutral/positive emotions
7. **Calibrating** (Last Resort): Only if truly no sensor data

**Key Improvement**: System now makes inferences even with partial data, preventing "Calibrating" lock.

### Data Logging

- **Periodic Logging**: Logs data every 3 seconds (configurable) for continuous charting
- **State Change Logging**: Also logs immediately when state changes
- **Memory Efficient**: Automatically prunes old data beyond configured limit

### Chart Behavior

- **Time Series**: Accumulates all historical data points
- **Continuous Plot**: Shows line connecting all points over time
- **Performance**: Limits to last N points (default 500) for smooth rendering
- **Auto-Update**: Refreshes every 2 seconds with new data

## Deployment Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure System

Edit `config.py` or set environment variables for your deployment needs.

### 3. Optional: Prepare Voice Model

If you have trained voice emotion models:

- Place `voice_model.h5` in project root
- Place `voice_emotion_encoder.joblib` in project root

If models are missing, system works without voice analysis.

### 4. Run Application

```bash
streamlit run app.py
```

### 5. Access Dashboard

- Open browser to `http://localhost:8501`
- Click "START" to activate webcam
- System begins monitoring automatically

## Production Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard
4. Deploy

### Local Production

```bash
# Run with custom config
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Monitoring & Maintenance

### Performance

- **Memory Usage**: System automatically manages DataFrame size
- **CPU Usage**: Frame skipping reduces DeepFace load
- **Storage**: In-memory only (add database for persistence if needed)

### Troubleshooting

1. **Stuck on "Calibrating"**:

   - Check webcam permissions
   - Verify face is visible
   - System should still work with partial data

2. **Chart not updating**:

   - Check data is being logged (every 3 seconds)
   - Verify DataFrame is accumulating

3. **High CPU**:
   - Increase `DEEPFACE_FRAME_SKIP` in config.py
   - Reduce `DATA_LOG_INTERVAL_SECONDS`

## Customization

All parameters in `config.py`:

- Detection thresholds
- Data logging intervals
- Chart display limits
- Heart rate ranges
- MediaPipe settings
- Audio processing parameters

## Next Steps

For production use:

1. Add database persistence (PostgreSQL, MongoDB)
2. Add user authentication
3. Add data export functionality
4. Add alert notifications (email, SMS)
5. Add multi-user support
6. Add API endpoints

## Support

See `README.md` for general information and `ARCHITECTURE.md` for system design details.

import numpy as np
from audio_features import extract_features, FEATURE_VECTOR_LENGTH


def test_extract_features_sine_wave():
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave at 440Hz
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    features = extract_features(y, sample_rate=sr)
    assert features is not None, "extract_features returned None for valid sine wave"
    assert features.shape[0] == FEATURE_VECTOR_LENGTH, f"Expected feature length {FEATURE_VECTOR_LENGTH}, got {features.shape[0]}"

import numpy as np
import librosa
import logging

# Public constants
FEATURE_VECTOR_LENGTH = 180
MIN_SAMPLES = 2048

logger = logging.getLogger(__name__)


def extract_features(audio_data, sample_rate=22050):
    """Extract MFCC, chroma, and mel features from audio_data.

    Returns a 1D numpy array of length FEATURE_VECTOR_LENGTH, or None on failure.
    This function is intentionally defensive: it logs detailed errors and returns None
    instead of raising so callers (training or runtime) can handle missing data.
    """
    try:
        if audio_data is None:
            logger.debug("extract_features: audio_data is None")
            return None

        # Ensure numpy array
        audio = np.asarray(audio_data, dtype=np.float32)

        if audio.size < MIN_SAMPLES:
            logger.debug(f"extract_features: audio too short ({audio.size} < {MIN_SAMPLES})")
            return None

        # MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)

        # Chroma
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)

        # Mel-spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

        features = np.hstack((mfccs, chroma, mel))

        if features.ndim != 1:
            logger.warning(f"extract_features: unexpected feature ndim {features.ndim}")
            return None

        if features.shape[0] != FEATURE_VECTOR_LENGTH:
            logger.warning(f"extract_features: feature length {features.shape[0]} != {FEATURE_VECTOR_LENGTH}")
            return None

        return features

    except Exception as e:
        logger.exception(f"Error extracting features: {e}")
        return None

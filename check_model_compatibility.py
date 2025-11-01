"""
Quick model compatibility checker.
Run this script to validate that the voice model and encoder can be loaded with the
current Python environment and installed packages. It prints diagnostic information
and suggestions if loading fails.

Usage:
    python check_model_compatibility.py

This script does NOT modify or delete models; it only attempts to load them and
reports errors to help you choose a remediation path.
"""
import os
import sys
import traceback

try:
    import joblib
except Exception:
    joblib = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import importlib.metadata as importlib_metadata
except Exception:
    import importlib_metadata

# Default paths (match app.py defaults)
VOICE_MODEL_PATH = os.environ.get('VOICE_MODEL_PATH', 'voice_model.h5')
VOICE_ENCODER_PATH = os.environ.get('VOICE_ENCODER_PATH', 'voice_emotion_encoder.joblib')


def print_pkg_version(name):
    try:
        v = importlib_metadata.version(name)
        print(f"{name}: {v}")
    except Exception:
        print(f"{name}: (not installed or version unknown)")


def main():
    print("== Environment versions ==")
    print_pkg_version('python')
    print_pkg_version('tensorflow')
    print_pkg_version('deepface')
    print_pkg_version('librosa')
    print_pkg_version('scikit-learn')
    print_pkg_version('joblib')
    print('')

    # Check TensorFlow availability
    if tf is None:
        print('TensorFlow is not importable. Please install the version used to train your model.')
    else:
        print(f"TensorFlow import OK, version: {tf.__version__}")

    # Check voice model
    print('\n== Voice model check ==')
    if not os.path.exists(VOICE_MODEL_PATH):
        print(f"Voice model not found at: {VOICE_MODEL_PATH}\nPlease train or place the model file there (see train_voice_model.py).")
    else:
        print(f"Found voice model at {VOICE_MODEL_PATH}. Attempting to load with compile=False...")
        try:
            model = None
            if tf is None:
                raise RuntimeError('TensorFlow not available to load model')
            model = tf.keras.models.load_model(VOICE_MODEL_PATH, compile=False)
            print('Successfully loaded model with compile=False')
            try:
                model.compile(loss='categorical_crossentropy', optimizer='adam')
                print('Model compiled successfully (basic check)')
            except Exception as e:
                print(f'Warning: model.compile() raised: {e}')
        except Exception as e:
            print('Failed to load voice model with compile=False:')
            traceback.print_exc()
            print('\nPossible causes: model saved with a different TF/Keras version, or file is corrupted.')
            print('Suggested actions:')
            print(' - Ensure your Python env has the same TensorFlow version used for training')
            print(' - If you have the original training environment, re-save the model in SavedModel format:')
            print('     model.save("my_model_saved", save_format="tf")')
            print(' - If that is not possible, consider retraining the model in the current environment')

    # Check encoder
    print('\n== Voice encoder check ==')
    if joblib is None:
        print('joblib not importable; cannot load encoder. Install joblib.')
    else:
        if not os.path.exists(VOICE_ENCODER_PATH):
            print(f'Encoder file not found at: {VOICE_ENCODER_PATH}')
        else:
            print(f'Found encoder at {VOICE_ENCODER_PATH}. Attempting to load...')
            try:
                enc = joblib.load(VOICE_ENCODER_PATH)
                print('Successfully loaded encoder. Encoder object type:', type(enc))
                if hasattr(enc, 'classes_'):
                    print('Encoder classes:', getattr(enc, 'classes_'))
            except Exception:
                print('Failed to load encoder:')
                traceback.print_exc()

    print('\nFinished compatibility check.')


if __name__ == '__main__':
    main()
